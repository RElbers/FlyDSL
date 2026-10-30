# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile many kernel specializations into the JIT disk cache in parallel worker processes.

The workers are fresh interpreters that only import FlyDSL and the kernels' modules, so they never run
the caller's script, and they stay up between :func:`precompile` calls. A later call to a compiled
specialization then loads it from the disk cache instead of compiling it.
"""

import atexit
import functools
import importlib
import os
import pickle
import queue
import subprocess
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..utils import env

try:
    import torch
except ImportError:
    torch = None


@dataclass
class CompileJob:
    """One specialization to compile: ``fn(*args, **kwargs)`` under ``compile_hints``.

    ``fn`` is a ``@flyc.jit`` function, or, with ``factory_kwargs``, a factory that ``fn(**factory_kwargs)``
    turns into one. Either must be importable by its module and qualified name.
    """

    fn: Any
    args: Tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    compile_hints: Dict[str, Any] = field(default_factory=dict)
    factory_kwargs: Optional[Dict[str, Any]] = None


def precompile(jobs: List[CompileJob], workers: int = 8) -> List[Optional[str]]:
    """Compile ``jobs`` in up to ``workers`` worker processes into the JIT disk cache.

    Returns one entry per job: None when it compiled, else why not. A job that cannot be sent to a worker
    (its function is not importable by name, or an argument does not pickle) is reported, not compiled.
    Tensor arguments are rebuilt as CPU tensors of the same shape, strides and dtype and streams as
    ``fx.Stream(None)``, which specialize a kernel exactly as the originals do; other arguments are
    sent pickled.
    """
    if not env.runtime.enable_cache:
        raise RuntimeError("precompile needs the JIT disk cache (FLYDSL_RUNTIME_ENABLE_CACHE)")
    results: List[Optional[str]] = [None] * len(jobs)
    tasks = queue.Queue()
    for index, job in enumerate(jobs):
        try:
            tasks.put((index, pickle.dumps(_encode_job(job))))
        except (TypeError, AttributeError, ImportError, pickle.PicklingError) as error:
            results[index] = f"cannot send to a compile worker: {error}"
    if tasks.empty():
        return results
    with _pool_lock:
        _pool.run(tasks, min(workers, tasks.qsize()), results)
    return results


# --------------------------------------------------------------------------- #
# Job encoding
# --------------------------------------------------------------------------- #
def _importable_name(obj):
    module, qualname = getattr(obj, "__module__", None), getattr(obj, "__qualname__", "")
    source = getattr(obj, "func", None)  # a @flyc.jit function names its wrapped function
    if source is not None:
        module, qualname = source.__module__, source.__qualname__
    if not module or not qualname or "<locals>" in qualname:
        raise TypeError(f"{obj!r} is not defined at module level")
    if module == "__main__":
        raise TypeError(f"{qualname} is defined in the main script, which compile workers do not run")
    if _resolve(module, qualname) is not obj:
        raise TypeError(f"{module}.{qualname} is not {obj!r}")
    return module, qualname


def _resolve(module, qualname):
    return functools.reduce(getattr, qualname.split("."), importlib.import_module(module))


def _encode_job(job):
    factory_kwargs = None
    if job.factory_kwargs is not None:
        factory_kwargs = {name: _worker_value(value) for name, value in job.factory_kwargs.items()}
    return (
        _importable_name(job.fn),
        factory_kwargs,
        [_worker_value(value) for value in job.args],
        {name: _worker_value(value) for name, value in job.kwargs.items()},
        dict(job.compile_hints),
    )


def _worker_value(value):
    """``value`` in a form a compile worker can rebuild, or raise TypeError.

    A value that does not round-trip just misses the disk cache later and compiles there.
    """
    if torch is not None and isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), tuple(value.stride()), value.dtype)
    if torch is not None and isinstance(value, torch.cuda.Stream):
        return ("stream",)
    if isinstance(value, (tuple, list)):
        return ("sequence", type(value), [_worker_value(item) for item in value])
    try:
        pickle.dumps(value)
    except Exception as error:
        raise TypeError(f"{type(value).__name__} does not pickle: {error}") from error
    return ("value", value)


def _rebuild_value(spec):
    kind = spec[0]
    if kind == "tensor":
        _, shape, stride, dtype = spec
        return torch.empty_strided(shape, stride, dtype=dtype, device="cpu")
    if kind == "stream":
        from ..expr.typing import Stream

        return Stream(None)
    if kind == "sequence":
        return spec[1](_rebuild_value(item) for item in spec[2])
    return spec[1]


# --------------------------------------------------------------------------- #
# Worker side
# --------------------------------------------------------------------------- #
def _compile_encoded(encoded):
    (module, qualname), factory_kwargs, args, kwargs, compile_hints = encoded
    try:
        fn = _resolve(module, qualname)
        if factory_kwargs is not None:
            fn = fn(**{name: _rebuild_value(spec) for name, spec in factory_kwargs.items()})
        if compile_hints:
            from .kernel_function import CompilationContext

            hints = CompilationContext.compile_hints(compile_hints)
        else:
            hints = nullcontext()
        with hints:
            fn._compile_only(
                *[_rebuild_value(spec) for spec in args],
                **{name: _rebuild_value(spec) for name, spec in kwargs.items()},
            )
        return None
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def _worker_main():
    """Entry point of a compile worker: compile the jobs read from stdin until it closes.

    Results go back on a private copy of stdout; stdout itself is pointed at stderr, so whatever a
    compile prints cannot corrupt them.
    """
    results = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    while True:
        try:
            encoded = pickle.load(sys.stdin.buffer)
        except EOFError:
            return
        pickle.dump(_compile_encoded(encoded), results)
        results.flush()


# --------------------------------------------------------------------------- #
# Parent side: the worker pool
# --------------------------------------------------------------------------- #
_WORKER_COMMAND = [sys.executable, "-c", "from flydsl.compiler.precompile import _worker_main; _worker_main()"]


class _WorkerPool:
    """Long-lived compile workers, restarted when the environment or import path they started with changes."""

    def __init__(self):
        self.workers = []
        self.started_with = None

    def run(self, tasks, count, results):
        # Workers read the cache directory and every other setting from the environment they start with.
        started_with = (dict(os.environ), list(sys.path))
        if started_with != self.started_with:
            self.close()
            self.started_with = started_with
        while len(self.workers) < count:
            worker_env = dict(os.environ, PYTHONPATH=os.pathsep.join(path for path in sys.path if path))
            self.workers.append(
                subprocess.Popen(_WORKER_COMMAND, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=worker_env)
            )
        drivers = [
            threading.Thread(target=self._drive, args=(worker, tasks, results)) for worker in self.workers[:count]
        ]
        for driver in drivers:
            driver.start()
        for driver in drivers:
            driver.join()
        self.workers = [worker for worker in self.workers if worker.poll() is None]
        while not tasks.empty():
            results[tasks.get_nowait()[0]] = "no compile worker was left to compile it"

    @staticmethod
    def _drive(worker, tasks, results):
        while True:
            try:
                index, encoded = tasks.get_nowait()
            except queue.Empty:
                return
            try:
                worker.stdin.write(encoded)
                worker.stdin.flush()
                results[index] = pickle.load(worker.stdout)
            except (EOFError, OSError, pickle.UnpicklingError):
                results[index] = "the compile worker exited"
                worker.kill()
                return

    def close(self):
        for worker in self.workers:
            try:
                worker.stdin.close()
            except OSError:
                pass
        for worker in self.workers:
            worker.wait()
        self.workers = []


_pool = _WorkerPool()
_pool_lock = threading.Lock()
atexit.register(_pool.close)
