# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile many kernel specializations into the JIT disk cache in parallel worker processes.

The workers are fresh interpreters that only import FlyDSL and the kernels' modules, so they never run
the caller's script. A later call to a compiled specialization then loads it from the disk cache instead
of compiling it.
"""

import io
import os
import pickle
import pkgutil
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..expr.struct import is_struct_type
from ..expr.typing import Stream
from ..utils import env
from .kernel_function import CompilationContext, merge_compile_hints

try:
    import torch
except ImportError:
    torch = None


@dataclass
class CompileJob:
    """One specialization to compile: ``fn(*args, **kwargs)`` under ``compile_hints``, on top of the hints
    the call would get from ``fn`` and the active compile-hint context.

    ``fn`` is a ``@flyc.jit`` function, or, with ``factory_kwargs``, a factory that ``fn(**factory_kwargs)``
    turns into one. Either must be importable by its module and qualified name.
    """

    fn: Any
    args: Tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    compile_hints: Dict[str, Any] = field(default_factory=dict)
    factory_kwargs: Optional[Dict[str, Any]] = None


def precompile(jobs: List[CompileJob], workers: int) -> List[Optional[str]]:
    """Compile ``jobs`` in up to ``workers`` worker processes into the JIT disk cache.

    Returns one entry per job: None when it compiled, else why not. A job that cannot be sent to a worker
    (its function is not importable by name, or an argument does not pickle) is reported, not compiled.
    Tensor arguments are sent as CPU tensors of the same shape, strides and dtype, streams as
    ``fx.Stream(None)`` and ``@fx.struct`` instances as their field values, which specialize a kernel exactly as
    the originals do.
    """
    reason = why_no_disk_cache()
    if reason is not None:
        raise RuntimeError(f"precompile needs the JIT disk cache: {reason}")
    results: List[Optional[str]] = [None] * len(jobs)
    tasks = queue.Queue()
    for index, job in enumerate(jobs):
        payload = io.BytesIO()
        try:
            # A factory's jit function gets its own hints from the factory, which runs in the worker.
            fn_hints = job.fn.compile_hints if job.factory_kwargs is None else None
            hints = merge_compile_hints(CompilationContext.get_compile_hints(), job.compile_hints)
            call = (_importable_name(job.fn), job.factory_kwargs, fn_hints, job.args, job.kwargs, hints)
            _JobPickler(payload).dump(call)
        except Exception as error:
            results[index] = f"cannot send to a compile worker: {error}"
            continue
        tasks.put((index, payload.getvalue()))
    if tasks.empty():
        return results

    _run_workers(tasks, min(workers, tasks.qsize()), results)
    return results


def why_no_disk_cache() -> Optional[str]:
    """Why a JIT call would not load a specialization from the disk cache that precompile fills, or None."""
    if not env.runtime.enable_cache:
        return "FLYDSL_RUNTIME_ENABLE_CACHE is off"
    if not env.runtime.cache_dir:
        return "FLYDSL_RUNTIME_CACHE_DIR is empty"
    if env.debug.dump_ir or env.debug.dump_asm:
        return "IR dumps bypass it"
    return None


def _importable_name(fn):
    source = getattr(fn, "func", fn)  # a @flyc.jit function is named by the function it wraps
    module = getattr(source, "__module__", None)
    qualname = getattr(source, "__qualname__", "")
    if not module or not qualname or "<locals>" in qualname:
        raise TypeError(f"{fn!r} is not defined at module level")
    if module == "__main__":
        raise TypeError(f"{qualname} is defined in the main script, which compile workers do not run")
    name = f"{module}:{qualname}"
    if getattr(pkgutil.resolve_name(name), "fn", None) is fn:
        name += ".fn"  # a stacked @autotune binds the name to the Autotuner around fn
    if pkgutil.resolve_name(name) is not fn:
        raise TypeError(f"{name} is not {fn!r}")
    return name


class _JobPickler(pickle.Pickler):
    def reducer_override(self, obj):
        # An @fx.struct instance's class is generated for its constexpr values, so pickle cannot name it; the worker
        # builds the instance again from the class it was built from.
        if is_struct_type(type(obj)):
            base = getattr(type(obj), "__dsl_base_type__", type(obj))
            return _struct, (base, {field.name: getattr(obj, field.name) for field in base.__dsl_field_defs__})
        if torch is None:
            return NotImplemented
        if isinstance(obj, torch.Tensor):
            return _cpu_tensor, (tuple(obj.shape), obj.stride(), obj.dtype)
        if isinstance(obj, torch.cuda.Stream):
            return Stream, (None,)
        return NotImplemented


def _cpu_tensor(shape, stride, dtype):
    return torch.empty_strided(shape, stride, dtype=dtype, device="cpu")


def _struct(base, values):
    return base(**values)


def _compile(payload):
    try:
        name, factory_kwargs, fn_hints, args, kwargs, compile_hints = pickle.loads(payload)
        fn = pkgutil.resolve_name(name)
        if factory_kwargs is None:
            fn.compile_hints = fn_hints  # the caller may have changed them since import
        else:
            fn = fn(**factory_kwargs)
        with CompilationContext.compile_hints(compile_hints):
            fn._compile_to_disk_cache(*args, **kwargs)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _worker_main():
    """Entry point of a compile worker: compile the jobs read from stdin until it closes.

    Results go back on a private copy of stdout; stdout itself is pointed at stderr, so whatever a
    compile prints cannot corrupt them.
    """
    results = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    while True:
        try:
            # Each job arrives as bytes, so one that does not unpickle fails in _compile, not here.
            payload = pickle.load(sys.stdin.buffer)
        except EOFError:
            return
        pickle.dump(_compile(payload), results)
        results.flush()


def _run_workers(tasks, count, results):
    # Set sys.path outright: python -c would put the working directory in front of it.
    command = [
        sys.executable,
        "-c",
        f"import sys; sys.path[:] = {sys.path!r}; from flydsl.compiler.precompile import _worker_main; _worker_main()",
    ]
    workers = []
    unsent = "no compile worker was left to compile it"
    try:
        while len(workers) < count:
            try:
                workers.append(subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE))
            except OSError as error:
                unsent = f"cannot start a compile worker: {error}"
                break
        drivers = [threading.Thread(target=_drive, args=(worker, tasks, results)) for worker in workers]
        for driver in drivers:
            driver.start()
        for driver in drivers:
            driver.join()
    finally:
        # Every job sent has been answered unless this call was interrupted, so no compile is cut short.
        for worker in workers:
            worker.kill()
            worker.wait()
    while not tasks.empty():
        results[tasks.get_nowait()[0]] = unsent


def _drive(worker, tasks, results):
    while True:
        try:
            index, payload = tasks.get_nowait()
        except queue.Empty:
            return
        try:
            pickle.dump(payload, worker.stdin)
            worker.stdin.flush()
            results[index] = pickle.load(worker.stdout)
        except Exception:
            # Also a worker whose output is not a result, which it can no longer be trusted to answer.
            results[index] = "the compile worker exited"
            worker.kill()
            return
