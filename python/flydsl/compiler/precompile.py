# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile many kernel specializations into the JIT disk cache in parallel worker processes.

The workers are forked from a fresh interpreter that only imports FlyDSL and the kernels' modules, so they
never run the caller's script and share those imports. A later call to a compiled specialization then loads it
from the disk cache instead of compiling it.
"""

import contextlib
import ctypes
import importlib
import io
import os
import pickle
import pkgutil
import queue
import signal
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
    modules = set()
    for index, job in enumerate(jobs):
        payload = io.BytesIO()
        try:
            # A factory's jit function gets its own hints from the factory, which runs in the worker.
            fn_hints = job.fn.compile_hints if job.factory_kwargs is None else None
            hints = merge_compile_hints(CompilationContext.get_compile_hints(), job.compile_hints)
            name = _importable_name(job.fn)
            _JobPickler(payload).dump((name, job.factory_kwargs, fn_hints, job.args, job.kwargs, hints))
        except Exception as error:
            results[index] = f"cannot send to a compile worker: {error}"
            continue
        tasks.put((index, payload.getvalue()))
        modules.add(name.partition(":")[0])
    failure = None
    while not tasks.empty():
        left = tasks.qsize()
        failure = _run_workers(tasks, min(workers, left), sorted(modules), results)
        # Jobs are left only when every worker died; each took at least the job it died on with it, so new workers
        # make progress unless none could start or none was sent a job.
        if failure is not None or tasks.qsize() == left:
            break
    while not tasks.empty():
        results[tasks.get_nowait()[0]] = failure or "no compile worker was left to compile it"
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


def _template_main(caller, modules, status_fd, worker_ends):
    """Entry point of the process the compile workers fork from.

    It imports what every worker needs once, so the workers share it instead of each importing it, then forks one
    worker per ``(jobs, results)`` pair of pipe ends in ``worker_ends`` and waits for them. It is a fresh interpreter
    that runs no other thread, which is what makes forking it safe. Before forking it writes to ``status_fd`` None,
    or why it cannot fork the workers.
    """
    _die_with(caller)
    os.dup2(2, 1)  # whatever a compile prints goes to stderr
    reason = None
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception:
            pass  # each of its jobs reports the error when its worker imports the module again
        if reason is None and torch is not None and torch.cuda.is_initialized():
            reason = f"importing {module} started the GPU runtime, which workers forked from it could not use"
    with os.fdopen(status_fd, "wb") as status:
        pickle.dump(reason, status)
    if reason is not None:
        return
    template = os.getpid()
    for index, (jobs_fd, results_fd) in enumerate(worker_ends):
        try:
            pid = os.fork()
        except OSError:
            pid = None  # closing this worker's ends below tells the caller it is gone
        if pid == 0:
            _die_with(template)
            # A worker holds only its own ends, so the caller sees end of file on a result pipe whose worker died.
            for later in worker_ends[index + 1 :]:
                for fd in later:
                    os.close(fd)
            try:
                _worker_main(jobs_fd, results_fd)
            finally:
                os._exit(0)
        os.close(jobs_fd)
        os.close(results_fd)
    while True:
        try:
            os.wait()
        except ChildProcessError:
            return


def _die_with(parent):
    """Have Linux kill this process when ``parent``, which started it, exits without stopping it first."""
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid() != parent:  # it exited before this process asked
        os._exit(1)


def _worker_main(jobs_fd, results_fd):
    """Compile the jobs read from ``jobs_fd`` until it closes, writing each result to ``results_fd``."""
    jobs = os.fdopen(jobs_fd, "rb")
    results = os.fdopen(results_fd, "wb")
    while True:
        try:
            # Each job arrives as bytes, so one that does not unpickle fails in _compile, not here.
            payload = pickle.load(jobs)
        except EOFError:
            return
        pickle.dump(_compile(payload), results)
        results.flush()


def _run_workers(tasks, count, modules, results):
    """Compile ``tasks`` in ``count`` workers until it is empty or every worker died; return why none started."""
    # Compiling uses neither pool, but torch and numpy start one thread per core for them at import, and those
    # threads spin and compete with the other workers' compiles. The template must not have threads to fork.
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    workers = []  # this process's (jobs, results) ends of each worker's pipes
    worker_ends = []
    status = None
    template = None
    drivers = []
    failure = None
    answered = False
    try:
        try:
            status_read, status_write = os.pipe()
            status = os.fdopen(status_read, "rb")
            worker_ends.append((status_write,))
            for _ in range(count):
                jobs_read, jobs_write = os.pipe()
                results_read, results_write = os.pipe()
                workers.append((os.fdopen(jobs_write, "wb"), os.fdopen(results_read, "rb")))
                worker_ends.append((jobs_read, results_write))
            # Set sys.path outright: python -c would put the working directory in front of it.
            command = [
                sys.executable,
                "-c",
                f"import sys; sys.path[:] = {sys.path!r}; from flydsl.compiler.precompile import _template_main; "
                f"_template_main({os.getpid()}, {modules!r}, {status_write}, {worker_ends[1:]!r})",
            ]
            # Its own session, so the workers form a process group that can be killed together.
            fds = [fd for ends in worker_ends for fd in ends]
            template = subprocess.Popen(command, pass_fds=fds, env=env, start_new_session=True)
        except OSError as error:
            failure = f"cannot start a compile worker: {error}"
        finally:
            for ends in worker_ends:
                for fd in ends:
                    os.close(fd)
        if template is not None:
            try:
                reason = pickle.load(status)
            except EOFError:
                reason = "it exited while importing the jobs' modules"
            if reason is not None:
                failure = f"cannot start the compile workers: {reason}"
        if failure is None:
            drivers = [threading.Thread(target=_drive, args=(worker, tasks, results)) for worker in workers]
            for driver in drivers:
                driver.start()
            for driver in drivers:
                driver.join()
            answered = True
    finally:
        if template is not None and not answered:
            # Interrupted, or the workers could not start: stop any compile in progress, which also ends the drivers
            # waiting for one.
            _kill(template)
            for driver in drivers:
                driver.join()
        # Idle workers exit when their job pipes close.
        for pipes in workers:
            for pipe in pipes:
                _close(pipe)
        if status is not None:
            _close(status)
        if template is not None:
            try:
                template.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _kill(template)
                template.wait()
    return failure


def _drive(worker, tasks, results):
    jobs, worker_results = worker
    while True:
        try:
            index, payload = tasks.get_nowait()
        except queue.Empty:
            return
        try:
            pickle.dump(payload, jobs)
            jobs.flush()
            results[index] = pickle.load(worker_results)
        except Exception:
            # Also a worker whose output is not a result, which it can no longer be trusted to answer; closing its
            # pipes makes it exit.
            results[index] = "the compile worker exited"
            _close(jobs)
            _close(worker_results)
            return


def _kill(template):
    # The template leads its own session, so its process group is the template and the workers.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(template.pid, signal.SIGKILL)


def _close(pipe):
    # Closing flushes what is left to write, which fails once the worker at the other end is gone.
    with contextlib.suppress(OSError):
        pipe.close()
