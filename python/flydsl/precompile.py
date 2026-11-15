# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Precompile kernels in parallel and save the results in the JIT disk cache for later calls.

``CompileJob`` describes a specialization with a target, arguments, and compile hints.
``precompile`` checks that the disk cache is enabled, serializes the jobs, and runs them in a worker pool.
It returns one result per job, in input order: ``None`` for success or an error message for failure.
"""

import contextlib
import ctypes
import importlib
import io
import os
import pickle
import pkgutil
import selectors
import signal
import subprocess
import sys
from collections import deque
from functools import partial

from . import compiler as flyc
from .compiler.jit_function import JitFunction
from .compiler.kernel_function import CompilationContext
from .expr.struct import is_struct_type
from .expr.typing import Stream
from .utils import env

try:
    import torch
except ImportError:
    torch = None


# public API


class CompileJob:
    """A kernel call to compile ahead of time, written like the call itself.

    Pass the ``@flyc.jit`` function and the arguments you will call it with::

        CompileJob(launch, x, out, N=4096, BLOCK=128)

    Give compile hints in brackets, as ``flyc.compile[hints]`` takes them::

        CompileJob[{"waves_per_eu": 2}](launch, x, out, N=4096, BLOCK=128)

    For a kernel that a factory builds, pass a ``functools.partial`` of the factory instead of the jit function.
    The worker calls the factory and compiles the call of the function it returns::

        CompileJob(functools.partial(build_kernel, BLOCK=128), x, out)

    The job records the compile-hint context that is active when you create it, and compiles what the call would
    compile in that context. It does not change the function's own hints.

    Workers import the jit function or the factory by name, so define it at the top level of an importable module.
    """

    def __init__(self, target, /, *args, **kwargs):
        if not isinstance(target, (JitFunction, partial)):
            raise TypeError(
                f"CompileJob expects a @flyc.jit function or a functools.partial of a factory, got {target!r}"
            )
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.compile_hints = {}
        # Read now, as flyc.compile reads them when it is called. A factory's jit function gets its own hints from
        # the factory, which runs in the worker.
        self.kernel_hints = dict(target.compile_hints) if isinstance(target, JitFunction) else None
        self.context_hints = dict(CompilationContext.get_compile_hints())

    def __class_getitem__(cls, compile_hints):
        """``CompileJob[hints]``: build jobs that compile as ``flyc.compile[hints]`` would."""
        if not isinstance(compile_hints, dict):
            raise TypeError(f"CompileJob[...] expects a dict of compile hints, got {type(compile_hints).__name__}")

        def with_hints(target, /, *args, **kwargs):
            job = cls(target, *args, **kwargs)
            job.compile_hints = dict(compile_hints)
            return job

        return with_hints


def precompile(jobs: list[CompileJob], max_workers: int) -> list[str | None]:
    """Compile ``jobs`` in up to ``max_workers`` worker processes into the JIT disk cache.

    Returns one entry per job: None when it compiled, else why not. A job that cannot be sent to a worker
    (its function is not importable by name, or an argument does not pickle) is reported, not compiled.
    ``precompile`` sends tensor arguments as CPU tensors of the same shape, strides and dtype, streams as
    ``fx.Stream(None)``, and ``@fx.struct`` instances as their field values. These specialize a kernel exactly as
    the originals do.

    Example::

        from functools import partial

        from kernels.norm.rmsnorm_kernel import build_rmsnorm_module, rmsnorm_direct

        # Only shape, strides and dtype matter, not device.
        x = torch.empty(64, 4096, dtype=torch.bfloat16)
        gamma = torch.empty(4096, dtype=torch.bfloat16)
        out = torch.empty_like(x)
        config = dict(N=4096, dtype_str="bf16", BLOCK_THREADS=256)
        jobs = [
            CompileJob(rmsnorm_direct, x, gamma, out, 64, **config),
            CompileJob[{"fast_fp_math": True}](rmsnorm_direct, x, gamma, out, 64, **config),
            CompileJob(partial(build_rmsnorm_module, N=4096, dtype_str="bf16", BLOCK_THREADS=512), x, gamma, out, 64),
        ]
        errors = precompile(jobs, max_workers=2)  # [None, None, None] when all compiled
    """
    if max_workers < 1:
        raise ValueError(f"precompile needs at least one worker, got {max_workers}")
    # The workers need fork, pipe descriptors passed to a child, and Linux's parent-death signal.
    if sys.platform != "linux":
        return [f"not compiled: precompile needs Linux, not {sys.platform}"] * len(jobs)
    # Calls load nothing from the disk cache when it is off, and IR dumps bypass it, so compiling would gain nothing.
    # Each job reports why instead of raising, so adding a precompile call changes nothing but speed.
    if not env.runtime.enable_cache:
        return ["not compiled: FLYDSL_RUNTIME_ENABLE_CACHE is off"] * len(jobs)
    if not env.runtime.cache_dir:
        return ["not compiled: FLYDSL_RUNTIME_CACHE_DIR is empty"] * len(jobs)
    if env.debug.dump_ir or env.debug.dump_asm:
        return ["not compiled: IR dumps bypass the JIT disk cache"] * len(jobs)

    errors: list[str | None] = [None] * len(jobs)
    pending_jobs = deque()  # (job index, payload), taken by the dispatch loop
    # The modules the worker parent imports before forking the workers. A set, because many jobs usually share one
    # module (every config of an autotune search does), and it only needs importing once.
    module_names = set()
    for job_index, job in enumerate(jobs):
        try:
            module_name, payload = _serialize_job(job)
        except Exception as error:
            errors[job_index] = f"cannot send to a compile worker: {error}"
            continue
        pending_jobs.append((job_index, payload))
        module_names.add(module_name)
    # A set's order of strings changes from run to run, so sort them for the same import order every time.
    module_names = sorted(module_names)

    if pending_jobs:
        num_workers = min(max_workers, len(pending_jobs))
        try:
            pool = _WorkerPool(num_workers, module_names)
        except OSError as error:
            _fail_pending(pending_jobs, errors, f"cannot start {num_workers} compile workers: {error}")
            return errors
        with pool:
            pool.run(pending_jobs, errors)

    # Jobs are left only when every worker died, which takes a crash, not a compile error. Compile them here, one at a
    # time, as a search without workers would. A crash then ends this process as it would there.
    while pending_jobs:
        job_index, payload = pending_jobs.popleft()
        errors[job_index] = _compile_job(payload)
    return errors


# job format: encoded by the caller, decoded by a worker


def _serialize_job(job):
    """Return the module ``job.target`` is imported from and the pickled call that ``_compile_job`` runs."""
    if isinstance(job.target, partial):
        import_path = _import_path(job.target.func)
        factory_call = (job.target.args, job.target.keywords)
    else:
        import_path = _import_path(job.target)
        factory_call = None
    call = (
        import_path,
        factory_call,
        job.kernel_hints,
        job.compile_hints,
        job.args,
        job.kwargs,
        job.context_hints,
    )
    buffer = io.BytesIO()
    _JobPickler(buffer).dump(call)
    return import_path.partition(":")[0], buffer.getvalue()


def _import_path(target):
    source_fn = getattr(target, "func", target)  # a @flyc.jit function is named by the function it wraps
    module_name = getattr(source_fn, "__module__", None)
    qualname = getattr(source_fn, "__qualname__", "")
    if not module_name or not qualname or "<locals>" in qualname:
        raise TypeError(f"{target!r} is not defined at module level")
    if module_name == "__main__":
        raise TypeError(f"{qualname} is defined in the main script, which compile workers do not run")

    import_path = f"{module_name}:{qualname}"
    if getattr(pkgutil.resolve_name(import_path), "fn", None) is target:
        import_path += ".fn"  # a stacked @autotune binds the name to the Autotuner around target

    # The name may be bound to something else, such as another decorator's wrapper, which the worker would compile.
    if pkgutil.resolve_name(import_path) is not target:
        raise TypeError(f"{import_path} is not {target!r}")
    return import_path


class _JobPickler(pickle.Pickler):
    def reducer_override(self, obj):
        # An @fx.struct instance's class is generated for its constexpr values, so pickle cannot name it. The worker
        # builds the instance again from the class it was built from.
        if is_struct_type(type(obj)):
            struct_type = getattr(type(obj), "__dsl_base_type__", type(obj))
            return _restore_struct, (
                struct_type,
                {field_def.name: getattr(obj, field_def.name) for field_def in struct_type.__dsl_field_defs__},
            )
        if torch is None:
            return NotImplemented
        # A compile reads only a tensor's shape, strides and dtype, and only that an argument is a stream. CPU
        # stand-ins carry that without copying data or touching the GPU in the worker.
        if isinstance(obj, torch.Tensor):
            return _empty_cpu_tensor, (tuple(obj.shape), obj.stride(), obj.dtype)
        if isinstance(obj, torch.cuda.Stream):
            return Stream, (None,)
        return NotImplemented


def _empty_cpu_tensor(shape, strides, dtype):
    return torch.empty_strided(shape, strides, dtype=dtype, device="cpu")


def _restore_struct(struct_type, field_values):
    return struct_type(**field_values)


def _compile_job(payload):
    try:
        import_path, factory_call, kernel_hints, job_hints, args, kwargs, context_hints = pickle.loads(payload)
        target = pkgutil.resolve_name(import_path)
        if factory_call is None:
            kernel = target
        else:
            factory_args, factory_kwargs = factory_call
            kernel = target(*factory_args, **factory_kwargs)
            kernel_hints = kernel.compile_hints
        # Merge the hints as flyc.compile[hints] does. For a jit function, the base is the caller's hints, which may
        # have changed since import.
        previous_hints = kernel.compile_hints
        kernel.compile_hints = {**kernel_hints, **job_hints}
        try:
            with CompilationContext.compile_hints(context_hints):
                flyc.compile(kernel, *args, compile_only=True, **kwargs)
        finally:
            # A factory may return the same jit function for every call, which later jobs then start from.
            kernel.compile_hints = previous_hints
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


# caller side of a worker pool


class _WorkerPool:
    """The caller's handle on one worker parent and its workers: the process and the pipe ends the caller keeps.

    It owns them from the moment it exists, which is only once the worker parent has started. When the ``with``
    block exits, the pool releases them, and it first kills the workers if an exception is leaving the block.
    """

    def __init__(self, num_workers, module_names):
        caller_ends = []
        child_fds = []  # the worker parent's ends, which this process closes once the worker parent has inherited them
        try:
            self.status, status_fd = self._open_pipe("rb", caller_ends, child_fds)
            self.workers, worker_fds = [], []
            for _ in range(num_workers):
                job_writer, jobs_fd = self._open_pipe("wb", caller_ends, child_fds)
                result_reader, results_fd = self._open_pipe("rb", caller_ends, child_fds)
                self.workers.append((job_writer, result_reader))
                worker_fds.append((jobs_fd, results_fd))

            # Set sys.path outright, because python -c would put the working directory in front of it. The entries of
            # sys.path may be path objects, whose repr the worker parent could not evaluate.
            import_paths = [os.fspath(entry) for entry in sys.path]
            code = (
                f"import sys; sys.path[:] = {import_paths!r}; from flydsl.precompile import _worker_parent_main; "
                f"_worker_parent_main({os.getpid()}, {module_names!r}, {status_fd}, {worker_fds!r})"
            )
            # Compiling uses neither pool, but torch and numpy start one thread per core for them at import, and those
            # threads spin and compete with the compiles. The worker parent must also have no threads when it forks.
            child_env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            # Its own session, so the workers form a process group that can be killed together.
            self.worker_parent = subprocess.Popen(
                [sys.executable, "-c", code], pass_fds=child_fds, env=child_env, start_new_session=True
            )
        except BaseException:
            for end in caller_ends:
                _close_pipe(end)
            raise
        finally:
            for fd in child_fds:
                os.close(fd)

    def run(self, pending_jobs, errors):
        """Compile pending jobs until none is pending or every worker has died.

        Fails every pending job when the workers cannot start, and kills them all when interrupted.
        """
        try:
            reason = pickle.load(self.status)
        except EOFError:
            reason = "it exited while importing the jobs' modules"
        if reason is not None:
            self.kill()
            _fail_pending(pending_jobs, errors, f"cannot start the compile workers: {reason}")
            return
        self._dispatch(pending_jobs, errors)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.kill()  # interrupted: stop any compile in progress
        self.close()

    def close(self):
        # Idle workers exit when their job pipes close.
        _close_pipe(self.status)
        for job_writer, result_reader in self.workers:
            _close_pipe(job_writer)
            _close_pipe(result_reader)
        try:
            self.worker_parent.wait(timeout=30)  # bounded, in case a worker hangs on its way out
        except subprocess.TimeoutExpired:
            self.kill()
            self.worker_parent.wait()

    def kill(self):
        # The worker parent leads its own session, so its process group is the worker parent and the workers.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.worker_parent.pid, signal.SIGKILL)

    @staticmethod
    def _open_pipe(caller_mode, caller_ends, child_fds):
        """Open a pipe whose ``caller_mode`` end this process keeps, recording both ends as soon as they exist."""
        read_fd, write_fd = os.pipe()
        caller_fd, child_fd = (read_fd, write_fd) if caller_mode == "rb" else (write_fd, read_fd)
        child_fds.append(child_fd)
        caller_ends.append(os.fdopen(caller_fd, caller_mode))
        return caller_ends[-1], child_fd

    def _dispatch(self, pending_jobs, errors):
        """Send each worker a job, then a new one whenever it returns a result, until no job is pending."""
        # One loop serves all workers. It waits for whichever result comes first, so a slow compile holds up no other
        # worker.
        with selectors.DefaultSelector() as busy:
            for job_writer, result_reader in self.workers:
                self._send_job(busy, job_writer, result_reader, pending_jobs, errors)
            while busy.get_map():
                for key, _ in busy.select():
                    result_reader = key.fileobj
                    job_writer, job_index = key.data
                    busy.unregister(result_reader)
                    try:
                        # A worker writes one result, then waits for its next job, so the pipe holds just this one.
                        errors[job_index] = pickle.load(result_reader)
                    except Exception:
                        _fail_worker(job_writer, result_reader, job_index, errors)
                        continue
                    self._send_job(busy, job_writer, result_reader, pending_jobs, errors)

    @staticmethod
    def _send_job(busy, job_writer, result_reader, pending_jobs, errors):
        # Nothing is queued once dispatch starts, so no pending job means the work is done.
        if not pending_jobs:
            return
        job_index, payload = pending_jobs.popleft()
        try:
            pickle.dump(payload, job_writer)
            job_writer.flush()
        except Exception:
            _fail_worker(job_writer, result_reader, job_index, errors)
            return
        busy.register(result_reader, selectors.EVENT_READ, (job_writer, job_index))


def _fail_worker(job_writer, result_reader, job_index, errors):
    # Closing a worker's pipes makes it exit.
    errors[job_index] = "the compile worker exited"
    _close_pipe(job_writer)
    _close_pipe(result_reader)


def _fail_pending(pending_jobs, errors, reason):
    for job_index, _ in pending_jobs:
        errors[job_index] = reason
    pending_jobs.clear()


def _close_pipe(pipe):
    # Closing flushes what is left to write, which fails once the worker at the other end is gone.
    with contextlib.suppress(OSError):
        pipe.close()


# child processes: the worker parent and its workers


def _worker_parent_main(caller_pid, module_names, status_fd, worker_fds):
    """Entry point of the process the compile workers fork from.

    It imports what every worker needs once, so the workers share it instead of each importing it, then forks one
    worker per ``(jobs_fd, results_fd)`` pair in ``worker_fds`` and waits for them. Forking is safe because this
    process is a fresh interpreter that runs no other thread, which it checks after each import. Before forking, it
    writes to ``status_fd`` None, or the reason it cannot fork the workers.
    """
    _kill_on_parent_exit(caller_pid)
    os.dup2(2, 1)  # compile prints are piped to stderr

    reason = None
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception:
            # Importing here only lets the workers share the module, so a failure is not reported here. Each worker
            # that runs a job from this module imports it again, fails the same way, and returns the error as that
            # job's result.
            pass
        if torch is not None and torch.cuda.is_initialized():
            reason = f"importing {module_name} started the GPU runtime, which workers forked from it could not use"
            break

        # A module can start the GPU runtime without torch knowing, for example through torch.cuda.is_available() on
        # ROCm, and the runtime's threads would not survive the fork.
        threads = len(os.listdir("/proc/self/task"))
        if threads > 1:
            reason = f"importing {module_name} left {threads - 1} other threads running, which a fork would not copy"
            break

    with os.fdopen(status_fd, "wb") as status:
        pickle.dump(reason, status)
    if reason is not None:
        return

    worker_parent_pid = os.getpid()
    for index, (jobs_fd, results_fd) in enumerate(worker_fds):
        try:
            pid = os.fork()
        except OSError:
            # Treat a failed fork as a worker that died at once. With a pid other than 0 the code below still closes
            # this worker's pipe ends, so the caller's first job for it fails with "the compile worker exited".
            pid = None

        if pid == 0:
            _kill_on_parent_exit(worker_parent_pid)
            # The fork copied the pipe ends of the workers not forked yet. The loop below has already closed the ends
            # of earlier workers. Close the copied ends. If this worker kept another worker's result pipe open, the
            # caller would not see that pipe close when the other worker dies, and would wait for its result forever.
            for later_fds in worker_fds[index + 1 :]:
                for fd in later_fds:
                    os.close(fd)
            try:
                _worker_main(jobs_fd, results_fd)
            finally:
                # Skip the worker parent's cleanup, which belongs to the worker parent, not to this copy of it.
                os._exit(0)

        os.close(jobs_fd)
        os.close(results_fd)

    while True:
        try:
            os.wait()
        except ChildProcessError:
            return


def _worker_main(jobs_fd, results_fd):
    """Compile the jobs read from ``jobs_fd`` until it closes, writing each result to ``results_fd``."""
    job_reader = os.fdopen(jobs_fd, "rb")
    result_writer = os.fdopen(results_fd, "wb")
    while True:
        try:
            # Each job arrives as bytes, so one that does not unpickle fails in _compile_job, not here.
            payload = pickle.load(job_reader)
        except EOFError:
            return
        pickle.dump(_compile_job(payload), result_writer)
        result_writer.flush()


def _kill_on_parent_exit(parent_pid):
    """Have Linux kill this process when ``parent_pid``, which started it, exits without stopping it first."""
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid() != parent_pid:  # it exited before this process asked
        os._exit(1)
