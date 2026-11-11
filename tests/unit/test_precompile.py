# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``precompile`` compiles kernel specializations in reusable worker processes into the JIT disk cache."""

import ast
import importlib
import inspect
import os
import resource
import signal
import subprocess
import sys
import textwrap
import threading
import time
import types
from functools import partial

import pytest

torch = pytest.importorskip("torch")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.compiler.jit_function as jit_function  # noqa: E402
import flydsl.precompile as precompile_module  # noqa: E402
from flydsl.autotune import Autotuner, Config  # noqa: E402
from flydsl.compiler.kernel_function import CompilationContext  # noqa: E402
from flydsl.precompile import CompileJob, precompile  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402

# Precompiling needs a compile target, not a GPU. Only the tests marked with _needs_gpu launch kernels or start the
# GPU runtime.
pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def _needs_gpu(test):
    return pytest.mark.l2_device(pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")(test))


CONFIGS = [Config(BLOCK_THREADS=64), Config(BLOCK_THREADS=128), Config(BLOCK_THREADS=128, waves_per_eu=2)]
N = 1024  # elements the tiny kernels copy, a multiple of every BLOCK_THREADS used here
OUTPUT_VAR = "FLYDSL_TEST_PRECOMPILE_OUTPUT"  # where a probe module writes what it observed


# Module bodies. The workers import a job's kernel by module name, so each test writes the modules it needs from
# these functions' bodies (see _module_source), and the code still gets linted and formatted here.


def _tiny_kernels():
    import functools

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.autotune import Config, autotune

    @fx.struct
    class CopyParams:
        block_threads: fx.Constexpr[int]

    @flyc.kernel
    def copy_kernel(src: fx.Tensor, dst: fx.Tensor, block_threads: fx.Constexpr[int]):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        tile_src = fx.slice(fx.logical_divide(src, fx.make_layout(block_threads, 1)), (None, bid))
        tile_dst = fx.slice(fx.logical_divide(dst, fx.make_layout(block_threads, 1)), (None, bid))
        tile_src = fx.logical_divide(tile_src, fx.make_layout(1, 1))
        tile_dst = fx.logical_divide(tile_dst, fx.make_layout(1, 1))
        atom = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
        value = fx.make_rmem_tensor(1, fx.Float32)
        fx.copy_atom_call(atom, fx.slice(tile_src, (None, tid)), value)
        fx.copy_atom_call(atom, value, fx.slice(tile_dst, (None, tid)))

    def launch_copy(src, dst, n, block_threads, stream):
        grid = (n + block_threads - 1) // block_threads
        copy_kernel(src, dst, block_threads).launch(grid=(grid, 1, 1), block=(block_threads, 1, 1), stream=stream)

    @flyc.jit
    def copy(
        src: fx.Tensor,
        dst: fx.Tensor,
        n: fx.Int32,
        BLOCK_THREADS: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        launch_copy(src, dst, n, BLOCK_THREADS, stream)

    @flyc.jit
    def copy_with_params(
        src: fx.Tensor, dst: fx.Tensor, n: fx.Int32, params: CopyParams, stream: fx.Stream = fx.Stream(None)
    ):
        launch_copy(src, dst, n, params.block_threads, stream)

    @flyc.compile[{"unsafe_fp_math": True}]
    @flyc.jit
    def hinted_copy(
        src: fx.Tensor,
        dst: fx.Tensor,
        n: fx.Int32,
        BLOCK_THREADS: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        launch_copy(src, dst, n, BLOCK_THREADS, stream)

    @autotune(configs=[Config(BLOCK_THREADS=64), Config(BLOCK_THREADS=128)], key=["n"])
    @flyc.jit
    def tuned_copy(
        src: fx.Tensor,
        dst: fx.Tensor,
        n: fx.Int32,
        BLOCK_THREADS: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        launch_copy(src, dst, n, BLOCK_THREADS, stream)

    def build_copy(block_threads):
        @flyc.jit
        def copy_built(src: fx.Tensor, dst: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
            launch_copy(src, dst, n, block_threads, stream)

        return copy_built

    @functools.lru_cache(maxsize=None)
    def build_cached_copy(block_threads):
        return build_copy(block_threads)


def _writes_its_thread_count_when_imported():
    import os

    with open(os.environ["FLYDSL_TEST_PRECOMPILE_OUTPUT"], "w") as output:
        output.write(str(len(os.listdir("/proc/self/task"))))


def _writes_its_thread_count_when_unpickled():
    import os

    def write_thread_count():
        with open(os.environ["FLYDSL_TEST_PRECOMPILE_OUTPUT"], "w") as output:
            output.write(str(len(os.listdir("/proc/self/task"))))

    class Probe:
        def __reduce__(self):
            return write_thread_count, ()


def _starts_the_gpu_runtime():
    import torch

    torch.cuda.init()


def _queries_the_gpu():
    import torch

    torch.cuda.is_available()


def _exits_in_the_worker_parent():
    import os
    import sys

    # Only the worker parent runs as python -c.
    if sys.argv[0] == "-c":
        os._exit(3)


def _hangs_in_the_worker_parent():
    import sys
    import time

    if sys.argv[0] == "-c":
        time.sleep(60)


def _caller_that_precompiles_for_a_minute():
    import importlib
    import os
    import sys
    import time

    from flydsl.precompile import CompileJob, precompile

    # Put the marker on the import path, which the worker parent's command line repeats, so the test can find it.
    sys.path.insert(0, os.environ["FLYDSL_TEST_PRECOMPILE_MARKER"])
    copy = importlib.import_module("precompile_kernels").copy

    class Sleeps:
        def __reduce__(self):
            return time.sleep, (60,)

    precompile([CompileJob(copy, Sleeps()) for _ in range(2)], 2)


def _module_source(*functions):
    """The bodies of ``functions``, dedented to module level and joined into the source of one module."""
    bodies = []
    for function in functions:
        lines, _ = inspect.getsourcelines(function)
        definition = ast.parse(textwrap.dedent("".join(lines))).body[0]
        assert isinstance(definition, ast.FunctionDef)
        first = definition.body[0]
        # A decorated statement's line number is its def or class line, after the decorators.
        start = min([first.lineno] + [decorator.lineno for decorator in getattr(first, "decorator_list", [])])
        bodies.append(textwrap.dedent("".join(lines[start - 1 :])))
    return "\n\n".join(bodies)


def _write_module(tmp_path, monkeypatch, name, *functions):
    """Write module ``name`` from the bodies of ``functions`` to ``tmp_path``, on the import path, and import it."""
    (tmp_path / f"{name}.py").write_text(_module_source(*functions))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


@pytest.fixture
def kernels(tmp_path, monkeypatch):
    return _write_module(tmp_path, monkeypatch, "precompile_kernels", _tiny_kernels)


@pytest.fixture
def compiles(tmp_path, monkeypatch):
    """The functions compiled in this process, starting from an empty cache in ``tmp_path``."""
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "jit"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    try:
        get_rocm_arch()
    except RuntimeError:
        monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx942")  # no GPU to detect; the workers inherit it
    compiled = []
    compile_module = jit_function.MlirCompiler.compile

    # Every call traces. Only a cache miss reaches MlirCompiler.compile.
    def counted(asm, *, func_name, config):
        compiled.append(func_name)
        return compile_module(asm, func_name=func_name, config=config)

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", counted)
    return compiled


def _copy_args(device="cpu"):
    src = torch.empty(N, dtype=torch.float32, device=device)
    return src, torch.empty_like(src), N


def _copy_job(fn, block_threads):
    return CompileJob(fn, *_copy_args(), BLOCK_THREADS=block_threads)


def _compile_in_this_process(job):
    job.target._compile_only(*job.args, **job.kwargs)


class _ExitsTheWorker:
    def __reduce__(self):
        return os._exit, (1,)


class _SleepsInTheWorker:
    def __init__(self, seconds):
        self.seconds = seconds

    def __reduce__(self):
        return time.sleep, (self.seconds,)


def _record_worker_parents(monkeypatch):
    started = []
    popen = subprocess.Popen

    def recorded(*args, **kwargs):
        started.append(popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(precompile_module.subprocess, "Popen", recorded)
    return started


def test_compiled_jobs_load_from_the_disk_cache(compiles, kernels):
    factory_job = CompileJob(partial(kernels.build_copy, 128), *_copy_args())

    assert precompile([_copy_job(kernels.copy, 64), factory_job], max_workers=2) == [None, None]

    _compile_in_this_process(_copy_job(kernels.copy, 64))
    factory_job.target()._compile_only(*factory_job.args)
    assert compiles == []


def test_workers_fork_from_one_interpreter_and_exit_when_their_jobs_are_done(compiles, kernels, monkeypatch):
    started = _record_worker_parents(monkeypatch)

    assert precompile([_copy_job(kernels.copy, 64), _copy_job(kernels.copy, 128)], max_workers=2) == [None, None]

    # The interpreter the workers fork from exits once it has waited for all of them.
    assert len(started) == 1
    assert started[0].returncode == 0


def test_workers_fork_from_an_interpreter_without_thread_pools(compiles, tmp_path, monkeypatch):
    # Forking a process with other threads running is unsafe, and torch and numpy start a pool of them at import
    # unless told not to. The worker parent imports the job's module just before it forks.
    monkeypatch.setenv(OUTPUT_VAR, str(tmp_path / "threads"))
    probe = _write_module(
        tmp_path,
        monkeypatch,
        "precompile_probe",
        _writes_its_thread_count_when_imported,
        _tiny_kernels,
    )

    assert precompile([_copy_job(probe.copy, 64)], max_workers=1) == [None]

    assert (tmp_path / "threads").read_text() == "1"


def test_workers_run_no_thread_pools_besides_the_compile(compiles, kernels, tmp_path, monkeypatch):
    # Idle OpenMP and BLAS pools started by torch and numpy compete with the compiles for the cores. The probe
    # writes the worker's thread count when the worker unpickles the job.
    monkeypatch.setenv(OUTPUT_VAR, str(tmp_path / "threads"))
    probe = _write_module(tmp_path, monkeypatch, "thread_probe", _writes_its_thread_count_when_unpickled)

    precompile([CompileJob(kernels.copy, probe.Probe())], max_workers=1)

    assert (tmp_path / "threads").read_text() == "1"


def test_an_interrupt_stops_the_compiles_in_progress_and_the_rest(compiles, kernels, monkeypatch):
    started = _record_worker_parents(monkeypatch)
    jobs = [CompileJob(kernels.copy, _SleepsInTheWorker(60)) for _ in range(2)]
    interrupt = threading.Timer(3, os.kill, (os.getpid(), signal.SIGINT))
    interrupt.start()
    start = time.perf_counter()

    try:
        with pytest.raises(KeyboardInterrupt):
            precompile(jobs, max_workers=1)
    finally:
        interrupt.cancel()  # a precompile that returned early must not interrupt a later test

    assert time.perf_counter() - start < 15
    assert started[0].poll() is not None


def _processes_naming(marker):
    found = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as file:
                if marker.encode() in file.read():
                    found.append(int(pid))
        except OSError:
            pass
    return found


def test_the_workers_die_with_a_caller_that_is_killed(compiles, kernels, tmp_path):
    # A killed caller runs no cleanup. The worker parent and its workers must not keep compiling for it.
    marker = str(tmp_path / "caller-marker")
    script = tmp_path / "caller.py"
    script.write_text(_module_source(_caller_that_precompiles_for_a_minute))
    # The marker goes in the environment, not on the caller's own command line. The script's directory, which holds
    # precompile_kernels, comes first on its import path.
    caller = subprocess.Popen([sys.executable, str(script)], env=dict(os.environ, FLYDSL_TEST_PRECOMPILE_MARKER=marker))
    deadline = time.monotonic() + 60
    while len(_processes_naming(marker)) < 3 and time.monotonic() < deadline:
        time.sleep(0.2)
    assert len(_processes_naming(marker)) == 3  # the caller's worker parent and its two workers

    caller.kill()
    caller.wait()
    deadline = time.monotonic() + 10
    while _processes_naming(marker) and time.monotonic() < deadline:
        time.sleep(0.2)

    assert _processes_naming(marker) == []


def test_jobs_left_when_every_worker_died_compile_in_this_process(compiles, kernels, monkeypatch):
    started = _record_worker_parents(monkeypatch)
    dying = [CompileJob(kernels.copy, _ExitsTheWorker()) for _ in range(2)]
    compiling = [_copy_job(kernels.copy, block_threads) for block_threads in (32, 64, 128, 64)]

    errors = precompile(dying + compiling, max_workers=2)

    assert errors == ["the compile worker exited"] * 2 + [None] * 4
    assert len(started) == 1
    # The second job for 64 threads finds the first one's artifact.
    assert compiles == ["copy"] * 3


def test_a_worker_dying_while_earlier_workers_compile_is_seen_at_once(compiles, kernels):
    # Each worker must hold only its own pipe ends: one that kept a later worker's result end open would hide
    # that worker's death from the caller, which would then wait for it forever.
    jobs = [CompileJob(kernels.copy, _SleepsInTheWorker(3)) for _ in range(2)]
    jobs += [CompileJob(kernels.copy, _ExitsTheWorker()) for _ in range(2)]
    errors = []
    call = threading.Thread(target=lambda: errors.extend(precompile(jobs, max_workers=4)), daemon=True)

    call.start()
    call.join(timeout=60)

    assert not call.is_alive()
    assert errors[2:] == ["the compile worker exited"] * 2


def test_jobs_report_a_platform_other_than_linux_and_start_no_workers(compiles, kernels, monkeypatch):
    started = _record_worker_parents(monkeypatch)
    monkeypatch.setattr(precompile_module.sys, "platform", "win32")

    errors = precompile([_copy_job(kernels.copy, 64)], max_workers=1)

    assert errors == ["not compiled: precompile needs Linux, not win32"]
    assert started == []


def test_jobs_report_an_empty_cache_directory_and_start_no_workers(compiles, kernels, monkeypatch):
    started = _record_worker_parents(monkeypatch)
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", "")

    errors = precompile([_copy_job(kernels.copy, 64)], max_workers=1)

    assert errors == ["not compiled: FLYDSL_RUNTIME_CACHE_DIR is empty"]
    assert started == []


def test_a_struct_argument_specializes_the_worker_compile_as_it_does_the_call(compiles, kernels):
    # The struct's class is generated for its constexpr values, so it is sent as the struct it was built from.
    job = CompileJob(kernels.copy_with_params, *_copy_args(), kernels.CopyParams(block_threads=64))

    assert precompile([job], max_workers=1) == [None]
    _compile_in_this_process(job)

    assert compiles == []


def test_a_job_that_cannot_be_sent_is_reported_and_the_rest_compile(compiles, kernels):
    unsendable = CompileJob(kernels.copy, lambda: None)
    errors = precompile([unsendable, _copy_job(kernels.copy, 64)], max_workers=2)

    assert errors[0].startswith("cannot send to a compile worker")
    assert errors[1] is None


def test_a_job_that_fails_in_a_worker_does_not_stop_the_rest(compiles, kernels, monkeypatch):
    # The class pickles here but its module does not exist in the worker, so the job fails to unpickle there.
    phantom = types.ModuleType("phantom")
    phantom.Marker = type("Marker", (), {"__module__": "phantom"})
    monkeypatch.setitem(sys.modules, "phantom", phantom)
    unloadable = CompileJob(kernels.copy, phantom.Marker())
    src, dst, _ = _copy_args()
    failing = CompileJob(kernels.copy, src, dst, "not a count", BLOCK_THREADS=64)

    errors = precompile([unloadable, failing, _copy_job(kernels.copy, 64)], max_workers=1)

    assert errors[0] is not None and errors[1] is not None
    assert errors[2] is None


@_needs_gpu
def test_a_kernel_module_that_starts_the_gpu_runtime_is_reported(compiles, tmp_path, monkeypatch):
    # Workers forked from an interpreter whose GPU runtime has started cannot use it.
    module = _write_module(tmp_path, monkeypatch, "gpu_runtime_kernels", _starts_the_gpu_runtime, _tiny_kernels)

    errors = precompile([_copy_job(module.copy, 64)], max_workers=1)

    assert errors[0].startswith("cannot start the compile workers: importing gpu_runtime_kernels started")


@_needs_gpu
def test_a_kernel_module_that_queries_the_gpu_at_import_is_reported(compiles, tmp_path, monkeypatch):
    # On ROCm this starts the GPU runtime's threads without torch.cuda.is_initialized() saying so.
    module = _write_module(tmp_path, monkeypatch, "gpu_query_kernels", _queries_the_gpu, _tiny_kernels)

    errors = precompile([_copy_job(module.copy, 64)], max_workers=1)

    assert errors[0].startswith("cannot start the compile workers: importing gpu_query_kernels left")


def test_an_interpreter_that_exits_while_starting_is_reported(compiles, tmp_path, monkeypatch):
    module = _write_module(tmp_path, monkeypatch, "exiting_kernels", _exits_in_the_worker_parent, _tiny_kernels)

    errors = precompile([_copy_job(module.copy, 64), _copy_job(module.copy, 128)], max_workers=2)

    assert errors == ["cannot start the compile workers: it exited while importing the jobs' modules"] * 2


def test_jobs_compile_with_the_hints_the_call_gets(compiles, kernels, monkeypatch):
    monkeypatch.setattr(kernels.copy, "compile_hints", {"unsafe_fp_math": True})
    with CompilationContext.compile_hints({"fast_fp_math": True}):
        job = _copy_job(kernels.copy, 64)

    # The job keeps the context it was created in, as flyc.compile uses the one it is called in.
    assert precompile([job], max_workers=1) == [None]
    with CompilationContext.compile_hints({"fast_fp_math": True}):
        _compile_in_this_process(job)

    assert compiles == []


def test_a_job_with_hints_compiles_what_flyc_compile_with_them_does(compiles, kernels, monkeypatch):
    # flyc.compile[hints] adds them to the function's own hints, which the active context overrides. The job's
    # unsafe_fp_math overrides the function's, and the context's fast_fp_math overrides the job's.
    monkeypatch.setattr(kernels.copy, "compile_hints", {"unsafe_fp_math": True})
    job_hints = {"fast_fp_math": True, "unsafe_fp_math": False}
    job = _copy_job(kernels.copy, 64)
    with CompilationContext.compile_hints({"fast_fp_math": False}):
        hinted = CompileJob[job_hints](kernels.copy, *job.args, **job.kwargs)

    assert precompile([hinted], max_workers=1) == [None]
    assert kernels.copy.compile_hints == {"unsafe_fp_math": True}
    with CompilationContext.compile_hints({"fast_fp_math": False}):
        flyc.compile[job_hints](kernels.copy, *job.args, **job.kwargs)

    assert compiles == []


def test_a_job_needs_a_jit_function_or_a_partial_factory(kernels):
    with pytest.raises(TypeError, match="functools.partial"):
        CompileJob(kernels.build_copy, 64)
    with pytest.raises(TypeError, match="dict of compile hints"):
        CompileJob["fast_fp_math"]


def test_a_jobs_hints_do_not_reach_later_jobs_of_a_cached_factory(compiles, kernels):
    # A cached factory returns the same jit function for every job, which one worker compiles in turn.
    factory = partial(kernels.build_cached_copy, 64)
    jobs = [CompileJob[{"fast_fp_math": True}](factory, *_copy_args()), CompileJob(factory, *_copy_args())]

    assert precompile(jobs, max_workers=1) == [None, None]
    kernels.build_cached_copy(64)._compile_only(*_copy_args())

    assert compiles == []


def test_jobs_compile_without_hints_the_caller_cleared(compiles, kernels, monkeypatch):
    monkeypatch.setattr(kernels.hinted_copy, "compile_hints", {})
    job = _copy_job(kernels.hinted_copy, 64)

    assert precompile([job], max_workers=1) == [None]
    _compile_in_this_process(job)

    assert compiles == []


def test_a_kernel_under_a_stacked_autotune_decorator_is_found(compiles, kernels):
    job = _copy_job(kernels.tuned_copy.fn, 64)

    assert precompile([job], max_workers=1) == [None]
    _compile_in_this_process(job)

    assert compiles == []


def _tune(tmp_path, monkeypatch, fn, **options):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "autotune"))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    tuner = Autotuner(fn, CONFIGS, key=["n"], warmup=1, rep=2, **options)
    tuner.resolve_config(*_copy_args("cuda"))


@_needs_gpu
def test_autotune_configs_precompiled_before_a_search_load_from_the_disk_cache(
    compiles, kernels, tmp_path, monkeypatch
):
    jobs = []
    for c in CONFIGS:
        with CompilationContext.compile_hints(c.compiler_opts()):
            jobs.append(CompileJob(kernels.copy, *_copy_args(), **c.all_kwargs()))
    assert precompile(jobs, max_workers=2) == [None] * len(CONFIGS)

    _tune(tmp_path, monkeypatch, kernels.copy)

    assert compiles == []


@_needs_gpu
def test_an_autotune_search_with_workers_compiles_nothing_in_this_process(compiles, kernels, tmp_path, monkeypatch):
    _tune(tmp_path, monkeypatch, kernels.copy, precompile_workers=2)

    assert compiles == []


def test_autotune_without_a_jit_function_warns_and_tunes_without_workers():
    with pytest.warns(UserWarning, match="precompile_workers is only supported on @jit functions"):
        tuner = Autotuner(lambda x: x, CONFIGS, key=[], warmup=1, rep=2, precompile_workers=2)

    assert tuner.precompile_workers == 0


@_needs_gpu
def test_autotune_warns_why_configs_did_not_precompile(compiles, kernels, tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path / "dump"))

    with pytest.warns(UserWarning, match="3 configs failed to precompile.*IR dumps bypass the JIT disk cache"):
        _tune(tmp_path, monkeypatch, kernels.copy, precompile_workers=2)


@_needs_gpu
def test_autotune_takes_its_precompile_workers_from_the_environment_when_unset(
    compiles, kernels, tmp_path, monkeypatch
):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_PRECOMPILE_WORKERS", "2")

    _tune(tmp_path, monkeypatch, kernels.copy)

    assert compiles == []


@_needs_gpu
def test_autotune_precompile_workers_set_on_the_tuner_override_the_environment(
    compiles, kernels, tmp_path, monkeypatch
):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_PRECOMPILE_WORKERS", "2")

    _tune(tmp_path, monkeypatch, kernels.copy, precompile_workers=0)

    assert compiles == ["copy"] * len(CONFIGS)


def test_workers_import_from_the_callers_path_not_their_working_directory(compiles, kernels, tmp_path, monkeypatch):
    # python -c puts the working directory first on its import path, where it would find this module instead.
    working_directory = tmp_path / "working_directory"
    working_directory.mkdir()
    (working_directory / "precompile_kernels.py").write_text("raise ImportError('not the module the caller imports')\n")
    monkeypatch.chdir(working_directory)

    assert precompile([_copy_job(kernels.copy, 64)], max_workers=1) == [None]


def test_jobs_report_a_worker_that_cannot_start(compiles, kernels, monkeypatch):
    def cannot_start(*args, **kwargs):
        raise OSError("Resource temporarily unavailable")

    monkeypatch.setattr(precompile_module.subprocess, "Popen", cannot_start)

    open_fds = set(os.listdir("/proc/self/fd"))
    errors = precompile([_copy_job(kernels.copy, 64), _copy_job(kernels.copy, 128)], max_workers=2)

    assert errors == ["cannot start 2 compile workers: Resource temporarily unavailable"] * 2
    assert set(os.listdir("/proc/self/fd")) == open_fds


@pytest.mark.parametrize("error", [RuntimeError("cannot start"), KeyboardInterrupt()])
def test_startup_exceptions_propagate_and_close_pipes(compiles, kernels, monkeypatch, error):
    def cannot_start(*args, **kwargs):
        raise error

    monkeypatch.setattr(precompile_module.subprocess, "Popen", cannot_start)

    open_fds = set(os.listdir("/proc/self/fd"))
    with pytest.raises(type(error)) as caught:
        precompile([_copy_job(kernels.copy, 64), _copy_job(kernels.copy, 128)], max_workers=2)

    assert caught.value is error
    assert set(os.listdir("/proc/self/fd")) == open_fds


def test_too_few_open_files_for_the_workers_is_reported(compiles, kernels):
    open_fds = set(os.listdir("/proc/self/fd"))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (len(open_fds) + 8, hard))
    try:
        errors = precompile([_copy_job(kernels.copy, 64), _copy_job(kernels.copy, 128)], max_workers=2)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    assert errors == ["cannot start 2 compile workers: [Errno 24] Too many open files"] * 2
    assert set(os.listdir("/proc/self/fd")) == open_fds


@pytest.mark.parametrize("failing_call", range(1, 6))
def test_a_pipe_that_cannot_be_opened_closes_the_ones_before_it(compiles, kernels, monkeypatch, failing_call):
    # Two workers need five pipes: the status pipe, then a job and a result pipe for each worker.
    calls = []
    pipe = os.pipe

    def fails_once(*args):
        calls.append(None)
        if len(calls) == failing_call:
            raise OSError("Too many open files")
        return pipe(*args)

    monkeypatch.setattr(precompile_module.os, "pipe", fails_once)

    open_fds = set(os.listdir("/proc/self/fd"))
    errors = precompile([_copy_job(kernels.copy, 64), _copy_job(kernels.copy, 128)], max_workers=2)

    assert errors == ["cannot start 2 compile workers: Too many open files"] * 2
    assert set(os.listdir("/proc/self/fd")) == open_fds


def test_an_interrupt_while_the_workers_start_stops_them(compiles, tmp_path, monkeypatch):
    started = _record_worker_parents(monkeypatch)
    module = _write_module(tmp_path, monkeypatch, "hanging_kernels", _hangs_in_the_worker_parent, _tiny_kernels)
    open_fds = set(os.listdir("/proc/self/fd"))
    interrupt = threading.Timer(3, os.kill, (os.getpid(), signal.SIGINT))
    interrupt.start()
    start = time.perf_counter()

    try:
        with pytest.raises(KeyboardInterrupt):
            precompile([_copy_job(module.copy, 64)], max_workers=1)
    finally:
        interrupt.cancel()  # a precompile that returned early must not interrupt a later test

    assert time.perf_counter() - start < 15
    assert started[0].poll() is not None
    assert set(os.listdir("/proc/self/fd")) == open_fds


def test_no_jobs_start_no_workers(compiles, monkeypatch):
    def cannot_start(*args, **kwargs):
        raise AssertionError("no workers should start")

    monkeypatch.setattr(precompile_module.subprocess, "Popen", cannot_start)

    assert precompile([], max_workers=1) == []


def test_precompile_refuses_fewer_than_one_worker(compiles, kernels):
    with pytest.raises(ValueError, match="at least one worker"):
        precompile([_copy_job(kernels.copy, 64)], max_workers=0)


def test_workers_take_path_objects_on_the_import_path(compiles, kernels, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", [tmp_path] + sys.path)

    assert precompile([_copy_job(kernels.copy, 64)], max_workers=1) == [None]


def test_jobs_report_ir_dumps_bypassing_the_disk_cache_and_start_no_workers(compiles, kernels, tmp_path, monkeypatch):
    started = _record_worker_parents(monkeypatch)
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path / "dump"))

    errors = precompile([_copy_job(kernels.copy, 64)], max_workers=1)

    assert errors == ["not compiled: IR dumps bypass the JIT disk cache"]
    assert started == []
