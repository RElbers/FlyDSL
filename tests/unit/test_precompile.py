# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``precompile`` compiles kernel specializations in reusable worker processes into the JIT disk cache."""

import functools
import gc
import importlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import types

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler as flyc  # noqa: E402
import flydsl.compiler.jit_function as jit_function  # noqa: E402
import flydsl.compiler.precompile as precompile_module  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.autotune import Autotuner, Config  # noqa: E402
from flydsl.compiler.extern_link import _register_link_metadata  # noqa: E402
from flydsl.compiler.kernel_function import CompilationContext  # noqa: E402
from flydsl.compiler.precompile import CompileJob, precompile  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.gemm_a16w16_gfx950 import (  # noqa: E402
    GEMM_A16W16_DTYPE_BF16,
    gemm_a16w16_gfx950,
    make_gemm_a16w16_gfx950_param,
)
from kernels.norm.rmsnorm_kernel import build_rmsnorm_module, rmsnorm_direct  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

CONFIGS = [Config(BLOCK_THREADS=256), Config(BLOCK_THREADS=512), Config(BLOCK_THREADS=512, waves_per_eu=2)]


@pytest.fixture
def compiles(tmp_path, monkeypatch):
    """The functions traced and compiled in this process, starting from an empty cache in ``tmp_path``."""
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "jit"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    for kernel in (rmsnorm_direct, gemm_a16w16_gfx950):
        monkeypatch.setattr(kernel, "_mem_cache", {})
        monkeypatch.setattr(kernel, "_call_state_cache", {})
        monkeypatch.setattr(kernel, "cache_manager", None)
        monkeypatch.setattr(kernel, "manager_key", None)
    compiled = []
    trace_and_compile = jit_function.JitFunction._trace_and_compile

    def counted(self, *args, **kwargs):
        compiled.append(self.func.__name__)
        return trace_and_compile(self, *args, **kwargs)

    monkeypatch.setattr(jit_function.JitFunction, "_trace_and_compile", counted)
    return compiled


def _rmsnorm_job(block_threads, fn=rmsnorm_direct):
    x = torch.empty(64, 4096, dtype=torch.bfloat16)
    gamma = torch.empty(4096, dtype=torch.bfloat16)
    return CompileJob(
        fn, (x, gamma, torch.empty_like(x), 64), dict(N=4096, dtype_str="bf16", BLOCK_THREADS=block_threads)
    )


def _compile_in_this_process(job):
    job.fn._compile_only(*job.args, **job.kwargs)


def _factory_job():
    x = torch.empty(64, 2048, dtype=torch.float32)
    gamma = torch.empty(2048, dtype=torch.float32)
    return CompileJob(
        build_rmsnorm_module,
        (x, gamma, torch.empty_like(x), 64),
        factory_kwargs=dict(N=2048, dtype_str="f32", BLOCK_THREADS=256),
    )


def test_compiled_jobs_load_from_the_disk_cache(compiles):
    assert precompile([_rmsnorm_job(256), _factory_job()], workers=2) == [None, None]

    _compile_in_this_process(_rmsnorm_job(256))
    job = _factory_job()
    build_rmsnorm_module(**job.factory_kwargs)._compile_only(*job.args)
    assert compiles == []


def test_workers_fork_from_one_interpreter_and_exit_when_their_jobs_are_done(compiles, monkeypatch):
    started = []
    popen = subprocess.Popen

    def recorded(*args, **kwargs):
        started.append(popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(precompile_module.subprocess, "Popen", recorded)

    assert precompile([_rmsnorm_job(256), _rmsnorm_job(512)], workers=2) == [None, None]

    # The interpreter the workers fork from exits once it has waited for all of them.
    assert len(started) == 1
    assert started[0].returncode == 0


class _ExitsTheWorker:
    def __reduce__(self):
        return os._exit, (1,)


def test_workers_run_no_thread_pools_besides_the_compile(compiles, tmp_path, monkeypatch):
    # Idle OpenMP and BLAS pools started by torch and numpy compete with the compiles for the cores. The probe
    # writes the worker's thread count when the worker unpickles the job.
    path = tmp_path / "threads"
    (tmp_path / "thread_probe.py").write_text(
        "import os\n\n\nclass Probe:\n    def __reduce__(self):\n        return write, ()\n\n\n"
        f"def write():\n    open({str(path)!r}, 'w').write(str(len(os.listdir('/proc/self/task'))))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "thread_probe", raising=False)
    probe = importlib.import_module("thread_probe").Probe()

    precompile([CompileJob(rmsnorm_direct, (probe,))], workers=1)

    assert path.read_text() == "1"


class _SleepsInTheWorker:
    def __init__(self, seconds):
        self.seconds = seconds

    def __reduce__(self):
        return time.sleep, (self.seconds,)


def test_an_interrupt_stops_the_compiles_in_progress_and_the_rest(compiles, monkeypatch):
    started = []
    popen = subprocess.Popen

    def recorded(*args, **kwargs):
        started.append(popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(precompile_module.subprocess, "Popen", recorded)
    jobs = [CompileJob(rmsnorm_direct, (_SleepsInTheWorker(60),)) for _ in range(2)]
    threading.Timer(3, os.kill, (os.getpid(), signal.SIGINT)).start()
    start = time.perf_counter()

    with pytest.raises(KeyboardInterrupt):
        precompile(jobs, workers=1)

    assert time.perf_counter() - start < 15
    assert started[0].poll() is not None


def test_jobs_left_when_every_worker_died_compile_in_new_workers(compiles):
    dying = [CompileJob(rmsnorm_direct, (_ExitsTheWorker(),)) for _ in range(2)]
    errors = precompile(dying + [_rmsnorm_job(block_threads) for block_threads in (128, 256, 512, 256)], workers=2)

    assert errors == ["the compile worker exited"] * 2 + [None] * 4


def test_precompile_refuses_an_empty_cache_directory(compiles, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", "")

    with pytest.raises(RuntimeError, match="FLYDSL_RUNTIME_CACHE_DIR"):
        precompile([_rmsnorm_job(256)], workers=1)


@pytest.mark.skipif(not str(get_rocm_arch()).startswith("gfx95"), reason="the struct-parameter GEMM is gfx95x only")
def test_a_struct_argument_specializes_the_worker_compile_as_it_does_the_call(compiles):
    # The struct's class is generated for its constexpr values, so it is sent as the struct it was built from.
    param = make_gemm_a16w16_gfx950_param(
        GEMM_A16W16_DTYPE_BF16, GEMM_A16W16_DTYPE_BF16, block_m=128, block_n=128, m_waves=2, n_waves=2
    )
    a = torch.empty(256, 256, dtype=torch.bfloat16)
    signal = torch.empty(64, dtype=torch.int32)
    job = CompileJob(gemm_a16w16_gfx950, (torch.empty_like(a), a, a, a, signal, signal, 1, param))

    assert precompile([job], workers=1) == [None]
    _compile_in_this_process(job)

    assert compiles == []


def test_a_job_that_cannot_be_sent_is_reported_and_the_rest_compile(compiles):
    unsendable = CompileJob(rmsnorm_direct, (lambda: None,))
    errors = precompile([unsendable, _rmsnorm_job(256)], workers=2)

    assert errors[0].startswith("cannot send to a compile worker")
    assert errors[1] is None


def test_a_job_that_fails_in_a_worker_does_not_stop_the_rest(compiles, monkeypatch):
    # The class pickles here but its module does not exist in the worker, so the job fails to unpickle there.
    phantom = types.ModuleType("phantom")
    phantom.Marker = type("Marker", (), {"__module__": "phantom"})
    monkeypatch.setitem(sys.modules, "phantom", phantom)
    unloadable = CompileJob(rmsnorm_direct, (phantom.Marker(),))
    failing = _rmsnorm_job(256)
    failing.kwargs["dtype_str"] = "f64"

    errors = precompile([unloadable, failing, _rmsnorm_job(256)], workers=1)

    assert errors[0] is not None and errors[1] is not None
    assert errors[2] is None


def _kernel_module(tmp_path, monkeypatch, decorators):
    """A module on the import path whose ``kernel`` is RMSNorm under ``decorators``."""
    source = f"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Config, autotune
from kernels.norm.rmsnorm_kernel import build_rmsnorm_module

{decorators}
def kernel(x: fx.Tensor, gamma: fx.Tensor, out: fx.Tensor, m: fx.Int32, N: fx.Constexpr[int],
           dtype_str: fx.Constexpr[str], BLOCK_THREADS: fx.Constexpr[int]):
    build_rmsnorm_module(N, dtype_str, BLOCK_THREADS=BLOCK_THREADS)(x, gamma, out, m)
"""
    (tmp_path / "test_precompile_kernel.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "test_precompile_kernel", raising=False)
    return importlib.import_module("test_precompile_kernel").kernel


def test_jobs_compile_with_the_hints_the_call_gets(compiles, monkeypatch):
    monkeypatch.setattr(rmsnorm_direct, "compile_hints", {"unsafe_fp_math": True})
    job = _rmsnorm_job(256)
    with CompilationContext.compile_hints({"fast_fp_math": True}):
        assert precompile([job], workers=1) == [None]
        _compile_in_this_process(job)

    assert compiles == []


def test_jobs_compile_without_hints_the_caller_cleared(compiles, tmp_path, monkeypatch):
    kernel = _kernel_module(tmp_path, monkeypatch, '@flyc.compile[{"unsafe_fp_math": True}]\n@flyc.jit')
    monkeypatch.setattr(kernel, "compile_hints", {})
    job = _rmsnorm_job(256, kernel)

    assert precompile([job], workers=1) == [None]
    _compile_in_this_process(job)

    assert compiles == []


def test_a_kernel_under_a_stacked_autotune_decorator_is_found(compiles, tmp_path, monkeypatch):
    tuned = _kernel_module(tmp_path, monkeypatch, f'@autotune(configs={CONFIGS!r}, key=["N"])\n@flyc.jit')
    job = _rmsnorm_job(256, tuned.fn)

    assert precompile([job], workers=1) == [None]
    _compile_in_this_process(job)

    assert compiles == []


def test_autotune_configs_precompiled_before_a_search_load_from_the_disk_cache(compiles, tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "autotune"))
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    args, kwargs = (x, gamma, torch.empty_like(x), 64), dict(N=4096, dtype_str="bf16")

    jobs = [CompileJob(rmsnorm_direct, args, {**kwargs, **c.all_kwargs()}, c.compiler_opts()) for c in CONFIGS]
    assert precompile(jobs, workers=2) == [None] * len(CONFIGS)
    Autotuner(rmsnorm_direct, CONFIGS, key=["N"], warmup=1, rep=2).resolve_config(*args, **kwargs)

    assert compiles == []


def test_workers_import_from_the_callers_path_not_their_working_directory(compiles, tmp_path, monkeypatch):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "__init__.py").write_text(
        "raise ImportError('not the kernels package the caller imports')\n"
    )
    monkeypatch.chdir(tmp_path)

    assert precompile([_rmsnorm_job(256)], workers=1) == [None]


def test_jobs_report_a_worker_that_cannot_start(compiles, monkeypatch):
    def cannot_start(*args, **kwargs):
        raise OSError("Resource temporarily unavailable")

    monkeypatch.setattr(precompile_module.subprocess, "Popen", cannot_start)

    errors = precompile([_rmsnorm_job(256), _rmsnorm_job(512)], workers=2)

    assert errors == ["cannot start a compile worker: Resource temporarily unavailable"] * 2


def test_precompile_refuses_while_ir_dumps_bypass_the_disk_cache(compiles, tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path / "dump"))

    with pytest.raises(RuntimeError, match="IR dumps"):
        precompile([_rmsnorm_job(256)], workers=1)


@flyc.jit
def _rmsnorm_with_a_post_load_step(x: fx.Tensor, gamma: fx.Tensor, out: fx.Tensor, m: fx.Int32):
    _register_link_metadata(None, print)
    build_rmsnorm_module(4096, "bf16", BLOCK_THREADS=256)(x, gamma, out, m)


def test_a_kernel_the_disk_cache_does_not_keep_fails_to_compile_into_it(compiles):
    job = _rmsnorm_job(256)

    with pytest.raises(RuntimeError, match="the disk cache does not keep"):
        _rmsnorm_with_a_post_load_step._compile_to_disk_cache(*job.args)


def test_a_failed_cache_write_fails_the_compile_into_the_disk_cache(compiles, monkeypatch):
    def disk_full(cache_file, value):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(jit_function.JitCacheManager, "_write_cache_file", staticmethod(disk_full))
    job = _rmsnorm_job(256)

    with pytest.raises(OSError, match="No space left on device"):
        job.fn._compile_to_disk_cache(*job.args, **job.kwargs)


def test_compiling_into_the_disk_cache_does_not_trust_the_in_memory_cache(compiles, tmp_path):
    job = _rmsnorm_job(256)
    job.fn._compile_to_disk_cache(*job.args, **job.kwargs)
    shutil.rmtree(tmp_path / "jit")

    job.fn._compile_to_disk_cache(*job.args, **job.kwargs)

    assert compiles == ["rmsnorm_direct", "rmsnorm_direct"]


def test_compiling_a_launched_specialization_into_the_disk_cache_keeps_the_launched_one(compiles, tmp_path):
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    launch = functools.partial(rmsnorm_direct, x, gamma, out, 64, N=4096, dtype_str="bf16", BLOCK_THREADS=256)
    launch()
    expected = out.clone()
    launched = list(rmsnorm_direct._mem_cache.values())
    shutil.rmtree(tmp_path / "jit")

    job = _rmsnorm_job(256)
    job.fn._compile_to_disk_cache(*job.args, **job.kwargs)

    assert compiles == ["rmsnorm_direct", "rmsnorm_direct"]
    # The launch's call state points into this artifact's code; replacing it would free that code.
    assert list(rmsnorm_direct._mem_cache.values()) == launched
    del launched
    gc.collect()
    out.zero_()
    launch()
    assert torch.equal(out, expected)
