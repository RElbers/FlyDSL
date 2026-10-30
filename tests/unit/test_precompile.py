# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``precompile`` compiles kernel specializations in reusable worker processes into the JIT disk cache."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler.jit_function as jit_function  # noqa: E402
import flydsl.compiler.precompile as precompile_module  # noqa: E402
from flydsl.compiler.precompile import CompileJob, precompile  # noqa: E402
from kernels.norm.rmsnorm_kernel import build_rmsnorm_module, rmsnorm_direct  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "jit"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.setattr(rmsnorm_direct, "_mem_cache", {})
    monkeypatch.setattr(rmsnorm_direct, "_call_state_cache", {})
    monkeypatch.setattr(rmsnorm_direct, "cache_manager", None)
    monkeypatch.setattr(rmsnorm_direct, "manager_key", None)
    compiles = []
    trace_and_compile = jit_function.JitFunction._trace_and_compile

    def counted(self, *args, **kwargs):
        compiles.append(self.func.__name__)
        return trace_and_compile(self, *args, **kwargs)

    monkeypatch.setattr(jit_function.JitFunction, "_trace_and_compile", counted)
    yield compiles
    precompile_module._pool.close()


def _rmsnorm_job(block_threads):
    x = torch.empty(64, 4096, dtype=torch.bfloat16)
    gamma = torch.empty(4096, dtype=torch.bfloat16)
    return CompileJob(
        rmsnorm_direct, (x, gamma, torch.empty_like(x), 64), dict(N=4096, dtype_str="bf16", BLOCK_THREADS=block_threads)
    )


def _factory_job():
    x = torch.empty(64, 2048, dtype=torch.float32)
    gamma = torch.empty(2048, dtype=torch.float32)
    return CompileJob(
        build_rmsnorm_module,
        (x, gamma, torch.empty_like(x), 64),
        factory_kwargs=dict(N=2048, dtype_str="f32", BLOCK_THREADS=256),
    )


def test_compiled_jobs_load_from_the_disk_cache(cache):
    assert precompile([_rmsnorm_job(256), _factory_job()], workers=2) == [None, None]

    job = _rmsnorm_job(256)
    job.fn._compile_only(*job.args, **job.kwargs)
    job = _factory_job()
    build_rmsnorm_module(**job.factory_kwargs)._compile_only(*job.args)
    assert cache == []


def test_workers_stay_up_between_calls_until_the_environment_changes(cache, monkeypatch):
    precompile([_rmsnorm_job(256), _rmsnorm_job(512)], workers=2)
    first = [worker.pid for worker in precompile_module._pool.workers]
    precompile([_rmsnorm_job(128)], workers=1)
    assert [worker.pid for worker in precompile_module._pool.workers] == first

    monkeypatch.setenv("FLYDSL_PRECOMPILE_TEST_MARKER", "1")
    precompile([_rmsnorm_job(128)], workers=1)
    assert not set(worker.pid for worker in precompile_module._pool.workers) & set(first)


def test_a_job_that_cannot_be_sent_is_reported_and_the_rest_compile(cache):
    unsendable = CompileJob(rmsnorm_direct, (lambda: None,))
    errors = precompile([unsendable, _rmsnorm_job(256)], workers=2)

    assert errors[0].startswith("cannot send to a compile worker")
    assert errors[1] is None
