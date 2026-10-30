# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``FLYDSL_AUTOTUNE_COMPILE_WORKERS`` compiles the configs in worker processes before they are benchmarked.

The benchmark loop must then load every config from the disk cache rather than compile it again.
"""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import flydsl.compiler.jit_function as jit_function  # noqa: E402
from flydsl.autotune import Autotuner, Config  # noqa: E402
from kernels.norm.rmsnorm_kernel import rmsnorm_direct  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

CONFIGS = [Config(BLOCK_THREADS=256), Config(BLOCK_THREADS=512), Config(BLOCK_THREADS=512, waves_per_eu=2)]


@pytest.fixture
def tuning(tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "jit"))
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "autotune"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    monkeypatch.setenv("FLYDSL_AUTOTUNE_COMPILE_WORKERS", "2")
    # Start from no compiled specializations, with a cache manager bound to this test's cache directory.
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
    return compiles


def _tune(tuner):
    x = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    return tuner.resolve_config(x, gamma, out, 64, N=4096, dtype_str="bf16")


def test_benchmark_loop_loads_every_config_compiled_by_the_workers(tuning, capsys):
    best = _tune(Autotuner(rmsnorm_direct, CONFIGS, key=["N"], warmup=1, rep=2))

    assert f"compiled {len(CONFIGS)}/{len(CONFIGS)} configs in 2 worker processes" in capsys.readouterr().out
    assert tuning == []
    assert best in CONFIGS


def test_a_kernel_the_workers_cannot_import_compiles_in_this_process(tuning, capsys, monkeypatch):
    tuner = Autotuner(rmsnorm_direct, CONFIGS, key=["N"], warmup=1, rep=2)
    monkeypatch.setattr(rmsnorm_direct.func, "__module__", "__main__")
    _tune(tuner)

    assert f"compiled 0/{len(CONFIGS)} configs" in capsys.readouterr().out
    assert len(tuning) == len(CONFIGS)
