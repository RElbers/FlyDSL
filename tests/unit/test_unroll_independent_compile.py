# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile-only checks of range(..., unroll=N, assume_independent=True[, pipeline=True]), with no GPU.

The device test compares results with plain range, which also passes when each load is waited on
before the next one issues. This test checks the final ISA for loads in flight.
"""

import re

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from kernels.common import buffer_ops

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

BLK = 64


def _scatter_launcher(assume_independent, name, pipeline=False):
    @flyc.kernel(name=name, known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        rout = buffer_ops.create_buffer_resource(out, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        for i in range(
            fx.Int32(tid), n, fx.Int32(BLK), unroll=4, assume_independent=assume_independent, pipeline=pipeline
        ):
            x = ArithValue(buffer_ops.buffer_load(rin, i, vec_width=1, dtype=T.f32))
            # Enough dependent work per element that LLVM computes the first element on its own.
            for _ in fx.range_constexpr(16):
                x = x * 0.5 + 1.0
            buffer_ops.buffer_store(x, rout, i)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32):
        k(inp, out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1))

    return launch


def _max_vmcnt(launch, tmp_path, monkeypatch):
    """The largest count of outstanding memory operations the ISA waits down to."""
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DEBUG_DUMP_ASM", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    launch(torch.empty(BLK * 8), torch.empty(BLK * 8), BLK * 8)
    dumped = list(tmp_path.glob("*/*_final_isa.s"))
    assert len(dumped) == 1, dumped
    return max((int(c) for c in re.findall(r"vmcnt\((\d+)\)", dumped[0].read_text())), default=0)


@pytest.mark.parametrize("assume_independent", [False, True])
def test_assume_independent_keeps_loads_in_flight(assume_independent, tmp_path, monkeypatch):
    # With the flag, a group's 4 loads issue before its first store, so the ISA waits for the first
    # load with 3 more in flight. LLVM's unroll=4 alone keeps each load behind the previous store.
    launch = _scatter_launcher(assume_independent, f"unroll_independent_isa_{assume_independent}")
    max_vmcnt = _max_vmcnt(launch, tmp_path, monkeypatch)
    assert (max_vmcnt >= 3) == assume_independent, max_vmcnt


@pytest.mark.parametrize("pipeline", [False, True], ids=["grouped", "pipelined"])
def test_pipeline_keeps_next_group_in_flight(pipeline, tmp_path, monkeypatch):
    # Pipelined, the loop issues the next group's 4 loads before it waits for the current group, so
    # it waits with at least those 4 still outstanding. Grouped, it never has more than 3.
    launch = _scatter_launcher(True, f"unroll_independent_pipeline_isa_{pipeline}", pipeline)
    max_vmcnt = _max_vmcnt(launch, tmp_path, monkeypatch)
    assert (max_vmcnt >= 4) == pipeline, max_vmcnt
