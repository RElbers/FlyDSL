# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile-only checks of range(..., prefetch_depth=N), with no GPU.

The device test compares results with plain range, which also passes when the loads end up waited on
one at a time. These tests check the final ISA for loads in flight.
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


def _sum_launcher(depth, name):
    @flyc.kernel(name=name, known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        acc = fx.Float32(0.0)
        for i in range(fx.Int32(tid), n, fx.Int32(BLK), prefetch_depth=depth):
            acc = ArithValue(acc) + ArithValue(buffer_ops.buffer_load(rin, i, vec_width=1, dtype=T.f32))
        buffer_ops.buffer_store(acc, buffer_ops.create_buffer_resource(out, max_size=True), tid)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32):
        k(inp, out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1))

    return launch


def _max_loads_left_in_flight(isa):
    """The largest count of outstanding loads the ISA waits down to (vmcnt, or loadcnt on gfx12)."""
    counts = [int(c) for c in re.findall(r"vmcnt\((\d+)\)", isa)]
    counts += [int(c, 0) for c in re.findall(r"s_wait_loadcnt\s+(0x[0-9a-fA-F]+|\d+)", isa)]
    return max(counts, default=0)


def _compiled_isa(launch, tmp_path, monkeypatch, arch):
    monkeypatch.setenv("FLYDSL_GPU_ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    launch(torch.empty(BLK * 8), torch.empty(BLK), BLK * 8)
    dumped = list(tmp_path.glob("*/*_final_isa.s"))
    assert len(dumped) == 1, dumped
    return dumped[0].read_text()


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1100", "gfx1201"])
def test_prefetch_depth_keeps_loads_in_flight(arch, tmp_path, monkeypatch):
    # At depth 4 the pipelined loop issues 4 loads before waiting on the first: the ISA waits down
    # to 3 outstanding loads somewhere. A loop that waits for every load reaches at most 0.
    isa = _compiled_isa(_sum_launcher(4, f"prefetch_isa_{arch}"), tmp_path, monkeypatch, arch)
    assert _max_loads_left_in_flight(isa) >= 3
