# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile-only check, with no GPU, that range(..., unroll=N) keeps loads in flight on a loop with i32
bounds. LLVM can only count the trips of such a loop, and so unroll it with a runtime trip count,
because fly-convert-scf-to-cf marks the increment nsw. Without the flag every unrolled copy keeps its
own exit test and waits for its load.
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


def _unrolled_sum_launcher(name):
    @flyc.kernel(name=name, known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        acc = fx.Float32(0.0)
        for i in range(fx.Int32(tid), n, fx.Int32(BLK), unroll=4):
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


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1100", "gfx1201"])
def test_unroll_keeps_loads_in_flight(arch, tmp_path, monkeypatch):
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    _unrolled_sum_launcher(f"unroll_isa_{arch}")(torch.empty(BLK * 8), torch.empty(BLK), BLK * 8)
    dumped = list(tmp_path.glob("*/*_final_isa.s"))
    assert len(dumped) == 1, dumped
    assert _max_loads_left_in_flight(dumped[0].read_text()) >= 3
