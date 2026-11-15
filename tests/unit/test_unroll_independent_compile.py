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


def _final_isa(launch, tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DEBUG_DUMP_ASM", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    launch(torch.empty(BLK * 8), torch.empty(BLK * 8), BLK * 8)
    dumped = list(tmp_path.glob("*/*_final_isa.s"))
    assert len(dumped) == 1, dumped
    return dumped[0].read_text()


def _max_vmcnt(launch, tmp_path, monkeypatch):
    """The largest count of outstanding memory operations the ISA waits down to."""
    return max((int(c) for c in re.findall(r"vmcnt\((\d+)\)", _final_isa(launch, tmp_path, monkeypatch))), default=0)


_VMEM_LOAD = re.compile(r"\s*(buffer|global|flat)_load")
_VMEM_STORE = re.compile(r"\s*(buffer|global|flat)_(store|atomic)")


def _main_loop_wait_drains(isa):
    """What each vmcnt wait of the main loop forces to complete, on gfx9, where loads and stores
    share the counter and complete in order. The main loop is the backward branch with the most
    vector loads; its body is replayed after a copy of itself, so a wait also sees the previous
    iteration's operations. A wait drains a store when the newest operation it forces is a store,
    and drains the prefetch when it forces a load of the same iteration that another vector memory
    operation of the iteration still follows."""
    lines = isa.split("\n")
    labels = {m.group(1): i for i, line in enumerate(lines) if (m := re.match(r"(\.LBB\d+_\d+):", line))}
    loops = []
    for i, line in enumerate(lines):
        m = re.search(r"s_cbranch_\w+\s+(\.LBB\d+_\d+)", line)
        if m and labels.get(m.group(1), i) < i:
            loops.append(lines[labels[m.group(1)] : i + 1])
    body = max(loops, key=lambda b: sum(bool(_VMEM_LOAD.match(x)) for x in b))
    events = []
    for line in body:
        if m := re.search(r"s_waitcnt\b.*vmcnt\((\d+)\)", line):
            events.append(int(m.group(1)))
        elif _VMEM_LOAD.match(line) or _VMEM_STORE.match(line):
            events.append("L" if _VMEM_LOAD.match(line) else "S")
    issued = [(False, e) for e in events if isinstance(e, str)]  # the previous iteration
    drains = []
    for pos, e in enumerate(events):
        if isinstance(e, str):
            issued.append((True, e))
            continue
        forced = issued[: max(0, len(issued) - e)]
        kinds = set()
        if forced and forced[-1][1] == "S":
            kinds.add("store")
        if any(isinstance(x, str) for x in events[pos + 1 :]) and any(cur and k == "L" for cur, k in forced):
            kinds.add("prefetch")
        drains.append((e, kinds))
    return drains


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


def test_pipelined_store_loop_waits_only_for_loads(tmp_path, monkeypatch):
    # In a pipelined loop that stores, no wait may hold the next group's loads back behind the
    # current group's work, or wait for a store: LLVM's wait at the loop's top must not merge in the
    # prologue's requirement, which would make it wait for the previous iteration's stores too.
    launch = _scatter_launcher(True, "unroll_independent_pipeline_drains", True)
    drains = _main_loop_wait_drains(_final_isa(launch, tmp_path, monkeypatch))
    assert drains and not any(kinds for _, kinds in drains), drains
