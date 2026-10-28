# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile-only checks of range(..., prefetch_depth=N), with no GPU.

The device test compares results with plain range, which also passes when the loads end up waited on
one at a time. These tests check the final ISA for loads in flight, and that a loop the pass leaves
unchanged is reported as a Python warning at its own source line, also without debug info.
"""

import re
import warnings
from pathlib import Path

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.ir import Context, Module
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler.diagnostics import dsl_ir_diagnostics
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
    monkeypatch.setenv("ARCH", arch)
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


def _barrier_launcher(name):
    @flyc.kernel(name=name, known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        acc = fx.Float32(0.0)
        for i in range(fx.Int32(tid), n, fx.Int32(BLK), prefetch_depth=4):  # remark line
            acc = ArithValue(acc) + ArithValue(buffer_ops.buffer_load(rin, i, vec_width=1, dtype=T.f32))
            gpu.barrier()
        buffer_ops.buffer_store(acc, buffer_ops.create_buffer_resource(out, max_size=True), tid)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32):
        k(inp, out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1))

    return launch


def test_prefetch_remark_names_loop_and_kernel_without_debug_info(monkeypatch):
    # Without debug info the IR loses its source locations before the pass runs. Each rejected loop
    # must still be reported at its own line and kernel, so two kernels give two distinct warnings.
    monkeypatch.setenv("ARCH", "gfx950")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", raising=False)
    line = next(i for i, text in enumerate(Path(__file__).read_text().splitlines(), 1) if "# remark line" in text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for name in ("prefetch_remark_a", "prefetch_remark_b"):
            _barrier_launcher(name)(torch.empty(BLK * 8), torch.empty(BLK), BLK * 8)
    remarks = [w for w in caught if "not pipelined" in str(w.message)]
    assert [(w.filename, w.lineno) for w in remarks] == [(__file__, line)] * 2
    assert "(in prefetch_remark_a)" in str(remarks[0].message)
    assert "(in prefetch_remark_b)" in str(remarks[1].message)


# A marked loop the pass leaves unchanged gets a remark, which FlyDSL reports as a Python warning at
# the user's source line.
_MARKED_LOOP = """
module attributes {{gpu.container_module}} {{
  gpu.module @m {{
    gpu.func @k(%in: !llvm.ptr<1>, %n: i32) kernel {{
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      scf.for %i = %c0 to %n step %c1 : i32 {{
        %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        %v = llvm.load %p : !llvm.ptr<1> -> f32
        {extra}
        llvm.store %v, %p : f32, !llvm.ptr<1>
      }} {{fly.prefetch_depth = 4 : i64{attrs}}} {loc}
      gpu.return
    }}
  }}
}}
"""


def _run_prefetch_pass(extra, *, assume_independent=True, attrs="", loc='loc("user_kernel.py":12:8)'):
    if assume_independent:
        attrs += ", fly.prefetch_assume_independent"
    with Context() as ctx, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = Module.parse(_MARKED_LOOP.format(extra=extra, attrs=attrs, loc=loc))
        with dsl_ir_diagnostics(ctx):
            PassManager.parse("builtin.module(gpu.module(fly-rocdl-prefetch-loops))").run(module.operation)
    return caught


def test_prefetch_loop_left_unchanged_warns_at_user_line():
    caught = _run_prefetch_pass("gpu.barrier")
    assert len(caught) == 1
    assert str(caught[0].message) == "loop not pipelined (prefetch_depth=4): it contains a barrier (in k)"
    assert (caught[0].filename, caught[0].lineno) == ("user_kernel.py", 12)


def test_prefetch_loop_warns_at_recorded_line_without_debug_info():
    # Without debug info the loop's location points into re-parsed IR text; the frontend records the
    # user's location in fly.prefetch_loc, and the remark uses it.
    caught = _run_prefetch_pass("gpu.barrier", attrs=', fly.prefetch_loc = loc("user_kernel.py":21:4)', loc="")
    assert len(caught) == 1
    assert (caught[0].filename, caught[0].lineno) == ("user_kernel.py", 21)


def test_prefetch_loop_pipelined_without_warning():
    assert _run_prefetch_pass("") == []


def test_prefetch_loop_that_stores_needs_assume_independent():
    caught = _run_prefetch_pass("", assume_independent=False)
    assert len(caught) == 1
    assert "it may write memory (llvm.store)" in str(caught[0].message)
    assert "prefetch_assume_independent=True" in str(caught[0].message)
