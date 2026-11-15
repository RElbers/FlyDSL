# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""range(..., unroll=N, assume_independent=True[, pipeline=True]) on the GPU: every output bit-identical to
plain range.

Each lane runs a grid-stride loop, so the sizes give the lanes trip counts of 0, 1, N - 1, N, N + 1 and
non-multiples of N, with neighbouring lanes one trip apart. Further tests cover runtime bounds with
empty and extreme ranges, an unbounded buffer that faults on any iteration past the trip count, loops
the pass rejects, and a check that the pass runs at all.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_warp_size
from kernels.common import buffer_ops

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

BLK = get_warp_size()
FACTORS = [2, 4, 8]
MAX_N = BLK * 64


def _sizes(factor):
    trips = [
        0,
        1,
        factor - 1,
        factor,
        factor + 1,
        2 * factor,
        2 * factor + 3,
        3 * factor,
        4 * factor,
        5 * factor + 1,
        37,
    ]
    return [0] + [BLK * t + BLK // 2 + 1 for t in trips]


def _directives(factor, pipeline):
    """The range keywords; `factor=None` is plain range, the reference. The AST rewriter forwards
    only literal keywords to the loop builder, so the kernels spell them out."""
    return (-1, False, False) if factor is None else (factor, True, pipeline)


def _load(rsrc, i):
    return buffer_ops.buffer_load(rsrc, i, vec_width=1, dtype=T.f32)


def _body_launcher(body, factor, i32_bounds, pipeline=False):
    """`factor=None` is plain range, the reference."""
    unroll, independent, pipe = _directives(factor, pipeline)

    if i32_bounds:
        # A plain `for` with reassigned variables lowers with i32 bounds.
        @flyc.kernel(known_block_size=[BLK, 1, 1])
        def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
            rin = buffer_ops.create_buffer_resource(inp, max_size=True)
            rout = buffer_ops.create_buffer_resource(out, max_size=True)
            tid = arith.index_cast(T.i32, gpu.thread_id("x"))
            acc = fx.Float32(0.0)
            for i in range(
                fx.Int32(tid), n, fx.Int32(BLK), unroll=unroll, assume_independent=independent, pipeline=pipe
            ):
                acc = body(rin, rout, i, acc)
            buffer_ops.buffer_store(acc, rout, ArithValue(tid) + MAX_N)

    else:
        # range(..., init=...) with yield lowers with index bounds.
        @flyc.kernel(known_block_size=[BLK, 1, 1])
        def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
            rin = buffer_ops.create_buffer_resource(inp, max_size=True)
            rout = buffer_ops.create_buffer_resource(out, max_size=True)
            tid = arith.index_cast(T.index, gpu.thread_id("x"))
            hi = ArithValue(arith.index_cast(T.index, n))
            acc = fx.Float32(0.0)
            for i, st in range(
                ArithValue(tid),
                hi,
                ArithValue(fx.Index(BLK)),
                init=[acc],
                unroll=unroll,
                assume_independent=independent,
                pipeline=pipe,
            ):
                new_acc = body(rin, rout, arith.index_cast(T.i32, i), st[0])
                acc = yield [new_acc]
            buffer_ops.buffer_store(acc, rout, ArithValue(arith.index_cast(T.i32, tid)) + MAX_N)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stream: fx.Stream) -> None:
        k(inp, out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


def _sum(rin, rout, i, acc):
    return ArithValue(acc) + ArithValue(_load(rin, i))


def _scatter(rin, rout, i, acc):
    buffer_ops.buffer_store(ArithValue(_load(rin, i)) * 2.0 + 1.0, rout, i)
    return acc


def _scatter_and_sum(rin, rout, i, acc):
    x = _load(rin, i)
    buffer_ops.buffer_store(ArithValue(x) * 3.0, rout, i)
    return ArithValue(acc) * 0.5 + ArithValue(x)


BODIES = {"sum": _sum, "scatter": _scatter, "scatter_and_sum": _scatter_and_sum}


def _run(launch, n, *, seed=0):
    inp = torch.rand(MAX_N, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed))
    out = torch.zeros(MAX_N + BLK, device="cuda")
    launch(inp, out, n, torch.cuda.current_stream())
    torch.cuda.synchronize()
    return out.cpu()


@pytest.mark.parametrize("pipeline", [False, True], ids=["grouped", "pipelined"])
@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("i32_bounds", [True, False], ids=["i32_bounds", "index_bounds"])
@pytest.mark.parametrize("body", list(BODIES))
def test_unroll_independent_matches_plain_range(body, i32_bounds, factor, pipeline):
    reference = _body_launcher(BODIES[body], None, i32_bounds)
    unrolled = _body_launcher(BODIES[body], factor, i32_bounds, pipeline)
    for n in _sizes(factor):
        assert torch.equal(_run(unrolled, n), _run(reference, n)), f"n={n}"


def _carried_offset_launcher(factor, pipeline=False):
    # The load's offset is carried from one iteration to the next and bumped after the load.
    unroll, independent, pipe = _directives(factor, pipeline)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        rout = buffer_ops.create_buffer_resource(out, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        acc = fx.Float32(0.0)
        for i, st in range(
            0, n, 1, init=[fx.Int32(tid), fx.Float32(0.0)], unroll=unroll, assume_independent=independent, pipeline=pipe
        ):
            x = _load(rin, st[0])
            buffer_ops.buffer_store(x, rout, st[0])
            new_off, acc = yield [ArithValue(st[0]) + BLK, ArithValue(st[1]) * 0.5 + ArithValue(x)]
        buffer_ops.buffer_store(acc, rout, ArithValue(tid) + MAX_N)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stream: fx.Stream):
        k(inp, out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("pipeline", [False, True], ids=["grouped", "pipelined"])
@pytest.mark.parametrize("factor", FACTORS)
def test_unroll_independent_carried_offset(factor, pipeline):
    reference, unrolled = _carried_offset_launcher(None), _carried_offset_launcher(factor, pipeline)
    for trips in [
        0,
        1,
        factor - 1,
        factor,
        factor + 1,
        2 * factor,
        2 * factor + 3,
        3 * factor,
        4 * factor,
        5 * factor + 1,
        37,
    ]:
        assert torch.equal(_run(unrolled, trips), _run(reference, trips)), f"trips={trips}"


def _bounds_launcher(factor, pipeline=False):
    unroll, independent, pipe = _directives(factor, pipeline)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, lb: fx.Int32, ub: fx.Int32, step: fx.Int32) -> None:
        # Bounded, so the loads at negative indices read zeros instead of faulting.
        rin = buffer_ops.create_buffer_resource(inp, max_size=False, num_records_bytes=4096)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        acc = fx.Float32(0.0)
        for i in range(lb, ub, step, unroll=unroll, assume_independent=independent, pipeline=pipe):
            acc = ArithValue(acc) * 0.5 + ArithValue(_load(rin, i))
        buffer_ops.buffer_store(acc, buffer_ops.create_buffer_resource(out, max_size=True), tid)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, lb: fx.Int32, ub: fx.Int32, step: fx.Int32, stream: fx.Stream):
        k(inp, out, lb, ub, step).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("pipeline", [False, True], ids=["grouped", "pipelined"])
@pytest.mark.parametrize("factor", FACTORS)
def test_unroll_independent_runtime_bounds(factor, pipeline):
    # Empty ranges with ub < lb stay empty, and neither the span, the trip count nor the step times N
    # may wrap in i32.
    reference, unrolled = _bounds_launcher(None), _bounds_launcher(factor, pipeline)
    inp = torch.arange(1, 1025, device="cuda", dtype=torch.float32)
    for bounds in [
        (100, 90, 4),
        (100, 99, 1),
        (5, 5, 1),
        (0, 3, 1),
        (90, 100, 4),
        (3, 1000, 7),
        (0, -(1 << 31), 1),
        (-(1 << 31), -(1 << 31), 1),
        (0, 1, 1 << 30),
        (-(1 << 31), 1 << 30, 1 << 29),
        (-(1 << 31), (1 << 31) - (1 << 29) - 1, 1 << 29),
        (0, 1, (1 << 31) - 1),
    ]:
        results = []
        for launch in (reference, unrolled):
            out = torch.full((BLK,), -1.0, device="cuda")
            launch(inp, out, *bounds, torch.cuda.current_stream())
            torch.cuda.synchronize()
            results.append(out.cpu())
        assert torch.equal(results[1], results[0]), bounds


def _strided_launcher(factor, pipeline=False):
    unroll, independent, pipe = _directives(factor, pipeline)

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stride: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        rout = buffer_ops.create_buffer_resource(out, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        for i in range(fx.Int32(0), n, fx.Int32(1), unroll=unroll, assume_independent=independent, pipeline=pipe):
            x = _load(rin, ArithValue(i) * ArithValue(stride) + ArithValue(tid))
            buffer_ops.buffer_store(x, rout, ArithValue(i) * BLK + ArithValue(tid))

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stride: fx.Int32, stream: fx.Stream):
        k(inp, out, n, stride).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


def _check_no_load_past_the_trip_count(factor, pipeline):
    inp = torch.ones(BLK, device="cuda")
    for n in (0, 1):
        outs = []
        for launch in (_strided_launcher(None), _strided_launcher(factor, pipeline)):
            out = torch.zeros(BLK * factor, device="cuda")
            launch(inp, out, n, 1 << 28, torch.cuda.current_stream())
            torch.cuda.synchronize()
            outs.append(out.cpu())
        assert torch.equal(outs[1], outs[0]), f"n={n}"


@pytest.mark.parametrize("pipeline", [False, True], ids=["grouped", "pipelined"])
@pytest.mark.parametrize("factor", FACTORS)
def test_unroll_independent_does_not_load_past_the_trip_count(factor, pipeline):
    # Iterations are 1 GiB apart and the buffer holds one row, with an unbounded (max_size) buffer
    # resource: any iteration past the trip count would fault the GPU. The check runs in a subprocess,
    # so a fault fails this test instead of aborting the pytest run.
    here = Path(__file__).resolve().parent
    code = (
        f"import sys; sys.path.insert(0, {str(here)!r}); "
        f"import test_unroll_independent as t; t._check_no_load_past_the_trip_count({factor}, {pipeline})"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]


def _accumulate_in_memory(rin, rout, i, acc):
    # Every iteration reads what the previous one stored, against the assertion; the pass rejects
    # the loop because the reload's address is the same in every iteration.
    tid = arith.index_cast(T.i32, gpu.thread_id("x"))
    total = ArithValue(_load(rout, ArithValue(tid) + MAX_N)) + ArithValue(_load(rin, i))
    buffer_ops.buffer_store(total, rout, ArithValue(tid) + MAX_N)
    return acc


def _shuffle(rin, rout, i, acc):
    # Lane-dependent bounds with a lane exchange: the pass rejects the loop.
    x = _load(rin, i)
    return ArithValue(acc) + ArithValue(x) * ArithValue(gpu.shuffle_xor(x, 1, BLK))


@pytest.mark.parametrize("pipeline", [False, True], ids=["grouped", "pipelined"])
@pytest.mark.parametrize("factor", [4])
@pytest.mark.parametrize("body", [_accumulate_in_memory, _shuffle], ids=["accumulate_in_memory", "shuffle"])
def test_unroll_independent_rejected_loops_match_plain_range(body, factor, pipeline):
    reference, candidate = _body_launcher(body, None, True), _body_launcher(body, factor, True, pipeline)
    for n in _sizes(factor):
        assert torch.equal(_run(candidate, n), _run(reference, n)), f"n={n}"


def _buffer_loads_in_llvm_ir(launch, tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    _run(launch, BLK * 8)
    dumped = list(tmp_path.glob("*/*_llvm_ir.ll"))
    assert len(dumped) == 1, dumped
    return sum("call" in line and "buffer.load" in line for line in dumped[0].read_text().splitlines())


@pytest.mark.parametrize(
    "factor, pipeline, want",
    [(None, False, 1), (4, False, 5), (4, True, 17)],
    ids=["plain", "grouped", "pipelined"],
)
def test_unroll_independent_runs_in_the_compile_pipeline(factor, pipeline, want, tmp_path, monkeypatch):
    # A comparison with plain range also passes if the pass never runs. Unrolled by N, the loop's
    # one load becomes N loads in the main loop and one in the remainder loop. Pipelined, the
    # prologue adds N, and the pipelined loop is unrolled by 2 with a one-iteration remainder loop.
    launch = _body_launcher(_scatter, factor, True, pipeline)
    assert _buffer_loads_in_llvm_ir(launch, tmp_path, monkeypatch) == want
