# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""range(..., prefetch_depth=N) on the GPU: every output bit-identical to plain range.

Each case is one loop body, compiled with plain range and with prefetch_depth. The sizes hit the
edges of the pipelined loop's prologue, epilogue and remainder: no iterations, lanes with no
iterations, fewer iterations than the depth, and lanes whose trip counts differ by one on either
side of a multiple of the depth (where a lane-exchange op would see the wrong lanes unless the
pass splits the loop at the wave's shortest trip count). Further tests cover runtime bounds with
empty ranges, loads issued past the trip count, lane-dependent bounds from mbcnt, and an address
carried from one iteration to the next.
"""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_warp_size
from kernels.common import buffer_ops, mem_ops

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

BLK = get_warp_size()
SIZES = [0, 5, BLK * 3 + 17, BLK * 47 + BLK // 2, BLK * 64]


def _load(rsrc, i):
    return buffer_ops.buffer_load(rsrc, i, vec_width=1, dtype=T.f32)


def _sum(r, i, acc):
    return ArithValue(acc) + ArithValue(_load(r["in"], i))


def _raw_sum(r, i, acc):
    x = mem_ops.global_load(r["in_ptr"], fx.Int64(i) * 4, T.f32, alignment=4)
    return ArithValue(acc) + ArithValue(x)


def _min(r, i, acc):
    return arith.min(acc, ArithValue(_load(r["in"], i)) + 1.0)


def _shuffle(r, i, acc):
    x = _load(r["in"], i)
    return ArithValue(acc) + ArithValue(x) * ArithValue(gpu.shuffle_xor(x, 1, BLK))


def _store_reload(r, i, acc):
    # Left unchanged by the pass: the reload would otherwise run ahead of the store.
    buffer_ops.buffer_store(ArithValue(_load(r["in"], i)) + 1.0, r["tmp"], i)
    return ArithValue(acc) + ArithValue(_load(r["tmp"], i))


def _intrinsic_reduce(r, i, acc):
    bits = arith.bitcast(T.i32, _load(r["in"], i))
    minimum = llvm.call_intrinsic(T.i32, "llvm.amdgcn.wave.reduce.umin", [bits, fx.Int32(0).ir_value()], [], [])
    return ArithValue(acc) + ArithValue(arith.bitcast(T.f32, minimum))


BODIES = {
    "sum": _sum,
    "raw_sum": _raw_sum,
    "min": _min,
    "shuffle": _shuffle,
    "store_reload": _store_reload,
    "intrinsic_reduce": _intrinsic_reduce,
}
INIT = {"min": 1.0e30}


def _launcher(body_name, depth, i32_bounds):
    body = BODIES[body_name]
    init = INIT.get(body_name, 0.0)

    def resources(inp, tmp):
        return {
            "in": buffer_ops.create_buffer_resource(inp, max_size=True),
            "tmp": buffer_ops.create_buffer_resource(tmp, max_size=True),
            "in_ptr": mem_ops.aligned_ptr(inp),
        }

    if i32_bounds:
        # A plain `for` with a reassigned variable lowers with i32 bounds.
        @flyc.kernel(known_block_size=[BLK, 1, 1])
        def k(inp: fx.Tensor, tmp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
            r = resources(inp, tmp)
            tid = arith.index_cast(T.i32, gpu.thread_id("x"))
            acc = fx.Float32(init)
            for i in range(fx.Int32(tid), n, fx.Int32(BLK), prefetch_depth=depth):
                acc = body(r, i, acc)
            buffer_ops.buffer_store(acc, buffer_ops.create_buffer_resource(out, max_size=True), tid)

    else:
        # range(..., init=...) with yield lowers with index bounds.
        @flyc.kernel(known_block_size=[BLK, 1, 1])
        def k(inp: fx.Tensor, tmp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
            r = resources(inp, tmp)
            tid = arith.index_cast(T.index, gpu.thread_id("x"))
            hi = ArithValue(arith.index_cast(T.index, n))
            acc = fx.Float32(init)
            for i, st in range(
                ArithValue(tid), hi, ArithValue(fx.Index(BLK)), init=[fx.Float32(init)], prefetch_depth=depth
            ):
                new_acc = body(r, arith.index_cast(T.i32, i), st[0])
                acc = yield [new_acc]
            out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
            buffer_ops.buffer_store(acc, out_rsrc, arith.index_cast(T.i32, gpu.thread_id("x")))

    @flyc.jit
    def launch(inp: fx.Tensor, tmp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stream: fx.Stream) -> None:
        k(inp, tmp, out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


def _run(launch, n):
    g = torch.Generator(device="cuda").manual_seed(n)
    inp = torch.rand(max(SIZES), device="cuda", generator=g)
    tmp = torch.zeros(max(SIZES), device="cuda")
    out = torch.zeros(BLK, device="cuda")
    launch(inp, tmp, out, n, torch.cuda.current_stream())
    torch.cuda.synchronize()
    return out.cpu(), tmp.cpu()


@pytest.mark.parametrize("depth", [1, 4, 16])
@pytest.mark.parametrize(
    "body_name, i32_bounds",
    [
        ("sum", False),
        ("sum", True),
        ("raw_sum", False),
        ("min", False),
        ("shuffle", False),
        ("shuffle", True),
        ("store_reload", False),
        ("intrinsic_reduce", False),
        ("intrinsic_reduce", True),
    ],
)
def test_prefetch_depth_matches_plain_range(body_name, i32_bounds, depth):
    reference = _launcher(body_name, None, i32_bounds)
    pipelined = _launcher(body_name, depth, i32_bounds)
    for n in SIZES:
        want = _run(reference, n)
        got = _run(pipelined, n)
        assert all(torch.equal(a, b) for a, b in zip(got, want)), f"n={n}"


def _launch_and_read(launch, *args, out_init=0.0):
    out = torch.full((BLK,), out_init, device="cuda")
    launch(*args[:1], out, *args[1:], torch.cuda.current_stream())
    torch.cuda.synchronize()
    return out.cpu()


def _bounds_launcher(depth):
    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, lb: fx.Int32, ub: fx.Int32, step: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        acc = fx.Float32(0.0)
        for i in range(lb, ub, step, prefetch_depth=depth):
            acc = ArithValue(acc) + ArithValue(_load(rin, i))
        buffer_ops.buffer_store(acc, buffer_ops.create_buffer_resource(out, max_size=True), tid)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, lb: fx.Int32, ub: fx.Int32, step: fx.Int32, stream: fx.Stream):
        k(inp, out, lb, ub, step).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("depth", [1, 4, 16])
def test_prefetch_depth_runtime_bounds(depth):
    # An empty range with ub < lb must stay empty (the unroller's trip count wraps without a clamp).
    reference, pipelined = _bounds_launcher(None), _bounds_launcher(depth)
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
        (0, 1, (1 << 31) - 1),
    ]:
        want = _launch_and_read(reference, inp, *bounds, out_init=-1.0)
        got = _launch_and_read(pipelined, inp, *bounds, out_init=-1.0)
        assert torch.equal(got, want), bounds


def _static_launcher(depth, trips):
    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        acc = fx.Float32(0.0)
        for i in range(trips, prefetch_depth=depth):
            acc = ArithValue(acc) + ArithValue(_load(rin, ArithValue(tid) + ArithValue(i) * BLK))
        buffer_ops.buffer_store(acc, buffer_ops.create_buffer_resource(out, max_size=True), tid)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, stream: fx.Stream):
        k(inp, out).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize("trips", [0, 1, 3, 4, 5, 8])
def test_prefetch_depth_static_bounds(depth, trips):
    inp = torch.arange(BLK * 8, device="cuda", dtype=torch.float32)
    want = _launch_and_read(_static_launcher(None, trips), inp)
    got = _launch_and_read(_static_launcher(depth, trips), inp)
    assert torch.equal(got, want)


def _strided_launcher(depth):
    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stride: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        acc = fx.Float32(0.0)
        for i in range(fx.Int32(0), n, fx.Int32(1), prefetch_depth=depth):
            acc = ArithValue(acc) + ArithValue(_load(rin, ArithValue(i) * ArithValue(stride) + ArithValue(tid)))
        buffer_ops.buffer_store(acc, buffer_ops.create_buffer_resource(out, max_size=True), tid)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stride: fx.Int32, stream: fx.Stream):
        k(inp, out, n, stride).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("depth", [1, 4, 16])
def test_prefetch_depth_does_not_load_past_the_trip_count(depth):
    # Iterations are 1 GiB apart and the buffer holds one row, with an unbounded (max_size) buffer
    # resource: an out-of-range iteration the pipelined loop let through would fault the GPU.
    inp = torch.ones(BLK, device="cuda")
    for n in (0, 1):
        want = _launch_and_read(_strided_launcher(None), inp, n, 1 << 28)
        got = _launch_and_read(_strided_launcher(depth), inp, n, 1 << 28)
        assert torch.equal(got, want), f"n={n}"


def _lane_id():
    lo = rocdl.mbcnt_lo(T.i32, fx.Int32(-1).ir_value(), fx.Int32(0).ir_value())
    if BLK == 32:
        return fx.Int32(lo)
    return fx.Int32(rocdl.mbcnt_hi(T.i32, fx.Int32(-1).ir_value(), lo))


def _mbcnt_shuffle_launcher(depth):
    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        lane = _lane_id()
        acc = fx.Float32(0.0)
        for i in range(lane, n, fx.Int32(BLK), prefetch_depth=depth):
            x = _load(rin, i)
            acc = ArithValue(acc) + ArithValue(x) * ArithValue(gpu.shuffle_xor(x, 1, BLK))
        buffer_ops.buffer_store(acc, buffer_ops.create_buffer_resource(out, max_size=True), lane)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stream: fx.Stream):
        k(inp, out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


def _carried_offset_launcher(depth):
    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        acc = fx.Float32(0.0)
        for i, st in range(0, n, 1, init=[fx.Int32(tid), fx.Float32(0.0)], prefetch_depth=depth):
            x = _load(rin, st[0])
            new_off, acc = yield [ArithValue(st[0]) + BLK, ArithValue(st[1]) + ArithValue(x)]
        buffer_ops.buffer_store(acc, buffer_ops.create_buffer_resource(out, max_size=True), tid)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stream: fx.Stream):
        k(inp, out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("depth", [1, 4, 16])
@pytest.mark.parametrize(
    "make_launcher, sizes",
    [
        # Lane-dependent bounds from mbcnt, not gpu.thread_id: the loop needs the wave-uniform split.
        (_mbcnt_shuffle_launcher, SIZES),
        # The load address is carried from the previous iteration and bumped after the load.
        (_carried_offset_launcher, [n // BLK for n in SIZES]),
    ],
    ids=["mbcnt_shuffle", "carried_offset"],
)
def test_prefetch_depth_loop_shapes(make_launcher, sizes, depth):
    reference, pipelined = make_launcher(None), make_launcher(depth)
    inp = torch.rand(max(SIZES), device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
    for n in sizes:
        want = _launch_and_read(reference, inp, n)
        got = _launch_and_read(pipelined, inp, n)
        assert torch.equal(got, want), f"n={n}"
