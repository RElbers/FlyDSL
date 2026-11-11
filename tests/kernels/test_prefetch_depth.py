# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""range(..., prefetch_depth=N) on the GPU: every output bit-identical to plain range.

Each case is one loop body, compiled with plain range and with prefetch_depth. The sizes hit the
edges of the pipelined loop's prologue, epilogue and remainder: no iterations, lanes with no
iterations, fewer iterations than the depth, and lanes whose trip counts differ by one on either
side of a multiple of the depth (where a lane-exchange op would see the wrong lanes unless the
pass splits the loop at the wave's shortest trip count). Further tests cover runtime bounds with
empty ranges, loads issued past the trip count, lane-dependent bounds from mbcnt, an address
carried from one iteration to the next, and loops that store, which need prefetch_assume_independent.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_warp_size, is_rdna_arch
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


def _gather(r, i, acc):
    # The second load's address comes from the first, so their guards can't be merged into one.
    j = (fx.Float32(_load(r["in"], i)) * 100.0).to(fx.Int32)
    return ArithValue(acc) + ArithValue(_load(r["in"], j))


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
    "gather": _gather,
    "intrinsic_reduce": _intrinsic_reduce,
}
INIT = {"min": 1.0e30}


def _body_launcher(body_name, depth, i32_bounds):
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
        ("gather", False),
        ("gather", True),
        ("intrinsic_reduce", False),
        ("intrinsic_reduce", True),
    ],
)
def test_prefetch_depth_matches_plain_range(body_name, i32_bounds, depth):
    reference = _body_launcher(body_name, None, i32_bounds)
    pipelined = _body_launcher(body_name, depth, i32_bounds)
    for n in SIZES:
        want = _run(reference, n)
        got = _run(pipelined, n)
        assert all(torch.equal(a, b) for a, b in zip(got, want)), f"n={n}"


def _launch_and_read(launch, inp, *scalars, out_init=0.0):
    out = torch.full((BLK,), out_init, device="cuda")
    launch(inp, out, *scalars, torch.cuda.current_stream())
    torch.cuda.synchronize()
    return out.cpu()


def _compare_on_sizes(reference, pipelined, inp, sizes, out_len):
    for n in sizes:
        outs = []
        for launch in (reference, pipelined):
            out = torch.zeros(out_len, device="cuda")
            launch(inp, out, n, torch.cuda.current_stream())
            torch.cuda.synchronize()
            outs.append(out.cpu())
        assert torch.equal(*outs), f"n={n}"


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


def _check_no_load_past_the_trip_count(depth):
    inp = torch.ones(BLK, device="cuda")
    for n in (0, 1):
        want = _launch_and_read(_strided_launcher(None), inp, n, 1 << 28)
        got = _launch_and_read(_strided_launcher(depth), inp, n, 1 << 28)
        assert torch.equal(got, want), f"n={n}"


@pytest.mark.parametrize("depth", [1, 4, 16])
def test_prefetch_depth_does_not_load_past_the_trip_count(depth):
    # Iterations are 1 GiB apart and the buffer holds one row, with an unbounded (max_size) buffer
    # resource: an out-of-range iteration the pipelined loop let through would fault the GPU. The
    # check runs in a subprocess, so a fault fails this test instead of aborting the pytest run.
    here = Path(__file__).resolve().parent
    code = (
        f"import sys; sys.path.insert(0, {str(here)!r}); "
        f"import test_prefetch_depth as t; t._check_no_load_past_the_trip_count({depth})"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]


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


def _scatter_launcher(depth):
    # The flag requires prefetch_depth, so the plain-range reference goes without it.
    assume = depth is not None

    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        rout = buffer_ops.create_buffer_resource(out, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        for i in range(fx.Int32(tid), n, fx.Int32(BLK), prefetch_depth=depth, prefetch_assume_independent=assume):
            buffer_ops.buffer_store(ArithValue(_load(rin, i)) * 2.0, rout, i)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stream: fx.Stream):
        k(inp, out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("depth", [1, 4, 16])
def test_prefetch_depth_assume_independent_scatter(depth):
    # A loop that stores is only pipelined with the flag; each iteration writes its own element.
    reference, pipelined = _scatter_launcher(None), _scatter_launcher(depth)
    inp = torch.rand(max(SIZES), device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
    _compare_on_sizes(reference, pipelined, inp, SIZES, max(SIZES))


def _accumulate_in_memory_launcher(depth, assume):
    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        rout = buffer_ops.create_buffer_resource(out, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        for i in range(fx.Int32(0), n, fx.Int32(1), prefetch_depth=depth, prefetch_assume_independent=assume):
            total = ArithValue(_load(rout, tid)) + ArithValue(_load(rin, i))
            buffer_ops.buffer_store(total, rout, tid)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stream: fx.Stream):
        k(inp, out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("assume", [False, True])
@pytest.mark.parametrize("depth", [1, 4])
def test_prefetch_depth_accumulator_in_memory_left_unchanged(depth, assume):
    # Every iteration reads what the previous one stored. Without the flag the loop stores, and with
    # it the load reads the same address in every iteration, so the pass leaves it unchanged either way.
    reference, candidate = _accumulate_in_memory_launcher(None, False), _accumulate_in_memory_launcher(depth, assume)
    inp = torch.arange(1, 1025, device="cuda", dtype=torch.float32)
    for n in (0, 1, 10, 37):
        want = _launch_and_read(reference, inp, n)
        got = _launch_and_read(candidate, inp, n)
        assert torch.equal(got, want), f"n={n}"


def _buffer_loads_in_llvm_ir(launch, args, tmp_path, monkeypatch):
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    launch(*args, torch.cuda.current_stream())
    torch.cuda.synchronize()
    dumped = list(tmp_path.glob("*/*_llvm_ir.ll"))
    assert len(dumped) == 1, dumped
    return sum("call" in line and "buffer.load" in line for line in dumped[0].read_text().splitlines())


@pytest.mark.parametrize("depth", [None, 4])
@pytest.mark.parametrize("body", ["sum", "scatter"])
def test_prefetch_depth_runs_in_the_compile_pipeline(body, depth, tmp_path, monkeypatch):
    # The other tests only compare against plain range, which also passes if the pass never runs.
    # Pipelined at depth N, the loop's one load becomes N guarded prologue loads, N steady-state
    # loads and one in the remainder loop.
    inp = torch.rand(BLK * 8, device="cuda")
    if body == "sum":
        launch = _body_launcher("sum", depth, i32_bounds=True)
        args = (inp, torch.zeros_like(inp), torch.zeros(BLK, device="cuda"), BLK * 8)
    else:
        launch = _scatter_launcher(depth)
        args = (inp, torch.zeros_like(inp), BLK * 8)
    want = 1 if depth is None else 2 * depth + 1
    assert _buffer_loads_in_llvm_ir(launch, args, tmp_path, monkeypatch) == want


def _multi_wave_launcher(depth, threads, blocks):
    # A grid-stride loop over all threads of the grid; the shuffle needs the wave-uniform split, which
    # works per wave.
    @flyc.kernel(known_block_size=[threads, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        g = ArithValue(arith.index_cast(T.i32, gpu.block_id("x"))) * threads + ArithValue(tid)
        acc = fx.Float32(0.0)
        for i in range(fx.Int32(g), n, fx.Int32(threads * blocks), prefetch_depth=depth):
            x = _load(rin, i)
            acc = ArithValue(acc) + ArithValue(x) * ArithValue(gpu.shuffle_xor(x, 1, BLK))
        buffer_ops.buffer_store(acc, buffer_ops.create_buffer_resource(out, max_size=True), g)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stream: fx.Stream):
        k(inp, out, n).launch(grid=(blocks, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("depth", [1, 4, 16])
@pytest.mark.parametrize(
    "threads, blocks",
    [(BLK * 4, 4), (BLK * 3 // 4, 1)],
    ids=["4_waves_4_blocks", "partial_wave"],
)
def test_prefetch_depth_multi_wave(threads, blocks, depth):
    reference, pipelined = _multi_wave_launcher(None, threads, blocks), _multi_wave_launcher(depth, threads, blocks)
    inp = torch.rand(BLK * 512, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
    total = threads * blocks
    _compare_on_sizes(reference, pipelined, inp, (0, 5, total * 3 + 77, total * 9 + total // 2), total)


def _nested_launcher(depth):
    # Both loops are marked; the inner one runs a constant 7 trips, below most depths.
    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        acc = fx.Float32(0.0)
        for j in range(fx.Int32(0), n, fx.Int32(1), prefetch_depth=depth):
            inner = fx.Float32(_load(rin, ArithValue(j) * BLK + ArithValue(tid)))
            for i in range(fx.Int32(0), fx.Int32(7), fx.Int32(1), prefetch_depth=depth):
                inner = ArithValue(inner) + ArithValue(
                    _load(rin, (ArithValue(j) + ArithValue(i)) * BLK + ArithValue(tid))
                )
            acc = ArithValue(acc) + ArithValue(inner)
        buffer_ops.buffer_store(acc, buffer_ops.create_buffer_resource(out, max_size=True), tid)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stream: fx.Stream):
        k(inp, out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


LAYOUT_ROW = BLK * 4  # each lane copies 4 floats of a row


def _layout_launcher(depth):
    # The layout API: a buffer tensor, a tiled copy into a register fragment, one row per iteration.
    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(A: fx.Tensor, Out: fx.Tensor, n: fx.Int32):
        tid = fx.thread_idx.x
        tiles = fx.zipped_divide(fx.rocdl.make_buffer_tensor(A), (1, LAYOUT_ROW))
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        tile_mn, tv = fx.make_layout_tv(fx.make_layout((1, BLK), (1, 1)), fx.make_layout((1, 4), (1, 1)))
        thr_copy = fx.make_tiled_copy(copy_atom, tv, tile_mn).get_slice(tid)
        acc = fx.Float32(0.0)
        for t, st in range(fx.Int32(0), n, fx.Int32(1), init=[fx.Float32(0.0)], prefetch_depth=depth):
            src = thr_copy.partition_S(fx.slice(tiles, (None, t)))
            frag = fx.make_fragment_like(src)
            fx.copy(copy_atom, src, frag)
            total = st[0]
            for a in fx.range_constexpr(4):
                total = total + frag[a]
            acc = yield [total]
        Out[tid] = acc

    @flyc.jit
    def launch(A: fx.Tensor, Out: fx.Tensor, n: fx.Int32, stream: fx.Stream):
        k(A, Out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


def _mfma_launcher(depth, lane_bounds):
    # Each iteration loads a vec4 of f16 per lane and accumulates one 16x16x16 MFMA. The guarded
    # prologue and epilogue feed the MFMA zeros for out-of-range iterations. With lane_bounds the
    # lower bound is computed from the lane id (tid // BLK, 0 in every lane), so the pass can't prove
    # it uniform and splits the loop at the wave's shortest trip count. The trip counts still match:
    # an MFMA ignores the exec mask, so with lanes on different trip counts even plain range would
    # leave unspecified values in the inactive lanes.
    @flyc.kernel(known_block_size=[BLK, 1, 1])
    def k(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32) -> None:
        rin = buffer_ops.create_buffer_resource(inp, max_size=True)
        tid = arith.index_cast(T.i32, gpu.thread_id("x"))
        lo = ArithValue(tid) // BLK if lane_bounds else fx.Int32(0)
        zero = arith.constant_vector(0.0, T.vec(4, T.f32))
        acc = zero
        for i, st in range(lo, n, fx.Int32(1), init=[zero], prefetch_depth=depth):
            e = ArithValue(arith.index_cast(T.i32, i)) * BLK + ArithValue(tid)
            a = buffer_ops.buffer_load(rin, e * 4, vec_width=4, dtype=T.f16)
            acc = yield [rocdl.mfma_f32_16x16x16f16(T.vec(4, T.f32), [a, a, st[0]])]
        buffer_ops.buffer_store(acc, buffer_ops.create_buffer_resource(out, max_size=True), ArithValue(tid) * 4)

    @flyc.jit
    def launch(inp: fx.Tensor, out: fx.Tensor, n: fx.Int32, stream: fx.Stream):
        k(inp, out, n).launch(grid=(1, 1, 1), block=(BLK, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("depth", [1, 4])
def test_prefetch_depth_nested_loops(depth):
    inp = torch.rand(BLK * 64, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
    _compare_on_sizes(_nested_launcher(None), _nested_launcher(depth), inp, (0, 1, 3, 37), BLK)


@pytest.mark.parametrize("depth", [1, 4, 16])
def test_prefetch_depth_layout_api(depth):
    inp = torch.rand(300, LAYOUT_ROW, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
    _compare_on_sizes(_layout_launcher(None), _layout_launcher(depth), inp, (0, 1, 3, 4, 5, 37, 300), BLK)


def test_prefetch_depth_layout_api_is_pipelined(tmp_path, monkeypatch):
    # The fragment of fx.copy is promoted to registers before the pass, so its load is pipelined.
    inp = torch.rand(8, LAYOUT_ROW, device="cuda")
    args = (inp, torch.zeros(BLK, device="cuda"), 8)
    assert _buffer_loads_in_llvm_ir(_layout_launcher(4), args, tmp_path, monkeypatch) == 9


@pytest.mark.skipif(is_rdna_arch() or BLK != 64, reason="MFMA needs a CDNA GPU")
@pytest.mark.parametrize("depth", [1, 4, 16])
@pytest.mark.parametrize("lane_bounds", [False, True], ids=["uniform_bounds", "lane_id_bounds"])
def test_prefetch_depth_mfma(lane_bounds, depth):
    gen = torch.Generator(device="cuda").manual_seed(0)
    inp = torch.rand(BLK * 4 * 64, device="cuda", generator=gen).to(torch.float16)
    reference, pipelined = _mfma_launcher(None, lane_bounds), _mfma_launcher(depth, lane_bounds)
    _compare_on_sizes(reference, pipelined, inp, (0, 1, 3, 5, 37), BLK * 4)
