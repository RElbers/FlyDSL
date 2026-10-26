# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""range(..., prefetch_depth=N) on the GPU: every output bit-identical to plain range.

Each case is one loop body, compiled with plain range and with prefetch_depth. The sizes hit the
edges of the pipelined loop's prologue, epilogue and remainder: no iterations, lanes with no
iterations, fewer iterations than the depth, and lanes whose trip counts differ by one on either
side of a multiple of the depth (where a lane-exchange op would see the wrong lanes unless the
pass splits the loop at the wave's shortest trip count).
"""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu
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


BODIES = {"sum": _sum, "raw_sum": _raw_sum, "min": _min, "shuffle": _shuffle, "store_reload": _store_reload}
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
    ],
)
def test_prefetch_depth_matches_plain_range(body_name, i32_bounds, depth):
    reference = _launcher(body_name, None, i32_bounds)
    pipelined = _launcher(body_name, depth, i32_bounds)
    for n in SIZES:
        want = _run(reference, n)
        got = _run(pipelined, n)
        assert all(torch.equal(a, b) for a, b in zip(got, want)), f"n={n}"
