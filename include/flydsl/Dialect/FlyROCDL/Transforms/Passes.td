// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYROCDL_PASSES
#define FLYROCDL_PASSES

include "mlir/Pass/PassBase.td"

def FlyROCDLExpandOpsPass : Pass<"fly-rocdl-expand-ops"> {
  let summary = "Expand fly_rocdl target specific ops into the ops";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "mlir::fly::FlyDialect"
  ];
}

def FlyROCDLClusterAttrPass : Pass<"fly-rocdl-cluster-attr"> {
  let summary = "Inject amdgpu-cluster-dims into llvm.func passthrough";
  let description = [{
    Converts the `rocdl.cluster_dims` discardable attribute on `llvm.func`
    into an LLVM IR `passthrough` entry containing `amdgpu-cluster-dims`.

    This is a hack to work around the upstream ROCDL dialect not supporting
    cluster_dims translation.  Run this pass inside `gpu.module(...)` after
    `convert-gpu-to-rocdl` so that `llvm.func` ops already exist -- it rewrites
    an attribute on those, and so is not part of the Fly -> ROCDL conversion.
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect"
  ];
}

def FlyROCDLUnrollIndependentPass : Pass<"fly-rocdl-unroll-independent"> {
  let summary = "Unroll fly.assume_independent loops, group their loads, and optionally pipeline";
  let description = [{
    Unrolls an `scf.for` that carries `fly.assume_independent` by the count
    N of its `loop_annotation` unroll hint, with a serial remainder loop.
    In each group of N iterations, the top-level loads and the pure ops that
    compute their addresses move above the group's other operations, in
    their original order. LLVM's unroller is disabled on both loops.

    A loop that also carries `fly.pipeline` is software-pipelined in two
    stages: each iteration issues the next group's load slice, then a
    `rocdl.sched.barrier` that keeps LLVM from sinking those loads, then the
    current group's other ops, then a second barrier that keeps LLVM from
    moving the back edge's copies of the loaded values into that compute. The pipelined loop runs inside a guard for at
    least one group, so its prologue and epilogue need no predicates. A loop
    that SCF's pipeliner rejects stays grouped and gets a remark. On gfx9,
    where loads and stores share vmcnt, the pass waits for the prologue's
    loads (`rocdl.s.waitcnt` with vmcnt(0)) right before the pipelined loop,
    so that LLVM's wait at the loop's top covers only the previous
    iteration's loads, not its stores.

    A scalar buffer load (`llvm.amdgcn.s.buffer.load`, as a call intrinsic)
    is a load like the others. It reads through the scalar cache, which
    vector stores do not update, so a load of what an earlier iteration
    stored could be stale; the marker's assertion rules that out.

    A pipelined loop that also carries `fly.pipeline_lds` stages its buffer
    loads through LDS: each becomes a `rocdl.raw.ptr.buffer.load.async.lds`
    into a two-slot, wave-private ring (one LDS global per loop), closed by
    `rocdl.asyncmark`, and its value is read back from LDS before its first
    use, after a `rocdl.wait.asyncmark`. A load stays in registers when it is
    not 4, 12 or 16 bytes, its resource is not proven wave-uniform, or its
    value addresses another load or is carried unchanged to the next
    iteration; a scalar buffer load stays in its scalar register. Staging needs a gfx950 target, a kernel `known_block_size`
    and room in the 160 KiB of LDS; otherwise the loop is pipelined in
    registers, with a remark. With `fly.pipeline_lds`, N may be 1.

    The marker asserts that no iteration reads memory that any thread wrote
    in an earlier iteration. The pass still rejects a loop with a barrier, a
    memory-counter wait, an LDS or asynchronous copy, a print or assert, a
    fence, a volatile or acquire-ordered access, private memory, or inline
    assembly with side effects; a loop whose
    top-level load follows a possible store in the same iteration, or reads
    the same address every iteration from a buffer the loop writes; a loop
    whose load addresses depend on other operations; and a loop with a lane
    exchange whose bounds are not proven wave-uniform.

    N must be from 2 to 32 (1 to 32 with `fly.pipeline_lds`). The pass rejects unsigned loops and loops whose
    constant step times N overflows a signed i64. A narrow loop keeps its
    type: the main loop's bounds are computed in i64, and when they do not
    fit the type, the remainder loop runs every iteration; when pipelining,
    the main loop's span plus one step must fit as well. The bounds of
    i64 and index loops are assumed not to overflow.

    The marker is consumed in every case. A rejected loop keeps its unroll
    hint, so it behaves as with plain `unroll=N`, and gets a remark saying
    why.

    Run inside `gpu.module(...)` after `convert-fly-to-rocdl` and before
    `convert-scf-to-cf`. The Python frontend sets the markers and the hint
    for `range(..., unroll=N, assume_independent=True, pipeline=True)`.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "ROCDL::ROCDLDialect",
    "scf::SCFDialect"
  ];
}

#endif // FLYROCDL_PASSES
