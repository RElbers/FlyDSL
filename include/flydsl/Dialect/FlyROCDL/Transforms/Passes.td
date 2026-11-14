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
  let summary = "Unroll loops marked fly.assume_independent and issue each group's loads first";
  let description = [{
    Unrolls an `scf.for` that carries `fly.assume_independent` by the count
    N of its `loop_annotation` unroll hint, with a serial remainder loop.
    In each group of N iterations, the top-level loads and the pure ops that
    compute their addresses move above the group's other operations, in
    their original order. LLVM's unroller is disabled on both loops.

    The marker asserts that no iteration reads memory that any thread wrote
    in an earlier iteration. The pass still rejects a loop with a barrier, a
    memory-counter wait, an LDS or asynchronous copy, a print or assert, a
    fence, a volatile or acquire-ordered access, private memory, or inline
    assembly with side effects; a loop whose
    top-level load follows a possible store in the same iteration, or reads
    the same address every iteration from a buffer the loop writes; a loop
    whose load addresses depend on other operations; and a loop with a lane
    exchange whose bounds are not proven wave-uniform.

    N must be from 2 to 32. The pass rejects unsigned loops and loops whose
    constant step times N overflows a signed i64. A narrow loop keeps its
    type: the main loop's bounds are computed in i64, and when they do not
    fit the type, the remainder loop runs every iteration. The bounds of
    i64 and index loops are assumed not to overflow.

    The marker is consumed in every case. A rejected loop keeps its unroll
    hint, so it behaves as with plain `unroll=N`, and gets a remark saying
    why.

    Run inside `gpu.module(...)` after `convert-fly-to-rocdl` and before
    `convert-scf-to-cf`. The Python frontend sets the marker and the hint
    for `range(..., unroll=N, assume_independent=True)`.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "scf::SCFDialect"
  ];
}

#endif // FLYROCDL_PASSES
