// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYROCDL_PASSES
#define FLYROCDL_PASSES

include "mlir/Pass/PassBase.td"

def FlyROCDLExpandOpsPass : Pass<"fly-rocdl-expand-ops"> {
  let summary = "Expand fly_rocdl target specific ops into the ops";
  let dependentDialects = [
    "arith::ArithDialect",
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

def FlyROCDLPrefetchLoopsPass : Pass<"fly-rocdl-prefetch-loops"> {
  let summary = "Software-pipeline scf.for loops marked with fly.prefetch_depth";
  let description = [{
    Rewrites each `scf.for` carrying `fly.prefetch_depth = N` (set by
    `range(..., prefetch_depth=N)`) so up to N loads per lane are in flight at
    once, instead of waiting for each iteration's load before the next one
    issues. The loop is unrolled by N and pipelined in two stages: each
    steady-state iteration issues the next N loads together, ahead of the
    previous N iterations' compute (it still waits for them before its
    backedge). In the
    prologue and epilogue of a runtime trip count, reads, side effects and
    integer division or remainder only run for in-range iterations. Loads move
    ahead of earlier iterations' writes, so a loop that may write memory is only
    pipelined with `fly.prefetch_assume_independent` (set by
    `prefetch_assume_independent=True`), which asserts that no iteration reads
    memory an earlier one wrote. A loop is left unchanged, with a remark naming
    the reason, when it contains a barrier, a print or assert, a copy into or
    out of LDS or an asynchronous copy, a fence, a volatile access or an atomic
    with acquire or stronger ordering, a private-memory access, a load after a
    possible store in the same iteration, a load from the same address in every
    iteration of a buffer the loop writes, no top-level load, a load whose
    address depends on a side-effecting op, or a guarded op of a type it can't
    stand in for. Narrow integer bounds are widened to i64; a loop is left
    unchanged if its step times the depth overflows a signed 64-bit integer (a
    runtime 64-bit step is assumed not to). Loops with a lane-exchange op
    (including calls, inline asm and AMDGPU intrinsic calls) and lane-dependent
    bounds are first split at the wave's shortest trip count. Run inside `gpu.module(...)` after
    `convert-fly-to-rocdl` and before `convert-scf-to-cf`.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "scf::SCFDialect"
  ];
}

#endif // FLYROCDL_PASSES
