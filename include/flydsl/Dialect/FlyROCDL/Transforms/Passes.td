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
    `range(..., prefetch_depth=N)`) so the loads of up to N future iterations
    stay in flight across the backedge, instead of draining `vmcnt(0)` every
    iteration. The loop is unrolled by N and pipelined in two stages; the
    prologue and epilogue of a runtime trip count are predicated. Loops with a
    barrier, an LDS store or a global->LDS DMA are left unchanged. Run inside
    `gpu.module(...)` after `convert-fly-to-rocdl` and before
    `convert-scf-to-cf`.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect"
  ];
}

#endif // FLYROCDL_PASSES
