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

def FlyROCDLPrefetchLoopsPass : Pass<"fly-rocdl-prefetch-loops"> {
  let summary = "Software-pipeline scf.for loops marked with fly.prefetch_depth";
  let description = [{
    Unrolls `scf.for` loops carrying `fly.prefetch_depth = N` by N, then
    pipelines them in two stages. The load stage contains top-level loads
    and their address computations. The compute stage contains the remaining
    operations, one unrolled iteration later. Loaded values cross the
    backedge in registers, so loads complete before the backedge. Dynamic
    prologues and epilogues guard reads, side effects and operations that
    cannot execute speculatively. Remainder loops remain serial.

    Loads move ahead of earlier iterations' writes, so a loop that may write
    memory requires `fly.prefetch_assume_independent`. This asserts that no
    iteration reads memory an earlier iteration wrote. Known memory,
    synchronization and load-address dependences still prevent pipelining.

    N must be between 1 and 32. Narrow integer bounds are widened to i64.
    The pass rejects loops whose step times N may overflow a signed i64,
    except that runtime i64 and index steps are assumed not to overflow.
    Loops with lane exchanges and bounds not proven wave-uniform are split
    at the wave's shortest trip count, rounded down to a multiple of N.

    Frontend markers are consumed on both success and rejection. Legality
    failures leave the loop unchanged apart from its markers and emit a
    remark. A clean SCF pipeliner rejection retains any loop preparation and
    unrolling already applied. Failure after SCF has changed the IR fails
    the pass.

    Run inside `gpu.module(...)` after `convert-fly-to-rocdl` and before
    `convert-scf-to-cf`. The Python frontend sets the markers for
    `range(..., prefetch_depth=N, prefetch_assume_independent=True)`.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "LLVM::LLVMDialect",
    "scf::SCFDialect"
  ];
}

#endif // FLYROCDL_PASSES
