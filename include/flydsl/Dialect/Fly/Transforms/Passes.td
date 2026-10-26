// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLY_PASSES
#define FLY_PASSES

include "mlir/Pass/PassBase.td"

def FlyCanonicalizePass : Pass<"fly-canonicalize"> {
  let summary = "Canonicalize Pattern";
  let description = [{
    Canonicalize Fly operations.
  }];
}

def FlyLayoutLoweringPass : Pass<"fly-layout-lowering"> {
  let summary = "Lower layout algebra operations";
  let description = [{
    Lowers layout algebra operations to simpler forms.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "vector::VectorDialect"
  ];
}

def FlyConvertAtomCallToSSAFormPass : Pass<"fly-convert-atom-call-to-ssa-form"> {
  let summary = "Convert copy_atom_call/mma_atom_call to SSA form";
  let description = [{
    Try to convert copy_atom_call and mma_atom_call to their SSA counterparts 
    (copy_atom_call_ssa / mma_atom_call_ssa) if possible.
    Promote register tensor to vector ssa value. Keep others memref unchanged.
  }];

  let dependentDialects = [
    "vector::VectorDialect"
  ];
}

def FlyIntSwizzleSimplifyPass : Pass<"fly-int-swizzle-simplify"> {
  let summary = "Algebraically simplify swizzle-shaped arith sequences";
  let description = [{
      Recognize the canonical swizzle sequence emitted by `applySwizzle`,

        %m  = arith.constant ((1<<M)-1) << (base+shift) : iN     // ring of 1s
        %s  = arith.constant shift                       : iN
        %a  = arith.andi %x, %m : iN
        %h  = arith.shrui %a, %s : iN
        %r  = arith.xori %x, %h : iN

      and apply the algebraic identity

        swizzle(a + d) = swizzle(a) + d   when d % period == 0,

      The peeled-off addends survive as plain `arith.addi`, exposing them to CSE and constant
      folding that the bit-level form would hide.
    }];

  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect"
  ];
}

def FlyConvertSCFToCFPass : Pass<"fly-convert-scf-to-cf"> {
  let summary = "Lower SCF to CF, marking scf.for increments as not wrapping";
  let description = [{
    The same conversion as upstream `convert-scf-to-cf`, plus `nsw` (or `nuw` for
    an unsigned loop) on each scf.for induction variable increment, and on any
    `iv + step` the body computes. scf.for makes overflow of the last increment
    undefined, so the flag only states what the loop guarantees. Without it,
    LLVM counts the trips of a loop with a constant step above 1 only when it
    can bound the loop's upper bound (a mask or a min, for example), since the
    induction variable could otherwise step past the bound and wrap; then it
    can't unroll the loop with a runtime trip count. A runtime step is also
    replaced by
    max(step, 1) inside the loop: scf.for requires a strictly positive step,
    and LLVM can only count the trips once it knows the step is positive.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "LLVM::LLVMDialect"
  ];
}

def FlyPromoteRegMemToVectorSSAPass : Pass<"fly-promote-regmem-to-vectorssa"> {
  let summary = "Promote register memory to vector SSA values";
  let description = [{
    Promotes fly.make_ptr(register) memory semantics to vector SSA values.
    Requires fly-convert-atom-call-to-ssa-form to have run first. 
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "scf::SCFDialect",
    "vector::VectorDialect",
    "ub::UBDialect"
  ];
}

#endif // FLY_PASSES
