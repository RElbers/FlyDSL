// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Lowers SCF to the CF dialect exactly as upstream `convert-scf-to-cf` does, and marks the
// increment of each scf.for induction variable as not wrapping: `nsw`, or `nuw` for an unsigned
// loop.
//
// scf.for guarantees this: the value after the last increment, LB + n * step, must be representable
// in the induction variable's type, or the loop is undefined. Without the flag LLVM can't compute
// the trip count of a loop with bounds narrower than 64 bits, so it can't unroll it with a runtime
// trip count (`range(..., unroll=N)` keeps one load in flight) or reason about its induction
// variable.
//
// Upstream's lowering creates the increment without flags and has no option for them, but it copies
// every LLVM-dialect attribute of the scf.for onto the loop's back-edge branch (that is how
// `llvm.loop_annotation` reaches LLVM), and that branch's first operand is the increment. So each
// loop gets an `#llvm.overflow` marker before the conversion, and the branch that carries it sets
// the flag.

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "flydsl/Dialect/Fly/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYCONVERTSCFTOCFPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

constexpr StringLiteral kNoWrapMarker = "fly.iv_no_wrap";

class FlyConvertSCFToCFPass
    : public mlir::fly::impl::FlyConvertSCFToCFPassBase<FlyConvertSCFToCFPass> {
public:
  using mlir::fly::impl::FlyConvertSCFToCFPassBase<
      FlyConvertSCFToCFPass>::FlyConvertSCFToCFPassBase;

  void runOnOperation() override {
    Operation *root = getOperation();
    MLIRContext *ctx = &getContext();
    root->walk([&](scf::ForOp forOp) {
      auto flags = forOp.getUnsignedCmp() ? LLVM::IntegerOverflowFlags::nuw
                                          : LLVM::IntegerOverflowFlags::nsw;
      forOp->setAttr(kNoWrapMarker, LLVM::IntegerOverflowFlagsAttr::get(ctx, flags));
    });

    // The same conversion as upstream's SCFToControlFlowPass.
    RewritePatternSet patterns(ctx);
    populateSCFToControlFlowConversionPatterns(patterns);
    ConversionTarget target(*ctx);
    target.addIllegalOp<scf::ForallOp, scf::ForOp, scf::IfOp, scf::IndexSwitchOp, scf::ParallelOp,
                        scf::WhileOp, scf::ExecuteRegionOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(root, target, std::move(patterns))))
      return signalPassFailure();

    root->walk([&](cf::BranchOp branch) {
      auto marker = branch->getAttrOfType<LLVM::IntegerOverflowFlagsAttr>(kNoWrapMarker);
      if (!marker)
        return;
      branch->removeAttr(kNoWrapMarker);
      if (branch.getDestOperands().empty())
        return;
      auto increment = branch.getDestOperands().front().getDefiningOp<arith::AddIOp>();
      if (!increment)
        return;
      bool isUnsigned = bitEnumContainsAny(marker.getValue(), LLVM::IntegerOverflowFlags::nuw);
      increment.setOverflowFlags(isUnsigned ? arith::IntegerOverflowFlags::nuw
                                            : arith::IntegerOverflowFlags::nsw);
    });
  }
};

} // namespace
