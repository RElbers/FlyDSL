// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Lowers SCF to the CF dialect exactly as upstream `convert-scf-to-cf` does, and marks the
// increment of each scf.for induction variable as not wrapping: `nsw`, or `nuw` for an unsigned
// loop.
//
// scf.for guarantees this: the value after the last increment, LB + n * step, must be representable
// in the induction variable's type, or the loop is undefined. Without the flag LLVM can't compute
// the trip count of a loop whose step isn't 1, since the induction variable could step past the
// bound and wrap, so it can't unroll the loop with a runtime trip count. Each copy
// `range(..., unroll=N)` makes then keeps its own exit test, and waits for its load before the next
// copy issues one.
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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

// Its value must be an LLVM-dialect attribute: upstream only copies those onto the back edge.
constexpr StringLiteral kNoWrapMarker = "fly.iv_no_wrap";

arith::IntegerOverflowFlags noWrapFlag(scf::ForOp forOp) {
  return forOp.getUnsignedCmp() ? arith::IntegerOverflowFlags::nuw
                                : arith::IntegerOverflowFlags::nsw;
}

bool isStep(Value v, scf::ForOp forOp) {
  if (v == forOp.getStep())
    return true;
  std::optional<int64_t> value = getConstantIntValue(v);
  return value && value == getConstantIntValue(forOp.getStep());
}

// Marks `iv + step` computed in the body, e.g. to load the next iteration's data, as not wrapping
// too. In the body iv <= LB + (n - 1) * step, so iv + step <= LB + n * step. Without the flag, LLVM
// merges this add with the flagged increment and keeps only the flags both have.
void markNextInductionValue(scf::ForOp forOp) {
  Value iv = forOp.getInductionVar();
  forOp.getBody()->walk([&](arith::AddIOp add) {
    if ((add.getLhs() == iv && isStep(add.getRhs(), forOp)) ||
        (add.getRhs() == iv && isStep(add.getLhs(), forOp)))
      add.setOverflowFlags(add.getOverflowFlags() | noWrapFlag(forOp));
  });
}

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
      markNextInductionValue(forOp);
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

    root->walk([&](Operation *op) {
      auto marker = op->getAttrOfType<LLVM::IntegerOverflowFlagsAttr>(kNoWrapMarker);
      if (!marker)
        return;
      op->removeAttr(kNoWrapMarker);
      auto branch = dyn_cast<cf::BranchOp>(op);
      if (!branch || branch.getDestOperands().empty())
        return;
      // The back edge passes iv + step as the condition block's first argument, the iv.
      auto increment = branch.getDestOperands().front().getDefiningOp<arith::AddIOp>();
      if (!increment || increment.getLhs() != branch.getDest()->getArgument(0))
        return;
      bool isUnsigned = bitEnumContainsAny(marker.getValue(), LLVM::IntegerOverflowFlags::nuw);
      increment.setOverflowFlags(isUnsigned ? arith::IntegerOverflowFlags::nuw
                                            : arith::IntegerOverflowFlags::nsw);
    });
  }
};

} // namespace
