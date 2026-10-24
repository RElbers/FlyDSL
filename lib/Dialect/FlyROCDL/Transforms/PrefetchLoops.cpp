// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Software-pipeline the `scf.for` loops that carry `fly.prefetch_depth = N` (set by
// `range(..., prefetch_depth=N)`) so the loads of up to N future iterations stay in flight across
// the backedge. Run inside gpu.module() AFTER convert-fly-to-rocdl, while the loads are ROCDL ops
// and the loop is still an scf.for.
//
// A rolled runtime-trip loop otherwise drains `s_waitcnt vmcnt(0)` at every backedge: each
// iteration's single load is waited on before the next one issues. The pass unrolls the loop by N,
// so the steady state has N distinct loads (N rotating destination registers), and then pipelines
// it in two stages: loads plus their address math first, everything else one iteration later.
// The prologue and epilogue of a runtime trip count are predicated by `predicateOp`.

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace fly_rocdl {
#define GEN_PASS_DEF_FLYROCDLPREFETCHLOOPSPASS
#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h.inc"
} // namespace fly_rocdl
} // namespace mlir

namespace {

constexpr StringLiteral kPrefetchDepthAttr = "fly.prefetch_depth";

// Loads lead their use by one iteration; depth comes from the unroll factor, not from more stages.
constexpr unsigned kNumStages = 2;

// A memory read whose issue we want to lead the compute (stage 0).
bool isPipelineableRead(Operation *op) {
  if (isa<LLVM::LoadOp, vector::TransferReadOp, vector::LoadOp>(op))
    return true;
  StringRef n = op->getName().getStringRef();
  return n.contains("buffer.load") || n.contains("global.load") || n.contains("ds.read");
}

// An op whose result depends on the values or the set of active lanes of the wave (shuffles,
// ballots, lane reads, DPP). Unrolling splits the iterations between the main loop and a serial
// remainder loop, so when lanes have different trip counts, lanes run the same iteration at
// different times and such an op sees a different set of active lanes than in the original loop.
bool isLaneExchangeOp(StringRef n) {
  return n.contains("shuffle") || n.contains("ballot") || n.contains("bpermute") ||
         n.contains("permlane") || n.contains("readlane") || n.contains("readfirstlane") ||
         n.contains("dpp") || n.contains("ds_swizzle") || n.contains("subgroup");
}

// An operand-free op whose value is the same in every lane of a wave.
bool isUniformSource(Operation *op) {
  if (op->hasTrait<OpTrait::ConstantLike>())
    return true;
  StringRef n = op->getName().getStringRef();
  return llvm::is_contained({"gpu.block_id", "gpu.block_dim", "gpu.grid_dim", "gpu.cluster_id",
                             "gpu.cluster_dim", "gpu.cluster_block_id", "gpu.subgroup_id",
                             "gpu.num_subgroups", "gpu.subgroup_size", "llvm.mlir.addressof"},
                            n) ||
         n.starts_with("rocdl.workgroup.") || n.starts_with("rocdl.grid.");
}

// Whether every lane of a wave holds the same value of `v`. Conservative: false when unsure. Kernel
// arguments, constants and block-level ids are uniform; so is a pure or read-only op whose operands
// all are, and a readfirstlane. Thread / lane ids, atomics, ops with regions and loop-carried
// values are not.
bool isWaveUniform(Value v, DenseMap<Value, bool> &memo) {
  if (auto it = memo.find(v); it != memo.end())
    return it->second;
  memo[v] = false;
  bool uniform = false;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Operation *parent = arg.getOwner()->getParentOp();
    if (parent->getName().getStringRef() == "gpu.func" && parent->hasAttr("kernel"))
      uniform = true;
    else if (auto forOp = dyn_cast<scf::ForOp>(parent); forOp && v == forOp.getInductionVar())
      uniform = isWaveUniform(forOp.getLowerBound(), memo) &&
                isWaveUniform(forOp.getUpperBound(), memo) &&
                isWaveUniform(forOp.getStep(), memo);
  } else {
    Operation *op = v.getDefiningOp();
    StringRef n = op->getName().getStringRef();
    if (n.contains("readfirstlane"))
      uniform = true;
    else if (op->getNumOperands() == 0)
      uniform = isUniformSource(op);
    else if (op->getNumRegions() == 0 && !isLaneExchangeOp(n) &&
             (isMemoryEffectFree(op) || hasSingleEffect<MemoryEffects::Read>(op)))
      uniform = llvm::all_of(op->getOperands(),
                             [&](Value operand) { return isWaveUniform(operand, memo); });
  }
  memo[v] = uniform;
  return uniform;
}

// Every lane of a wave runs the same number of iterations.
bool hasWaveUniformTripCount(scf::ForOp forOp) {
  DenseMap<Value, bool> memo;
  return isWaveUniform(forOp.getLowerBound(), memo) && isWaveUniform(forOp.getUpperBound(), memo) &&
         isWaveUniform(forOp.getStep(), memo);
}

// Mechanical whole-loop rotation cannot safely pipeline a loop that contains an intra-loop barrier
// or a single-buffered global->LDS DMA (`buffer.load.lds`) / LDS store: rotating the DMA across the
// backedge would clobber the shared buffer (no double buffering), and a dynamic-trip epilogue would
// have to predicate the barrier, which is illegal. A loop with a lane-exchange op is skipped unless
// every lane has the same trip count, since then all lanes still run each iteration together. Such
// loops are left as they are.
bool isMechanicallyPipelineable(scf::ForOp forOp) {
  bool hasLaneExchange = false;
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    StringRef n = op->getName().getStringRef();
    if (n.contains("barrier") || n.contains("buffer.load.lds") || n.contains("ds.write"))
      return WalkResult::interrupt();
    hasLaneExchange |= isLaneExchangeOp(n);
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return false;
  return !hasLaneExchange || hasWaveUniformTripCount(forOp);
}

// Stage 0 = reads + the pure address math feeding them; last stage = everything else.
void buildLoadComputeSchedule(scf::ForOp forOp,
                              std::vector<std::pair<Operation *, unsigned>> &schedule) {
  llvm::SmallPtrSet<Operation *, 16> stage0;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!isPipelineableRead(&op))
      continue;
    stage0.insert(&op);
    SetVector<Operation *> slice;
    BackwardSliceOptions sliceOpts;
    sliceOpts.inclusive = true;
    sliceOpts.filter = [&](Operation *o) {
      return o->getBlock() == forOp.getBody() && isMemoryEffectFree(o);
    };
    // Root the slice at the read's operands, not the read itself: getBackwardSlice returns an empty
    // slice when the root fails the filter (a read is not memory-effect-free), which would leave the
    // address math in the last stage ("operation scheduled before its operands").
    for (Value operand : op.getOperands())
      (void)getBackwardSlice(operand, &slice, sliceOpts);
    stage0.insert(slice.begin(), slice.end());
  }
  for (Operation &op : forOp.getBody()->without_terminator())
    schedule.push_back({&op, stage0.contains(&op) ? 0u : kNumStages - 1});
}

// A buffer load past its resource returns 0 instead of faulting, so it may issue unguarded.
bool isBoundsCheckedRead(Operation *op) {
  return op->getName().getStringRef().contains("buffer.load");
}

// Predicate an op cloned into the prologue/epilogue of a dynamic-trip loop.
//
// A buffer read is masked to zero (`select(pred, load, 0)`); any other read is sunk into
// `scf.if(pred)` yielding zero, because it can fault when issued past the trip count. Pure ops run
// unchanged on the zeroed inputs. The zeros never reach the loop results: the pipeliner selects
// each epilogue live-out against the previous iteration's value under the same predicate. Any
// other side-effecting op (a store, an atomicrmw, an scf.if or scf.for that reads) is sunk into
// `scf.if(pred)`, yielding zeros when it does not run; those zeros are discarded the same way. A
// side-effecting op with a result type that cannot be zeroed returns nullptr, which makes
// pipelineForLoop leave the loop un-pipelined.
//
// The pipeliner calls this on each clone before any later op uses its results, and maps the
// original op's results to the returned op's results by position.
Operation *predicateOp(RewriterBase &rewriter, Operation *op, Value pred) {
  Location loc = op->getLoc();
  if (isPipelineableRead(op) && op->getNumResults() == 1) {
    Type ty = op->getResult(0).getType();
    if (!isa<IntegerType, FloatType, VectorType>(ty))
      return isBoundsCheckedRead(op) ? op : nullptr;
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(op);
    auto zero = arith::ConstantOp::create(rewriter, loc, cast<TypedAttr>(rewriter.getZeroAttr(ty)));
    if (isBoundsCheckedRead(op))
      return arith::SelectOp::create(rewriter, loc, pred, op->getResult(0), zero);
    auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{ty}, pred, /*withElseRegion=*/true);
    op->moveBefore(ifOp.thenBlock(), ifOp.thenBlock()->end());
    zero->moveBefore(ifOp.elseBlock(), ifOp.elseBlock()->end());
    rewriter.setInsertionPointToEnd(ifOp.thenBlock());
    scf::YieldOp::create(rewriter, loc, op->getResults());
    rewriter.setInsertionPointToEnd(ifOp.elseBlock());
    scf::YieldOp::create(rewriter, loc, zero.getResult());
    return ifOp;
  }
  if (isMemoryEffectFree(op))
    return op;

  bool zeroable = llvm::all_of(op->getResultTypes(), [](Type t) {
    return isa<IntegerType, FloatType, VectorType>(t);
  });
  if (!zeroable)
    return nullptr;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);
  if (op->getNumResults() == 0) {
    auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, pred, /*withElseRegion=*/false);
    op->moveBefore(ifOp.thenBlock()->getTerminator());
    return ifOp;
  }
  // With result types the builder leaves both blocks unterminated, so build the yields here. The
  // else-branch zeros only reach ops of the same out-of-range iteration.
  auto ifOp = scf::IfOp::create(rewriter, loc, op->getResultTypes(), pred, /*withElseRegion=*/true);
  op->moveBefore(ifOp.thenBlock(), ifOp.thenBlock()->end());
  rewriter.setInsertionPointToEnd(ifOp.thenBlock());
  scf::YieldOp::create(rewriter, loc, op->getResults());
  rewriter.setInsertionPointToEnd(ifOp.elseBlock());
  SmallVector<Value> zeros;
  for (Type t : op->getResultTypes())
    zeros.push_back(arith::ConstantOp::create(rewriter, loc, cast<TypedAttr>(rewriter.getZeroAttr(t))));
  scf::YieldOp::create(rewriter, loc, zeros);
  return ifOp;
}

class FlyROCDLPrefetchLoopsPass
    : public mlir::fly_rocdl::impl::FlyROCDLPrefetchLoopsPassBase<FlyROCDLPrefetchLoopsPass> {
public:
  using mlir::fly_rocdl::impl::FlyROCDLPrefetchLoopsPassBase<
      FlyROCDLPrefetchLoopsPass>::FlyROCDLPrefetchLoopsPassBase;

  void runOnOperation() override {
    SmallVector<std::pair<scf::ForOp, int64_t>> requested;
    getOperation()->walk([&](scf::ForOp forOp) {
      if (auto depth = forOp->getAttrOfType<IntegerAttr>(kPrefetchDepthAttr))
        requested.push_back({forOp, depth.getInt()});
    });

    IRRewriter rewriter(&getContext());
    for (auto [forOp, depth] : requested) {
      forOp->removeAttr(kPrefetchDepthAttr);
      if (depth < 1 || !isMechanicallyPipelineable(forOp))
        continue;

      scf::ForOp target = forOp;
      if (depth > 1) {
        // The remainder loop for a runtime trip count stays serial.
        FailureOr<UnrolledLoopInfo> info = loopUnrollByFactor(forOp, depth);
        if (succeeded(info) && info->mainLoopOp)
          target = *info->mainLoopOp;
      }

      scf::PipeliningOption options;
      options.supportDynamicLoops = true;
      // peelEpilogue (default true) must stay true: folding the epilogue into the kernel would
      // predicate every steady-state load, forcing a wait right after each one.
      options.predicateFn = predicateOp;
      options.getScheduleFn = buildLoadComputeSchedule;
      rewriter.setInsertionPoint(target);
      bool modified = false;
      if (failed(scf::pipelineForLoop(rewriter, target, options, &modified)) && modified)
        return signalPassFailure(); // IR mutated, then failed
    }
  }
};

} // namespace
