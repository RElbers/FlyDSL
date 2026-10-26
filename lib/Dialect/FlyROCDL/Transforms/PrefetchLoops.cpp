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
// In the prologue and epilogue of a runtime trip count, `predicateOp` guards reads and side effects
// so they only run for in-range iterations.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#include <limits>

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
// Calls, inline asm and AMDGPU intrinsic calls are opaque (a callee or an `llvm.amdgcn.icmp` can
// exchange values across lanes), so they count as lane exchanges.
bool isLaneExchangeOp(Operation *op) {
  if (isa<LLVM::CallOp, LLVM::InlineAsmOp>(op))
    return true;
  if (auto intrinsic = dyn_cast<LLVM::CallIntrinsicOp>(op))
    return intrinsic.getIntrin().starts_with("llvm.amdgcn.");
  StringRef n = op->getName().getStringRef();
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

// An op known to give the same result in every lane when its operands are the same in every lane.
// An allowlist: an unknown op could read the lane id (rocdl.mbcnt, an intrinsic call, inline asm).
bool propagatesUniformity(Operation *op) {
  StringRef dialect = op->getDialect() ? op->getDialect()->getNamespace() : "";
  StringRef n = op->getName().getStringRef();
  if (llvm::is_contained({"arith", "index", "math", "vector"}, dialect))
    return true;
  if (dialect == "llvm")
    return !isa<LLVM::CallOp, LLVM::CallIntrinsicOp, LLVM::InlineAsmOp>(op);
  if (dialect == "rocdl")
    return n.contains("make.buffer.rsrc") || n.contains("buffer.load");
  return n == "builtin.unrealized_conversion_cast";
}

// Whether every lane of a wave holds the same value of `v`. Conservative: false when unsure. Kernel
// arguments, constants and block-level ids are uniform; so is a pure or read-only op from
// `propagatesUniformity` whose operands all are, and a readfirstlane. Thread / lane ids, atomics,
// ops with regions and loop-carried values are not.
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
                isWaveUniform(forOp.getUpperBound(), memo) && isWaveUniform(forOp.getStep(), memo);
  } else {
    Operation *op = v.getDefiningOp();
    StringRef n = op->getName().getStringRef();
    if (n.contains("readfirstlane"))
      uniform = true;
    else if (op->getNumOperands() == 0)
      uniform = isUniformSource(op);
    else if (op->getNumRegions() == 0 && !isLaneExchangeOp(op) && propagatesUniformity(op) &&
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

bool hasLaneExchangeOp(scf::ForOp forOp) {
  WalkResult result = forOp.getBody()->walk([](Operation *op) {
    return isLaneExchangeOp(op) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return result.wasInterrupted();
}

// A leaf op that may write memory: a write effect, or no effect information at all.
bool mayWriteMemory(Operation *op) {
  return !isPipelineableRead(op) && !isMemoryEffectFree(op) &&
         !hasSingleEffect<MemoryEffects::Read>(op);
}

// Stage 0 runs a top-level read one iteration group ahead of everything else, so a read that
// follows a write in the body would move ahead of that write, which may store to the same address
// (an LDS round trip, or a global store and reload).
bool hasReadAfterWrite(scf::ForOp forOp) {
  bool sawWrite = false;
  WalkResult result = forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getNumRegions() > 0)
      return WalkResult::advance();
    if (sawWrite && isPipelineableRead(op) && op->getBlock() == forOp.getBody())
      return WalkResult::interrupt();
    sawWrite |= mayWriteMemory(op);
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

Value toI64(RewriterBase &rewriter, Location loc, Value v) {
  Type i64 = rewriter.getI64Type();
  if (v.getType().isIndex())
    return arith::IndexCastOp::create(rewriter, loc, i64, v);
  if (v.getType().getIntOrFloatBitWidth() < 64)
    return arith::ExtSIOp::create(rewriter, loc, i64, v);
  return v;
}

Value fromI64(RewriterBase &rewriter, Location loc, Value v, Type ty) {
  if (ty.isIndex())
    return arith::IndexCastOp::create(rewriter, loc, ty, v);
  if (ty.getIntOrFloatBitWidth() < 64)
    return arith::TruncIOp::create(rewriter, loc, ty, v);
  return v;
}

// Keep the unroller's scaled step and the pipeliner's peeled bounds out of narrow integer types.
// The body still sees the original induction type, including its original wrapping arithmetic.
void widenLoopBounds(RewriterBase &rewriter, scf::ForOp forOp) {
  auto ty = dyn_cast<IntegerType>(forOp.getLowerBound().getType());
  if (!ty || ty.getWidth() >= 64)
    return;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forOp);
  auto extend = [&](Value v) -> Value {
    if (forOp.getUnsignedCmp())
      return rewriter.createOrFold<arith::ExtUIOp>(forOp.getLoc(), rewriter.getI64Type(), v);
    return rewriter.createOrFold<arith::ExtSIOp>(forOp.getLoc(), rewriter.getI64Type(), v);
  };
  forOp.setLowerBound(extend(forOp.getLowerBound()));
  forOp.setUpperBound(extend(forOp.getUpperBound()));
  forOp.setStep(extend(forOp.getStep()));
  Value iv = forOp.getInductionVar();
  iv.setType(rewriter.getI64Type());
  rewriter.setInsertionPointToStart(forOp.getBody());
  auto narrow = arith::TruncIOp::create(rewriter, forOp.getLoc(), ty, iv);
  rewriter.replaceAllUsesExcept(iv, narrow.getResult(), narrow);
}

// Ends `forOp` after the wave's shortest trip count, rounded down to a multiple of `depth`, and
// runs the remaining iterations of each lane in a plain tail loop after it. Every lane then runs
// `forOp` the same number of times, and the tail runs in lockstep, so each iteration index sees the
// same active lanes as in the original loop. Costs one wave reduction before the loop.
LogicalResult splitAtWaveUniformTripCount(RewriterBase &rewriter, scf::ForOp forOp, int64_t depth) {
  if (forOp.getUnsignedCmp())
    return failure();
  Location loc = forOp.getLoc();
  Type boundTy = forOp.getLowerBound().getType();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forOp);
  Value lb = toI64(rewriter, loc, forOp.getLowerBound());
  Value ub = toI64(rewriter, loc, forOp.getUpperBound());
  Value step = toI64(rewriter, loc, forOp.getStep());
  Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
  Value span = arith::SubIOp::create(rewriter, loc, ub, lb);
  Value tripCount = arith::MaxSIOp::create(
      rewriter, loc, arith::CeilDivSIOp::create(rewriter, loc, span, step), zero);
  // With bounds of at most 32 bits the trip count is below 2^32, so reduce it as an unsigned i32,
  // which halves the cross-lane work.
  bool narrow = !boundTy.isIndex() && boundTy.getIntOrFloatBitWidth() <= 32;
  Type reduceTy = narrow ? rewriter.getI32Type() : rewriter.getI64Type();
  if (narrow)
    tripCount = arith::TruncIOp::create(rewriter, loc, reduceTy, tripCount);
  Value strategy = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
  Value minTripCount =
      LLVM::CallIntrinsicOp::create(rewriter, loc, reduceTy,
                                    rewriter.getStringAttr("llvm.amdgcn.wave.reduce.umin"),
                                    ValueRange{tripCount, strategy})
          .getResult(0);
  if (narrow)
    minTripCount = arith::ExtUIOp::create(rewriter, loc, rewriter.getI64Type(), minTripCount);
  Value depthVal = arith::ConstantIntOp::create(rewriter, loc, depth, 64);
  Value groups = arith::DivUIOp::create(rewriter, loc, minTripCount, depthVal);
  Value mainTrips = arith::MulIOp::create(rewriter, loc, groups, depthVal);
  Value split = arith::AddIOp::create(rewriter, loc, lb,
                                      arith::MulIOp::create(rewriter, loc, mainTrips, step));
  split = fromI64(rewriter, loc, split, boundTy);

  rewriter.setInsertionPointAfter(forOp);
  auto tail = cast<scf::ForOp>(rewriter.clone(*forOp));
  for (auto [from, to] : llvm::zip(forOp.getResults(), tail.getResults()))
    rewriter.replaceAllUsesExcept(from, to, tail);
  tail.setLowerBound(split);
  tail.getInitArgsMutable().assign(forOp.getResults());
  forOp.setUpperBound(split);
  return success();
}

// Stage 0 = reads + the pure address math feeding them, including the math that computes a
// loop-carried address for the next iteration (a pointer bumped after its load); last stage =
// everything else.
void buildLoadComputeSchedule(scf::ForOp forOp,
                              std::vector<std::pair<Operation *, unsigned>> &schedule) {
  Block *body = forOp.getBody();
  llvm::SmallPtrSet<Operation *, 16> stage0;
  SmallVector<Value> worklist;
  for (Operation &op : body->without_terminator()) {
    if (!isPipelineableRead(&op))
      continue;
    stage0.insert(&op);
    llvm::append_range(worklist, op.getOperands());
  }
  // Visit each value once, following pure address math, region captures and loop-carried edges.
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    if (auto arg = dyn_cast<BlockArgument>(v); arg && arg.getOwner() == body) {
      if (arg != forOp.getInductionVar())
        worklist.push_back(body->getTerminator()->getOperand(arg.getArgNumber() - 1));
      continue;
    }
    Operation *op = v.getDefiningOp();
    if (!op || op->getBlock() != body || !isMemoryEffectFree(op) || !stage0.insert(op).second)
      continue;
    llvm::append_range(worklist, op->getOperands());
    SetVector<Value> captures;
    getUsedValuesDefinedAbove(op->getRegions(), captures);
    llvm::append_range(worklist, captures);
  }
  for (Operation &op : body->without_terminator())
    schedule.push_back({&op, stage0.contains(&op) ? 0u : kNumStages - 1});
}

// A type `zeroValue` can build a placeholder for.
bool hasZeroValue(Type t) {
  return isa<IntegerType, IndexType, FloatType, VectorType, LLVM::LLVMPointerType>(t);
}

Value zeroValue(OpBuilder &b, Location loc, Type t) {
  if (isa<LLVM::LLVMPointerType>(t))
    return LLVM::ZeroOp::create(b, loc, t);
  return arith::ConstantOp::create(b, loc, cast<TypedAttr>(b.getZeroAttr(t)));
}

// Whether predicateOp must guard `op` when it is cloned into the prologue or epilogue.
// Upstream marks floordivsi, srem and urem speculatable although a zero divisor is undefined
// behavior, so guard them explicitly.
bool needsPredication(Operation *op) {
  return isPipelineableRead(op) || !isMemoryEffectFree(op) || !isSpeculatable(op) ||
         isa<arith::FloorDivSIOp, LLVM::SRemOp, LLVM::URemOp>(op);
}

// Predicate an op cloned into the prologue/epilogue of a dynamic-trip loop.
//
// Reads and side-effecting ops (stores, atomics, an scf.if or scf.for that reads) are sunk into
// `scf.if(pred)` and yield placeholder zeros when the iteration is out of range: a read issued past
// the trip count can fault, buffer loads included, since a buffer resource is usually created
// without a size limit. Only safely speculatable ops run unchanged on the zeros. The zeros never
// reach the loop results: the pipeliner selects each epilogue live-out against the previous
// iteration's value under the same predicate. `whyNotPipelineable` rejects a loop with an op this
// can't guard, so it never returns nullptr after the pipeliner has changed the IR.
//
// The pipeliner calls this on each clone before any later op uses its results, and maps the
// original op's results to the returned op's results by position.
Operation *predicateOp(RewriterBase &rewriter, Operation *op, Value pred) {
  // A constant-true predicate (a constant trip count) needs no guard.
  if (!needsPredication(op) || matchPattern(pred, m_One()))
    return op;
  if (!llvm::all_of(op->getResultTypes(), hasZeroValue))
    return nullptr;
  Location loc = op->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);
  if (op->getNumResults() == 0) {
    auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, pred, /*withElseRegion=*/false);
    op->moveBefore(ifOp.thenBlock()->getTerminator());
    return ifOp;
  }
  // With result types the builder leaves both blocks unterminated, so build the yields here.
  auto ifOp = scf::IfOp::create(rewriter, loc, op->getResultTypes(), pred, /*withElseRegion=*/true);
  op->moveBefore(ifOp.thenBlock(), ifOp.thenBlock()->end());
  rewriter.setInsertionPointToEnd(ifOp.thenBlock());
  scf::YieldOp::create(rewriter, loc, op->getResults());
  rewriter.setInsertionPointToEnd(ifOp.elseBlock());
  SmallVector<Value> zeros;
  for (Type t : op->getResultTypes())
    zeros.push_back(zeroValue(rewriter, loc, t));
  scf::YieldOp::create(rewriter, loc, zeros);
  return ifOp;
}

// A copy into or out of LDS issued by the memory unit, or an asynchronous copy whose completion is
// tied to a separate wait. Pipelining would move it relative to the reads of the same buffer and
// to its wait.
bool isAsyncOrLdsCopy(StringRef n) {
  return (n.contains("lds") && (n.contains("load") || n.contains("store"))) ||
         n.contains("async") || n.contains("tensor.load") || n.contains("tensor.store");
}

// Whether a stage-0 op (a load or its address math) uses a value the last stage computes, in the
// same iteration or through a loop-carried value: it would run before that value exists. Upstream
// rejects such a schedule with an error diagnostic, so check it before changing the loop.
bool readsLateValue(scf::ForOp forOp) {
  std::vector<std::pair<Operation *, unsigned>> schedule;
  buildLoadComputeSchedule(forOp, schedule);
  DenseMap<Operation *, unsigned> stageOf(schedule.begin(), schedule.end());
  Block *body = forOp.getBody();
  auto isLate = [&](Value v) {
    if (auto arg = dyn_cast<BlockArgument>(v); arg && arg.getOwner() == body) {
      if (arg == forOp.getInductionVar())
        return false;
      v = body->getTerminator()->getOperand(arg.getArgNumber() - 1);
    }
    Operation *def = v.getDefiningOp();
    if (!def || !forOp->isProperAncestor(def))
      return false;
    def = body->findAncestorOpInBlock(*def);
    return def && def != body->getTerminator() && stageOf.lookup(def) > 0;
  };
  return llvm::any_of(schedule, [&](auto &entry) {
    if (entry.second != 0)
      return false;
    WalkResult result = entry.first->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands())
        if (isLate(operand))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return result.wasInterrupted();
  });
}

// Why `forOp` can't be pipelined, or an empty string when it can. Mechanical whole-loop rotation
// cannot safely pipeline a loop that contains an intra-loop barrier or a single-buffered copy into
// LDS: rotating the copy across the backedge would clobber the shared buffer (no double buffering),
// and a dynamic-trip epilogue would have to predicate the barrier, which is illegal. Prints and
// asserts would run out of order, and a read after a write could move ahead of it.
StringRef whyNotPipelineable(scf::ForOp forOp) {
  StringRef reason;
  forOp.getBody()->walk([&](Operation *op) {
    StringRef n = op->getName().getStringRef();
    if (n.contains("barrier"))
      reason = "it contains a barrier";
    else if (isAsyncOrLdsCopy(n))
      reason = "it contains a copy into or out of LDS, or an asynchronous copy";
    else if (n.contains("print") || n.contains("assert"))
      reason = "it contains a print or an assert";
    return reason.empty() ? WalkResult::advance() : WalkResult::interrupt();
  });
  if (!reason.empty())
    return reason;
  Block *body = forOp.getBody();
  if (llvm::none_of(body->without_terminator(),
                    [](Operation &op) { return isPipelineableRead(&op); }))
    return "its body has no load at the top level to issue early";
  if (hasReadAfterWrite(forOp))
    return "a load follows a possible store in the same iteration and would move ahead of it";
  if (llvm::any_of(body->without_terminator(), [](Operation &op) {
        return needsPredication(&op) && !llvm::all_of(op.getResultTypes(), hasZeroValue);
      }))
    return "a load or side-effecting op returns a type the out-of-range iterations can't stand in "
           "for (only integers, floats, vectors, index and pointers)";
  if (readsLateValue(forOp))
    return "a load's address depends on a value the loop computes after its loads (in the same "
           "iteration or carried from the previous one)";
  return {};
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
      // The loop was marked explicitly, so say why it stays as it is.
      auto notPipelined = [&, depth = depth](scf::ForOp loop, StringRef why) {
        loop.emitRemark() << "loop not pipelined (prefetch_depth=" << depth << "): " << why;
      };
      if (depth < 1) {
        notPipelined(forOp, "the depth must be at least 1");
        continue;
      }
      if (StringRef why = whyNotPipelineable(forOp); !why.empty()) {
        notPipelined(forOp, why);
        continue;
      }
      if (depth > 1) {
        // Narrow bounds are widened below, so a narrow runtime step is bounded by its type. A
        // runtime 64-bit or index step is assumed to stay below 2^63 / depth.
        std::optional<int64_t> maxStep = getConstantIntValue(forOp.getStep());
        if (!maxStep) {
          if (auto ty = dyn_cast<IntegerType>(forOp.getStep().getType()); ty && ty.getWidth() < 64)
            maxStep = forOp.getUnsignedCmp()
                          ? APInt::getMaxValue(ty.getWidth()).getZExtValue()
                          : APInt::getSignedMaxValue(ty.getWidth()).getSExtValue();
        }
        if (maxStep && (*maxStep <= 0 || depth > std::numeric_limits<int64_t>::max() / *maxStep)) {
          notPipelined(forOp,
                       "its step multiplied by the depth may overflow a signed 64-bit integer");
          continue;
        }
      }
      // Unrolling and peeling split each lane's iterations at its own trip count; a lane-exchange
      // op needs that split to fall at the same iteration in every lane of the wave.
      if (hasLaneExchangeOp(forOp) && !hasWaveUniformTripCount(forOp) &&
          failed(splitAtWaveUniformTripCount(rewriter, forOp, depth))) {
        notPipelined(forOp, "it exchanges values across lanes and compares its bounds unsigned, so "
                            "its trip count can't be split the same way in every lane");
        continue;
      }

      widenLoopBounds(rewriter, forOp);
      // Both unrolling and peeling assume a nonnegative span. Clamp even at depth one, where
      // subtracting a step from an empty range's original upper bound could underflow.
      rewriter.setInsertionPoint(forOp);
      Value lb = forOp.getLowerBound(), ub = forOp.getUpperBound();
      // createOrFold keeps a constant trip count constant, so its loop needs no predication.
      Value clamped = forOp.getUnsignedCmp()
                          ? rewriter.createOrFold<arith::MaxUIOp>(forOp.getLoc(), lb, ub)
                          : rewriter.createOrFold<arith::MaxSIOp>(forOp.getLoc(), lb, ub);
      forOp.setUpperBound(clamped);
      scf::ForOp target = forOp;
      if (depth > 1) {
        // The remainder loop stays serial. Keep a one-iteration main loop alive for the pipeliner.
        FailureOr<UnrolledLoopInfo> info = loopUnrollByFactor(
            forOp, depth, /*annotateFn=*/nullptr, /*shouldPromoteIfSingleIteration=*/false);
        if (succeeded(info)) {
          if (!info->mainLoopOp)
            continue;
          target = *info->mainLoopOp;
        }
      }

      scf::PipeliningOption options;
      options.supportDynamicLoops = true;
      // peelEpilogue (default true) must stay true: folding the epilogue into the kernel would
      // predicate every steady-state load, forcing a wait right after each one.
      options.predicateFn = predicateOp;
      options.getScheduleFn = buildLoadComputeSchedule;
      rewriter.setInsertionPoint(target);
      bool modified = false;
      // whyNotPipelineable rejects the loops the pipeliner can't handle, so it should not fail
      // here. If it does, keep its diagnostic away from callers: an error the pass then ignores
      // aborts a Python PassManager.run, which asserts that no captured error goes unreported.
      std::string pipelinerMessage;
      LogicalResult pipelined = failure();
      {
        ScopedDiagnosticHandler capture(&getContext(), [&](Diagnostic &diag) {
          if (pipelinerMessage.empty())
            pipelinerMessage = diag.str();
          return success();
        });
        pipelined = scf::pipelineForLoop(rewriter, target, options, &modified);
      }
      if (failed(pipelined)) {
        if (modified) {
          target.emitError() << "prefetch_depth=" << depth
                             << ": pipelining failed after changing the loop: " << pipelinerMessage;
          return signalPassFailure();
        }
        notPipelined(target, "the pipeliner rejected it (" + pipelinerMessage +
                                 "); the loop is only unrolled");
      }
    }
  }
};

} // namespace
