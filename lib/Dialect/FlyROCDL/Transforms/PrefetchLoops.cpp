// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Software-pipeline scf.for loops marked with fly.prefetch_depth = N by the frontend.
// Run inside gpu.module() after convert-fly-to-rocdl.
//
// Unroll by N, then use SCF's two-stage pipeliner to issue the next group's loads ahead of
// the current group's compute. Values cross the backedge in registers, so loads still complete
// before the backedge. Dynamic prologues and epilogues guard reads and side effects.
//
// Consume the frontend markers, analyze each loop, then prepare its bounds and pipeline it.
// Legality failures leave the loop unchanged apart from its markers and emit a remark. Loads
// move ahead of earlier iterations' writes, so writable loops require the frontend's
// fly.prefetch_assume_independent assertion of iteration independence.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h"

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace mlir {
namespace fly_rocdl {
#define GEN_PASS_DEF_FLYROCDLPREFETCHLOOPSPASS
#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h.inc"
} // namespace fly_rocdl
} // namespace mlir

namespace {

/// Loop markers set by the frontend for `range(..., prefetch_depth=N)`.
constexpr StringLiteral kPrefetchDepthAttr = "fly.prefetch_depth";
/// Set by `prefetch_assume_independent=True`: no iteration reads memory an earlier one wrote.
constexpr StringLiteral kAssumeIndependentAttr = "fly.prefetch_assume_independent";
/// The loop's source location, for remarks: without debug info (FlyDSL's default) the loop's own
/// location points into re-parsed IR text instead of the user's source.
constexpr StringLiteral kSourceLocAttr = "fly.prefetch_loc";

/// Unrolling by more grows the IR and the registers without hiding more latency: vec4 bodies
/// already need ~265 VGPRs at 32 and spill at 64 on gfx950.
constexpr int64_t kMaxDepth = 32;

constexpr unsigned kPrivateAddressSpace = 5;

//===----------------------------------------------------------------------===//
// Op classification
//===----------------------------------------------------------------------===//

// Many ROCDL ops carry no useful traits: barriers and waits declare an empty effect list, so MLIR
// treats them as pure, and MFMA has no effect interface, so MLIR assumes it writes memory. Unknown
// ops must conservatively be treated as possible memory writes and lane exchanges.

constexpr StringRef kMatrixPrefixes[] = {"mfma.", "smfmac.", "wmma.", "swmmac."};
constexpr StringRef kTransposedLoadPrefixes[] = {"global.load.tr", "ds.read.tr", "ds.load.tr"};
constexpr StringRef kLaneLocalComputePrefixes[] = {"sdot",  "udot", "sudot", "fdot",
                                                   "dot4.", "cvt.", "mbcnt."};
constexpr StringRef kLaneExchangePrefixes[] = {
    "readlane",   "readfirstlane", "permlane",    "ds_bpermute", "ds.bpermute",
    "ds_swizzle", "ds.swizzle",    "update.dpp",  "mov.dpp",     "ballot",
    "icmp",       "fcmp",          "wave.reduce."};
constexpr StringRef kLaneLocalMemoryPrefixes[] = {
    "raw.ptr.buffer.", "raw.buffer.", "struct.ptr.buffer.", "struct.buffer.",
    "global.",         "ds.read",     "ds.write",           "make.buffer.rsrc"};
constexpr StringRef kLaneLocalControlPrefixes[] = {"workitem.", "workgroup.", "grid.",
                                                   "sched.",    "s.sleep",    "s.setprio"};

constexpr StringRef kWaveUniformGpuIds[] = {
    "gpu.block_id",    "gpu.block_dim",     "gpu.grid_dim",
    "gpu.cluster_id",  "gpu.cluster_dim",   "gpu.cluster_block_id",
    "gpu.subgroup_id", "gpu.num_subgroups", "gpu.subgroup_size"};
constexpr StringRef kPerLaneGpuIds[] = {"gpu.thread_id", "gpu.lane_id", "gpu.global_id"};

/// Whether the op's name, or an intrinsic call's intrinsic, starts with one of `prefixes` after
/// `rocdl.` or `llvm.amdgcn.`.
bool hasAmdgpuName(Operation *op, ArrayRef<StringRef> prefixes) {
  StringRef n = op->getName().getStringRef();
  if (auto intrinsic = dyn_cast<LLVM::CallIntrinsicOp>(op))
    n = intrinsic.getIntrin();
  if (!n.consume_front("rocdl.") && !n.consume_front("llvm.amdgcn."))
    return false;
  return llvm::any_of(prefixes, [&](StringRef prefix) { return n.starts_with(prefix); });
}

/// Atomic and volatile loads must retain their ordering.
bool isPrefetchableLoad(Operation *op) {
  if (auto load = dyn_cast<LLVM::LoadOp>(op))
    return load.getOrdering() == LLVM::AtomicOrdering::not_atomic && !load.getVolatile_();
  if (isa<vector::TransferReadOp, vector::LoadOp>(op))
    return true;
  StringRef n = op->getName().getStringRef();
  return n.contains("buffer.load") || n.contains("global.load") || n.contains("ds.read");
}

/// Scheduling hints neither synchronize threads nor access memory.
bool isSchedulingHint(Operation *op) {
  return op->getName().getStringRef().starts_with("rocdl.sched.");
}

bool isBarrier(Operation *op) {
  return op->getName().getStringRef().contains("barrier") && !isSchedulingHint(op);
}

/// A wait would also drain the prefetched loads, leaving none in flight.
bool isCounterWait(Operation *op) {
  if (auto inlineAsm = dyn_cast<LLVM::InlineAsmOp>(op))
    return inlineAsm.getAsmString().contains("s_wait");
  return hasAmdgpuName(op, {"s.waitcnt", "s.wait."});
}

/// These copies cannot move relative to accesses to the same buffer or their completion waits.
bool isLdsOrAsyncCopy(Operation *op) {
  StringRef n = op->getName().getStringRef();
  return (n.contains("lds") && (n.contains("load") || n.contains("store"))) ||
         n.contains("async") || n.contains("tensor.load") || n.contains("tensor.store");
}

bool isPrintOrAssert(Operation *op) {
  StringRef n = op->getName().getStringRef();
  return n.contains("print") || n.contains("assert");
}

bool isKnownNonMemoryOp(Operation *op) {
  return isSchedulingHint(op) || hasAmdgpuName(op, kMatrixPrefixes) ||
         hasAmdgpuName(op, kLaneLocalComputePrefixes) || hasAmdgpuName(op, kLaneExchangePrefixes);
}

bool isMatrixOrTransposeOp(Operation *op) {
  return hasAmdgpuName(op, kMatrixPrefixes) || hasAmdgpuName(op, kTransposedLoadPrefixes);
}

/// Whether each lane's result depends only on its own operands, regardless of the active lanes.
/// Unrolling splits iterations between the main loop and a serial remainder. With divergent trip
/// counts, lane exchanges could then see different active lanes. Unknown ops require a uniform
/// split; treating a lane-local op as an exchange only costs a wave reduction.
bool isLaneLocal(Operation *op) {
  if (isMatrixOrTransposeOp(op))
    return false;
  StringRef dialect = op->getDialect() ? op->getDialect()->getNamespace() : "";
  StringRef n = op->getName().getStringRef();
  if (llvm::is_contained(
          {"arith", "builtin", "cf", "index", "math", "memref", "scf", "ub", "vector"}, dialect))
    return true;
  if (dialect == "llvm") {
    if (auto intrinsic = dyn_cast<LLVM::CallIntrinsicOp>(op))
      return !intrinsic.getIntrin().starts_with("llvm.amdgcn.");
    return !isa<LLVM::CallOp, LLVM::InlineAsmOp>(op);
  }
  if (dialect == "gpu")
    return llvm::is_contained(kWaveUniformGpuIds, n) || llvm::is_contained(kPerLaneGpuIds, n);
  if (dialect == "rocdl")
    return hasAmdgpuName(op, kLaneLocalMemoryPrefixes) ||
           hasAmdgpuName(op, kLaneLocalComputePrefixes) ||
           hasAmdgpuName(op, kLaneLocalControlPrefixes);
  return false;
}

/// An operand-free op whose value is the same in every lane of a wave.
bool isUniformSource(Operation *op) {
  if (op->hasTrait<OpTrait::ConstantLike>())
    return true;
  StringRef n = op->getName().getStringRef();
  return llvm::is_contained(kWaveUniformGpuIds, n) || n == "llvm.mlir.addressof" ||
         n.starts_with("rocdl.workgroup.") || n.starts_with("rocdl.grid.");
}

/// Unknown operations may read the lane id even when all explicit operands are uniform.
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

/// Upstream marks floordivsi, srem and urem speculatable although a zero divisor is undefined
/// behavior, so guard them explicitly.
bool needsPredication(Operation *op) {
  return isPrefetchableLoad(op) || !isMemoryEffectFree(op) || !isSpeculatable(op) ||
         isa<arith::FloorDivSIOp, LLVM::SRemOp, LLVM::URemOp>(op);
}

/// Whether createZeroValue can build an out-of-range placeholder of this type.
bool hasZeroValue(Type t) {
  if (auto vec = dyn_cast<VectorType>(t))
    t = vec.getElementType();
  return isa<IntegerType, IndexType, FloatType, LLVM::LLVMPointerType>(t);
}

//===----------------------------------------------------------------------===//
// Wave uniformity
//===----------------------------------------------------------------------===//

/// Conservative, demand-driven uniformity analysis. Keep the cache local to one loop analysis:
/// splitting or widening a loop changes the facts used to prove its induction variable uniform.
class WaveUniformityAnalysis {
public:
  bool hasUniformTripCount(scf::ForOp forOp);

private:
  bool isUniform(Value value);
  DenseMap<Value, bool> knownUniform;
};

bool WaveUniformityAnalysis::hasUniformTripCount(scf::ForOp forOp) {
  return isUniform(forOp.getLowerBound()) && isUniform(forOp.getUpperBound()) &&
         isUniform(forOp.getStep());
}

bool WaveUniformityAnalysis::isUniform(Value v) {
  if (auto it = knownUniform.find(v); it != knownUniform.end())
    return it->second;
  knownUniform[v] = false; // A value reached again through a cycle is not uniform.
  bool uniform = false;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Operation *parent = arg.getOwner()->getParentOp();
    // Matched by name rather than dyn_cast, as elsewhere in FlyROCDL, with both spellings that
    // GPUFuncOp::isKernel accepts.
    if (parent->getName().getStringRef() == "gpu.func" &&
        (parent->hasAttr("kernel") || parent->hasAttr("gpu.kernel")))
      uniform = true;
    else if (auto forOp = dyn_cast<scf::ForOp>(parent); forOp && v == forOp.getInductionVar())
      uniform = hasUniformTripCount(forOp);
  } else {
    Operation *op = v.getDefiningOp();
    StringRef n = op->getName().getStringRef();
    if (n.contains("readfirstlane"))
      uniform = true;
    else if (op->getNumOperands() == 0)
      uniform = isUniformSource(op);
    else if (op->getNumRegions() == 0 && isLaneLocal(op) && propagatesUniformity(op) &&
             (isMemoryEffectFree(op) || hasSingleEffect<MemoryEffects::Read>(op)))
      uniform = llvm::all_of(op->getOperands(), [&](Value operand) { return isUniform(operand); });
  }
  knownUniform[v] = uniform;
  return uniform;
}

/// A lane exchange requires the main/tail split to fall at the same iteration in every lane.
bool needsWaveUniformSplit(scf::ForOp forOp) {
  WalkResult result = forOp.getBody()->walk([](Operation *op) {
    return isLaneLocal(op) ? WalkResult::advance() : WalkResult::interrupt();
  });
  WaveUniformityAnalysis uniformity;
  return result.wasInterrupted() && !uniformity.hasUniformTripCount(forOp);
}

//===----------------------------------------------------------------------===//
// Memory dependences
//===----------------------------------------------------------------------===//

/// Check only leaves; callers walk nested regions separately.
bool mayWriteMemory(Operation *op) {
  return op->getNumRegions() == 0 && !isPrefetchableLoad(op) && !isMemoryEffectFree(op) &&
         !hasSingleEffect<MemoryEffects::Read>(op) && !isKnownNonMemoryOp(op);
}

/// A fence, a volatile access, or an atomic whose ordering also orders other memory accesses:
/// moving loads across it changes what they may observe.
bool isOrderedOrVolatile(Operation *op) {
  auto orders = [](LLVM::AtomicOrdering o) {
    return o != LLVM::AtomicOrdering::not_atomic && o != LLVM::AtomicOrdering::unordered &&
           o != LLVM::AtomicOrdering::monotonic;
  };
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<LLVM::FenceOp>([](auto) { return true; })
      .Case<LLVM::LoadOp, LLVM::StoreOp, LLVM::AtomicRMWOp>(
          [&](auto access) { return access.getVolatile_() || orders(access.getOrdering()); })
      .Case<LLVM::AtomicCmpXchgOp>([&](auto cmpxchg) {
        return cmpxchg.getVolatile_() || orders(cmpxchg.getSuccessOrdering()) ||
               orders(cmpxchg.getFailureOrdering());
      })
      .Default(false);
}

bool isPrivatePointer(Type t) {
  auto ptr = dyn_cast<LLVM::LLVMPointerType>(t);
  return ptr && ptr.getAddressSpace() == kPrivateAddressSpace;
}

/// Return the access root, or null when unknown. ROCDL resources and addresses are the first
/// pointer operand, after any store or atomic data.
Value getAccessRoot(Operation *op) {
  Value ptr = llvm::TypeSwitch<Operation *, Value>(op)
                  .Case<LLVM::LoadOp, LLVM::StoreOp>([](auto access) { return access.getAddr(); })
                  .Case<LLVM::AtomicRMWOp, LLVM::AtomicCmpXchgOp>(
                      [](auto atomic) { return atomic.getPtr(); })
                  .Default([](Operation *op) -> Value {
                    if (!op->getName().getStringRef().starts_with("rocdl."))
                      return nullptr;
                    auto it = llvm::find_if(op->getOperands(), [](Value operand) {
                      return isa<LLVM::LLVMPointerType>(operand.getType());
                    });
                    return it == op->operand_end() ? Value() : *it;
                  });
  while (ptr) {
    Operation *def = ptr.getDefiningOp();
    if (auto gep = dyn_cast_or_null<LLVM::GEPOp>(def))
      ptr = gep.getBase();
    else if (auto cast = dyn_cast_or_null<LLVM::AddrSpaceCastOp>(def))
      ptr = cast.getArg();
    else if (def && def->getName().getStringRef() == "rocdl.make.buffer.rsrc")
      ptr = def->getOperand(0);
    else
      break;
  }
  return ptr;
}

/// Includes pure expressions of values defined outside the loop.
bool isLoopInvariant(Value v, scf::ForOp forOp, DenseMap<Value, bool> &memo) {
  if (forOp.isDefinedOutsideOfLoop(v))
    return true;
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (auto it = memo.find(v); it != memo.end())
    return it->second;
  memo[v] = false; // Provisional: a value reached again through a cycle is not invariant.
  bool invariant = def->getNumRegions() == 0 && isMemoryEffectFree(def) &&
                   llvm::all_of(def->getOperands(), [&](Value operand) {
                     return isLoopInvariant(operand, forOp, memo);
                   });
  memo[v] = invariant;
  return invariant;
}

/// An invariant-address load may read an earlier iteration's write to the same buffer. Moving it
/// into the load stage would bypass that write.
bool readsInvariantAddressOfWrittenBuffer(scf::ForOp forOp) {
  DenseSet<Value> written;
  forOp.getBody()->walk([&](Operation *op) {
    if (mayWriteMemory(op))
      if (Value root = getAccessRoot(op))
        written.insert(root);
  });
  DenseMap<Value, bool> memo;
  return llvm::any_of(forOp.getBody()->without_terminator(), [&](Operation &op) {
    if (!isPrefetchableLoad(&op))
      return false;
    Value root = getAccessRoot(&op);
    return root && written.contains(root) && llvm::all_of(op.getOperands(), [&](Value operand) {
             return isLoopInvariant(operand, forOp, memo);
           });
  });
}

/// Moving a top-level load into the load stage crosses preceding stores even within its own
/// iteration; iteration independence does not rule out these aliases.
bool hasLoadAfterPossibleStore(scf::ForOp forOp) {
  bool sawWrite = false;
  WalkResult result = forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (sawWrite && isPrefetchableLoad(op) && op->getBlock() == forOp.getBody())
      return WalkResult::interrupt();
    sawWrite |= mayWriteMemory(op);
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

//===----------------------------------------------------------------------===//
// Pipeline schedule
//===----------------------------------------------------------------------===//

using Schedule = std::vector<std::pair<Operation *, unsigned>>;
constexpr unsigned kLoadStage = 0;
constexpr unsigned kComputeStage = 1;

/// The two-stage schedule issues loads and their address computations one iteration before the
/// remaining operations. The prefetch depth is the unroll factor, not the number of stages.
///
/// Use the same dependency closure for legality checks and schedule construction. This analysis
/// is valid only until the loop changes; construct it again after preparing and unrolling the loop.
class LoadStageAnalysis {
public:
  explicit LoadStageAnalysis(scf::ForOp forOp);

  bool empty() const { return loadStage.empty(); }
  bool dependsOnComputeStage();
  void appendSchedule(Schedule &schedule);

private:
  scf::ForOp forOp;
  llvm::SmallPtrSet<Operation *, 16> loadStage;
};

/// Collect top-level loads and the pure address math feeding them, including the math that
/// computes a loop-carried address for the next iteration (a pointer bumped after its load).
LoadStageAnalysis::LoadStageAnalysis(scf::ForOp forOp) : forOp(forOp) {
  Block *body = forOp.getBody();
  SmallVector<Value> worklist;
  for (Operation &op : body->without_terminator()) {
    if (!isPrefetchableLoad(&op))
      continue;
    loadStage.insert(&op);
    llvm::append_range(worklist, op.getOperands());
  }
  // A generic backward slice follows a block argument to its parent op. Here we need the tied
  // yield instead, to include the next iteration's address without including the whole loop.
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    if (auto arg = dyn_cast<BlockArgument>(v); arg && arg.getOwner() == body) {
      if (OpOperand *yielded = forOp.getTiedLoopYieldedValue(arg))
        worklist.push_back(yielded->get());
      continue;
    }
    Operation *op = v.getDefiningOp();
    if (!op || op->getBlock() != body || !isMemoryEffectFree(op) || !loadStage.insert(op).second)
      continue;
    llvm::append_range(worklist, op->getOperands());
    SetVector<Value> captures;
    getUsedValuesDefinedAbove(op->getRegions(), captures);
    llvm::append_range(worklist, captures);
  }
}

void LoadStageAnalysis::appendSchedule(Schedule &schedule) {
  // Emit in body order so scheduling does not reorder operations within a stage.
  for (Operation &op : forOp.getBody()->without_terminator())
    schedule.emplace_back(&op, loadStage.contains(&op) ? kLoadStage : kComputeStage);
}

/// Whether a load-stage op (a load or its address math) uses a value the compute stage computes, in
/// the same iteration or through a loop-carried value: it would run before that value exists.
/// Upstream rejects such a schedule with an error diagnostic, so check it before changing the loop.
bool LoadStageAnalysis::dependsOnComputeStage() {
  Block *body = forOp.getBody();
  auto isComputedLate = [&](Value v) {
    if (auto arg = dyn_cast<BlockArgument>(v); arg && arg.getOwner() == body) {
      OpOperand *yielded = forOp.getTiedLoopYieldedValue(arg);
      if (!yielded) // The induction variable.
        return false;
      v = yielded->get();
    }
    Operation *def = v.getDefiningOp();
    if (!def || !forOp->isProperAncestor(def))
      return false;
    def = body->findAncestorOpInBlock(*def);
    return def && def != body->getTerminator() && !loadStage.contains(def);
  };
  return llvm::any_of(body->without_terminator(), [&](Operation &op) {
    if (!loadStage.contains(&op))
      return false;
    WalkResult result = op.walk([&](Operation *nested) {
      return llvm::any_of(nested->getOperands(), isComputedLate) ? WalkResult::interrupt()
                                                                 : WalkResult::advance();
    });
    return result.wasInterrupted();
  });
}

//===----------------------------------------------------------------------===//
// Prefetch requests and legality
//===----------------------------------------------------------------------===//

/// A loop marked by `range(..., prefetch_depth=N)`, with its markers already removed.
struct PrefetchRequest {
  scf::ForOp loop;
  int64_t depth;
  bool assumeIndependent;
  Location remarkLoc;
  std::string kernelName;

  /// An active InFlightDiagnostic converts to failure in legality checks. The pass treats these
  /// remarks as clean rejections, not pass failures.
  InFlightDiagnostic emitRemark(const Twine &why) const {
    InFlightDiagnostic remark = mlir::emitRemark(remarkLoc);
    remark << "loop not pipelined (prefetch_depth=" << depth << "): " << why;
    if (!kernelName.empty())
      remark << " (in " << kernelName << ")";
    return remark;
  }
};

/// Decisions needed to prepare an accepted loop. Do not retain operations from the body here:
/// widening and unrolling invalidate the load-stage analysis.
struct PrefetchPlan {
  int64_t depth;
  bool splitAtUniformTrips;
};

/// Reads the loop's markers and removes them, so no marked loop keeps them whatever happens next.
std::optional<PrefetchRequest> takePrefetchRequest(RewriterBase &rewriter, scf::ForOp forOp) {
  auto depth = forOp->getAttrOfType<IntegerAttr>(kPrefetchDepthAttr);
  if (!depth)
    return std::nullopt;
  PrefetchRequest request{
      forOp, depth.getInt(), forOp->hasAttr(kAssumeIndependentAttr), forOp.getLoc(), {}};
  if (auto recorded = forOp->getAttrOfType<LocationAttr>(kSourceLocAttr))
    request.remarkLoc = recorded;
  if (auto symbol = forOp->getParentOfType<SymbolOpInterface>())
    request.kernelName = symbol.getName().str();
  rewriter.modifyOpInPlace(forOp, [&] {
    forOp->removeAttr(kPrefetchDepthAttr);
    forOp->removeAttr(kAssumeIndependentAttr);
    forOp->removeAttr(kSourceLocAttr);
  });
  return request;
}

/// The loop's constant step as the loop compares it: zero-extended for an unsigned loop (clamped to
/// INT64_MAX when it doesn't fit), sign-extended otherwise.
std::optional<int64_t> getConstantStepValue(scf::ForOp forOp) {
  std::optional<APInt> step = forOp.getConstantStep();
  if (!step)
    return std::nullopt;
  if (!forOp.getUnsignedCmp())
    return step->getSExtValue();
  if (step->getActiveBits() >= 64)
    return std::numeric_limits<int64_t>::max();
  return static_cast<int64_t>(step->getZExtValue());
}

/// Whether the unroller's step, the step times `depth`, may overflow a signed 64-bit integer.
/// Narrow bounds are widened first, so a narrow runtime step is bounded by its type; a runtime
/// 64-bit or index step is assumed to stay below 2^63 / depth.
bool mayOverflowUnrolledStep(scf::ForOp forOp, int64_t depth) {
  std::optional<int64_t> maxStep = getConstantStepValue(forOp);
  auto ty = dyn_cast<IntegerType>(forOp.getStep().getType());
  if (!maxStep && ty && ty.getWidth() < 64)
    maxStep = forOp.getUnsignedCmp() ? APInt::getMaxValue(ty.getWidth()).getZExtValue()
                                     : APInt::getSignedMaxValue(ty.getWidth()).getSExtValue();
  return maxStep && depth > std::numeric_limits<int64_t>::max() / *maxStep;
}

/// Check hazards that rule out pipelining anywhere in the body, including nested regions.
/// Rotating an LDS copy can clobber a single-buffered allocation, and a dynamic epilogue cannot
/// predicate a barrier. Private memory may hold compiler-generated register arrays, so the user's
/// independence assertion cannot justify reordering its accesses.
LogicalResult checkPrefetchHazards(Operation *op, const PrefetchRequest &request) {
  if (isBarrier(op))
    return request.emitRemark("it contains a barrier");
  if (isCounterWait(op))
    return request.emitRemark("it waits on a memory counter (" + op->getName().getStringRef() +
                              "), which would also wait for the prefetched loads");
  if (isLdsOrAsyncCopy(op))
    return request.emitRemark("it contains a copy into or out of LDS, or an asynchronous copy");
  if (isPrintOrAssert(op))
    return request.emitRemark("it contains a print or an assert");
  if (isOrderedOrVolatile(op))
    return request.emitRemark(
        "it contains a fence, a volatile access or an atomic with acquire or stronger ordering");
  if (llvm::any_of(op->getOperandTypes(), isPrivatePointer) ||
      llvm::any_of(op->getResultTypes(), isPrivatePointer))
    return request.emitRemark(
        "it accesses private memory, which an earlier iteration may have written");
  return success();
}

/// Check memory ordering, predication and load-stage dependencies. The independence assertion
/// permits moving loads across iterations, but does not override known dependences.
LogicalResult checkPrefetchableBody(const PrefetchRequest &request) {
  scf::ForOp forOp = request.loop;
  Block *body = forOp.getBody();
  WalkResult hazards =
      body->walk([&](Operation *op) -> WalkResult { return checkPrefetchHazards(op, request); });
  if (hazards.wasInterrupted())
    return failure();

  LoadStageAnalysis loadStage(forOp);
  if (loadStage.empty())
    return request.emitRemark("its body has no load at the top level to issue early");
  if (!request.assumeIndependent) {
    WalkResult writes = body->walk([&](Operation *op) -> WalkResult {
      if (!mayWriteMemory(op))
        return WalkResult::advance();
      request.emitRemark(
          "it may write memory (" + op->getName().getStringRef() +
          ") and its loads would move ahead of earlier iterations' writes; set "
          "prefetch_assume_independent=True if no iteration reads memory an earlier one wrote");
      return WalkResult::interrupt();
    });
    if (writes.wasInterrupted())
      return failure();
  }
  if (hasLoadAfterPossibleStore(forOp))
    return request.emitRemark(
        "a load follows a possible store in the same iteration and would move ahead of it");
  if (readsInvariantAddressOfWrittenBuffer(forOp))
    return request.emitRemark("a load reads the same address in every iteration from a buffer "
                              "the loop writes, so it would read before an earlier iteration's "
                              "write");
  if (llvm::any_of(body->without_terminator(), [](Operation &op) {
        return needsPredication(&op) && !llvm::all_of(op.getResultTypes(), hasZeroValue);
      }))
    return request.emitRemark("a load or side-effecting op returns a type the out-of-range "
                              "iterations can't stand in for (only integers, floats, vectors, "
                              "index and pointers)");
  if (loadStage.dependsOnComputeStage())
    return request.emitRemark("a load's address depends on a value the loop computes after its "
                              "loads (in the same iteration or carried from the previous one)");
  return success();
}

/// Analyze a request before changing bounds or body. Preserve the order of checks so a loop with
/// several hazards continues to report the same reason for rejection.
FailureOr<PrefetchPlan> analyzePrefetchLoop(const PrefetchRequest &request) {
  scf::ForOp forOp = request.loop;
  if (request.depth < 1 || request.depth > kMaxDepth)
    return request.emitRemark("the depth must be between 1 and " + Twine(kMaxDepth));
  if (failed(checkPrefetchableBody(request)))
    return failure();
  if (std::optional<int64_t> step = getConstantStepValue(forOp); step && *step <= 0)
    return request.emitRemark("its step is not positive, which scf.for does not allow");
  if (request.depth > 1 && mayOverflowUnrolledStep(forOp, request.depth))
    return request.emitRemark(
        "its step multiplied by the depth may overflow a signed 64-bit integer");
  bool splitAtUniformTrips = needsWaveUniformSplit(forOp);
  if (forOp.getUnsignedCmp()) {
    if (splitAtUniformTrips)
      return request.emitRemark("it exchanges values across lanes and compares its bounds "
                                "unsigned, so its trip count can't be split the same way in "
                                "every lane");
    // Widening makes a narrow unsigned loop signed; a 64-bit or index one stays unsigned.
    auto boundTy = dyn_cast<IntegerType>(forOp.getLowerBound().getType());
    if (!boundTy || boundTy.getWidth() >= 64)
      return request.emitRemark(
          "it compares 64-bit bounds unsigned, which the pipeliner does not support");
  }
  return PrefetchPlan{request.depth, splitAtUniformTrips};
}

//===----------------------------------------------------------------------===//
// Loop preparation
//===----------------------------------------------------------------------===//

/// Fold casts so constant bounds remain available to the unroller and pipeliner.
Value extendToI64(RewriterBase &rewriter, Location loc, Value v, bool isUnsigned) {
  Type i64 = rewriter.getI64Type();
  if (v.getType().isIndex())
    return rewriter.createOrFold<arith::IndexCastOp>(loc, i64, v);
  if (v.getType().getIntOrFloatBitWidth() == 64)
    return v;
  if (isUnsigned)
    return rewriter.createOrFold<arith::ExtUIOp>(loc, i64, v);
  return rewriter.createOrFold<arith::ExtSIOp>(loc, i64, v);
}

/// Keep the unroller's scaled step and the pipeliner's peeled bounds out of narrow integer types.
/// The body still sees the original induction type, including its original wrapping arithmetic.
void widenLoopBounds(RewriterBase &rewriter, scf::ForOp forOp) {
  auto ty = dyn_cast<IntegerType>(forOp.getLowerBound().getType());
  if (!ty || ty.getWidth() >= 64)
    return;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  bool isUnsigned = forOp.getUnsignedCmp();
  Value lb = extendToI64(rewriter, loc, forOp.getLowerBound(), isUnsigned);
  Value ub = extendToI64(rewriter, loc, forOp.getUpperBound(), isUnsigned);
  Value step = extendToI64(rewriter, loc, forOp.getStep(), isUnsigned);
  Value iv = forOp.getInductionVar();
  rewriter.modifyOpInPlace(forOp, [&] {
    forOp.setLowerBound(lb);
    forOp.setUpperBound(ub);
    forOp.setStep(step);
    // Zero-extended narrow bounds fit in signed i64, which the pipeliner requires.
    forOp.setUnsignedCmp(false);
    iv.setType(rewriter.getI64Type());
  });
  rewriter.setInsertionPointToStart(forOp.getBody());
  auto narrow = arith::TruncIOp::create(rewriter, loc, ty, iv);
  rewriter.replaceAllUsesExcept(iv, narrow.getResult(), narrow);
}

/// Ends `forOp` after the wave's shortest trip count, rounded down to a multiple of `depth`, and
/// runs the remaining iterations of each lane in a plain tail loop after it. Every lane then runs
/// `forOp` the same number of times, and the tail runs in lockstep, so each iteration index sees
/// the same active lanes as in the original loop. Costs one wave reduction before the loop.
void splitAtWaveUniformTripCount(RewriterBase &rewriter, scf::ForOp forOp, int64_t depth) {
  assert(!forOp.getUnsignedCmp() && "analysis rejects unsigned loops that need a split");
  Location loc = forOp.getLoc();
  Type boundTy = forOp.getLowerBound().getType();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);
  Value lb = extendToI64(rewriter, loc, forOp.getLowerBound(), /*isUnsigned=*/false);
  Value ub = extendToI64(rewriter, loc, forOp.getUpperBound(), /*isUnsigned=*/false);
  Value step = extendToI64(rewriter, loc, forOp.getStep(), /*isUnsigned=*/false);
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
  Value splitBound = getValueOrCreateCastToIndexLike(
      rewriter, loc, boundTy,
      arith::AddIOp::create(rewriter, loc, lb,
                            arith::MulIOp::create(rewriter, loc, mainTrips, step)));

  // The tail loop, built the way scf::peelForLoop builds its partial iteration.
  rewriter.setInsertionPointAfter(forOp);
  auto tail = cast<scf::ForOp>(rewriter.clone(*forOp));
  rewriter.modifyOpInPlace(tail, [&] { tail.setLowerBound(splitBound); });
  rewriter.replaceAllUsesWith(forOp.getResults(), tail.getResults());
  rewriter.modifyOpInPlace(tail, [&] { tail.getInitArgsMutable().assign(forOp.getResults()); });
  rewriter.modifyOpInPlace(forOp, [&] { forOp.setUpperBound(splitBound); });
}

/// Both unrolling and peeling assume a nonnegative span. Clamp even at depth one, where subtracting
/// a step from an empty range's original upper bound could underflow. createOrFold keeps a constant
/// trip count constant, so its loop needs no predication.
void clampUpperBoundToLowerBound(RewriterBase &rewriter, scf::ForOp forOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);
  Value ub = rewriter.createOrFold<arith::MaxSIOp>(forOp.getLoc(), forOp.getLowerBound(),
                                                   forOp.getUpperBound());
  rewriter.modifyOpInPlace(forOp, [&] { forOp.setUpperBound(ub); });
}

/// A null loop means unrolling has promoted or erased the entire main loop. Track whether
/// unrolling succeeded so a later SCF rejection can describe the transformation already applied.
struct PreparedLoop {
  scf::ForOp loop;
  bool unrolled = false;
};

/// Prepare an accepted loop for SCF's pipeliner. The serial tail keeps the original bounds and
/// induction type, so split it off before widening the main loop's bounds.
PreparedLoop prepareLoop(RewriterBase &rewriter, scf::ForOp forOp, const PrefetchPlan &plan) {
  if (plan.splitAtUniformTrips)
    splitAtWaveUniformTripCount(rewriter, forOp, plan.depth);
  widenLoopBounds(rewriter, forOp);
  clampUpperBoundToLowerBound(rewriter, forOp);
  if (plan.depth == 1)
    return {forOp};

  FailureOr<UnrolledLoopInfo> info = loopUnrollByFactor(forOp, plan.depth);
  if (failed(info))
    return {forOp};
  if (!info->mainLoopOp)
    return {};
  // The uniform split ends the main loop at a multiple of the depth. The unroller's remainder
  // loop never runs; the serial tail already handles the remaining iterations.
  if (plan.splitAtUniformTrips && info->epilogueLoopOp)
    rewriter.replaceOp(*info->epilogueLoopOp, info->epilogueLoopOp->getInitArgs());
  return {*info->mainLoopOp, /*unrolled=*/true};
}

//===----------------------------------------------------------------------===//
// Predication
//===----------------------------------------------------------------------===//

Value createZeroValue(OpBuilder &builder, Location loc, Type type) {
  // getZeroAttr has no pointer attribute, for a pointer or a vector of them.
  if (isa<LLVM::LLVMPointerType>(getElementTypeOrSelf(type)))
    return LLVM::ZeroOp::create(builder, loc, type);
  return arith::ConstantOp::create(builder, loc, cast<TypedAttr>(builder.getZeroAttr(type)));
}

/// Create unterminated guard blocks for the caller to populate.
scf::IfOp createEmptyGuard(RewriterBase &rewriter, Location loc, TypeRange types, Value cond) {
  return scf::IfOp::create(rewriter, loc, types, cond, /*addThenBlock=*/true,
                           /*addElseBlock=*/!types.empty());
}

/// Guard an operation cloned into a dynamic prologue or epilogue by SCF's pipeliner.
///
/// Even buffer loads can fault out of range: their resources usually have no size limit. Guarded
/// ops yield zeros out of range, and only safely speculatable ops may consume those zeros without
/// a guard. The zeros cannot escape as loop results because the pipeliner selects each epilogue
/// live-out against the previous iteration's value under the same predicate.
///
/// Body analysis checks the placeholder types before mutation. SCF calls this before creating
/// uses of the clone's results and maps results to the returned op by position.
Operation *predicateOp(RewriterBase &rewriter, Operation *op, Value pred) {
  if (!needsPredication(op) || matchPattern(pred, m_One()))
    return op;
  if (!llvm::all_of(op->getResultTypes(), hasZeroValue))
    return nullptr;
  Location loc = op->getLoc();
  OpBuilder::InsertionGuard insertionGuard(rewriter);
  rewriter.setInsertionPointAfter(op);
  scf::IfOp guard = createEmptyGuard(rewriter, loc, op->getResultTypes(), pred);
  rewriter.moveOpBefore(op, guard.thenBlock(), guard.thenBlock()->end());
  rewriter.setInsertionPointToEnd(guard.thenBlock());
  scf::YieldOp::create(rewriter, loc, op->getResults());
  if (op->getNumResults() == 0)
    return guard;
  rewriter.setInsertionPointToEnd(guard.elseBlock());
  SmallVector<Value> zeros;
  for (Type t : op->getResultTypes())
    zeros.push_back(createZeroValue(rewriter, loc, t));
  scf::YieldOp::create(rewriter, loc, zeros);
  return guard;
}

/// Find the operations to hoist before merging two guards in the same block, with `first` before
/// `second`. Reject memory effects between them and dependencies on the first guard (e.g. a load
/// address from a guarded load). Return the hoist list in block order.
FailureOr<SmallVector<Operation *>> getGuardHoistOps(scf::IfOp first, scf::IfOp second) {
  llvm::SmallPtrSet<Operation *, 16> between;
  for (Operation *op = first->getNextNode(); op != second; op = op->getNextNode()) {
    if (!isMemoryEffectFree(op))
      return failure();
    between.insert(op);
  }
  SetVector<Value> used;
  getUsedValuesDefinedAbove(second->getRegions(), used);
  SmallVector<Value> worklist(used.begin(), used.end());
  llvm::SmallPtrSet<Operation *, 16> needed;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (def == first.getOperation())
      return failure();
    if (!def || !between.contains(def) || !needed.insert(def).second)
      continue;
    if (def->getNumRegions() > 0)
      return failure();
    llvm::append_range(worklist, def->getOperands());
  }
  SmallVector<Operation *> toHoist;
  for (Operation *op = first->getNextNode(); op != second; op = op->getNextNode())
    if (needed.contains(op))
      toHoist.push_back(op);
  return toHoist;
}

FailureOr<scf::IfOp> mergeGuard(RewriterBase &rewriter, scf::IfOp first, scf::IfOp second) {
  FailureOr<SmallVector<Operation *>> toHoist = getGuardHoistOps(first, second);
  if (failed(toHoist))
    return failure();
  for (Operation *op : *toHoist)
    rewriter.moveOpBefore(op, first);

  OpBuilder::InsertionGuard insertionGuard(rewriter);
  SmallVector<Type> types(first->getResultTypes());
  llvm::append_range(types, second->getResultTypes());
  Location loc = first.getLoc();
  rewriter.setInsertionPoint(first);
  scf::IfOp merged = createEmptyGuard(rewriter, loc, types, first.getCondition());
  SmallVector<Value> thenValues, elseValues;
  for (scf::IfOp guard : {first, second}) {
    llvm::append_range(thenValues, guard.thenYield().getOperands());
    rewriter.eraseOp(guard.thenYield());
    rewriter.mergeBlocks(guard.thenBlock(), merged.thenBlock());
    if (guard.getNumResults() == 0)
      continue;
    llvm::append_range(elseValues, guard.elseYield().getOperands());
    rewriter.eraseOp(guard.elseYield());
    rewriter.mergeBlocks(guard.elseBlock(), merged.elseBlock());
  }
  rewriter.setInsertionPointToEnd(merged.thenBlock());
  scf::YieldOp::create(rewriter, loc, thenValues);
  if (!types.empty()) {
    rewriter.setInsertionPointToEnd(merged.elseBlock());
    scf::YieldOp::create(rewriter, loc, elseValues);
  }
  unsigned numFirst = first.getNumResults();
  rewriter.replaceOp(first, merged.getResults().take_front(numFirst));
  rewriter.replaceOp(second, merged.getResults().drop_front(numFirst));
  return merged;
}

/// `predicateOp` guards each op on its own, so N loads in the prologue would become N branch
/// regions, each saving and restoring the exec mask. Merge each guard into the previous one with
/// the same predicate where `mergeGuard` can. Upstream's CombineIfs only merges adjacent ifs, and
/// each guard is preceded by its own address math.
void mergeGuards(RewriterBase &rewriter, ArrayRef<scf::IfOp> guards) {
  scf::IfOp head;
  for (scf::IfOp guard : guards) {
    if (!head || head->getBlock() != guard->getBlock() ||
        head.getCondition() != guard.getCondition() || !head->isBeforeInBlock(guard)) {
      head = guard;
      continue;
    }
    FailureOr<scf::IfOp> merged = mergeGuard(rewriter, head, guard);
    head = succeeded(merged) ? *merged : guard;
  }
}

//===----------------------------------------------------------------------===//
// Pipeline expansion
//===----------------------------------------------------------------------===//

/// A clean SCF rejection emits a remark; failure means SCF modified the IR before failing.
LogicalResult pipelineLoadsAhead(RewriterBase &rewriter, scf::ForOp loop,
                                 const PrefetchRequest &request, bool unrolled) {
  scf::PipeliningOption options;
  options.supportDynamicLoops = true;
  // Folding the epilogue into the kernel would predicate every steady-state load, forcing a wait
  // right after each one.
  options.peelEpilogue = true;
  // Rebuild the schedule after bounds preparation and unrolling, which may create address ops.
  options.getScheduleFn = [](scf::ForOp forOp, Schedule &schedule) {
    LoadStageAnalysis(forOp).appendSchedule(schedule);
  };
  SmallVector<scf::IfOp> guards;
  options.predicateFn = [&](RewriterBase &r, Operation *op, Value pred) {
    Operation *guarded = predicateOp(r, op, pred);
    if (guarded && guarded != op)
      guards.push_back(cast<scf::IfOp>(guarded));
    return guarded;
  };

  // pipelineForLoop emits the prologue at the rewriter's insertion point.
  rewriter.setInsertionPoint(loop);
  // SCF can still reject a loop, e.g. an unsupported loop-carried dependence. Capture its
  // diagnostics so a clean rejection becomes a remark: an ignored error would abort Python's
  // PassManager.run, which requires every captured error to be reported as a pass failure.
  std::string pipelinerMessage;
  bool changedIR = false;
  FailureOr<scf::ForOp> pipelined = failure();
  {
    ScopedDiagnosticHandler capture(loop.getContext(), [&](Diagnostic &diag) {
      if (pipelinerMessage.empty())
        pipelinerMessage = diag.str();
      return success();
    });
    pipelined = scf::pipelineForLoop(rewriter, loop, options, &changedIR);
  }
  if (succeeded(pipelined)) {
    mergeGuards(rewriter, guards);
    return success();
  }
  if (changedIR)
    return loop.emitError() << "loop not pipelined (prefetch_depth=" << request.depth
                            << "): the pipeliner failed after changing the loop: "
                            << pipelinerMessage;
  std::string why = "the pipeliner rejected it";
  if (!pipelinerMessage.empty())
    why += " (" + pipelinerMessage + ")";
  if (unrolled)
    why += "; the loop is only unrolled";
  // The loop stays correct, so the remark is not a pass failure.
  request.emitRemark(why);
  return success();
}

class FlyROCDLPrefetchLoopsPass
    : public mlir::fly_rocdl::impl::FlyROCDLPrefetchLoopsPassBase<FlyROCDLPrefetchLoopsPass> {
public:
  using mlir::fly_rocdl::impl::FlyROCDLPrefetchLoopsPassBase<
      FlyROCDLPrefetchLoopsPass>::FlyROCDLPrefetchLoopsPassBase;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    // Post-order, so an inner marked loop is rewritten before an outer one's unrolling clones it.
    SmallVector<PrefetchRequest> requests;
    getOperation()->walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
      if (std::optional<PrefetchRequest> request = takePrefetchRequest(rewriter, forOp))
        requests.push_back(std::move(*request));
    });

    for (const PrefetchRequest &request : requests) {
      // Analyze just before rewriting: an inner request may already have changed this body.
      FailureOr<PrefetchPlan> plan = analyzePrefetchLoop(request);
      if (failed(plan))
        continue;
      PreparedLoop prepared = prepareLoop(rewriter, request.loop, *plan);
      if (!prepared.loop)
        continue;
      if (failed(pipelineLoadsAhead(rewriter, prepared.loop, request, prepared.unrolled)))
        return signalPassFailure();
    }
  }
};

} // namespace
