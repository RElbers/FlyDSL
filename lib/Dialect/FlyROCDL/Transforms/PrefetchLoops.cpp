// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Software-pipeline the `scf.for` loops that carry `fly.prefetch_depth = N` (set by
// `range(..., prefetch_depth=N)`) so up to N loads per lane are in flight at once. Run inside
// gpu.module() AFTER convert-fly-to-rocdl, while the loads are ROCDL ops and the loop is still an
// scf.for.
//
// A rolled runtime-trip loop otherwise waits for each iteration's single load before the next one
// issues. The pass unrolls the loop by N and then pipelines it in two stages: loads plus their
// address math first, everything else one iteration later. Each steady-state iteration issues the
// N loads of the next group together, ahead of the previous group's compute; how much of that
// compute they overlap is up to register allocation. The loop still waits for them before its
// backedge, since the loaded values are carried into the next iteration in registers.
// In the prologue and epilogue of a runtime trip count, `predicateOp` guards reads and side effects
// so they only run for in-range iterations. Loads move ahead of earlier iterations' writes, so a
// loop that may write memory is only pipelined when `fly.prefetch_assume_independent` (set by
// `prefetch_assume_independent=True`) asserts that no iteration reads memory an earlier one wrote.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
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

// Loop markers set by the frontend for `range(..., prefetch_depth=N)`.
constexpr StringLiteral kPrefetchDepthAttr = "fly.prefetch_depth";
// Set by `prefetch_assume_independent=True`: no iteration reads memory an earlier one wrote.
constexpr StringLiteral kAssumeIndependentAttr = "fly.prefetch_assume_independent";
// The loop's source location, for remarks: without debug info (FlyDSL's default) the loop's own
// location points into re-parsed IR text instead of the user's source.
constexpr StringLiteral kSourceLocAttr = "fly.prefetch_loc";

// Unrolling by more grows the IR and the registers without hiding more latency: vec4 bodies already
// need ~265 VGPRs at 32 and spill at 64 on gfx950.
constexpr int64_t kMaxDepth = 32;

// The pipeline has two stages: the loads with their address math, then everything else, one
// iteration later. The depth comes from the unroll factor, not from more stages.
constexpr unsigned kLoadStage = 0;
constexpr unsigned kComputeStage = 1;
using Schedule = std::vector<std::pair<Operation *, unsigned>>;

// The AMDGPU private (scratch) address space.
constexpr unsigned kPrivateAddressSpace = 5;

//===----------------------------------------------------------------------===//
// Op classification
//
// Every decision the pass makes from an op's name is in this section. Many ROCDL ops carry no
// useful traits: barriers and waits declare an empty effect list, so MLIR treats them as pure, and
// MFMA has no effect interface, so MLIR assumes it writes memory. Where a wrong answer would
// miscompile, an op the tables don't know gets the safe answer: it may write memory, and it may
// exchange values across lanes.
//===----------------------------------------------------------------------===//

// Name prefixes of ROCDL ops and AMDGPU intrinsics, after `rocdl.` or `llvm.amdgcn.`.
constexpr StringRef kMatrixPrefixes[] = {"mfma.", "smfmac.", "wmma.", "swmmac."};
constexpr StringRef kTransposedLoadPrefixes[] = {"global.load.tr", "ds.read.tr", "ds.load.tr"};
// Dot products, conversions and the lane id: lane-local, and touching no memory.
constexpr StringRef kLaneLocalComputePrefixes[] = {"sdot",  "udot", "sudot", "fdot",
                                                   "dot4.", "cvt.", "mbcnt."};
// Lane reads, permutes, DPP, ballots and wave reductions: touching no memory.
constexpr StringRef kLaneExchangePrefixes[] = {
    "readlane",   "readfirstlane", "permlane",    "ds_bpermute", "ds.bpermute",
    "ds_swizzle", "ds.swizzle",    "update.dpp",  "mov.dpp",     "ballot",
    "icmp",       "fcmp",          "wave.reduce."};
// Per-lane memory accesses and buffer resources.
constexpr StringRef kLaneLocalMemoryPrefixes[] = {
    "raw.ptr.buffer.", "raw.buffer.", "struct.ptr.buffer.", "struct.buffer.",
    "global.",         "ds.read",     "ds.write",           "make.buffer.rsrc"};
// Workgroup and grid ids, scheduling hints and wave priority.
constexpr StringRef kLaneLocalControlPrefixes[] = {"workitem.", "workgroup.", "grid.",
                                                   "sched.",    "s.sleep",    "s.setprio"};

// GPU dialect ids that are the same in every lane of a wave.
constexpr StringRef kWaveUniformGpuIds[] = {
    "gpu.block_id",    "gpu.block_dim",     "gpu.grid_dim",
    "gpu.cluster_id",  "gpu.cluster_dim",   "gpu.cluster_block_id",
    "gpu.subgroup_id", "gpu.num_subgroups", "gpu.subgroup_size"};
// GPU dialect ids that differ between lanes.
constexpr StringRef kPerLaneGpuIds[] = {"gpu.thread_id", "gpu.lane_id", "gpu.global_id"};

// Whether the op's name, or an intrinsic call's intrinsic, starts with one of `prefixes` after
// `rocdl.` or `llvm.amdgcn.`.
bool hasAmdgpuName(Operation *op, ArrayRef<StringRef> prefixes) {
  StringRef n = op->getName().getStringRef();
  if (auto intrinsic = dyn_cast<LLVM::CallIntrinsicOp>(op))
    n = intrinsic.getIntrin();
  if (!n.consume_front("rocdl.") && !n.consume_front("llvm.amdgcn."))
    return false;
  return llvm::any_of(prefixes, [&](StringRef prefix) { return n.starts_with(prefix); });
}

// A memory read whose issue we want to lead the compute (the load stage). An atomic or volatile
// load keeps its place.
bool isPrefetchableLoad(Operation *op) {
  if (auto load = dyn_cast<LLVM::LoadOp>(op))
    return load.getOrdering() == LLVM::AtomicOrdering::not_atomic && !load.getVolatile_();
  if (isa<vector::TransferReadOp, vector::LoadOp>(op))
    return true;
  StringRef n = op->getName().getStringRef();
  return n.contains("buffer.load") || n.contains("global.load") || n.contains("ds.read");
}

// An instruction-scheduling hint (rocdl.sched.barrier, rocdl.sched.group.barrier): it neither
// synchronizes nor touches memory.
bool isSchedulingHint(Operation *op) {
  return op->getName().getStringRef().starts_with("rocdl.sched.");
}

bool isBarrier(Operation *op) {
  return op->getName().getStringRef().contains("barrier") && !isSchedulingHint(op);
}

// A wait on a memory counter (s_waitcnt, s_wait_loadcnt, ...), or inline asm containing one. In a
// pipelined loop it would wait for the prefetched loads too, so none would stay in flight.
bool isCounterWait(Operation *op) {
  if (auto inlineAsm = dyn_cast<LLVM::InlineAsmOp>(op))
    return inlineAsm.getAsmString().contains("s_wait");
  return hasAmdgpuName(op, {"s.waitcnt", "s.wait."});
}

// A copy into or out of LDS issued by the memory unit, or an asynchronous copy whose completion is
// tied to a separate wait. Pipelining would move it relative to the reads of the same buffer and
// to its wait.
bool isLdsOrAsyncCopy(Operation *op) {
  StringRef n = op->getName().getStringRef();
  return (n.contains("lds") && (n.contains("load") || n.contains("store"))) ||
         n.contains("async") || n.contains("tensor.load") || n.contains("tensor.store");
}

bool isPrintOrAssert(Operation *op) {
  StringRef n = op->getName().getStringRef();
  return n.contains("print") || n.contains("assert");
}

// ROCDL matrix, dot, conversion and lane ops, and scheduling hints, don't implement the memory
// effect interface, so MLIR assumes they may access memory; none of them does.
bool isKnownNonMemoryOp(Operation *op) {
  return isSchedulingHint(op) || hasAmdgpuName(op, kMatrixPrefixes) ||
         hasAmdgpuName(op, kLaneLocalComputePrefixes) || hasAmdgpuName(op, kLaneExchangePrefixes);
}

// A matrix op (MFMA, WMMA, SMFMAC) or a transposed load: each lane's result is built from the
// registers or addresses of other lanes.
bool isMatrixOrTransposeOp(Operation *op) {
  return hasAmdgpuName(op, kMatrixPrefixes) || hasAmdgpuName(op, kTransposedLoadPrefixes);
}

// Whether `op` gives each lane a result from that lane's own operands, independent of the other
// lanes and of which lanes are active. An allowlist: any op it doesn't know may exchange values
// across lanes (shuffles, ballots, lane reads, DPP, matrix ops, transposed loads, calls, inline
// asm, AMDGPU intrinsic calls). Unrolling splits the iterations between the main loop and a serial
// remainder loop, so when lanes have different trip counts they run an iteration at different
// times, and such an op would see other active lanes than in the original loop. Treating a
// lane-local op as an exchange only costs a wave reduction.
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

// An operand-free op whose value is the same in every lane of a wave.
bool isUniformSource(Operation *op) {
  if (op->hasTrait<OpTrait::ConstantLike>())
    return true;
  StringRef n = op->getName().getStringRef();
  return llvm::is_contained(kWaveUniformGpuIds, n) || n == "llvm.mlir.addressof" ||
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

//===----------------------------------------------------------------------===//
// Wave uniformity
//===----------------------------------------------------------------------===//

bool isWaveUniform(Value v, DenseMap<Value, bool> &memo);

// Every lane of a wave runs `forOp` the same number of times.
bool hasWaveUniformTripCount(scf::ForOp forOp, DenseMap<Value, bool> &memo) {
  return isWaveUniform(forOp.getLowerBound(), memo) && isWaveUniform(forOp.getUpperBound(), memo) &&
         isWaveUniform(forOp.getStep(), memo);
}

bool hasWaveUniformTripCount(scf::ForOp forOp) {
  DenseMap<Value, bool> memo;
  return hasWaveUniformTripCount(forOp, memo);
}

// Whether every lane of a wave holds the same value of `v`. Conservative: false when unsure. Kernel
// arguments, constants and block-level ids are uniform; so is a pure or read-only op from
// `propagatesUniformity` whose operands all are, and a readfirstlane. Thread / lane ids, atomics,
// ops with regions and loop-carried values are not.
bool isWaveUniform(Value v, DenseMap<Value, bool> &memo) {
  if (auto it = memo.find(v); it != memo.end())
    return it->second;
  memo[v] = false; // Provisional: a value reached again through a cycle is not uniform.
  bool uniform = false;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Operation *parent = arg.getOwner()->getParentOp();
    // Matched by name rather than dyn_cast, as elsewhere in FlyROCDL, with both spellings that
    // GPUFuncOp::isKernel accepts.
    if (parent->getName().getStringRef() == "gpu.func" &&
        (parent->hasAttr("kernel") || parent->hasAttr("gpu.kernel")))
      uniform = true;
    else if (auto forOp = dyn_cast<scf::ForOp>(parent); forOp && v == forOp.getInductionVar())
      uniform = hasWaveUniformTripCount(forOp, memo);
  } else {
    Operation *op = v.getDefiningOp();
    StringRef n = op->getName().getStringRef();
    if (n.contains("readfirstlane"))
      uniform = true;
    else if (op->getNumOperands() == 0)
      uniform = isUniformSource(op);
    else if (op->getNumRegions() == 0 && isLaneLocal(op) && propagatesUniformity(op) &&
             (isMemoryEffectFree(op) || hasSingleEffect<MemoryEffects::Read>(op)))
      uniform = llvm::all_of(op->getOperands(),
                             [&](Value operand) { return isWaveUniform(operand, memo); });
  }
  memo[v] = uniform;
  return uniform;
}

bool hasLaneExchangeOp(scf::ForOp forOp) {
  WalkResult result = forOp.getBody()->walk([](Operation *op) {
    return isLaneLocal(op) ? WalkResult::advance() : WalkResult::interrupt();
  });
  return result.wasInterrupted();
}

//===----------------------------------------------------------------------===//
// Memory dependences
//===----------------------------------------------------------------------===//

// A leaf op that may write memory: a write effect, or no effect information at all. An op with
// regions is never one; walks visit the ops inside it.
bool mayWriteMemory(Operation *op) {
  return op->getNumRegions() == 0 && !isPrefetchableLoad(op) && !isMemoryEffectFree(op) &&
         !hasSingleEffect<MemoryEffects::Read>(op) && !isKnownNonMemoryOp(op);
}

// A fence, a volatile access, or an atomic whose ordering also orders other memory accesses:
// moving loads across it changes what they may observe.
bool isOrderedOrVolatile(Operation *op) {
  auto orders = [](LLVM::AtomicOrdering o) {
    return o != LLVM::AtomicOrdering::not_atomic && o != LLVM::AtomicOrdering::unordered &&
           o != LLVM::AtomicOrdering::monotonic;
  };
  if (isa<LLVM::FenceOp>(op))
    return true;
  if (auto load = dyn_cast<LLVM::LoadOp>(op))
    return load.getVolatile_() || orders(load.getOrdering());
  if (auto store = dyn_cast<LLVM::StoreOp>(op))
    return store.getVolatile_() || orders(store.getOrdering());
  if (auto rmw = dyn_cast<LLVM::AtomicRMWOp>(op))
    return rmw.getVolatile_() || orders(rmw.getOrdering());
  if (auto cmpxchg = dyn_cast<LLVM::AtomicCmpXchgOp>(op))
    return cmpxchg.getVolatile_() || orders(cmpxchg.getSuccessOrdering()) ||
           orders(cmpxchg.getFailureOrdering());
  return false;
}

bool isPrivatePointer(Type t) {
  auto ptr = dyn_cast<LLVM::LLVMPointerType>(t);
  return ptr && ptr.getAddressSpace() == kPrivateAddressSpace;
}

// The pointer or buffer an access goes through, looking through GEPs and buffer resources, or
// null when unknown. A ROCDL buffer or global intrinsic takes its resource or address as its first
// pointer operand (after the data of a store or atomic).
Value getAccessRoot(Operation *op) {
  Value ptr;
  if (auto load = dyn_cast<LLVM::LoadOp>(op))
    ptr = load.getAddr();
  else if (auto store = dyn_cast<LLVM::StoreOp>(op))
    ptr = store.getAddr();
  else if (auto rmw = dyn_cast<LLVM::AtomicRMWOp>(op))
    ptr = rmw.getPtr();
  else if (auto cmpxchg = dyn_cast<LLVM::AtomicCmpXchgOp>(op))
    ptr = cmpxchg.getPtr();
  else if (op->getName().getStringRef().starts_with("rocdl."))
    for (Value operand : op->getOperands())
      if (isa<LLVM::LLVMPointerType>(operand.getType())) {
        ptr = operand;
        break;
      }
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

// Whether `v` has the same value in every iteration: defined outside the loop, or computed in it
// by pure ops from such values.
bool isLoopInvariant(Value v, scf::ForOp forOp, DenseMap<Value, bool> &memo) {
  Operation *def = v.getDefiningOp();
  if (!forOp->isAncestor(def ? def : cast<BlockArgument>(v).getOwner()->getParentOp()))
    return true;
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

// A load whose address is the same in every iteration, from a buffer the loop writes: unless every
// write misses that address, each iteration reads what an earlier one wrote (an accumulator kept in
// memory), and the pipelined load would read before that write.
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

// The load stage runs a top-level load one iteration group ahead of everything else, so a load
// that follows a store in the body would move ahead of that store, which may write the same address
// (an LDS round trip, or a global store and reload).
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
// Loop bounds
//===----------------------------------------------------------------------===//

// `v`, an index or an integer of at most 64 bits, as an i64.
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

Value narrowFromI64(RewriterBase &rewriter, Location loc, Value v, Type ty) {
  if (ty.isIndex())
    return arith::IndexCastOp::create(rewriter, loc, ty, v);
  if (ty.getIntOrFloatBitWidth() < 64)
    return arith::TruncIOp::create(rewriter, loc, ty, v);
  return v;
}

// The loop's constant step as the loop compares it: zero-extended for an unsigned loop (clamped to
// INT64_MAX when it doesn't fit), sign-extended otherwise.
std::optional<int64_t> getConstantStep(scf::ForOp forOp) {
  APInt step;
  if (!matchPattern(forOp.getStep(), m_ConstantInt(&step)))
    return std::nullopt;
  if (!forOp.getUnsignedCmp())
    return step.getSExtValue();
  if (step.getActiveBits() >= 64)
    return std::numeric_limits<int64_t>::max();
  return static_cast<int64_t>(step.getZExtValue());
}

// Keep the unroller's scaled step and the pipeliner's peeled bounds out of narrow integer types.
// The body still sees the original induction type, including its original wrapping arithmetic.
void widenLoopBounds(RewriterBase &rewriter, scf::ForOp forOp) {
  auto ty = dyn_cast<IntegerType>(forOp.getLowerBound().getType());
  if (!ty || ty.getWidth() >= 64)
    return;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  bool isUnsigned = forOp.getUnsignedCmp();
  forOp.setLowerBound(extendToI64(rewriter, loc, forOp.getLowerBound(), isUnsigned));
  forOp.setUpperBound(extendToI64(rewriter, loc, forOp.getUpperBound(), isUnsigned));
  forOp.setStep(extendToI64(rewriter, loc, forOp.getStep(), isUnsigned));
  // Zero-extended bounds are below 2^32, so a signed i64 comparison gives the same trip count, and
  // the unroller and the pipeliner only handle signed ones.
  forOp.setUnsignedCmp(false);
  Value iv = forOp.getInductionVar();
  iv.setType(rewriter.getI64Type());
  rewriter.setInsertionPointToStart(forOp.getBody());
  auto narrow = arith::TruncIOp::create(rewriter, loc, ty, iv);
  rewriter.replaceAllUsesExcept(iv, narrow.getResult(), narrow);
}

// Ends `forOp` after the wave's shortest trip count, rounded down to a multiple of `depth`, and
// runs the remaining iterations of each lane in a plain tail loop after it. Every lane then runs
// `forOp` the same number of times, and the tail runs in lockstep, so each iteration index sees the
// same active lanes as in the original loop. Costs one wave reduction before the loop.
void splitAtWaveUniformTripCount(RewriterBase &rewriter, scf::ForOp forOp, int64_t depth) {
  assert(!forOp.getUnsignedCmp() && "whyNotPrefetchable rejects unsigned loops that need a split");
  Location loc = forOp.getLoc();
  Type boundTy = forOp.getLowerBound().getType();
  OpBuilder::InsertionGuard g(rewriter);
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
  Value splitBound =
      narrowFromI64(rewriter, loc,
                    arith::AddIOp::create(rewriter, loc, lb,
                                          arith::MulIOp::create(rewriter, loc, mainTrips, step)),
                    boundTy);

  rewriter.setInsertionPointAfter(forOp);
  auto tail = cast<scf::ForOp>(rewriter.clone(*forOp));
  for (auto [from, to] : llvm::zip(forOp.getResults(), tail.getResults()))
    rewriter.replaceAllUsesExcept(from, to, tail);
  tail.setLowerBound(splitBound);
  tail.getInitArgsMutable().assign(forOp.getResults());
  forOp.setUpperBound(splitBound);
}

//===----------------------------------------------------------------------===//
// Pipeline schedule and guards
//===----------------------------------------------------------------------===//

// The load stage holds the loads and the pure address math feeding them, including the math that
// computes a loop-carried address for the next iteration (a pointer bumped after its load); the
// compute stage holds everything else.
void buildLoadComputeSchedule(scf::ForOp forOp, Schedule &schedule) {
  Block *body = forOp.getBody();
  llvm::SmallPtrSet<Operation *, 16> loadStage;
  SmallVector<Value> worklist;
  for (Operation &op : body->without_terminator()) {
    if (!isPrefetchableLoad(&op))
      continue;
    loadStage.insert(&op);
    llvm::append_range(worklist, op.getOperands());
  }
  // Visit each value once, following pure address math, region captures and loop-carried edges.
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
  for (Operation &op : body->without_terminator())
    schedule.push_back({&op, loadStage.contains(&op) ? kLoadStage : kComputeStage});
}

// A type `zeroValue` can build a placeholder for.
bool hasZeroValue(Type t) {
  if (auto vec = dyn_cast<VectorType>(t))
    t = vec.getElementType();
  return isa<IntegerType, IndexType, FloatType, LLVM::LLVMPointerType>(t);
}

Value zeroValue(OpBuilder &b, Location loc, Type t) {
  // getZeroAttr has no pointer attribute, for a pointer or a vector of them.
  if (isa<LLVM::LLVMPointerType>(getElementTypeOrSelf(t)))
    return LLVM::ZeroOp::create(b, loc, t);
  return arith::ConstantOp::create(b, loc, cast<TypedAttr>(b.getZeroAttr(t)));
}

// Whether predicateOp must guard `op` when it is cloned into the prologue or epilogue.
// Upstream marks floordivsi, srem and urem speculatable although a zero divisor is undefined
// behavior, so guard them explicitly.
bool needsPredication(Operation *op) {
  return isPrefetchableLoad(op) || !isMemoryEffectFree(op) || !isSpeculatable(op) ||
         isa<arith::FloorDivSIOp, LLVM::SRemOp, LLVM::URemOp>(op);
}

// An scf.if on `cond` with an empty then block, and an empty else block when it has results. The
// caller fills both and terminates them.
scf::IfOp createGuard(RewriterBase &rewriter, Location loc, TypeRange types, Value cond) {
  return scf::IfOp::create(rewriter, loc, types, cond, /*addThenBlock=*/true,
                           /*addElseBlock=*/!types.empty());
}

// Predicate an op cloned into the prologue/epilogue of a dynamic-trip loop.
//
// Reads and side-effecting ops (stores, atomics, an scf.if or scf.for that reads) are sunk into
// `scf.if(pred)` and yield placeholder zeros when the iteration is out of range: a read issued past
// the trip count can fault, buffer loads included, since a buffer resource is usually created
// without a size limit. Only safely speculatable ops run unchanged on the zeros. The zeros never
// reach the loop results: the pipeliner selects each epilogue live-out against the previous
// iteration's value under the same predicate. `whyBodyNotPipelineable` rejects a loop with an op
// this can't guard, so it never returns nullptr after the pipeliner has changed the IR.
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
  scf::IfOp guard = createGuard(rewriter, loc, op->getResultTypes(), pred);
  op->moveBefore(guard.thenBlock(), guard.thenBlock()->end());
  rewriter.setInsertionPointToEnd(guard.thenBlock());
  scf::YieldOp::create(rewriter, loc, op->getResults());
  if (op->getNumResults() == 0)
    return guard;
  rewriter.setInsertionPointToEnd(guard.elseBlock());
  SmallVector<Value> zeros;
  for (Type t : op->getResultTypes())
    zeros.push_back(zeroValue(rewriter, loc, t));
  scf::YieldOp::create(rewriter, loc, zeros);
  return guard;
}

// Merges `second` into `first`, an earlier guard in the same block with the same predicate, and
// returns the merged guard. Every op between them must be free of memory effects, so the guarded
// ops of `second` can move above them; the ones `second` uses move above `first`, which fails
// when they depend on its results (a load whose address another guarded load produced).
FailureOr<scf::IfOp> mergeGuard(RewriterBase &rewriter, scf::IfOp first, scf::IfOp second) {
  DenseSet<Operation *> between;
  for (Operation *op = first->getNextNode(); op != second; op = op->getNextNode()) {
    if (!isMemoryEffectFree(op))
      return failure();
    between.insert(op);
  }
  SetVector<Value> used;
  getUsedValuesDefinedAbove(second->getRegions(), used);
  SmallVector<Value> worklist(used.begin(), used.end());
  DenseSet<Operation *> needed;
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
  for (Operation *op = first->getNextNode(); op != second;) {
    Operation *next = op->getNextNode();
    if (needed.contains(op))
      rewriter.moveOpBefore(op, first);
    op = next;
  }

  SmallVector<Type> types(first->getResultTypes());
  llvm::append_range(types, second->getResultTypes());
  Location loc = first.getLoc();
  rewriter.setInsertionPoint(first);
  scf::IfOp merged = createGuard(rewriter, loc, types, first.getCondition());
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

// `predicateOp` guards each op on its own, so N loads in the prologue would become N branch
// regions, each saving and restoring the exec mask. Merge each guard into the previous one with
// the same predicate where `mergeGuard` can. Upstream's CombineIfs only merges adjacent ifs, and
// each guard is preceded by its own address math.
void mergeGuards(RewriterBase &rewriter, ArrayRef<scf::IfOp> guards) {
  scf::IfOp head;
  for (scf::IfOp guard : guards) {
    if (head && head->getBlock() == guard->getBlock() &&
        head.getCondition() == guard.getCondition() && head->isBeforeInBlock(guard))
      if (FailureOr<scf::IfOp> merged = mergeGuard(rewriter, head, guard); succeeded(merged)) {
        head = *merged;
        continue;
      }
    head = guard;
  }
}

//===----------------------------------------------------------------------===//
// Legality
//===----------------------------------------------------------------------===//

// Whether a load-stage op (a load or its address math) uses a value the compute stage computes, in
// the same iteration or through a loop-carried value: it would run before that value exists.
// Upstream rejects such a schedule with an error diagnostic, so check it before changing the loop.
bool loadDependsOnComputeStage(scf::ForOp forOp) {
  Schedule schedule;
  buildLoadComputeSchedule(forOp, schedule);
  DenseMap<Operation *, unsigned> stageOf(schedule.begin(), schedule.end());
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
    return def && def != body->getTerminator() && stageOf.lookup(def) == kComputeStage;
  };
  return llvm::any_of(schedule, [&](auto &entry) {
    if (entry.second != kLoadStage)
      return false;
    WalkResult result = entry.first->walk([&](Operation *nested) {
      for (Value operand : nested->getOperands())
        if (isComputedLate(operand))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return result.wasInterrupted();
  });
}

// Why the body of `forOp` can't be pipelined, or an empty string when it can. Mechanical whole-loop
// rotation cannot safely pipeline a loop that contains an intra-loop barrier or a single-buffered
// copy into LDS: rotating the copy across the backedge would clobber the shared buffer (no double
// buffering), and a dynamic-trip epilogue would have to predicate the barrier, which is illegal.
// Prints and asserts would run out of order, and a read after a write could move ahead of it.
//
// Loads move ahead of earlier iterations' writes, which is only correct when no iteration reads
// memory an earlier one wrote. Without `assumeIndependent` a loop that may write memory is left
// unchanged; with it, only a load that certainly reads such a write is rejected. Private memory
// may hold a register array the compiler could not keep in registers, which the user can't see, so
// it is rejected either way.
std::string whyBodyNotPipelineable(scf::ForOp forOp, bool assumeIndependent) {
  std::string reason;
  forOp.getBody()->walk([&](Operation *op) {
    if (isBarrier(op))
      reason = "it contains a barrier";
    else if (isCounterWait(op))
      reason = ("it waits on a memory counter (" + op->getName().getStringRef() +
                "), which would also wait for the prefetched loads")
                   .str();
    else if (isLdsOrAsyncCopy(op))
      reason = "it contains a copy into or out of LDS, or an asynchronous copy";
    else if (isPrintOrAssert(op))
      reason = "it contains a print or an assert";
    else if (isOrderedOrVolatile(op))
      reason = "it contains a fence, a volatile access or an atomic with acquire or stronger "
               "ordering";
    else if (llvm::any_of(op->getOperandTypes(), isPrivatePointer) ||
             llvm::any_of(op->getResultTypes(), isPrivatePointer))
      reason = "it accesses private memory, which an earlier iteration may have written";
    return reason.empty() ? WalkResult::advance() : WalkResult::interrupt();
  });
  if (!reason.empty())
    return reason;
  Block *body = forOp.getBody();
  if (llvm::none_of(body->without_terminator(),
                    [](Operation &op) { return isPrefetchableLoad(&op); }))
    return "its body has no load at the top level to issue early";
  if (!assumeIndependent) {
    Operation *write = nullptr;
    body->walk([&](Operation *op) {
      if (mayWriteMemory(op))
        write = op;
      return write ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (write)
      return ("it may write memory (" + write->getName().getStringRef() +
              ") and its loads would move ahead of earlier iterations' writes; set "
              "prefetch_assume_independent=True if no iteration reads memory an earlier one wrote")
          .str();
  }
  if (hasLoadAfterPossibleStore(forOp))
    return "a load follows a possible store in the same iteration and would move ahead of it";
  if (readsInvariantAddressOfWrittenBuffer(forOp))
    return "a load reads the same address in every iteration from a buffer the loop writes, so it "
           "would read before an earlier iteration's write";
  if (llvm::any_of(body->without_terminator(), [](Operation &op) {
        return needsPredication(&op) && !llvm::all_of(op.getResultTypes(), hasZeroValue);
      }))
    return "a load or side-effecting op returns a type the out-of-range iterations can't stand in "
           "for (only integers, floats, vectors, index and pointers)";
  if (loadDependsOnComputeStage(forOp))
    return "a load's address depends on a value the loop computes after its loads (in the same "
           "iteration or carried from the previous one)";
  return {};
}

// Where a remark about `forOp` goes: the location the frontend recorded on the loop, or the loop's
// own location. Without debug info the loop's own location points into FlyDSL's re-parsed IR text
// rather than the user's source, but a location held in an attribute survives the round trip.
Location getRemarkLoc(scf::ForOp forOp) {
  if (auto recorded = forOp->getAttrOfType<LocationAttr>(kSourceLocAttr))
    return recorded;
  return forOp.getLoc();
}

// The name of the function (the kernel) containing `op`, or an empty string.
std::string getEnclosingSymbolName(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent; parent = parent->getParentOp())
    if (auto name = parent->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      return name.str();
  return "";
}

// A loop marked by `range(..., prefetch_depth=N)`, with its markers already removed.
struct PrefetchRequest {
  scf::ForOp loop;
  int64_t depth;
  bool assumeIndependent;
  Location remarkLoc;
  std::string kernelName;

  // The loop was marked explicitly, so say why it stays as it is, and where.
  void remarkNotPipelined(const Twine &why) const {
    InFlightDiagnostic remark = mlir::emitRemark(remarkLoc);
    remark << "loop not pipelined (prefetch_depth=" << depth << "): " << why;
    if (!kernelName.empty())
      remark << " (in " << kernelName << ")";
  }
};

// Reads the loop's markers and removes them, so no marked loop keeps them whatever happens next.
std::optional<PrefetchRequest> takePrefetchRequest(scf::ForOp forOp) {
  auto depth = forOp->getAttrOfType<IntegerAttr>(kPrefetchDepthAttr);
  if (!depth)
    return std::nullopt;
  PrefetchRequest request{forOp, depth.getInt(), forOp->hasAttr(kAssumeIndependentAttr),
                          getRemarkLoc(forOp), getEnclosingSymbolName(forOp)};
  forOp->removeAttr(kPrefetchDepthAttr);
  forOp->removeAttr(kAssumeIndependentAttr);
  forOp->removeAttr(kSourceLocAttr);
  return request;
}

// Unrolling and peeling split each lane's iterations at its own trip count; a lane-exchange op
// needs that split to fall at the same iteration in every lane of the wave.
bool needsWaveUniformSplit(scf::ForOp forOp) {
  return hasLaneExchangeOp(forOp) && !hasWaveUniformTripCount(forOp);
}

// Whether the unroller's step, the step times `depth`, may overflow a signed 64-bit integer. Narrow
// bounds are widened first, so a narrow runtime step is bounded by its type; a runtime 64-bit or
// index step is assumed to stay below 2^63 / depth.
bool mayOverflowUnrolledStep(scf::ForOp forOp, int64_t depth) {
  std::optional<int64_t> maxStep = getConstantStep(forOp);
  auto ty = dyn_cast<IntegerType>(forOp.getStep().getType());
  if (!maxStep && ty && ty.getWidth() < 64)
    maxStep = forOp.getUnsignedCmp() ? APInt::getMaxValue(ty.getWidth()).getZExtValue()
                                     : APInt::getSignedMaxValue(ty.getWidth()).getSExtValue();
  return maxStep && depth > std::numeric_limits<int64_t>::max() / *maxStep;
}

// Why the request can't be carried out, or an empty string when it can. Checked before anything
// changes the loop, so a rejected loop is left exactly as it was.
std::string whyNotPrefetchable(const PrefetchRequest &request) {
  scf::ForOp forOp = request.loop;
  if (request.depth < 1 || request.depth > kMaxDepth)
    return "the depth must be between 1 and " + std::to_string(kMaxDepth);
  if (std::string why = whyBodyNotPipelineable(forOp, request.assumeIndependent); !why.empty())
    return why;
  // A Python range that counts down (range(n, 0, -1)) lowers to such a step.
  if (std::optional<int64_t> step = getConstantStep(forOp); step && *step <= 0)
    return "its step is not positive, which scf.for does not allow";
  if (request.depth > 1 && mayOverflowUnrolledStep(forOp, request.depth))
    return "its step multiplied by the depth may overflow a signed 64-bit integer";
  if (!forOp.getUnsignedCmp())
    return {};
  if (needsWaveUniformSplit(forOp))
    return "it exchanges values across lanes and compares its bounds unsigned, so its trip count "
           "can't be split the same way in every lane";
  // widenLoopBounds makes a narrow unsigned loop signed; a 64-bit or index one stays unsigned.
  auto boundTy = dyn_cast<IntegerType>(forOp.getLowerBound().getType());
  if (!boundTy || boundTy.getWidth() >= 64)
    return "it compares 64-bit bounds unsigned, which the pipeliner does not support";
  return {};
}

//===----------------------------------------------------------------------===//
// Transformation
//===----------------------------------------------------------------------===//

// Both unrolling and peeling assume a nonnegative span. Clamp even at depth one, where subtracting
// a step from an empty range's original upper bound could underflow. createOrFold keeps a constant
// trip count constant, so its loop needs no predication.
void clampUpperBoundToLowerBound(RewriterBase &rewriter, scf::ForOp forOp) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forOp);
  forOp.setUpperBound(rewriter.createOrFold<arith::MaxSIOp>(forOp.getLoc(), forOp.getLowerBound(),
                                                            forOp.getUpperBound()));
}

// Runs the pipeliner with the load/compute schedule and merges the guards it creates. Returns
// failure only when the pipeliner failed after changing the IR; a clean rejection is a remark.
LogicalResult pipelineLoadsAhead(RewriterBase &rewriter, scf::ForOp loop,
                                 const PrefetchRequest &request, bool unrolled) {
  scf::PipeliningOption options;
  options.supportDynamicLoops = true;
  // peelEpilogue (default true) must stay true: folding the epilogue into the kernel would
  // predicate every steady-state load, forcing a wait right after each one.
  options.getScheduleFn = buildLoadComputeSchedule;
  SmallVector<scf::IfOp> guards;
  options.predicateFn = [&](RewriterBase &r, Operation *op, Value pred) {
    Operation *guarded = predicateOp(r, op, pred);
    if (guarded && guarded != op)
      guards.push_back(cast<scf::IfOp>(guarded));
    return guarded;
  };

  // pipelineForLoop emits the prologue at the rewriter's insertion point.
  rewriter.setInsertionPoint(loop);
  // whyBodyNotPipelineable rejects the loops the pipeliner can't handle, so it should not fail
  // here. If it does, keep its diagnostic away from callers: an error the pass then ignores aborts
  // a Python PassManager.run, which asserts that no captured error goes unreported.
  std::string pipelinerMessage;
  bool changedIR = false;
  LogicalResult pipelined = failure();
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
  request.remarkNotPipelined(why);
  return success();
}

// Unrolls the loop by the depth and pipelines the main loop. The request has passed
// whyNotPrefetchable.
LogicalResult prefetchLoop(RewriterBase &rewriter, const PrefetchRequest &request) {
  scf::ForOp forOp = request.loop;
  bool splitAtUniformTrips = needsWaveUniformSplit(forOp);
  if (splitAtUniformTrips)
    splitAtWaveUniformTripCount(rewriter, forOp, request.depth);
  widenLoopBounds(rewriter, forOp);
  clampUpperBoundToLowerBound(rewriter, forOp);
  if (request.depth == 1)
    return pipelineLoadsAhead(rewriter, forOp, request, /*unrolled=*/false);

  // The remainder loop stays serial. Keep a one-iteration main loop alive for the pipeliner.
  FailureOr<UnrolledLoopInfo> info = loopUnrollByFactor(
      forOp, request.depth, /*annotateFn=*/nullptr, /*shouldPromoteIfSingleIteration=*/false);
  if (failed(info) || !info->mainLoopOp)
    return pipelineLoadsAhead(rewriter, forOp, request, /*unrolled=*/false);
  // The split already ends the main loop at a multiple of the depth, so the unroller's remainder
  // loop never runs; the plain tail loop after it runs the rest.
  if (splitAtUniformTrips && info->epilogueLoopOp)
    rewriter.replaceOp(*info->epilogueLoopOp, info->epilogueLoopOp->getInitArgs());
  return pipelineLoadsAhead(rewriter, *info->mainLoopOp, request, /*unrolled=*/true);
}

class FlyROCDLPrefetchLoopsPass
    : public mlir::fly_rocdl::impl::FlyROCDLPrefetchLoopsPassBase<FlyROCDLPrefetchLoopsPass> {
public:
  using mlir::fly_rocdl::impl::FlyROCDLPrefetchLoopsPassBase<
      FlyROCDLPrefetchLoopsPass>::FlyROCDLPrefetchLoopsPassBase;

  void runOnOperation() override {
    // Post-order, so an inner marked loop is rewritten before an outer one's unrolling clones it.
    SmallVector<PrefetchRequest> requests;
    getOperation()->walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
      if (std::optional<PrefetchRequest> request = takePrefetchRequest(forOp))
        requests.push_back(std::move(*request));
    });

    IRRewriter rewriter(&getContext());
    for (const PrefetchRequest &request : requests) {
      if (std::string why = whyNotPrefetchable(request); !why.empty()) {
        request.remarkNotPipelined(why);
        continue;
      }
      if (failed(prefetchLoop(rewriter, request)))
        return signalPassFailure();
    }
  }
};

} // namespace
