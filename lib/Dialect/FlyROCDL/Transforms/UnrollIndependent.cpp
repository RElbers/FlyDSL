// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Unroll scf.for loops marked by `range(..., unroll=N, assume_independent=True)` by N and issue
// each group's loads before the group's other operations. Run inside gpu.module() after
// convert-fly-to-rocdl.
//
// LLVM cannot move a load above an earlier iteration's store unless it proves that they access
// different memory, so its unroll of a loop that stores still waits for each load in turn. The
// frontend's fly.assume_independent asserts that no iteration reads memory any thread wrote in an
// earlier iteration, which makes the reordering legal. Nothing runs speculatively: the unrolled
// loop runs whole groups only, and a serial remainder loop runs the rest.
//
// A rejected loop keeps the frontend's LLVM unroll hint, so it behaves as with plain unroll=N, and
// gets a remark saying why.
//
// pipeline="lds" (fly.pipeline_lds) carries the pipelined loop's next group in an LDS ring instead
// of registers: see "LDS staging" below.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h"

#include <limits>
#include <optional>
#include <string>

using namespace mlir;

namespace mlir {
namespace fly_rocdl {
#define GEN_PASS_DEF_FLYROCDLUNROLLINDEPENDENTPASS
#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h.inc"
} // namespace fly_rocdl
} // namespace mlir

namespace {

/// Set by `assume_independent=True`: no iteration reads memory any thread wrote in an earlier one.
constexpr StringLiteral kAssumeIndependentAttr = "fly.assume_independent";
/// Set by `pipeline=True`: also issue the next group's loads before the current group's other ops.
constexpr StringLiteral kPipelineAttr = "fly.pipeline";
/// Set with fly.pipeline by `pipeline="lds"`: carry the next group's loads in LDS, not registers.
constexpr StringLiteral kPipelineLdsAttr = "fly.pipeline_lds";
/// The frontend's LLVM unroll hint, which carries N.
constexpr StringLiteral kLoopAnnotationAttr = "loop_annotation";

/// Larger factors grow the IR and the live registers without hiding more latency.
constexpr int64_t kMaxFactor = 32;

constexpr unsigned kSharedAddressSpace = 3;
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

/// `fx.rocdl.s_buffer_load` emits the intrinsic as a call, which MLIR assumes writes memory. It
/// reads through the scalar cache, which vector stores do not update: a load that reads what an
/// earlier iteration stored could return stale data even in program order. assume_independent
/// rules that out, so moving it above an earlier iteration's stores is covered as for other loads.
bool isScalarBufferLoadCall(Operation *op) {
  auto intrinsic = dyn_cast<LLVM::CallIntrinsicOp>(op);
  return intrinsic && intrinsic.getIntrin().starts_with("llvm.amdgcn.s.buffer.load");
}

/// Atomic and volatile loads must retain their ordering.
bool isMovableLoad(Operation *op) {
  if (isScalarBufferLoadCall(op))
    return true;
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

/// A wait in an earlier iteration would also wait for the loads moved above it.
bool isCounterWait(Operation *op) { return hasAmdgpuName(op, {"s.waitcnt", "s.wait."}); }

/// Assembly with side effects can be a barrier, a wait, a fence or a cache invalidation, which the
/// op-name checks cannot see. Assembly without them is no more movable for us than for LLVM.
bool isSideEffectingAsm(Operation *op) {
  auto inlineAsm = dyn_cast<LLVM::InlineAsmOp>(op);
  return inlineAsm && inlineAsm.getHasSideEffects();
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
/// With lane-dependent trip counts, the unrolled loop and its remainder each run an iteration
/// index with a different set of active lanes than the original loop did.
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

//===----------------------------------------------------------------------===//
// Wave uniformity
//===----------------------------------------------------------------------===//

/// Conservative, demand-driven uniformity analysis.
class WaveUniformityAnalysis {
public:
  bool hasUniformTripCount(scf::ForOp forOp);
  bool isUniform(Value value);

private:
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

//===----------------------------------------------------------------------===//
// Memory dependences
//===----------------------------------------------------------------------===//

/// Check only leaves; callers walk nested regions separately.
bool mayWriteMemory(Operation *op) {
  return op->getNumRegions() == 0 && !isMovableLoad(op) && !isMemoryEffectFree(op) &&
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
      .Default([](Operation *op) {
        // Bit 31 of a buffer access's integer aux operand, or of a scalar buffer load's cache
        // policy operand, marks it volatile.
        APInt policy;
        if (isScalarBufferLoadCall(op))
          return !matchPattern(op->getOperand(2), m_ConstantInt(&policy)) ||
                 policy.getZExtValue() & (1u << 31);
        auto aux = op->getAttrOfType<IntegerAttr>("aux");
        return aux && op->getName().getStringRef().contains("buffer.") &&
               aux.getValue().getZExtValue() & (1u << 31);
      });
}

bool isPrivatePointer(Type t) {
  auto ptr = dyn_cast<LLVM::LLVMPointerType>(t);
  return ptr && ptr.getAddressSpace() == kPrivateAddressSpace;
}

/// Return the access root, or null when unknown. ROCDL resources and addresses are the first
/// pointer operand, after any store or atomic data; a scalar buffer load's resource is its first
/// operand, a pointer cast to a vector of i32.
Value getAccessRoot(Operation *op) {
  Value ptr = llvm::TypeSwitch<Operation *, Value>(op)
                  .Case<LLVM::LoadOp, LLVM::StoreOp>([](auto access) { return access.getAddr(); })
                  .Case<LLVM::CallIntrinsicOp>([](LLVM::CallIntrinsicOp call) -> Value {
                    return isScalarBufferLoadCall(call) ? call->getOperand(0) : Value();
                  })
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
    else if (isa_and_nonnull<LLVM::BitcastOp, LLVM::PtrToIntOp>(def))
      ptr = def->getOperand(0); // A scalar buffer load's resource, as a vector of i32.
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

/// An invariant-address load from a buffer the loop writes reads an earlier iteration's write
/// unless the write goes elsewhere, which the assertion cannot be trusted to cover.
bool readsInvariantAddressOfWrittenBuffer(scf::ForOp forOp) {
  DenseSet<Value> written;
  forOp.getBody()->walk([&](Operation *op) {
    if (mayWriteMemory(op))
      if (Value root = getAccessRoot(op))
        written.insert(root);
  });
  DenseMap<Value, bool> memo;
  return llvm::any_of(forOp.getBody()->without_terminator(), [&](Operation &op) {
    if (!isMovableLoad(&op))
      return false;
    Value root = getAccessRoot(&op);
    return root && written.contains(root) && llvm::all_of(op.getOperands(), [&](Value operand) {
             return isLoopInvariant(operand, forOp, memo);
           });
  });
}

/// A top-level load after a possible store would move above that store in its own iteration,
/// which the assertion does not cover.
bool hasLoadAfterPossibleStore(scf::ForOp forOp) {
  bool sawWrite = false;
  WalkResult result = forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (sawWrite && isMovableLoad(op) && op->getBlock() == forOp.getBody())
      return WalkResult::interrupt();
    sawWrite |= mayWriteMemory(op);
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

//===----------------------------------------------------------------------===//
// Load slices
//===----------------------------------------------------------------------===//

using OpSet = llvm::SmallPtrSet<Operation *, 32>;

/// The top-level loads among `ops` and the pure ops among `ops` that compute their operands. With
/// `loop`, a carried operand is followed to the value the loop yields for it, which the previous
/// iteration computed.
OpSet collectLoadSlice(ArrayRef<Operation *> ops, scf::ForOp loop) {
  OpSet candidates(ops.begin(), ops.end());
  OpSet slice;
  SmallVector<Value> worklist;
  for (Operation *op : ops) {
    if (!isMovableLoad(op))
      continue;
    slice.insert(op);
    llvm::append_range(worklist, op->getOperands());
  }
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    if (auto arg = dyn_cast<BlockArgument>(v); loop && arg && arg.getOwner() == loop.getBody()) {
      if (OpOperand *yielded = loop.getTiedLoopYieldedValue(arg))
        worklist.push_back(yielded->get());
      continue;
    }
    Operation *def = v.getDefiningOp();
    if (!def || !candidates.contains(def) || !isMemoryEffectFree(def) || !slice.insert(def).second)
      continue;
    llvm::append_range(worklist, def->getOperands());
    SetVector<Value> captures;
    getUsedValuesDefinedAbove(def->getRegions(), captures);
    llvm::append_range(worklist, captures);
  }
  return slice;
}

/// Whether an op of the load slice uses a value computed by an op outside it, in the same iteration
/// or carried from the previous one. Such a slice cannot move above the rest of the body.
bool sliceDependsOnRestOfBody(scf::ForOp forOp, const OpSet &slice) {
  Block *body = forOp.getBody();
  auto isComputedOutsideSlice = [&](Value v) {
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
    return def && def != body->getTerminator() && !slice.contains(def);
  };
  return llvm::any_of(body->without_terminator(), [&](Operation &op) {
    if (!slice.contains(&op))
      return false;
    WalkResult result = op.walk([&](Operation *nested) {
      return llvm::any_of(nested->getOperands(), isComputedOutsideSlice) ? WalkResult::interrupt()
                                                                         : WalkResult::advance();
    });
    return result.wasInterrupted();
  });
}

/// Move the load slice of `body` above its first op outside the slice, keeping the order of the
/// moved ops.
void hoistLoadSlice(RewriterBase &rewriter, Block *body) {
  SmallVector<Operation *> ops;
  for (Operation &op : body->without_terminator())
    ops.push_back(&op);
  OpSet slice = collectLoadSlice(ops, /*loop=*/nullptr);
  Operation *firstOther = nullptr;
  for (Operation *op : ops) {
    if (!slice.contains(op))
      firstOther = firstOther ? firstOther : op;
    else if (firstOther)
      rewriter.moveOpBefore(op, firstOther);
  }
}

//===----------------------------------------------------------------------===//
// Requests and legality
//===----------------------------------------------------------------------===//

/// A loop marked by `range(..., unroll=N, assume_independent=True)`, with its marker removed.
struct UnrollRequest {
  scf::ForOp loop;
  int64_t factor;
  bool pipeline;
  bool lds;
  std::string kernelName;

  /// An active InFlightDiagnostic converts to failure in legality checks. The pass treats these
  /// remarks as clean rejections, not pass failures.
  InFlightDiagnostic emitRemark(const Twine &why) const {
    InFlightDiagnostic remark = mlir::emitRemark(loop->getLoc());
    remark << "assume_independent ignored (unroll=" << factor
           << "), the loop is left to LLVM: " << why;
    if (!kernelName.empty())
      remark << " (in " << kernelName << ")";
    return remark;
  }
};

/// N is the count of the LLVM unroll hint, or 0 when the loop has none.
int64_t getUnrollCount(scf::ForOp forOp) {
  auto annotation = forOp->getAttrOfType<LLVM::LoopAnnotationAttr>(kLoopAnnotationAttr);
  if (!annotation || !annotation.getUnroll() || !annotation.getUnroll().getCount())
    return 0;
  return annotation.getUnroll().getCount().getInt();
}

/// Reads the markers and removes them, so no loop keeps them whatever happens next.
std::optional<UnrollRequest> takeUnrollRequest(scf::ForOp forOp) {
  bool lds = static_cast<bool>(forOp->removeAttr(kPipelineLdsAttr));
  bool pipeline = static_cast<bool>(forOp->removeAttr(kPipelineAttr)) || lds;
  if (!forOp->removeAttr(kAssumeIndependentAttr))
    return std::nullopt;
  UnrollRequest request{forOp, getUnrollCount(forOp), pipeline, lds, {}};
  if (auto symbol = forOp->getParentOfType<SymbolOpInterface>())
    request.kernelName = symbol.getName().str();
  return request;
}

/// A narrow loop's bounds are computed in i64, where the step times N cannot overflow. A runtime
/// 64-bit or index step is assumed to stay below 2^63 / N.
bool mayOverflowUnrolledStep(scf::ForOp forOp, int64_t factor) {
  std::optional<APInt> step = forOp.getConstantStep();
  return step && step->getBitWidth() >= 64 &&
         step->getSExtValue() > std::numeric_limits<int64_t>::max() / factor;
}

/// Hazards anywhere in the body, including nested regions. Private memory may hold
/// compiler-generated register arrays, so the user's assertion cannot cover its accesses.
LogicalResult checkHazards(Operation *op, const UnrollRequest &request) {
  if (isBarrier(op))
    return request.emitRemark("it contains a barrier");
  if (isSideEffectingAsm(op))
    return request.emitRemark("it contains inline assembly with side effects (\"" +
                              cast<LLVM::InlineAsmOp>(op).getAsmString() +
                              "\"), which may be a barrier, a wait or a fence");
  if (isCounterWait(op))
    return request.emitRemark("it waits on a memory counter (" + op->getName().getStringRef() +
                              "), which would also wait for the loads moved above it");
  if (isLdsOrAsyncCopy(op))
    return request.emitRemark("it contains a copy into or out of LDS, or an asynchronous copy");
  if (isPrintOrAssert(op))
    return request.emitRemark("it contains a print or an assert");
  if (isOrderedOrVolatile(op))
    return request.emitRemark(
        "it contains a fence, a volatile access or an atomic with acquire or stronger ordering");
  if (llvm::any_of(op->getOperandTypes(), isPrivatePointer) ||
      llvm::any_of(op->getResultTypes(), isPrivatePointer))
    return request.emitRemark("it accesses private memory");
  return success();
}

/// Check the loop before changing it. Keep the order of checks so a loop with several problems
/// keeps reporting the same one.
LogicalResult checkLoop(const UnrollRequest &request) {
  scf::ForOp forOp = request.loop;
  Block *body = forOp.getBody();
  // With LDS staging, the groups of one iteration still get a pipeline; in registers they would
  // only add copies.
  int64_t minFactor = request.lds ? 1 : 2;
  if (request.factor < minFactor || request.factor > kMaxFactor)
    return request.emitRemark("it needs an unroll count from " + Twine(minFactor) + " to " +
                              Twine(kMaxFactor));
  WalkResult hazards =
      body->walk([&](Operation *op) -> WalkResult { return checkHazards(op, request); });
  if (hazards.wasInterrupted())
    return failure();

  SmallVector<Operation *> ops;
  for (Operation &op : body->without_terminator())
    ops.push_back(&op);
  OpSet slice = collectLoadSlice(ops, forOp);
  if (slice.empty())
    return request.emitRemark("its body has no load at the top level");
  if (hasLoadAfterPossibleStore(forOp))
    return request.emitRemark(
        "a load follows a possible store in the same iteration and would move above it");
  if (readsInvariantAddressOfWrittenBuffer(forOp))
    return request.emitRemark("a load reads the same address in every iteration from a buffer "
                              "the loop writes");
  if (sliceDependsOnRestOfBody(forOp, slice))
    return request.emitRemark("a load's address depends on a value the loop computes after its "
                              "loads (in the same iteration or carried from the previous one)");

  // The frontend never emits an unsigned loop, so the bounds arithmetic below is signed only.
  if (forOp.getUnsignedCmp())
    return request.emitRemark("it compares its bounds unsigned");
  if (std::optional<APInt> step = forOp.getConstantStep(); step && !step->isStrictlyPositive())
    return request.emitRemark("its step is not positive, which scf.for does not allow");
  if (mayOverflowUnrolledStep(forOp, request.factor))
    return request.emitRemark("its step times N may overflow a signed 64-bit integer");
  WalkResult exchange = body->walk([](Operation *op) {
    return isLaneLocal(op) ? WalkResult::advance() : WalkResult::interrupt();
  });
  if (exchange.wasInterrupted() && !WaveUniformityAnalysis().hasUniformTripCount(forOp))
    return request.emitRemark("it exchanges values across lanes (or calls an unknown op) and its "
                              "bounds are not proven wave-uniform, so the unrolled loop and its "
                              "remainder would run an iteration with other lanes active");
  return success();
}

//===----------------------------------------------------------------------===//
// Transformation
//===----------------------------------------------------------------------===//

/// The main loop's upper bound and step: the end of the last whole group of N iterations, and the
/// step times N. A narrow loop computes them in i64, where they cannot wrap, and keeps its own
/// type, whose arithmetic must not wrap either: if they do not fit, the main loop runs no
/// iteration and the remainder loop runs them all. Folds when the bounds are constant.
/// The pipeliner computes the main loop's span plus one step in the loop's type, so a pipelined
/// loop also needs that to fit.
std::pair<Value, Value> buildMainLoopBounds(RewriterBase &rewriter, scf::ForOp forOp,
                                            int64_t factor, bool pipeline) {
  Location loc = forOp.getLoc();
  auto narrowTy = dyn_cast<IntegerType>(forOp.getLowerBound().getType());
  if (narrowTy && narrowTy.getWidth() >= 64)
    narrowTy = nullptr;
  Type wideTy = narrowTy ? rewriter.getI64Type() : forOp.getLowerBound().getType();
  auto widen = [&](Value v) {
    return narrowTy ? rewriter.createOrFold<arith::ExtSIOp>(loc, wideTy, v) : v;
  };
  auto constant = [&](int64_t c) -> Value {
    return arith::ConstantOp::create(rewriter, loc, rewriter.getIntegerAttr(wideTy, c));
  };
  Value lb = widen(forOp.getLowerBound());
  Value ub = widen(forOp.getUpperBound());
  Value step = widen(forOp.getStep());
  Value span = rewriter.createOrFold<arith::MaxSIOp>(
      loc, rewriter.createOrFold<arith::SubIOp>(loc, ub, lb), constant(0));
  Value trips = rewriter.createOrFold<arith::DivUIOp>(
      loc,
      rewriter.createOrFold<arith::AddIOp>(
          loc, span, rewriter.createOrFold<arith::SubIOp>(loc, step, constant(1))),
      step);
  Value groups = rewriter.createOrFold<arith::DivUIOp>(loc, trips, constant(factor));
  Value mainStep = rewriter.createOrFold<arith::MulIOp>(loc, step, constant(factor));
  Value mainUb = rewriter.createOrFold<arith::AddIOp>(
      loc, lb, rewriter.createOrFold<arith::MulIOp>(loc, groups, mainStep));
  if (!narrowTy)
    return {mainUb, mainStep};

  Value max = constant(APInt::getSignedMaxValue(narrowTy.getWidth()).getSExtValue());
  auto fitsNarrow = [&](Value v) {
    return rewriter.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, v, max);
  };
  auto narrowOr = [&](Value fits, Value wide, Value fallback) {
    Value narrow = rewriter.createOrFold<arith::TruncIOp>(loc, narrowTy, wide);
    return rewriter.createOrFold<arith::SelectOp>(loc, fits, narrow, fallback);
  };
  // Separate conditions, so a constant step stays constant.
  Value stepFits = fitsNarrow(mainStep);
  Value fits = rewriter.createOrFold<arith::AndIOp>(loc, stepFits, fitsNarrow(mainUb));
  if (pipeline) {
    Value spanAndStep = rewriter.createOrFold<arith::AddIOp>(
        loc, rewriter.createOrFold<arith::SubIOp>(loc, mainUb, lb), mainStep);
    fits = rewriter.createOrFold<arith::AndIOp>(loc, fits, fitsNarrow(spanAndStep));
  }
  return {narrowOr(fits, mainUb, forOp.getLowerBound()),
          narrowOr(stepFits, mainStep, forOp.getStep())};
}

/// The unrolled loop and its remainder must not be unrolled again by LLVM: the user asked for N.
void disableLlvmUnroll(scf::ForOp forOp) {
  MLIRContext *ctx = forOp.getContext();
  auto old = forOp->getAttrOfType<LLVM::LoopAnnotationAttr>(kLoopAnnotationAttr);
  auto unroll =
      LLVM::LoopUnrollAttr::get(ctx, /*disable=*/BoolAttr::get(ctx, true), {}, {}, {}, {}, {}, {});
  if (!old) {
    forOp->setAttr(kLoopAnnotationAttr,
                   LLVM::LoopAnnotationAttr::get(ctx, {}, {}, {}, unroll, {}, {}, {}, {}, {}, {},
                                                 {}, {}, {}, {}, {}));
    return;
  }
  auto annotation = LLVM::LoopAnnotationAttr::get(
      ctx, old.getDisableNonforced(), old.getVectorize(), old.getInterleave(), unroll,
      old.getUnrollAndJam(), old.getLicm(), old.getDistribute(), old.getPipeline(), old.getPeeled(),
      old.getUnswitch(), old.getMustProgress(), old.getIsVectorized(), old.getStartLoc(),
      old.getEndLoc(), old.getParallelAccesses());
  forOp->setAttr(kLoopAnnotationAttr, annotation);
}

/// Unroll by N with a serial remainder loop, then hoist each group's loads. The main loop runs only
/// whole groups, so every hoisted load also ran in the original loop.
void unrollAndHoistLoads(RewriterBase &rewriter, scf::ForOp forOp, int64_t factor, bool pipeline) {
  Location loc = forOp.getLoc();
  Value step = forOp.getStep();
  rewriter.setInsertionPoint(forOp);
  auto [mainUb, mainStep] = buildMainLoopBounds(rewriter, forOp, factor, pipeline);

  // The remainder loop runs the original loop from where the main loop stops.
  rewriter.setInsertionPointAfter(forOp);
  auto remainder = cast<scf::ForOp>(rewriter.clone(*forOp));
  rewriter.replaceAllUsesWith(forOp.getResults(), remainder.getResults());
  rewriter.modifyOpInPlace(remainder, [&] {
    remainder.setLowerBound(mainUb);
    remainder.getInitArgsMutable().assign(forOp.getResults());
  });
  rewriter.modifyOpInPlace(forOp, [&] {
    forOp.setUpperBound(mainUb);
    forOp.setStep(mainStep);
  });

  Block *body = forOp.getBody();
  if (factor > 1)
    generateUnrolledLoop(
        body, forOp.getInductionVar(), factor,
        [&](unsigned i, Value iv, OpBuilder b) -> Value {
          Value offset = arith::MulIOp::create(
              b, loc, step, arith::ConstantOp::create(b, loc, b.getIntegerAttr(iv.getType(), i)));
          return arith::AddIOp::create(b, loc, iv, offset);
        },
        /*annotateFn=*/nullptr, forOp.getRegionIterArgs(), body->getTerminator()->getOperands());
  disableLlvmUnroll(forOp);
  disableLlvmUnroll(remainder);
  hoistLoadSlice(rewriter, body);
}

//===----------------------------------------------------------------------===//
// LDS staging
//===----------------------------------------------------------------------===//

// With pipeline="lds", each staged buffer load of the grouped loop becomes an asynchronous DMA into
// a two-slot LDS ring, and its value is read back from LDS right before its first use. The
// pipeliner then issues the next group's DMAs in place of its register loads, so no register holds
// a value in flight across the back edge. The ring is wave-private: a DMA lane writes its value at
// the wave's base plus its lane times the lane stride, and the same lane reads it back, so no
// barrier is needed.

constexpr int64_t kWaveSize = 64;
constexpr int64_t kRingSlots = 2;
constexpr int64_t kGfx950LdsBytes = 160 * 1024;

/// The ops the pipeline schedule places explicitly after staging.
struct LdsStaging {
  scf::ForOp loop;
  DenseSet<Operation *> readbacks;
  Operation *mark = nullptr;
};

/// The byte size of a scalar or 1-D vector of integers or floats.
std::optional<int64_t> getByteSize(Type type) {
  int64_t count = 1;
  if (auto vec = dyn_cast<VectorType>(type)) {
    if (vec.getRank() != 1 || vec.isScalable())
      return std::nullopt;
    count = vec.getNumElements();
    type = vec.getElementType();
  }
  if (!type.isIntOrFloat() || type.getIntOrFloatBitWidth() % 8)
    return std::nullopt;
  return count * type.getIntOrFloatBitWidth() / 8;
}

/// Why `load` stays in registers, or an empty string when it can be staged.
StringRef whyNotStaged(Operation *load, const OpSet &slice, Block *body,
                       WaveUniformityAnalysis &uniformity) {
  std::optional<int64_t> bytes = getByteSize(load->getResult(0).getType());
  if (!bytes || (*bytes != 4 && *bytes != 12 && *bytes != 16))
    return "the DMA moves only 4, 12 or 16 bytes per lane";
  if (!isa_and_nonnull<IntegerAttr>(load->getAttr("aux")))
    return "its cache policy is not a plain integer";
  if (!uniformity.isUniform(load->getOperand(0)))
    return "its buffer resource is not proven wave-uniform";
  if (load->use_empty())
    return "its value is unused";
  for (Operation *user : load->getUsers()) {
    Operation *ancestor = body->findAncestorOpInBlock(*user);
    if (slice.contains(ancestor))
      return "its value feeds another load's address";
    if (ancestor == body->getTerminator())
      return "its value is carried to the next iteration unchanged";
  }
  return "";
}

/// The chip of the module's ROCDL target, or empty.
StringRef getChip(Operation *op) {
  auto module = op->getParentOfType<gpu::GPUModuleOp>();
  ArrayAttr targets = module ? module.getTargetsAttr() : ArrayAttr();
  for (Attribute target : targets ? targets.getValue() : ArrayRef<Attribute>())
    if (auto rocdl = dyn_cast<ROCDL::ROCDLTargetAttr>(target))
      return rocdl.getChip();
  return "";
}

/// The bytes of the module's static LDS globals, which LLVM lays out together for each kernel.
int64_t getStaticLdsBytes(gpu::GPUModuleOp module) {
  int64_t bytes = 0;
  for (auto global : module.getBody()->getOps<LLVM::GlobalOp>())
    if (global.getAddrSpace() == kSharedAddressSpace)
      bytes += DataLayout::closest(global).getTypeSize(global.getGlobalType());
  return bytes;
}

/// The same kind of global that a static SharedAllocator leaf lowers to.
LLVM::GlobalOp createRing(RewriterBase &rewriter, gpu::GPUModuleOp module, Location loc,
                          int64_t bytes) {
  SymbolTable symbols(module);
  unsigned counter = 0;
  SmallString<32> name = SymbolTable::generateSymbolName<32>(
      "__fly_pipeline_ring", [&](StringRef candidate) { return symbols.lookup(candidate); },
      counter);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto type = LLVM::LLVMArrayType::get(rewriter.getI8Type(), bytes);
  auto ring = LLVM::GlobalOp::create(rewriter, loc, type, /*isConstant=*/false,
                                     LLVM::Linkage::External, name, /*value=*/Attribute(),
                                     /*alignment=*/16, kSharedAddressSpace);
  ring.setDsoLocal(true);
  return ring;
}

/// Replaces the grouped loop's stageable loads with DMAs into the ring and LDS readbacks, and adds
/// the slot offset as an iter arg. Fails with a remark, before changing anything, when no load can
/// be staged; the loop is then pipelined in registers.
FailureOr<LdsStaging> stageLoadsInLds(RewriterBase &rewriter, scf::ForOp forOp,
                                      const UnrollRequest &request) {
  Location loc = forOp.getLoc();
  Block *body = forOp.getBody();
  auto fallBack = [&](const Twine &why) {
    InFlightDiagnostic remark = mlir::emitRemark(loc);
    remark << "pipeline=\"lds\" falls back to registers (unroll=" << request.factor << "): " << why;
    if (!request.kernelName.empty())
      remark << " (in " << request.kernelName << ")";
    return failure();
  };

  StringRef chip = getChip(forOp);
  if (!chip.starts_with("gfx95"))
    return fallBack("staging needs gfx950's 12- and 16-byte LDS DMA, and the target is " +
                    (chip.empty() ? StringRef("unknown") : chip));
  auto kernel = forOp->getParentOfType<gpu::GPUFuncOp>();
  std::optional<ArrayRef<int32_t>> blockSize;
  if (kernel)
    blockSize = kernel.getKnownBlockSize();
  if (!blockSize)
    return fallBack("the kernel has no known_block_size, which sizes the ring");

  SmallVector<Operation *> ops;
  for (Operation &op : body->without_terminator())
    ops.push_back(&op);
  OpSet slice = collectLoadSlice(ops, forOp);
  WaveUniformityAnalysis uniformity;
  SmallVector<Operation *> staged;
  llvm::MapVector<StringRef, int> unstaged;
  for (Operation *op : ops) {
    if (!isa<ROCDL::RawPtrBufferLoadOp>(op)) {
      if (isScalarBufferLoadCall(op))
        ++unstaged["it is a scalar load, which stays in a scalar register"];
      else if (isMovableLoad(op))
        ++unstaged["it is not a buffer load"];
      continue;
    }
    StringRef why = whyNotStaged(op, slice, body, uniformity);
    if (why.empty())
      staged.push_back(op);
    else
      ++unstaged[why];
  }
  std::string reasons;
  for (auto [why, count] : unstaged)
    reasons += (reasons.empty() ? "" : "; ") + std::to_string(count) +
               (count == 1 ? " load: " : " loads: ") + why.str();
  if (staged.empty())
    return fallBack("no load can be staged (" + reasons + ")");

  // ring[wave][slot][load][lane]: a DMA of up to 4 bytes per lane spaces lanes 4 bytes apart in
  // LDS, a larger one 16 bytes.
  SmallVector<int64_t> laneStrides, loadOffsets;
  int64_t slotBytes = 0;
  for (Operation *load : staged) {
    laneStrides.push_back(*getByteSize(load->getResult(0).getType()) > 4 ? 16 : 4);
    loadOffsets.push_back(slotBytes);
    slotBytes += kWaveSize * laneStrides.back();
  }
  int64_t threads = int64_t((*blockSize)[0]) * (*blockSize)[1] * (*blockSize)[2];
  int64_t waves = llvm::divideCeil(threads, kWaveSize);
  int64_t ringBytes = waves * kRingSlots * slotBytes;
  auto module = forOp->getParentOfType<gpu::GPUModuleOp>();
  int64_t ldsBytes = getStaticLdsBytes(module) + ringBytes;
  if (ldsBytes > kGfx950LdsBytes)
    return fallBack("the ring needs " + Twine(ringBytes) + " bytes, and the module's LDS would " +
                    "total " + Twine(ldsBytes) + " of " + Twine(kGfx950LdsBytes));
  if (!unstaged.empty()) {
    InFlightDiagnostic remark = mlir::emitRemark(loc);
    remark << "pipeline=\"lds\" keeps some loads in registers (unroll=" << request.factor
           << "): " << reasons;
    if (!request.kernelName.empty())
      remark << " (in " << request.kernelName << ")";
  }

  // Before the loop: the wave's part of the ring and each lane's readback offsets.
  rewriter.setInsertionPoint(forOp);
  Type i32 = rewriter.getI32Type();
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), kSharedAddressSpace);
  auto constant = [&](int64_t c) -> Value {
    return arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(c));
  };
  auto offsetPtr = [&](Value base, Value bytes) -> Value {
    return LLVM::GEPOp::create(rewriter, loc, ptrTy, rewriter.getI8Type(), base,
                               ArrayRef<LLVM::GEPArg>{bytes});
  };
  LLVM::GlobalOp ring = createRing(rewriter, module, loc, ringBytes);
  Value tid = ROCDL::ThreadIdXOp::create(rewriter, loc, i32);
  if ((*blockSize)[1] * (*blockSize)[2] > 1) {
    Value y = ROCDL::ThreadIdYOp::create(rewriter, loc, i32);
    Value z = ROCDL::ThreadIdZOp::create(rewriter, loc, i32);
    Value yz = arith::AddIOp::create(
        rewriter, loc, y, arith::MulIOp::create(rewriter, loc, z, constant((*blockSize)[1])));
    tid = arith::AddIOp::create(
        rewriter, loc, tid, arith::MulIOp::create(rewriter, loc, yz, constant((*blockSize)[0])));
  }
  // The DMA's LDS base must be wave-uniform.
  Value wave = ROCDL::ReadfirstlaneOp::create(
      rewriter, loc, i32, arith::DivUIOp::create(rewriter, loc, tid, constant(kWaveSize)));
  Value wavePtr =
      offsetPtr(LLVM::AddressOfOp::create(rewriter, loc, ring),
                arith::MulIOp::create(rewriter, loc, wave, constant(kRingSlots * slotBytes)));
  Value lane =
      ROCDL::MbcntHiOp::create(rewriter, loc, i32, constant(-1),
                               ROCDL::MbcntLoOp::create(rewriter, loc, i32, constant(-1),
                                                        constant(0), ArrayAttr(), ArrayAttr()),
                               ArrayAttr(), ArrayAttr());
  SmallVector<Value> readbackOffsets, dmaOffsets;
  for (auto [stride, offset] : llvm::zip(laneStrides, loadOffsets)) {
    readbackOffsets.push_back(arith::AddIOp::create(
        rewriter, loc, arith::MulIOp::create(rewriter, loc, lane, constant(stride)),
        constant(offset)));
    dmaOffsets.push_back(constant(offset));
  }
  Value zero = constant(0);
  Value slotStep = constant(slotBytes);

  // The slot that a group's DMAs write: 0 for the first group, then alternating.
  auto withSlot = cast<scf::ForOp>(*forOp.replaceWithAdditionalYields(
      rewriter, zero, /*replaceInitOperandUsesInLoop=*/false,
      [&](OpBuilder &b, Location yieldLoc, ArrayRef<BlockArgument> args) -> SmallVector<Value> {
        return {arith::XOrIOp::create(b, yieldLoc, args[0], slotStep)};
      }));
  body = withSlot.getBody();
  Value slot = withSlot.getRegionIterArgs().back();
  Operation *nextSlot =
      cast<scf::YieldOp>(body->getTerminator()).getResults().back().getDefiningOp();
  rewriter.moveOpBefore(nextSlot, &body->front());
  rewriter.setInsertionPointAfter(nextSlot);
  Value slotPtr = offsetPtr(wavePtr, slot);

  LdsStaging result{withSlot, {}, nullptr};
  Operation *firstReadback = nullptr, *lastDma = nullptr;
  for (auto [i, load] : llvm::enumerate(staged)) {
    Location loadLoc = load->getLoc();
    rewriter.setInsertionPoint(load);
    std::optional<int64_t> bytes = getByteSize(load->getResult(0).getType());
    lastDma = ROCDL::RawPtrBufferLoadAsyncLdsOp::create(
        rewriter, loadLoc, load->getOperand(0), offsetPtr(slotPtr, dmaOffsets[i]), constant(*bytes),
        load->getOperand(1), load->getOperand(2), zero, load->getAttr("aux"), ArrayAttr(),
        ArrayAttr(), ArrayAttr());

    Operation *firstUse = nullptr;
    for (Operation *user : load->getUsers()) {
      Operation *ancestor = body->findAncestorOpInBlock(*user);
      if (!firstUse || ancestor->isBeforeInBlock(firstUse))
        firstUse = ancestor;
    }
    rewriter.setInsertionPoint(firstUse);
    auto readback = LLVM::LoadOp::create(rewriter, loadLoc, load->getResult(0).getType(),
                                         offsetPtr(slotPtr, readbackOffsets[i]),
                                         /*alignment=*/laneStrides[i]);
    rewriter.replaceAllUsesWith(load->getResult(0), readback.getResult());
    rewriter.eraseOp(load);
    result.readbacks.insert(readback);
    if (!firstReadback || readback->isBeforeInBlock(firstReadback))
      firstReadback = readback;
  }

  // One async group per group of iterations. The wait count is for a loop that is not pipelined;
  // the pipeliner's kernel loop raises it to keep the next group in flight.
  rewriter.setInsertionPointAfter(lastDma);
  result.mark = ROCDL::AsyncmarkOp::create(rewriter, loc);
  rewriter.setInsertionPoint(firstReadback);
  ROCDL::WaitAsyncmarkOp::create(rewriter, loc, uint16_t(0));
  return result;
}

/// Moves `forOp` into `scf.if (lb < ub)`, which yields the loop's inits when it runs no iteration.
void guardByFirstIteration(RewriterBase &rewriter, scf::ForOp forOp) {
  Location loc = forOp.getLoc();
  rewriter.setInsertionPoint(forOp);
  Value runs = rewriter.createOrFold<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                    forOp.getLowerBound(), forOp.getUpperBound());
  bool hasResults = forOp.getNumResults() > 0;
  auto guard = scf::IfOp::create(rewriter, loc, forOp.getResultTypes(), runs,
                                 /*addThenBlock=*/true, /*addElseBlock=*/hasResults);
  rewriter.replaceAllUsesWith(forOp.getResults(), guard.getResults());
  rewriter.moveOpBefore(forOp, guard.thenBlock(), guard.thenBlock()->end());
  rewriter.setInsertionPointToEnd(guard.thenBlock());
  scf::YieldOp::create(rewriter, loc, forOp.getResults());
  if (hasResults) {
    rewriter.setInsertionPointToEnd(guard.elseBlock());
    scf::YieldOp::create(rewriter, loc, forOp.getInitArgs());
  }
}

/// Without the first barrier, LLVM's scheduler sinks the next group's loads into the current
/// group's compute, which shortens the time they have to complete. Without the second, it hoists
/// the copies of the next group's loaded values into the back edge's registers into the compute,
/// and each copy waits for its load there.
/// With LDS staging, the next group's loads, DMAs and async mark precede the current group's wait,
/// and the LDS readbacks after the wait are loads too.
void pinLoadStage(RewriterBase &rewriter, scf::ForOp loop) {
  Operation *lastLoad = nullptr;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (isa<ROCDL::WaitAsyncmarkOp>(op))
      break;
    if (isMovableLoad(&op) || isa<ROCDL::AsyncmarkOp>(op))
      lastLoad = &op;
  }
  assert(lastLoad && "the legality checks require a top-level load");
  rewriter.setInsertionPointAfter(lastLoad);
  ROCDL::SchedBarrier::create(rewriter, lastLoad->getLoc(), ROCDL::SchedGroupMask::none);
  Operation *yield = loop.getBody()->getTerminator();
  rewriter.setInsertionPoint(yield);
  ROCDL::SchedBarrier::create(rewriter, yield->getLoc(), ROCDL::SchedGroupMask::none);
}

/// On gfx9, loads and stores share vmcnt. The prologue's last load must complete before the
/// loop's first use, and LLVM's waitcnt insertion merges that requirement with the back edge's at
/// the loop header, so the loop waits vmcnt(0) at its top in every iteration, also for the
/// previous iteration's stores. Waiting for the prologue's loads before the loop leaves only the
/// back edge's requirement. gfx10 and later count stores separately, and gfx12 splits vmcnt, so
/// the flush is gfx9 only.
void flushBeforeLoop(RewriterBase &rewriter, scf::ForOp loop) {
  if (!getChip(loop).starts_with("gfx9"))
    return;
  // vmcnt(0), with expcnt (bits 6:4) and lgkmcnt (bits 11:8) at their maxima, so it waits for
  // nothing else.
  constexpr uint32_t kGfx9WaitVmcnt0 = 0x0F70;
  rewriter.setInsertionPoint(loop);
  ROCDL::SWaitcntOp::create(rewriter, loop.getLoc(), kGfx9WaitVmcnt0);
}

/// Pipelines the grouped main loop in two stages: each group's load slice, then the rest, so the
/// next group's loads issue before the current group's other ops. The loop runs only inside a guard
/// for at least one group, so the pipeliner's prologue and epilogue never run past the trip count
/// and need no predicate. Failure means the pipeliner changed the IR before failing.
LogicalResult pipelineGroups(RewriterBase &rewriter, scf::ForOp forOp,
                             const UnrollRequest &request) {
  LdsStaging staging;
  if (request.lds) {
    if (FailureOr<LdsStaging> staged = stageLoadsInLds(rewriter, forOp, request);
        succeeded(staged)) {
      staging = std::move(*staged);
      forOp = staging.loop;
    }
  }

  // The pipeliner rejects a yield of the iter arg itself, which always holds the init value.
  auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (auto [arg, init, yielded] :
       llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs(), yield->getOpOperands()))
    if (yielded.get() == arg)
      rewriter.modifyOpInPlace(yield, [&] { yielded.set(init); });
  guardByFirstIteration(rewriter, forOp);

  scf::PipeliningOption options;
  options.supportDynamicLoops = true;
  options.peelEpilogue = true;
  // The LDS readbacks belong to the current group's stage, and the async mark to the next group's.
  options.getScheduleFn = [&staging](scf::ForOp loop,
                                     std::vector<std::pair<Operation *, unsigned>> &schedule) {
    SmallVector<Operation *> ops, sliceCandidates;
    for (Operation &op : loop.getBody()->without_terminator()) {
      ops.push_back(&op);
      if (!staging.readbacks.contains(&op))
        sliceCandidates.push_back(&op);
    }
    OpSet slice = collectLoadSlice(sliceCandidates, loop);
    for (Operation *op : ops)
      schedule.emplace_back(op, slice.contains(op) || op == staging.mark ? 0 : 1);
  };
  options.predicateFn = [](RewriterBase &, Operation *op, Value) { return op; };
  // In the kernel loop, the next group's DMAs are issued before the current group's wait.
  options.annotateFn = [](Operation *op, scf::PipeliningOption::PipelinerPart part, unsigned) {
    if (auto wait = dyn_cast<ROCDL::WaitAsyncmarkOp>(op);
        wait && part == scf::PipeliningOption::PipelinerPart::Kernel)
      wait.setCount(kRingSlots - 1);
  };

  rewriter.setInsertionPoint(forOp);
  std::string message;
  bool changedIR = false;
  FailureOr<scf::ForOp> pipelined = failure();
  {
    // A clean rejection becomes a remark: Python's PassManager.run fails on any reported error.
    // The handler is the context's, so it lets through what other threads report about other loops.
    DenseSet<Location> loopLocs;
    forOp->walk([&](Operation *op) { loopLocs.insert(op->getLoc()); });
    ScopedDiagnosticHandler capture(forOp.getContext(), [&](Diagnostic &diag) {
      if (!loopLocs.contains(diag.getLocation()))
        return failure();
      if (message.empty())
        message = diag.str();
      return success();
    });
    pipelined = scf::pipelineForLoop(rewriter, forOp, options, &changedIR);
  }
  if (succeeded(pipelined)) {
    disableLlvmUnroll(*pipelined);
    pinLoadStage(rewriter, *pipelined);
    flushBeforeLoop(rewriter, *pipelined);
    return success();
  }
  if (changedIR)
    return forOp.emitError("the pipeliner failed after changing the loop: ") << message;
  mlir::emitRemark(forOp.getLoc())
      << "pipeline ignored (unroll=" << request.factor
      << "), the loop's groups are not pipelined: the pipeliner rejected it"
      << (message.empty() ? "" : " (" + message + ")");
  return success();
}

class FlyROCDLUnrollIndependentPass
    : public mlir::fly_rocdl::impl::FlyROCDLUnrollIndependentPassBase<
          FlyROCDLUnrollIndependentPass> {
public:
  using mlir::fly_rocdl::impl::FlyROCDLUnrollIndependentPassBase<
      FlyROCDLUnrollIndependentPass>::FlyROCDLUnrollIndependentPassBase;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    // Post-order, so an inner marked loop is rewritten before an outer one's unrolling clones it.
    SmallVector<UnrollRequest> requests;
    getOperation()->walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
      if (std::optional<UnrollRequest> request = takeUnrollRequest(forOp))
        requests.push_back(std::move(*request));
    });
    for (const UnrollRequest &request : requests) {
      if (failed(checkLoop(request)))
        continue;
      unrollAndHoistLoads(rewriter, request.loop, request.factor, request.pipeline);
      if (request.pipeline && failed(pipelineGroups(rewriter, request.loop, request)))
        return signalPassFailure();
    }
  }
};

} // namespace
