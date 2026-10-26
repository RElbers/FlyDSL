// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-prefetch-loops),canonicalize)' | FileCheck %s --implicit-check-not=fly.prefetch_depth --implicit-check-not=fly.prefetch_assume_independent
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-prefetch-loops))' -verify-diagnostics -o /dev/null

// The FileCheck run adds canonicalize only to fold constant bounds and trip counts, which some
// checks below rely on. The compile pipeline runs no canonicalize after the pass, so these checks
// must not depend on it merging guards: the pass merges guards with the same predicate itself, and
// upstream's CombineIfs only merges adjacent ones.

gpu.module @prefetch_edges {
  // Unrolling leaves exactly one main-loop iteration; it must not erase the operation before
  // pipelining. Canonicalization folds the resulting straight-line computation.
  // CHECK-LABEL: gpu.func @one_unrolled_iteration
  // CHECK-NOT: scf.for
  // CHECK-COUNT-4: llvm.load
  // CHECK: llvm.store
  // CHECK: gpu.return
  gpu.func @one_unrolled_iteration(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c0) -> i32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      %v = llvm.load %p : !llvm.ptr<1> -> i32
      %next = arith.addi %acc, %v : i32
      scf.yield %next : i32
    } {fly.prefetch_depth = 4 : i64}
    llvm.store %result, %out : i32, !llvm.ptr<1>
    gpu.return
  }

  // Both the load and the division need guards, since the skipped load produces a zero placeholder.
  // CHECK-LABEL: gpu.func @division
  // CHECK: scf.if {{.*}} -> (i32) {
  // CHECK-NEXT: llvm.load
  // CHECK: scf.for
  // CHECK: arith.divsi
  // CHECK: scf.yield
  // CHECK: scf.if {{.*}} -> (i32) {
  // CHECK-NEXT: arith.divsi
  // CHECK-NEXT: scf.yield
  // CHECK: llvm.store
  // CHECK: gpu.return
  gpu.func @division(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0) -> i32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      %v = llvm.load %p : !llvm.ptr<1> -> i32
      %q = arith.divsi %c1, %v : i32
      %next = arith.addi %acc, %q : i32
      scf.yield %next : i32
    } {fly.prefetch_depth = 1 : i64}
    llvm.store %result, %out : i32, !llvm.ptr<1>
    gpu.return
  }

  // Upstream marks floordivsi speculatable, but a zero divisor is still undefined behavior.
  // CHECK-LABEL: gpu.func @floor_division
  // CHECK: scf.if {{.*}} -> (i32) {
  // CHECK-NEXT: llvm.load
  // CHECK: scf.for
  // CHECK: arith.floordivsi
  // CHECK: scf.yield
  // CHECK: scf.if {{.*}} -> (i32) {
  // CHECK-NEXT: arith.floordivsi
  // CHECK-NEXT: scf.yield
  // CHECK: llvm.store
  // CHECK: gpu.return
  gpu.func @floor_division(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c0) -> i32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      %v = llvm.load %p : !llvm.ptr<1> -> i32
      %q = arith.floordivsi %c1, %v : i32
      %next = arith.addi %acc, %q : i32
      scf.yield %next : i32
    } {fly.prefetch_depth = 1 : i64}
    llvm.store %result, %out : i32, !llvm.ptr<1>
    gpu.return
  }

  // Peeling must not turn the empty i32 range into 0..INT_MAX.
  // CHECK-LABEL: gpu.func @empty_minimum_bound
  // CHECK-NOT: llvm.load
  // CHECK-NOT: scf.for
  // CHECK: gpu.return
  gpu.func @empty_minimum_bound(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %min = arith.constant -2147483648 : i32
    scf.for %i = %c0 to %min step %c1 : i32 {
      %v = llvm.load %in : !llvm.ptr<1> -> i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 1 : i64}
    gpu.return
  }

  // The scaled i32 step would be zero without widening the loop control arithmetic.
  // CHECK-LABEL: gpu.func @large_step
  // CHECK: %[[STEP:.*]] = arith.constant 4294967296 : i64
  // CHECK: arith.extsi %arg2 : i32 to i64
  // CHECK: scf.for {{.*}} step %[[STEP]]
  // CHECK: gpu.return
  gpu.func @large_step(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1073741824 : i32
    scf.for %i = %c0 to %n step %step : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      %v = llvm.load %p : !llvm.ptr<1> -> i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // CHECK-LABEL: gpu.func @overflowing_wide_step
  // CHECK: %[[STEP:.*]] = arith.constant 4611686018427387904 : i64
  // CHECK: scf.for {{.*}} step %[[STEP]]
  // CHECK-NEXT: llvm.load
  // CHECK: gpu.return
  gpu.func @overflowing_wide_step(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i64) kernel {
    %c0 = arith.constant 0 : i64
    %step = arith.constant 4611686018427387904 : i64
    // expected-remark @below {{loop not pipelined (prefetch_depth=4): its step multiplied by the depth may overflow a signed 64-bit integer}}
    scf.for %i = %c0 to %n step %step : i64 {
      %v = llvm.load %in : !llvm.ptr<1> -> i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // A runtime index step is assumed to be far below 2^63 / depth, so the loop is still pipelined.
  // CHECK-LABEL: gpu.func @runtime_wide_step
  // CHECK: scf.if {{.*}} -> (i32, i32, i32, i32) {
  // CHECK-NEXT: llvm.load
  // CHECK: scf.for
  // CHECK: gpu.return
  gpu.func @runtime_wide_step(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: index, %step: index) kernel {
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %n step %step {
      %v = llvm.load %in : !llvm.ptr<1> -> i32
      llvm.store %v, %out : i32, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // Identify the reduction from its intrinsic attribute and split at the wave's shortest trip count.
  // CHECK-LABEL: gpu.func @intrinsic_reduction
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"
  // CHECK: scf.for
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"
  // CHECK: scf.for
  // CHECK: gpu.return
  gpu.func @intrinsic_reduction(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %tid = gpu.thread_id x
    %lane = arith.index_cast %tid : index to i32
    scf.for %i = %lane to %n step %c64 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      %v = llvm.load %p : !llvm.ptr<1> -> i32
      %min = llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"(%v, %c0) : (i32, i32) -> i32
      llvm.store %min, %out : i32, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // Calls, inline asm and AMDGPU intrinsics are opaque, so lane-dependent bounds get the wave split.
  // CHECK-LABEL: gpu.func @inline_asm_exchange
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"
  // CHECK: scf.for
  // CHECK: gpu.return
  gpu.func @inline_asm_exchange(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c64 = arith.constant 64 : i32
    %tid = gpu.thread_id x
    %lane = arith.index_cast %tid : index to i32
    scf.for %i = %lane to %n step %c64 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      %v = llvm.load %p : !llvm.ptr<1> -> i32
      %r = llvm.inline_asm has_side_effects "v_readlane_b32 $0, $1, 0", "=s,v" %v : (i32) -> i32
      llvm.store %r, %out : i32, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // CHECK-LABEL: gpu.func @extern_call_exchange
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"
  // CHECK: scf.for
  // CHECK: gpu.return
  llvm.func @extern_reduce(i32) -> i32
  gpu.func @extern_call_exchange(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c64 = arith.constant 64 : i32
    %tid = gpu.thread_id x
    %lane = arith.index_cast %tid : index to i32
    scf.for %i = %lane to %n step %c64 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      %v = llvm.load %p : !llvm.ptr<1> -> i32
      %r = llvm.call @extern_reduce(%v) : (i32) -> i32
      llvm.store %r, %out : i32, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // llvm.amdgcn.icmp returns a wave mask, like a ballot.
  // CHECK-LABEL: gpu.func @amdgcn_icmp_exchange
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"
  // CHECK: scf.for
  // CHECK: gpu.return
  gpu.func @amdgcn_icmp_exchange(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %eq = arith.constant 32 : i32
    %tid = gpu.thread_id x
    %lane = arith.index_cast %tid : index to i32
    scf.for %i = %lane to %n step %c64 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      %v = llvm.load %p : !llvm.ptr<1> -> i32
      %mask = llvm.call_intrinsic "llvm.amdgcn.icmp.i64.i32"(%v, %c0, %eq) : (i32, i32, i32) -> i64
      llvm.store %mask, %out : i64, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // A wave reduction declares no memory effects but does not write memory, so a loop that only
  // loads and reduces is pipelined without the flag, after the wave-uniform split.
  // CHECK-LABEL: gpu.func @intrinsic_reduction_without_flag
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"
  // CHECK: scf.if {{.*}} -> (i32, i32, i32, i32) {
  // CHECK: scf.for
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"
  // CHECK: gpu.return
  gpu.func @intrinsic_reduction_without_flag(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %tid = gpu.thread_id x
    %lane = arith.index_cast %tid : index to i32
    %r = scf.for %i = %lane to %n step %c64 iter_args(%acc = %c0) -> i32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      %v = llvm.load %p : !llvm.ptr<1> -> i32
      %min = llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"(%v, %c0) : (i32, i32) -> i32
      %s = arith.addi %acc, %min : i32
      scf.yield %s : i32
    } {fly.prefetch_depth = 4 : i64}
    llvm.store %r, %out : i32, !llvm.ptr<1>
    gpu.return
  }

  // The worklist must follow captures of pure region ops and the next value of a carried address.
  // CHECK-LABEL: gpu.func @captured_address
  // CHECK: scf.if {{.*}} -> (i32)
  // CHECK: scf.if {{.*}} -> (i32)
  // CHECK: llvm.load
  // CHECK: scf.for
  // CHECK: gpu.return
  gpu.func @captured_address(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32, %choose: i1) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result:2 = scf.for %i = %c0 to %n step %c1 iter_args(%off = %c0, %acc = %c0) -> (i32, i32) : i32 {
      %next = arith.addi %off, %c1 : i32
      %addr = scf.if %choose -> i32 {
        scf.yield %off : i32
      } else {
        scf.yield %next : i32
      }
      %p = llvm.getelementptr %in[%addr] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      %v = llvm.load %p : !llvm.ptr<1> -> i32
      %sum = arith.addi %acc, %v : i32
      scf.yield %next, %sum : i32, i32
    } {fly.prefetch_depth = 2 : i64}
    llvm.store %result#1, %out : i32, !llvm.ptr<1>
    gpu.return
  }

  // Without prefetch_assume_independent, a loop that may write memory is left unchanged: its loads
  // would move ahead of earlier iterations' writes.
  // CHECK-LABEL: gpu.func @store_without_flag
  // CHECK:      scf.for
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.load
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.store
  // CHECK-NEXT: }
  gpu.func @store_without_flag(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{loop not pipelined (prefetch_depth=4): it may write memory (llvm.store) and its loads would move ahead of earlier iterations' writes; set prefetch_assume_independent=True}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // MFMA declares no memory effects but does not write memory, so a load-only loop that feeds it
  // is pipelined without the flag.
  // CHECK-LABEL: gpu.func @mfma_without_flag
  // CHECK: scf.if {{.*}} -> (vector<4xf16>
  // CHECK: scf.for
  // CHECK: rocdl.mfma.f32.16x16x16f16
  // CHECK: gpu.return
  gpu.func @mfma_without_flag(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : vector<4xf32>
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> vector<4xf32> : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<4xf16>
      %a = llvm.load %p : !llvm.ptr<1> -> vector<4xf16>
      %d = rocdl.mfma.f32.16x16x16f16 %a, %a, %acc, 0, 0, none : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      scf.yield %d : vector<4xf32>
    } {fly.prefetch_depth = 4 : i64}
    llvm.store %result, %out : vector<4xf32>, !llvm.ptr<1>
    gpu.return
  }

  // An accumulator kept in memory: the flag asserts independence, but every iteration reads the
  // address the previous one wrote.
  // CHECK-LABEL: gpu.func @invariant_accumulator
  // CHECK:      scf.for
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.load
  // CHECK-NEXT:   llvm.load
  gpu.func @invariant_accumulator(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{loop not pipelined (prefetch_depth=4): a load reads the same address in every iteration from a buffer the loop writes}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %old = llvm.load %out : !llvm.ptr<1> -> f32
      %sum = arith.addf %old, %v : f32
      llvm.store %sum, %out : f32, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // The same through a buffer resource, with the offset computed in the loop from invariant values.
  // CHECK-LABEL: gpu.func @invariant_buffer_accumulator
  // CHECK:      scf.for
  // CHECK:        rocdl.raw.ptr.buffer.store
  // CHECK-NEXT: }
  gpu.func @invariant_buffer_accumulator(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32, %lane: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1>, i64 to <8>
    %rout = rocdl.make.buffer.rsrc %out, %c0_i16, %num_records, %flags : <1>, i64 to <8>
    // expected-remark @below {{loop not pipelined (prefetch_depth=4): a load reads the same address in every iteration from a buffer the loop writes}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : f32
      %off = arith.muli %lane, %c4 : i32
      %old = rocdl.raw.ptr.buffer.load %rout, %off, %c0, 0 : f32
      %sum = arith.addf %old, %v : f32
      rocdl.raw.ptr.buffer.store %sum, %rout, %off, %c0, 0 : f32
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // Ordering and private memory are checked even with the flag.
  // CHECK-LABEL: gpu.func @seq_cst_load
  // CHECK:      scf.for
  // CHECK-NEXT:   llvm.load {{.*}} atomic seq_cst
  gpu.func @seq_cst_load(%flag: !llvm.ptr<1>, %in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{loop not pipelined (prefetch_depth=2): it contains a fence, a volatile access or an atomic with acquire or stronger ordering}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %f = llvm.load %flag atomic seq_cst {alignment = 4 : i64} : !llvm.ptr<1> -> i32
      %p = llvm.getelementptr %in[%f] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 2 : i64}
    gpu.return
  }

  // CHECK-LABEL: gpu.func @volatile_load
  // CHECK:      scf.for
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.load volatile
  gpu.func @volatile_load(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    // expected-remark @below {{loop not pipelined (prefetch_depth=2): it contains a fence, a volatile access or an atomic with acquire or stronger ordering}}
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load volatile %p : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    } {fly.prefetch_depth = 2 : i64}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // CHECK-LABEL: gpu.func @private_memory
  // CHECK:      scf.for
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.load
  gpu.func @private_memory(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %slot = llvm.alloca %c1 x f32 : (i32) -> !llvm.ptr<5>
    // expected-remark @below {{loop not pipelined (prefetch_depth=4): it accesses private memory}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %old = llvm.load %slot : !llvm.ptr<5> -> f32
      %sum = arith.addf %old, %v : f32
      llvm.store %sum, %slot : f32, !llvm.ptr<5>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    %final = llvm.load %slot : !llvm.ptr<5> -> f32
    llvm.store %final, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // A guarded load of a vector of pointers gets a zero placeholder from llvm.mlir.zero, since
  // getZeroAttr has no pointer attribute.
  // CHECK-LABEL: gpu.func @vector_of_pointers
  // CHECK: llvm.mlir.zero : vector<2x!llvm.ptr<1>>
  // CHECK: scf.for
  // CHECK: gpu.return
  gpu.func @vector_of_pointers(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<2x!llvm.ptr<1>>
      %v = llvm.load %p : !llvm.ptr<1> -> vector<2x!llvm.ptr<1>>
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<2x!llvm.ptr<1>>
      llvm.store %v, %q : vector<2x!llvm.ptr<1>>, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 2 : i64}
    gpu.return
  }

  // Narrow unsigned bounds are zero-extended to i64 and compared signed, which the unroller and
  // the pipeliner handle; the static loop is unrolled by 4 with a two-iteration remainder.
  // CHECK-LABEL: gpu.func @unsigned_static
  // CHECK-NOT: scf.for unsigned
  // CHECK: scf.for %{{.*}} = %c8_i64 to %c10_i64 step %c1_i64 : i64
  // CHECK: gpu.return
  gpu.func @unsigned_static(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    scf.for unsigned %i = %c0 to %c10 step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // CHECK-LABEL: gpu.func @unsigned_dynamic
  // CHECK: arith.extui %arg2 : i32 to i64
  // CHECK-NOT: scf.for unsigned
  // CHECK: scf.if {{.*}} -> (f32
  // CHECK: scf.for {{.*}} : i64
  // CHECK: gpu.return
  gpu.func @unsigned_dynamic(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for unsigned %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // CHECK-LABEL: gpu.func @unsigned_wide
  // CHECK: scf.for unsigned
  gpu.func @unsigned_wide(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i64) kernel {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    // expected-remark @below {{loop not pipelined (prefetch_depth=4): it compares 64-bit bounds unsigned, which the pipeliner does not support}}
    scf.for unsigned %i = %c0 to %n step %c1 : i64 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.prefetch_assume_independent, fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  // A scheduling hint is neither a barrier nor a write, so a load-only loop with one is pipelined
  // without the flag.
  // CHECK-LABEL: gpu.func @sched_barrier_hint
  // CHECK: scf.if {{.*}} -> (f32, f32, f32, f32) {
  // CHECK: scf.for
  // CHECK: rocdl.sched.barrier
  // CHECK: gpu.return
  gpu.func @sched_barrier_hint(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      rocdl.sched.barrier none
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    } {fly.prefetch_depth = 4 : i64}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // An MFMA combines the registers of all lanes, so lane-dependent bounds get the wave split.
  // CHECK-LABEL: gpu.func @mfma_lane_dependent
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"
  // CHECK: scf.for
  // CHECK: rocdl.mfma.f32.16x16x16f16
  // CHECK: gpu.return
  gpu.func @mfma_lane_dependent(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c64 = arith.constant 64 : i32
    %zero = arith.constant dense<0.0> : vector<4xf32>
    %tid = gpu.thread_id x
    %lane = arith.index_cast %tid : index to i32
    %result = scf.for %i = %lane to %n step %c64 iter_args(%acc = %zero) -> vector<4xf32> : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<4xf16>
      %a = llvm.load %p : !llvm.ptr<1> -> vector<4xf16>
      %d = rocdl.mfma.f32.16x16x16f16 %a, %a, %acc, 0, 0, none : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      scf.yield %d : vector<4xf32>
    } {fly.prefetch_depth = 4 : i64}
    llvm.store %result, %out : vector<4xf32>, !llvm.ptr<1>
    gpu.return
  }

  // A transposed load gathers data addressed by other lanes.
  // CHECK-LABEL: gpu.func @transposed_load_lane_dependent
  // CHECK: llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"
  // CHECK: scf.for
  // CHECK: rocdl.global.load.tr4.b64
  // CHECK: gpu.return
  gpu.func @transposed_load_lane_dependent(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c64 = arith.constant 64 : i32
    %zero = arith.constant dense<0> : vector<2xi32>
    %tid = gpu.thread_id x
    %lane = arith.index_cast %tid : index to i32
    %result = scf.for %i = %lane to %n step %c64 iter_args(%acc = %zero) -> vector<2xi32> : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<2xi32>
      %v = rocdl.global.load.tr4.b64 %p : !llvm.ptr<1> -> vector<2xi32>
      %s = arith.addi %acc, %v : vector<2xi32>
      scf.yield %s : vector<2xi32>
    } {fly.prefetch_depth = 4 : i64}
    llvm.store %result, %out : vector<2xi32>, !llvm.ptr<1>
    gpu.return
  }

  // Arguments of a function marked with the gpu.kernel attribute are wave-uniform too, so a shuffle
  // loop over them needs no wave split.
  // CHECK-LABEL: gpu.func @kernel_attribute_uniform
  // CHECK-NOT: wave.reduce
  // CHECK: scf.for
  // CHECK: gpu.return
  gpu.func @kernel_attribute_uniform(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %lo: i32, %hi: i32) attributes {gpu.kernel} {
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %f0 = arith.constant 0.0 : f32
    %r = scf.for %i = %lo to %hi step %c1 iter_args(%acc = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %sh, %valid = gpu.shuffle xor %v, %c1, %c64 : f32
      %s = arith.addf %acc, %sh : f32
      scf.yield %s : f32
    } {fly.prefetch_depth = 4 : i64}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // CHECK-LABEL: gpu.func @too_deep
  // CHECK: scf.for
  // CHECK-NEXT: llvm.getelementptr
  gpu.func @too_deep(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    // expected-remark @below {{loop not pipelined (prefetch_depth=65): the depth must be between 1 and 64}}
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    } {fly.prefetch_depth = 65 : i64}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // A Python range counting down lowers to a negative step, which scf.for does not allow; the
  // remark names that rather than an overflow, at any depth.
  // CHECK-LABEL: gpu.func @negative_step
  // CHECK: scf.for
  // CHECK-NEXT: llvm.getelementptr
  // CHECK: scf.for
  // CHECK-NEXT: llvm.getelementptr
  gpu.func @negative_step(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %cm1 = arith.constant -1 : i32
    %f0 = arith.constant 0.0 : f32
    // expected-remark @below {{loop not pipelined (prefetch_depth=4): its step is not positive, which scf.for does not allow}}
    %r = scf.for %i = %n to %c0 step %cm1 iter_args(%acc = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    } {fly.prefetch_depth = 4 : i64}
    // expected-remark @below {{loop not pipelined (prefetch_depth=1): its step is not positive, which scf.for does not allow}}
    %r1 = scf.for %i = %n to %c0 step %cm1 iter_args(%acc = %r) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    } {fly.prefetch_depth = 1 : i64}
    llvm.store %r1, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // An unsigned loop compares its step unsigned, so an all-ones i32 step is 2^32 - 1, not -1. It is
  // widened to i64, so step * depth fits and the loop is pipelined.
  // CHECK-LABEL: gpu.func @unsigned_large_step
  // CHECK: scf.if
  // CHECK: scf.for {{.*}} : i64
  gpu.func @unsigned_large_step(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %big = arith.constant -1 : i32
    %f0 = arith.constant 0.0 : f32
    %r = scf.for unsigned %i = %c0 to %n step %big iter_args(%acc = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    } {fly.prefetch_depth = 4 : i64}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }
}
