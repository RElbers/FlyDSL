// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-prefetch-loops),canonicalize)' | FileCheck %s --implicit-check-not=fly.prefetch_depth
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-prefetch-loops))' -verify-diagnostics -o /dev/null

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
    } {fly.prefetch_depth = 1 : i64}
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
    } {fly.prefetch_depth = 4 : i64}
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
    } {fly.prefetch_depth = 4 : i64}
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
    } {fly.prefetch_depth = 4 : i64}
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
    } {fly.prefetch_depth = 4 : i64}
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
    } {fly.prefetch_depth = 4 : i64}
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
    } {fly.prefetch_depth = 4 : i64}
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
    } {fly.prefetch_depth = 4 : i64}
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
}
