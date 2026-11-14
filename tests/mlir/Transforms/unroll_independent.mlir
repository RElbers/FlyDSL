// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-unroll-independent))' --mlir-print-local-scope | FileCheck %s --implicit-check-not=fly.assume_independent
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-unroll-independent))' -verify-diagnostics -o /dev/null

// Loops the pass unrolls: each group of N iterations issues its loads, with their address
// computations, before its other operations. A serial remainder loop runs the rest. LLVM's unroller
// is disabled on both loops.

gpu.module @unroll_independent {
  // The main loop keeps the i32 type. Its bound, computed in i64, falls back to the lower bound
  // when it does not fit, and the remainder loop then runs every iteration.
  // CHECK-LABEL: gpu.func @scatter
  // CHECK:      %[[FITS:.*]] = arith.cmpi sle, %[[WIDE_UB:.*]], %c2147483647_i64 : i64
  // CHECK-NEXT: %[[NARROW_UB:.*]] = arith.trunci %[[WIDE_UB]] : i64 to i32
  // CHECK-NEXT: %[[MAIN_UB:.*]] = arith.select %[[FITS]], %[[NARROW_UB]], %c0_i32 : i32
  // CHECK:      scf.for %{{.*}} = %c0_i32 to %[[MAIN_UB]] step %c4_i32 : i32 {
  // CHECK-COUNT-4: llvm.load
  // CHECK-NOT:  llvm.load
  // CHECK-COUNT-4: llvm.store
  // CHECK-NEXT: } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  // CHECK-NEXT: scf.for %{{.*}} = %[[MAIN_UB]] to %arg2 step %c1_i32 : i32 {
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.load
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.store
  // CHECK-NEXT: } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  // CHECK-NEXT: gpu.return
  gpu.func @scatter(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // Buffer loads, a computation between load and store, and the loads' offsets: every copy's offset
  // and load come first, in the original order.
  // CHECK-LABEL: gpu.func @buffer_scatter
  // CHECK:      scf.for %[[I0:.*]] = {{.*}} : i32 {
  // CHECK-NEXT:   %[[V0:.*]] = rocdl.raw.ptr.buffer.load %{{.*}}, %[[I0]],
  // CHECK:        %[[I1:.*]] = arith.addi %[[I0]],
  // CHECK-NEXT:   %[[V1:.*]] = rocdl.raw.ptr.buffer.load %{{.*}}, %[[I1]],
  // CHECK-NEXT:   %[[W0:.*]] = arith.mulf %[[V0]], %[[V0]]
  // CHECK-NEXT:   rocdl.raw.ptr.buffer.store %[[W0]], %{{.*}}, %[[I0]],
  // CHECK-NEXT:   %[[W1:.*]] = arith.mulf %[[V1]], %[[V1]]
  // CHECK-NEXT:   rocdl.raw.ptr.buffer.store %[[W1]], %{{.*}}, %[[I1]],
  // CHECK-NEXT: } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  gpu.func @buffer_scatter(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %lo: i32, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    %rout = rocdl.make.buffer.rsrc %out, %c0_i16, %num_records, %flags : <1> to <8>
    scf.for %i = %lo to %n step %c64 : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : f32
      %w = arith.mulf %v, %v : f32
      rocdl.raw.ptr.buffer.store %w, %rout, %i, %c0, 0 : f32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // A reduction carries its accumulator through the group in order, after all of the group's loads.
  // CHECK-LABEL: gpu.func @reduce
  // CHECK:      scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (f32) : i32 {
  // CHECK-COUNT-4: llvm.load
  // CHECK-NEXT:   %[[S0:.*]] = arith.addf %[[ACC]],
  // CHECK-NEXT:   %[[S1:.*]] = arith.addf %[[S0]],
  // CHECK-NEXT:   %[[S2:.*]] = arith.addf %[[S1]],
  // CHECK-NEXT:   %[[S3:.*]] = arith.addf %[[S2]],
  // CHECK-NEXT:   scf.yield %[[S3]] : f32
  // CHECK:      scf.for
  // CHECK:      gpu.return
  gpu.func @reduce(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // A gather: the second load's address comes from the first load. Both move, in order.
  // CHECK-LABEL: gpu.func @gather
  // CHECK:      scf.for {{.*}} : i32 {
  // CHECK:        %[[J0:.*]] = llvm.load
  // CHECK:        llvm.getelementptr %{{.*}}[%[[J0]]]
  // CHECK-NEXT:   llvm.load
  // CHECK:        %[[J1:.*]] = llvm.load
  // CHECK:        llvm.getelementptr %{{.*}}[%[[J1]]]
  // CHECK-NEXT:   llvm.load
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.store
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.store
  // CHECK-NEXT: }
  gpu.func @gather(%idx: !llvm.ptr<1>, %in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pj = llvm.getelementptr %idx[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      %j = llvm.load %pj : !llvm.ptr<1> -> i32
      %pin = llvm.getelementptr %in[%j] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // An address carried from one iteration to the next and bumped after its load: a bump that feeds
  // the next copy's load moves up with it.
  // CHECK-LABEL: gpu.func @carried_address
  // CHECK:      scf.for {{.*}} iter_args(%[[P:.*]] = %{{.*}}, %[[ACC:.*]] = %{{.*}}) -> (!llvm.ptr<1>, f32) : i32 {
  // CHECK-NEXT:   %[[V0:.*]] = llvm.load %[[P]]
  // CHECK-NEXT:   %[[P1:.*]] = llvm.getelementptr %[[P]][1]
  // CHECK-NEXT:   %[[V1:.*]] = llvm.load %[[P1]]
  // CHECK:        %[[S0:.*]] = arith.addf %[[ACC]], %[[V0]]
  // CHECK:        %[[S1:.*]] = arith.addf %[[S0]], %[[V1]]
  // CHECK-NEXT:   %[[P2:.*]] = llvm.getelementptr %[[P1]][1]
  // CHECK-NEXT:   scf.yield %[[P2]], %[[S1]]
  gpu.func @carried_address(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%p = %in, %acc = %f0) -> (!llvm.ptr<1>, f32) : i32 {
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %v : f32
      %next = llvm.getelementptr %p[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f32
      scf.yield %next, %s : !llvm.ptr<1>, f32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    llvm.store %r#1, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // Six constant trips at N = 4: the bounds fold, the main loop runs one group and the remainder loop
  // the two remaining trips.
  // CHECK-LABEL: gpu.func @one_group_and_remainder
  // CHECK:      scf.for %{{.*}} = %c0_i32 to %c4_i32 step %c4_i32{{.*}} : i32 {
  // CHECK-COUNT-4: llvm.load
  // CHECK-NOT:  llvm.load
  // CHECK-COUNT-4: llvm.store
  // CHECK-NEXT: } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  // CHECK-NEXT: scf.for %{{.*}} = %c4_i32 to %c6_i32 step %c1_i32 : i32 {
  // CHECK:      } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  gpu.func @one_group_and_remainder(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c6 = arith.constant 6 : i32
    scf.for %i = %c0 to %c6 step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // The i32 step times N does not fit i32, so the main loop runs no iteration and the remainder loop
  // runs them all.
  // CHECK-LABEL: gpu.func @large_step
  // CHECK:      scf.for %{{.*}} = %c0_i32 to %c0_i32 step %c1073741824_i32 : i32 {
  // CHECK:      scf.for %{{.*}} = %c0_i32 to %arg2 step %c1073741824_i32 : i32 {
  gpu.func @large_step(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %step = arith.constant 1073741824 : i32
    scf.for %i = %c0 to %n step %step : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // Index bounds, as range(..., init=...) emits them, stay index; a runtime index step is assumed
  // not to overflow when multiplied by N.
  // CHECK-LABEL: gpu.func @index_bounds
  // CHECK:      arith.maxsi %{{.*}} : index
  // CHECK-NOT:  arith.select
  // CHECK:      scf.for {{.*}} {
  // CHECK-COUNT-4: llvm.load
  // CHECK:      scf.for
  gpu.func @index_bounds(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %lo: index, %n: index, %step: index) kernel {
    scf.for %i = %lo to %n step %step {
      %i32 = arith.index_cast %i : index to i32
      %pin = llvm.getelementptr %in[%i32] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %pout = llvm.getelementptr %out[%i32] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // A shuffle is fine when the bounds are wave-uniform: block ids and kernel arguments.
  // CHECK-LABEL: gpu.func @shuffle_uniform_bounds
  // CHECK:      scf.for {{.*}} : i32 {
  // CHECK-COUNT-2: llvm.load
  // CHECK:        gpu.shuffle
  gpu.func @shuffle_uniform_bounds(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %bid = gpu.block_id x
    %bid_i32 = arith.index_cast %bid : index to i32
    %lo = arith.muli %bid_i32, %n : i32
    %hi = arith.addi %lo, %n : i32
    scf.for %i = %lo to %hi step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %s, %valid = gpu.shuffle xor %v, %c1, %c64 : f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %s, %pout : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // Inline assembly without side effects declares no memory effect, so LLVM may move it freely and
  // the loads may move above it too. As an opaque op it needs wave-uniform bounds.
  // CHECK-LABEL: gpu.func @pure_asm
  // CHECK:      scf.for {{.*}} : i32 {
  // CHECK-COUNT-2: llvm.load
  // CHECK:        llvm.inline_asm "v_mov_b32
  gpu.func @pure_asm(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %w = llvm.inline_asm "v_mov_b32 $0, $1", "=v,v" %v : (f32) -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %w, %q : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // A scheduling hint is not a barrier; the loads move above it.
  // CHECK-LABEL: gpu.func @sched_barrier_hint
  // CHECK:      scf.for {{.*}} : i32 {
  // CHECK-COUNT-2: llvm.load
  // CHECK-NEXT:   rocdl.sched.barrier
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
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // Both loops are marked. The inner one is unrolled first; the outer one moves only its own
  // top-level load.
  // CHECK-LABEL: gpu.func @nested
  // CHECK:      scf.for {{.*}} : i32 {
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.load
  // CHECK:        arith.muli
  // CHECK-NEXT:   arith.addi
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.load
  // CHECK:        scf.for {{.*}} : i32 {
  // CHECK-COUNT-2: llvm.load
  // CHECK:        scf.for
  // CHECK:        scf.for {{.*}} : i32 {
  // CHECK-COUNT-2: llvm.load
  // CHECK:        scf.for
  gpu.func @nested(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32, %m: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %j = %c0 to %n step %c1 : i32 {
      %pj = llvm.getelementptr %in[%j] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %init = llvm.load %pj : !llvm.ptr<1> -> f32
      %sum = scf.for %i = %c0 to %m step %c1 iter_args(%acc = %init) -> f32 : i32 {
        %k = arith.addi %i, %j : i32
        %p = llvm.getelementptr %in[%k] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        %v = llvm.load %p : !llvm.ptr<1> -> f32
        %s = arith.addf %acc, %v : f32
        scf.yield %s : f32
      } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
      %pout = llvm.getelementptr %out[%j] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %sum, %pout : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }
}
