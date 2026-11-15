// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-unroll-independent))' --mlir-print-local-scope | FileCheck %s --implicit-check-not=fly.assume_independent --implicit-check-not=fly.pipeline
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-unroll-independent))' -verify-diagnostics -o /dev/null

// Loops marked fly.pipeline: the grouped main loop is software-pipelined in two stages, so each
// iteration issues the next group's loads, then a scheduling barrier, then the current group's other
// ops, then a second barrier before the back edge. The pipelined loop runs inside a guard for at
// least one group, so its prologue and epilogue run unconditionally inside it. The serial remainder
// loop follows.

gpu.module @unroll_independent_pipeline {
  // CHECK-LABEL: gpu.func @scatter
  // CHECK:      %[[MAIN_UB:.*]] = arith.select
  // CHECK:      %[[HAS_GROUP:.*]] = arith.cmpi slt, %arg2, %[[MAIN_UB]] : i32
  // CHECK-NEXT: scf.if %[[HAS_GROUP]] {
  // CHECK-NOT:    scf.if
  // CHECK:        %[[V0:.*]] = rocdl.raw.ptr.buffer.load
  // CHECK:        %[[V1:.*]] = rocdl.raw.ptr.buffer.load
  // CHECK:        scf.for {{.*}} iter_args(%[[A0:.*]] = %[[V0]], %[[A1:.*]] = %[[V1]], %{{.*}} = %{{.*}}) -> (f32, f32, i32) : i32 {
  // CHECK:          %[[N0:.*]] = rocdl.raw.ptr.buffer.load
  // CHECK:          %[[N1:.*]] = rocdl.raw.ptr.buffer.load
  // CHECK-NEXT:     rocdl.sched.barrier none
  // CHECK-NEXT:     arith.mulf %[[A0]], %[[A0]]
  // CHECK:          rocdl.raw.ptr.buffer.store
  // CHECK:          rocdl.raw.ptr.buffer.store
  // CHECK-NEXT:     rocdl.sched.barrier none
  // CHECK-NEXT:     scf.yield %[[N0]], %[[N1]],
  // CHECK-NEXT:   } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  // CHECK-NOT:    rocdl.raw.ptr.buffer.load
  // CHECK:        rocdl.raw.ptr.buffer.store
  // CHECK:        rocdl.raw.ptr.buffer.store
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.for %{{.*}} = %[[MAIN_UB]] to %arg3 step %c64_i32 : i32 {
  gpu.func @scatter(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %lo: i32, %n: i32) kernel {
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
    } {fly.assume_independent, fly.pipeline, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // A pipelined loop needs its span plus one step to fit i32 as well, since the pipeliner computes
  // its epilogue's trip count in the loop's type.
  // CHECK-LABEL: gpu.func @span_must_fit
  // CHECK:      %[[WIDE_UB:.*]] = arith.addi %[[LB:[0-9]+]], %[[SPAN:[0-9]+]] : i64
  // CHECK:      %[[UB_FITS:.*]] = arith.cmpi sle, %[[WIDE_UB]], %c2147483647_i64 : i64
  // CHECK-NEXT: %[[SPAN_AND_STEP:.*]] = arith.addi %[[SPAN]], %c128_i64 : i64
  // CHECK-NEXT: %[[SPAN_FITS:.*]] = arith.cmpi sle, %[[SPAN_AND_STEP]], %c2147483647_i64 : i64
  // CHECK-NEXT: %[[FITS:.*]] = arith.andi %[[UB_FITS]], %[[SPAN_FITS]] : i1
  // CHECK:      arith.select %[[FITS]],
  gpu.func @span_must_fit(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %lo: i32, %n: i32) kernel {
    %c64 = arith.constant 64 : i32
    scf.for %i = %lo to %n step %c64 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.assume_independent, fly.pipeline, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // A reduction: the accumulator stays in the compute stage, and the loaded values cross the
  // back edge. The epilogue adds the last group.
  // CHECK-LABEL: gpu.func @reduce
  // CHECK:      scf.if %{{.*}} -> (f32) {
  // CHECK:        %[[R:.*]]:3 = scf.for {{.*}} iter_args(%[[ACC:.*]] = %cst, %[[A0:.*]] = %{{.*}}, %[[A1:.*]] = %{{.*}}) -> (f32, f32, f32) : i32 {
  // CHECK:          llvm.load
  // CHECK:          llvm.load
  // CHECK-NEXT:     rocdl.sched.barrier none
  // CHECK-NEXT:     %[[S0:.*]] = arith.addf %[[ACC]], %[[A0]]
  // CHECK-NEXT:     %[[S1:.*]] = arith.addf %[[S0]], %[[A1]]
  // CHECK-NEXT:     rocdl.sched.barrier none
  // CHECK-NEXT:     scf.yield %[[S1]],
  // CHECK:        %[[E0:.*]] = arith.addf %[[R]]#0, %[[R]]#1
  // CHECK-NEXT:   %[[E1:.*]] = arith.addf %[[E0]], %[[R]]#2
  // CHECK:        scf.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   scf.yield %cst : f32
  // CHECK-NEXT: }
  // CHECK:      scf.for
  gpu.func @reduce(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    } {fly.assume_independent, fly.pipeline, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // An address carried from one iteration to the next: its bump is in the load stage, so it runs a
  // group ahead with the loads.
  // CHECK-LABEL: gpu.func @carried_address
  // CHECK:        scf.for {{.*}} iter_args(%[[P:.*]] = %{{.*}}, %{{.*}} = %cst, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (!llvm.ptr<1>, f32, f32, f32) : i32 {
  // CHECK-NEXT:     %[[N0:.*]] = llvm.load %[[P]]
  // CHECK-NEXT:     %[[P1:.*]] = llvm.getelementptr %[[P]][1]
  // CHECK-NEXT:     %[[N1:.*]] = llvm.load %[[P1]]
  // CHECK-NEXT:     rocdl.sched.barrier none
  // CHECK:          %[[P2:.*]] = llvm.getelementptr %[[P1]][1]
  // CHECK-NEXT:     rocdl.sched.barrier none
  // CHECK-NEXT:     scf.yield %[[P2]], %{{.*}}, %[[N0]], %[[N1]]
  gpu.func @carried_address(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%p = %in, %acc = %f0) -> (!llvm.ptr<1>, f32) : i32 {
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %v : f32
      %next = llvm.getelementptr %p[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f32
      scf.yield %next, %s : !llvm.ptr<1>, f32
    } {fly.assume_independent, fly.pipeline, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    llvm.store %r#1, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // The pipeliner rejects a yield of the iter arg itself, which always holds its init value, so the
  // pass yields the init value instead.
  // CHECK-LABEL: gpu.func @unchanged_iter_arg
  // CHECK:        scf.for {{.*}} iter_args(%{{.*}} = %cst, {{.*}}) -> (f32, f32, i32, f32) : i32 {
  // CHECK:          rocdl.sched.barrier none
  // CHECK:          scf.yield %cst,
  gpu.func @unchanged_iter_arg(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
      scf.yield %acc : f32
    } {fly.assume_independent, fly.pipeline, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // Constant bounds with three groups and a remainder of one: the guard and the bounds fold.
  // CHECK-LABEL: gpu.func @constant_bounds
  // CHECK:      scf.if %true{{.*}} {
  // CHECK:        scf.for %{{.*}} = %c0_i32 to %{{.*}} step %c2_i32 {{.*}} : i32 {
  // CHECK:      scf.for %{{.*}} = %c6_i32 to %c7_i32 step %c1_i32 : i32 {
  gpu.func @constant_bounds(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c7 = arith.constant 7 : i32
    scf.for %i = %c0 to %c7 step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.assume_independent, fly.pipeline, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // A loop the grouping rules reject drops both markers and keeps its LLVM unroll hint.
  // CHECK-LABEL: gpu.func @rejected
  // CHECK:      gpu.barrier
  // CHECK-NEXT: } {loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
  gpu.func @rejected(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{assume_independent ignored (unroll=2), the loop is left to LLVM: it contains a barrier (in rejected)}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
      gpu.barrier
    } {fly.assume_independent, fly.pipeline, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // Without fly.pipeline the loop is only grouped.
  // CHECK-LABEL: gpu.func @grouped_only
  // CHECK-NOT:  scf.if
  // CHECK-NOT:  rocdl.sched.barrier
  // CHECK:      gpu.return
  gpu.func @grouped_only(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }
}

// On gfx9, loads and stores share vmcnt, and the pass waits for the prologue's loads before the
// pipelined loop, so that LLVM's wait at the loop's top only covers the back edge. A scalar buffer
// load, as fx.rocdl.s_buffer_load emits it, is a load like the others.
gpu.module @gfx9_flush [#rocdl.target<chip = "gfx950">] {
  // CHECK-LABEL: gpu.func @scalar_gather
  // CHECK:      scf.if
  // CHECK:        llvm.call_intrinsic "llvm.amdgcn.s.buffer.load.i32"
  // CHECK:        rocdl.raw.ptr.buffer.load
  // CHECK:        llvm.call_intrinsic "llvm.amdgcn.s.buffer.load.i32"
  // CHECK:        rocdl.raw.ptr.buffer.load
  // CHECK-NOT:    rocdl.raw.ptr.buffer.store
  // CHECK:        rocdl.s.waitcnt 3952
  // CHECK-NEXT:   scf.for
  // CHECK:          %[[J0:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.buffer.load.i32"
  // CHECK-NEXT:     rocdl.raw.ptr.buffer.load %{{.*}}, %[[J0]],
  // CHECK:          %[[J1:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.buffer.load.i32"
  // CHECK-NEXT:     rocdl.raw.ptr.buffer.load %{{.*}}, %[[J1]],
  // CHECK-NEXT:     rocdl.sched.barrier none
  // CHECK-NEXT:     rocdl.raw.ptr.buffer.store
  // CHECK-NEXT:     rocdl.raw.ptr.buffer.store
  // CHECK-NEXT:     rocdl.sched.barrier none
  // CHECK-NOT:  rocdl.s.waitcnt
  // CHECK:      gpu.return
  gpu.func @scalar_gather(%idx: !llvm.ptr<1>, %in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %ridx = rocdl.make.buffer.rsrc %idx, %c0_i16, %num_records, %flags : <1> to <8>
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    %rout = rocdl.make.buffer.rsrc %out, %c0_i16, %num_records, %flags : <1> to <8>
    %ridx_int = llvm.ptrtoint %ridx : !llvm.ptr<8> to i128
    %ridx_vec = llvm.bitcast %ridx_int : i128 to vector<4xi32>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %off = arith.muli %i, %c4 : i32
      %j = llvm.call_intrinsic "llvm.amdgcn.s.buffer.load.i32"(%ridx_vec, %off, %c0) : (vector<4xi32>, i32, i32) -> i32
      %v = rocdl.raw.ptr.buffer.load %rin, %j, %c0, 0 : f32
      rocdl.raw.ptr.buffer.store %v, %rout, %off, %c0, 0 : f32
    } {fly.assume_independent, fly.pipeline, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // A loop that is only grouped gets no flush.
  // CHECK-LABEL: gpu.func @grouped_gfx9
  // CHECK-NOT:  rocdl.s.waitcnt
  // CHECK:      gpu.return
  gpu.func @grouped_gfx9(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }
}

// gfx10 and later count stores separately, and gfx12 splits vmcnt: no flush.
gpu.module @gfx1250_no_flush [#rocdl.target<chip = "gfx1250">] {
  // CHECK-LABEL: gpu.func @pipelined_gfx1250
  // CHECK:      scf.if
  // CHECK-NOT:  rocdl.s.waitcnt
  // CHECK:      gpu.return
  gpu.func @pipelined_gfx1250(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.assume_independent, fly.pipeline, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }
}
