// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-unroll-independent))' --mlir-print-local-scope | FileCheck %s --implicit-check-not=fly.assume_independent --implicit-check-not=fly.pipeline
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-unroll-independent))' -verify-diagnostics -o /dev/null

// Loops marked fly.pipeline_lds: each staged buffer load of the grouped loop becomes an asynchronous
// DMA into a two-slot, wave-private LDS ring, read back right before its first use. The pipelined
// loop issues the next group's DMAs and closes their async group, then waits for the current group
// and reads it back; the epilogue waits for every group. Only an i32 slot offset and its LDS pointer
// are carried for the staged loads.

gpu.module @unroll_independent_pipeline_lds [#rocdl.target<chip = "gfx950">] {
  // One ring per staged loop, each inserted at the start of the module. @gather: 1 wave x 2 slots x
  // 2 loads x 64 lanes x 4 bytes. @sizes: 1 wave x 2 slots x (256 + 1024 + 1024) bytes, then 2 x 256
  // bytes. @scatter: 2 waves x 2 slots x 2 loads x 64 lanes x 16 bytes.
  // CHECK: llvm.mlir.global external @__fly_pipeline_ring_3() {addr_space = 3 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i8>
  // CHECK: llvm.mlir.global external @__fly_pipeline_ring_2() {addr_space = 3 : i32, alignment = 16 : i64, dso_local} : !llvm.array<512 x i8>
  // CHECK: llvm.mlir.global external @__fly_pipeline_ring_1() {addr_space = 3 : i32, alignment = 16 : i64, dso_local} : !llvm.array<4608 x i8>
  // CHECK: llvm.mlir.global external @__fly_pipeline_ring_0() {addr_space = 3 : i32, alignment = 16 : i64, dso_local} : !llvm.array<8192 x i8>

  // CHECK-LABEL: gpu.func @scatter
  // CHECK:      %[[TID:.*]] = rocdl.workitem.id.x : i32
  // CHECK:      %[[WAVE:.*]] = arith.divui %[[TID]], %{{.*}} : i32
  // CHECK-NEXT: %[[UWAVE:.*]] = rocdl.readfirstlane %[[WAVE]] : i32
  // CHECK:      %[[WAVE_OFF:.*]] = arith.muli %[[UWAVE]], %c4096_i32 : i32
  // CHECK-NEXT: %[[RING:.*]] = llvm.mlir.addressof @__fly_pipeline_ring_0 : !llvm.ptr<3>
  // CHECK-NEXT: %[[WAVE_PTR:.*]] = llvm.getelementptr %[[RING]][%[[WAVE_OFF]]]
  // CHECK:      %[[LO:.*]] = rocdl.mbcnt.lo
  // CHECK:      %[[LANE:.*]] = rocdl.mbcnt.hi %{{.*}}, %[[LO]]
  // CHECK:      %[[READ0:.*]] = arith.addi %{{.*}}, %c0_i32{{.*}} : i32
  // CHECK:      %[[READ1:.*]] = arith.addi %{{.*}}, %c1024_i32 : i32
  // CHECK:      scf.if
  // CHECK:        rocdl.raw.ptr.buffer.load.async.lds
  // CHECK:        rocdl.raw.ptr.buffer.load.async.lds
  // CHECK-NEXT:   rocdl.asyncmark
  // CHECK-NOT:    rocdl.wait.asyncmark
  // CHECK:        rocdl.s.waitcnt 3952
  // CHECK-NEXT:   %[[LOOP:.*]]:3 = scf.for {{.*}} iter_args(%[[SLOT:.*]] = %{{.*}}, %[[SLOT_PTR:.*]] = %{{.*}}, %{{.*}} = %{{.*}}) -> (i32, !llvm.ptr<3>, i32) : i32 {
  // CHECK-NEXT:     %[[NEXT_SLOT:.*]] = arith.xori %[[SLOT]], %c2048_i32 : i32
  // CHECK-NEXT:     %[[NEXT_PTR:.*]] = llvm.getelementptr %[[WAVE_PTR]][%[[SLOT]]]
  // CHECK:          rocdl.raw.ptr.buffer.load.async.lds %{{.*}}, %{{.*}}, %c16_i32
  // CHECK:          %[[DMA1_PTR:.*]] = llvm.getelementptr %[[NEXT_PTR]][%c1024_i32{{.*}}]
  // CHECK-NEXT:     rocdl.raw.ptr.buffer.load.async.lds %{{.*}}, %[[DMA1_PTR]], %c16_i32
  // CHECK-NEXT:     rocdl.asyncmark
  // CHECK-NEXT:     rocdl.sched.barrier none
  // CHECK-NEXT:     %[[R0_PTR:.*]] = llvm.getelementptr %[[SLOT_PTR]][%[[READ0]]]
  // CHECK-NEXT:     rocdl.wait.asyncmark 1
  // CHECK-NEXT:     %[[R0:.*]] = llvm.load %[[R0_PTR]] {alignment = 16 : i64} : !llvm.ptr<3> -> vector<4xf32>
  // CHECK-NEXT:     arith.mulf %[[R0]], %[[R0]]
  // CHECK:          rocdl.raw.ptr.buffer.store
  // CHECK-NEXT:     %[[R1_PTR:.*]] = llvm.getelementptr %[[SLOT_PTR]][%[[READ1]]]
  // CHECK-NEXT:     %[[R1:.*]] = llvm.load %[[R1_PTR]] {alignment = 16 : i64} : !llvm.ptr<3> -> vector<4xf32>
  // CHECK-NEXT:     arith.mulf %[[R1]], %[[R1]]
  // CHECK-NEXT:     rocdl.raw.ptr.buffer.store
  // CHECK-NEXT:     rocdl.sched.barrier none
  // CHECK-NEXT:     scf.yield %[[NEXT_SLOT]], %[[NEXT_PTR]],
  // CHECK-NEXT:   } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
  // CHECK-NOT:    rocdl.raw.ptr.buffer.load
  // CHECK:        rocdl.wait.asyncmark 0
  // CHECK-NEXT:   llvm.load %{{.*}} {alignment = 16 : i64} : !llvm.ptr<3> -> vector<4xf32>
  // CHECK:        llvm.load %{{.*}} {alignment = 16 : i64} : !llvm.ptr<3> -> vector<4xf32>
  // CHECK:      scf.for
  // CHECK-NEXT:   rocdl.raw.ptr.buffer.load %{{.*}} : vector<4xf32>
  gpu.func @scatter(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %lo: i32, %n: i32) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    %rout = rocdl.make.buffer.rsrc %out, %c0_i16, %num_records, %flags : <1> to <8>
    scf.for %i = %lo to %n step %c64 : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : vector<4xf32>
      %w = arith.mulf %v, %v : vector<4xf32>
      rocdl.raw.ptr.buffer.store %w, %rout, %i, %c0, 0 : vector<4xf32>
    } {fly.assume_independent, fly.pipeline, fly.pipeline_lds, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // A 4-byte load spaces the lanes 4 bytes apart, a 12- or 16-byte load 16 bytes, and unroll=1 runs
  // one iteration per group. A 3-D block linearizes the thread id.
  // CHECK-LABEL: gpu.func @sizes
  // CHECK:      %[[X:.*]] = rocdl.workitem.id.x : i32
  // CHECK-NEXT: %[[Y:.*]] = rocdl.workitem.id.y : i32
  // CHECK-NEXT: %[[Z:.*]] = rocdl.workitem.id.z : i32
  // CHECK:      llvm.mlir.addressof @__fly_pipeline_ring_1
  // CHECK:      scf.for
  // CHECK:        rocdl.raw.ptr.buffer.load.async.lds %{{.*}}, %{{.*}}, %c4_i32
  // CHECK:        rocdl.raw.ptr.buffer.load.async.lds %{{.*}}, %{{.*}}, %c12_i32
  // CHECK:        rocdl.raw.ptr.buffer.load.async.lds %{{.*}}, %{{.*}}, %c16_i32
  // CHECK:        rocdl.wait.asyncmark 1
  // CHECK-NEXT:   llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr<3> -> f32
  // CHECK:        llvm.load %{{.*}} {alignment = 16 : i64} : !llvm.ptr<3> -> vector<3xf32>
  // CHECK:        llvm.load %{{.*}} {alignment = 16 : i64} : !llvm.ptr<3> -> vector<4xi32>
  // CHECK:      llvm.mlir.addressof @__fly_pipeline_ring_2
  gpu.func @sizes(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %lo: i32, %n: i32) kernel attributes {known_block_size = array<i32: 16, 2, 2>} {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    %rout = rocdl.make.buffer.rsrc %out, %c0_i16, %num_records, %flags : <1> to <8>
    scf.for %i = %lo to %n step %c64 : i32 {
      %a = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : f32
      %b = rocdl.raw.ptr.buffer.load %rin, %i, %c64, 0 : vector<3xf32>
      %c = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : vector<4xi32>
      rocdl.raw.ptr.buffer.store %a, %rout, %i, %c0, 0 : f32
      rocdl.raw.ptr.buffer.store %b, %rout, %i, %c0, 0 : vector<3xf32>
      rocdl.raw.ptr.buffer.store %c, %rout, %i, %c0, 0 : vector<4xi32>
    } {fly.assume_independent, fly.pipeline, fly.pipeline_lds, loop_annotation = #llvm.loop_annotation<unroll = <count = 1 : i32, disable = true>>}
    scf.for %i = %lo to %n step %c64 : i32 {
      %a = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : f32
      rocdl.raw.ptr.buffer.store %a, %rout, %i, %c0, 0 : f32
    } {fly.assume_independent, fly.pipeline, fly.pipeline_lds, loop_annotation = #llvm.loop_annotation<unroll = <count = 1 : i32, disable = true>>}
    gpu.return
  }

  // A load whose value addresses another load stays in registers; the other load is staged.
  // CHECK-LABEL: gpu.func @gather
  // CHECK:      scf.for {{.*}} iter_args({{.*}}) -> (i32, !llvm.ptr<3>, i32) : i32 {
  // CHECK:        %[[J0:.*]] = rocdl.raw.ptr.buffer.load %{{.*}} : i32
  // CHECK:        rocdl.raw.ptr.buffer.load.async.lds %{{.*}}, %{{.*}}, %c4_i32{{.*}}, %[[J0]],
  // CHECK:        %[[J1:.*]] = rocdl.raw.ptr.buffer.load %{{.*}} : i32
  // CHECK:        rocdl.raw.ptr.buffer.load.async.lds %{{.*}}, %{{.*}}, %c4_i32{{.*}}, %[[J1]],
  // CHECK-NEXT:   rocdl.asyncmark
  // CHECK-NEXT:   rocdl.sched.barrier none
  gpu.func @gather(%in: !llvm.ptr<1>, %idx: !llvm.ptr<1>, %out: !llvm.ptr<1>, %lo: i32, %n: i32) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    %ridx = rocdl.make.buffer.rsrc %idx, %c0_i16, %num_records, %flags : <1> to <8>
    %rout = rocdl.make.buffer.rsrc %out, %c0_i16, %num_records, %flags : <1> to <8>
    // expected-remark @below {{pipeline="lds" keeps some loads in registers (unroll=2): 2 loads: its value feeds another load's address (in gather)}}
    scf.for %i = %lo to %n step %c64 : i32 {
      %j = rocdl.raw.ptr.buffer.load %ridx, %i, %c0, 0 : i32
      %v = rocdl.raw.ptr.buffer.load %rin, %j, %c0, 0 : f32
      rocdl.raw.ptr.buffer.store %v, %rout, %i, %c0, 0 : f32
    } {fly.assume_independent, fly.pipeline, fly.pipeline_lds, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // A scalar load cannot be DMA'd: it stays in a scalar register, and the buffer load it addresses
  // is staged.
  // CHECK-LABEL: gpu.func @scalar_gather
  // CHECK:      scf.for
  // CHECK:        %[[J:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.buffer.load.i32"
  // CHECK:        rocdl.raw.ptr.buffer.load.async.lds %{{.*}}, %{{.*}}, %c4_i32{{.*}}, %[[J]],
  gpu.func @scalar_gather(%idx: !llvm.ptr<1>, %in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
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
    // expected-remark @below {{pipeline="lds" keeps some loads in registers (unroll=1): 1 load: it is a scalar load, which stays in a scalar register (in scalar_gather)}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %off = arith.muli %i, %c4 : i32
      %j = llvm.call_intrinsic "llvm.amdgcn.s.buffer.load.i32"(%ridx_vec, %off, %c0) : (vector<4xi32>, i32, i32) -> i32
      %v = rocdl.raw.ptr.buffer.load %rin, %j, %c0, 0 : f32
      rocdl.raw.ptr.buffer.store %v, %rout, %off, %c0, 0 : f32
    } {fly.assume_independent, fly.pipeline, fly.pipeline_lds, loop_annotation = #llvm.loop_annotation<unroll = <count = 1 : i32, disable = true>>}
    gpu.return
  }

  // A value carried to the next iteration as loaded may still be in flight from LDS when the next
  // iteration's DMA overwrites its slot, so the loop falls back to registers. In the unrolled group,
  // the first iteration's value is unused.
  // CHECK-LABEL: gpu.func @carried_value
  // CHECK-NOT:  async
  // CHECK:      gpu.return
  gpu.func @carried_value(%in: !llvm.ptr<1>, %lo: i32, %n: i32) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %zero = arith.constant 0.0 : f32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    // expected-remark @below {{pipeline="lds" falls back to registers (unroll=2): no load can be staged (1 load: its value is unused; 1 load: its value is carried to the next iteration unchanged) (in carried_value)}}
    %last = scf.for %i = %lo to %n step %c64 iter_args(%prev = %zero) -> (f32) : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : f32
      scf.yield %v : f32
    } {fly.assume_independent, fly.pipeline, fly.pipeline_lds, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // Loops that cannot be staged at all are pipelined in registers.
  // CHECK-LABEL: gpu.func @fallbacks
  // CHECK-NOT:  async
  // CHECK:      gpu.return
  gpu.func @fallbacks(%in: !llvm.ptr<1>, %lo: i32, %n: i32) kernel attributes {known_block_size = array<i32: 1024, 1, 1>} {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    // 16 waves x 2 slots x 8 loads x 1 KiB.
    // expected-remark @below {{pipeline="lds" falls back to registers (unroll=8): the ring needs 262144 bytes, and the module's LDS would total 276992 of 163840 (in fallbacks)}}
    scf.for %i = %lo to %n step %c64 : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : vector<4xf32>
      rocdl.raw.ptr.buffer.store %v, %rin, %i, %c0, 0 : vector<4xf32>
    } {fly.assume_independent, fly.pipeline, fly.pipeline_lds, loop_annotation = #llvm.loop_annotation<unroll = <count = 8 : i32>>}
    // expected-remark @below {{pipeline="lds" falls back to registers (unroll=2): no load can be staged (2 loads: the DMA moves only 4, 12 or 16 bytes per lane) (in fallbacks)}}
    scf.for %i = %lo to %n step %c64 : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : vector<2xf32>
      rocdl.raw.ptr.buffer.store %v, %rin, %i, %c0, 0 : vector<2xf32>
    } {fly.assume_independent, fly.pipeline, fly.pipeline_lds, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    // expected-remark @below {{pipeline="lds" falls back to registers (unroll=2): no load can be staged (2 loads: its buffer resource is not proven wave-uniform) (in fallbacks)}}
    scf.for %i = %lo to %n step %c64 : i32 {
      %tid = rocdl.workitem.id.x : i32
      %base = llvm.getelementptr %in[%tid] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
      %r = rocdl.make.buffer.rsrc %base, %c0_i16, %num_records, %flags : <1> to <8>
      %v = rocdl.raw.ptr.buffer.load %r, %i, %c0, 0 : f32
      rocdl.raw.ptr.buffer.store %v, %rin, %i, %c0, 0 : f32
    } {fly.assume_independent, fly.pipeline, fly.pipeline_lds, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    // Only pipeline="lds" allows unroll=1.
    // expected-remark @below {{assume_independent ignored (unroll=1), the loop is left to LLVM: it needs an unroll count from 2 to 32 (in fallbacks)}}
    scf.for %i = %lo to %n step %c64 : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : f32
      rocdl.raw.ptr.buffer.store %v, %rin, %i, %c0, 0 : f32
    } {fly.assume_independent, fly.pipeline, loop_annotation = #llvm.loop_annotation<unroll = <count = 1 : i32, disable = true>>}
    gpu.return
  }

  // CHECK-LABEL: gpu.func @no_block_size
  // CHECK-NOT:  async
  // CHECK:      gpu.return
  gpu.func @no_block_size(%in: !llvm.ptr<1>, %lo: i32, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    // expected-remark @below {{pipeline="lds" falls back to registers (unroll=2): the kernel has no known_block_size, which sizes the ring (in no_block_size)}}
    scf.for %i = %lo to %n step %c64 : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : f32
      rocdl.raw.ptr.buffer.store %v, %rin, %i, %c0, 0 : f32
    } {fly.assume_independent, fly.pipeline, fly.pipeline_lds, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }
}

// Other targets: gfx942's DMA moves at most 4 bytes, and gfx1250's takes a per-lane LDS address.
gpu.module @other_targets [#rocdl.target<chip = "gfx942">] {
  // CHECK-LABEL: gpu.func @gfx942
  // CHECK-NOT:  async
  // CHECK:      gpu.return
  gpu.func @gfx942(%in: !llvm.ptr<1>, %lo: i32, %n: i32) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    // expected-remark @below {{pipeline="lds" falls back to registers (unroll=2): staging needs gfx950's 12- and 16-byte LDS DMA, and the target is gfx942 (in gfx942)}}
    scf.for %i = %lo to %n step %c64 : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : f32
      rocdl.raw.ptr.buffer.store %v, %rin, %i, %c0, 0 : f32
    } {fly.assume_independent, fly.pipeline, fly.pipeline_lds, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }
}

gpu.module @gfx1250_target [#rocdl.target<chip = "gfx1250">] {
  gpu.func @gfx1250(%in: !llvm.ptr<1>, %lo: i32, %n: i32) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    // expected-remark @below {{pipeline="lds" falls back to registers (unroll=2): staging needs gfx950's 12- and 16-byte LDS DMA, and the target is gfx1250 (in gfx1250)}}
    scf.for %i = %lo to %n step %c64 : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : f32
      rocdl.raw.ptr.buffer.store %v, %rin, %i, %c0, 0 : f32
    } {fly.assume_independent, fly.pipeline, fly.pipeline_lds, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }
}
