// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-prefetch-loops))' | FileCheck %s --implicit-check-not=fly.prefetch_depth
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-prefetch-loops))' -verify-diagnostics -o /dev/null

// Tests for fly-rocdl-prefetch-loops:
//   - a marked loop is unrolled by its depth and pipelined: guarded prologue loads, a steady
//     state that issues the next group's loads before storing the current group, predicated
//     epilogue stores, and a serial remainder loop for the runtime trip count
//   - a raw load in the prologue only issues under scf.if, since it can fault past the trip count;
//     a buffer load issues unguarded and its result is masked to zero
//   - a loop with a barrier or a print is left unchanged, and so is a loop that reads after a write
//     in the same iteration (global or LDS), since the read would move ahead of the write; a read
//     whose address comes from another read is still pipelined
//   - a loop with a lane-exchange op (gpu.shuffle) is pipelined as is when its bounds are
//     wave-uniform; when they depend on the lane, its main loop first ends at the wave's shortest
//     trip count (wave.reduce.umin, as i32 for 32-bit bounds and i64 for index) and a plain tail
//     loop runs each lane's remaining iterations
//   - a loop whose loads are all nested (here in an inner loop) is left unchanged, not unrolled
//   - every marked loop left unchanged gets a remark saying why (checked by -verify-diagnostics)
//   - the marker is consumed in every case; unmarked loops are untouched

// CHECK-LABEL: gpu.func @scatter
// CHECK-COUNT-4: scf.if {{.*}} -> (f32) {
// CHECK-NEXT:      llvm.load
// CHECK: scf.for {{.*}} iter_args(%[[V0:[a-z0-9_]+]] = {{.*}}) -> (f32, i32, f32, i32, f32, i32, f32)
// CHECK:   llvm.load
// CHECK:   llvm.store %[[V0]],
// CHECK-COUNT-3: llvm.load
// CHECK:   scf.yield
// CHECK-COUNT-4: scf.if
// CHECK: scf.for
// CHECK:   llvm.load
// CHECK:   llvm.store
// CHECK: gpu.return

// CHECK-LABEL: gpu.func @buffer_scatter
// CHECK-NOT: scf.if
// CHECK-COUNT-4: rocdl.raw.ptr.buffer.load
// CHECK: arith.select
// CHECK: scf.for
// CHECK: gpu.return

// CHECK-LABEL: gpu.func @with_barrier
// CHECK:      scf.for
// CHECK-NEXT:   llvm.getelementptr
// CHECK-NEXT:   llvm.load
// CHECK-NEXT:   gpu.barrier
// CHECK-NEXT:   llvm.getelementptr
// CHECK-NEXT:   llvm.store
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return

// CHECK-LABEL: gpu.func @with_shuffle_uniform
// CHECK-COUNT-4: scf.if {{.*}} -> (f32) {
// CHECK-NEXT:      llvm.load
// CHECK: scf.for
// CHECK:   gpu.shuffle
// CHECK: gpu.return

// CHECK-LABEL: gpu.func @with_shuffle
// CHECK:      gpu.thread_id
// CHECK:      %[[TRIPS:.*]] = llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"({{.*}}) : (i32, i32) -> i32
// CHECK:      arith.extui %[[TRIPS]] : i32 to i64
// CHECK-COUNT-4: scf.if {{.*}} -> (f32) {
// CHECK-NEXT:      llvm.load
// CHECK:      scf.for
// CHECK:        gpu.shuffle
// CHECK:      scf.for %{{.*}} = %{{.*}} to %[[SPLIT:[a-z0-9_]+]] step
// CHECK:      scf.for %{{.*}} = %[[SPLIT]] to %arg2 step
// CHECK-NEXT:   llvm.getelementptr
// CHECK-NEXT:   llvm.load
// CHECK-NEXT:   gpu.shuffle
// CHECK:      gpu.return

// CHECK-LABEL: gpu.func @with_shuffle_index
// CHECK:      llvm.call_intrinsic "llvm.amdgcn.wave.reduce.umin"({{.*}}) : (i64, i32) -> i64
// CHECK:      gpu.return

// CHECK-LABEL: gpu.func @gather
// CHECK-COUNT-4: scf.if {{.*}} -> (i32) {
// CHECK-NEXT:      llvm.load
// CHECK:      scf.for
// CHECK:      gpu.return

// CHECK-LABEL: gpu.func @store_then_reload
// CHECK:      scf.for
// CHECK-NEXT:   llvm.getelementptr
// CHECK-NEXT:   llvm.load
// CHECK-NEXT:   llvm.store
// CHECK-NEXT:   llvm.load
// CHECK-NEXT:   llvm.store
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return

// CHECK-LABEL: gpu.func @lds_round_trip
// CHECK:      scf.for
// CHECK-NEXT:   llvm.getelementptr
// CHECK-NEXT:   llvm.load
// CHECK-NEXT:   llvm.store
// CHECK-NEXT:   llvm.load
// CHECK-NEXT:   llvm.getelementptr
// CHECK-NEXT:   llvm.store
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return

// CHECK-LABEL: gpu.func @with_printf
// CHECK:      scf.for
// CHECK-NEXT:   llvm.getelementptr
// CHECK-NEXT:   llvm.load
// CHECK-NEXT:   gpu.printf
// CHECK-NEXT:   llvm.getelementptr
// CHECK-NEXT:   llvm.store
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return

// CHECK-LABEL: gpu.func @nested_reads
// CHECK:      scf.for
// CHECK-NEXT:   scf.for
// CHECK:      gpu.return

// CHECK-LABEL: gpu.func @unmarked
// CHECK:      scf.for
// CHECK-NEXT:   llvm.getelementptr
// CHECK-NEXT:   llvm.load
// CHECK-NEXT:   llvm.getelementptr
// CHECK-NEXT:   llvm.store
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return
gpu.module @prefetch_loops {
  gpu.func @scatter(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
    } {fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  gpu.func @buffer_scatter(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1>, i64 to <8>
    %rout = rocdl.make.buffer.rsrc %out, %c0_i16, %num_records, %flags : <1>, i64 to <8>
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : f32
      rocdl.raw.ptr.buffer.store %v, %rout, %i, %c0, 0 : f32
    } {fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  gpu.func @with_barrier(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{loop not pipelined (prefetch_depth=4): it contains a barrier}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      gpu.barrier
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
    } {fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  gpu.func @with_shuffle_uniform(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
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
    } {fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  gpu.func @with_shuffle(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %tid = gpu.thread_id x
    %lo = arith.index_cast %tid : index to i32
    scf.for %i = %lo to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %s, %valid = gpu.shuffle xor %v, %c1, %c64 : f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %s, %pout : f32, !llvm.ptr<1>
    } {fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  gpu.func @with_shuffle_index(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: index) kernel {
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %tid = gpu.thread_id x
    scf.for %i = %tid to %n step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %pin = llvm.getelementptr %in[%i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %s, %valid = gpu.shuffle xor %v, %c1_i32, %c64 : f32
      %pout = llvm.getelementptr %out[%i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %s, %pout : f32, !llvm.ptr<1>
    } {fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  gpu.func @gather(%idx: !llvm.ptr<1>, %in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pidx = llvm.getelementptr %idx[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      %j = llvm.load %pidx : !llvm.ptr<1> -> i32
      %pin = llvm.getelementptr %in[%j] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
    } {fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  gpu.func @store_then_reload(%in: !llvm.ptr<1>, %tmp: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{loop not pipelined (prefetch_depth=4): a load follows a possible store in the same iteration and would move ahead of it}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      llvm.store %v, %tmp : f32, !llvm.ptr<1>
      %w = llvm.load %tmp : !llvm.ptr<1> -> f32
      llvm.store %w, %p : f32, !llvm.ptr<1>
    } {fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  gpu.func @lds_round_trip(%in: !llvm.ptr<1>, %lds: !llvm.ptr<3>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{loop not pipelined (prefetch_depth=4): a load follows a possible store in the same iteration and would move ahead of it}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      llvm.store %v, %lds : f32, !llvm.ptr<3>
      %w = llvm.load %lds : !llvm.ptr<3> -> f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %w, %pout : f32, !llvm.ptr<1>
    } {fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  gpu.func @with_printf(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{loop not pipelined (prefetch_depth=4): it contains a print or an assert}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      gpu.printf "%f", %v : f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
    } {fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  gpu.func @nested_reads(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    // expected-remark @below {{loop not pipelined (prefetch_depth=4): its body has no load at the top level to issue early}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      scf.for %j = %c0 to %c2 step %c1 : i32 {
        %k = arith.addi %i, %j : i32
        %p = llvm.getelementptr %in[%k] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        %v = llvm.load %p : !llvm.ptr<1> -> f32
        %q = llvm.getelementptr %out[%k] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
        llvm.store %v, %q : f32, !llvm.ptr<1>
      }
    } {fly.prefetch_depth = 4 : i64}
    gpu.return
  }

  gpu.func @unmarked(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
    }
    gpu.return
  }
}
