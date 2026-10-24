// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-prefetch-loops))' | FileCheck %s --implicit-check-not=fly.prefetch_depth

// Tests for fly-rocdl-prefetch-loops:
//   - a marked loop is unrolled by its depth and pipelined: masked prologue loads, a steady
//     state that issues the next group's loads before storing the current group, predicated
//     epilogue stores, and a serial remainder loop for the runtime trip count
//   - a loop with a barrier is left unchanged
//   - the marker is consumed in every case; unmarked loops are untouched

// CHECK-LABEL: gpu.func @scatter
// CHECK-COUNT-4: llvm.load
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

// CHECK-LABEL: gpu.func @with_barrier
// CHECK:      scf.for
// CHECK-NEXT:   llvm.getelementptr
// CHECK-NEXT:   llvm.load
// CHECK-NEXT:   gpu.barrier
// CHECK-NEXT:   llvm.getelementptr
// CHECK-NEXT:   llvm.store
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return

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

  gpu.func @with_barrier(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      gpu.barrier
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
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
