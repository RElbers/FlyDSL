// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-convert-scf-to-cf | FileCheck %s --implicit-check-not=fly.iv_no_wrap --implicit-check-not=scf.

// fly-convert-scf-to-cf lowers SCF like upstream convert-scf-to-cf and marks each scf.for induction
// variable increment nsw (nuw for an unsigned loop), which scf.for's semantics guarantee; the
// loop's other LLVM attributes still reach its back-edge branch.

#unroll = #llvm.loop_unroll<count = 4>
#annotation = #llvm.loop_annotation<unroll = #unroll>

// CHECK-LABEL: func.func @signed_i32
// CHECK: %[[NEXT:.*]] = arith.addi %{{.*}}, %{{.*}} overflow<nsw> : i32
// CHECK: cf.br ^{{.*}}(%[[NEXT]] : i32)
func.func @signed_i32(%n: i32, %p: memref<?xf32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %j = arith.index_cast %i : i32 to index
    %v = memref.load %p[%j] : memref<?xf32>
    memref.store %v, %p[%j] : memref<?xf32>
  }
  return
}

// CHECK-LABEL: func.func @unsigned_i32
// CHECK: arith.cmpi ult
// CHECK: arith.addi %{{.*}}, %{{.*}} overflow<nuw> : i32
func.func @unsigned_i32(%n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  scf.for unsigned %i = %c0 to %n step %c1 : i32 {
  }
  return
}

// CHECK-LABEL: func.func @index_loop
// CHECK: arith.addi %{{.*}}, %{{.*}} overflow<nsw> : index
func.func @index_loop(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
  }
  return
}

// Both increments of a nested loop get the flag, and a loop annotation still reaches the back edge.
// CHECK-LABEL: func.func @nested_with_annotation
// CHECK-COUNT-2: arith.addi %{{.*}}, %{{.*}} overflow<nsw> : i32
// CHECK: cf.br {{.*}} {loop_annotation = #{{.*}}}
func.func @nested_with_annotation(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%a = %c0) -> i32 : i32 {
    %s = scf.for %j = %c0 to %n step %c1 iter_args(%b = %a) -> i32 : i32 {
      %t = arith.addi %b, %j : i32
      scf.yield %t : i32
    }
    scf.yield %s : i32
  } {loop_annotation = #annotation}
  return %r : i32
}

// scf.if lowers as before; only loop increments get a flag.
// CHECK-LABEL: func.func @branch
// CHECK: cf.cond_br
// CHECK: arith.addi %arg1, %arg1 : i32
func.func @branch(%c: i1, %x: i32) -> i32 {
  %r = scf.if %c -> i32 {
    %y = arith.addi %x, %x : i32
    scf.yield %y : i32
  } else {
    scf.yield %x : i32
  }
  return %r : i32
}
