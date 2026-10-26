// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-promote-regmem-to-vectorssa | FileCheck %s

// Rebuilding an scf.for to thread register state must keep its discardable attributes:
// `loop_annotation` (from range(..., unroll=)) and `fly.prefetch_depth` are read by later passes.

// CHECK-LABEL: gpu.func @loop_attrs_survive_promotion
// CHECK: scf.for {{.*}} iter_args({{.*}}) -> (vector<1xi32>) {
// CHECK: } {fly.prefetch_depth = 4 : i64, loop_annotation = #loop_annotation}
gpu.module @promote_rmem_loop_attrs {
  gpu.func @loop_attrs_survive_promotion(%out: !fly.ptr<i32, global>, %n: index) kernel {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %reg = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i32, register>
    fly.ptr.store(%c0_i32, %reg) : (i32, !fly.ptr<i32, register>) -> ()
    scf.for %i = %c0 to %n step %c1 {
      %cur = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
      %next = arith.addi %cur, %c1_i32 : i32
      fly.ptr.store(%next, %reg) : (i32, !fly.ptr<i32, register>) -> ()
    } {fly.prefetch_depth = 4 : i64, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    %final = fly.ptr.load(%reg) : (!fly.ptr<i32, register>) -> i32
    fly.ptr.store(%final, %out) : (i32, !fly.ptr<i32, global>) -> ()
    gpu.return
  }
}
