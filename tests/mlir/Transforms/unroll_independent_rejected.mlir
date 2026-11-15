// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-unroll-independent))' --mlir-print-local-scope | FileCheck %s --implicit-check-not=fly.assume_independent --implicit-check-not='<disable = true>>'
// RUN: %fly-opt %s --pass-pipeline='builtin.module(gpu.module(fly-rocdl-unroll-independent))' -verify-diagnostics -o /dev/null

// Loops the pass rejects: each keeps its body and its LLVM unroll hint, so it behaves as with plain
// unroll=N, and gets a remark saying why (checked by -verify-diagnostics). The marker is consumed.

gpu.module @unroll_independent_rejected {
  // CHECK-LABEL: gpu.func @barrier
  // CHECK:      scf.for
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.load
  // CHECK-NEXT:   gpu.barrier
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.store
  // CHECK-NEXT: } {loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
  gpu.func @barrier(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{assume_independent ignored (unroll=4), the loop is left to LLVM: it contains a barrier (in barrier)}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      gpu.barrier
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // CHECK-LABEL: gpu.func @counter_wait
  // CHECK:      rocdl.s.waitcnt
  // CHECK:      } {loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
  gpu.func @counter_wait(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    // expected-remark @below {{it waits on a memory counter (rocdl.s.waitcnt), which would also wait for the loads moved above it}}
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%a = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      rocdl.s.waitcnt 0
      %s = arith.addf %a, %v : f32
      scf.yield %s : f32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // Inline assembly with side effects is rejected whatever it contains: a wait, a barrier or a fence.
  // CHECK-LABEL: gpu.func @counter_wait_asm
  // CHECK:      llvm.inline_asm
  gpu.func @counter_wait_asm(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    // expected-remark @below {{it contains inline assembly with side effects ("s_waitcnt vmcnt(0)"), which may be a barrier, a wait or a fence}}
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%a = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      llvm.inline_asm has_side_effects "s_waitcnt vmcnt(0)", "" : () -> ()
      %s = arith.addf %a, %v : f32
      scf.yield %s : f32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // A barrier in inline assembly with no wait in the same string, as an `s_barrier` helper emits.
  // CHECK-LABEL: gpu.func @barrier_asm
  // CHECK:      scf.for
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.load
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.store
  // CHECK-NEXT:   llvm.inline_asm has_side_effects "s_barrier"
  // CHECK-NEXT: } {loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
  gpu.func @barrier_asm(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{it contains inline assembly with side effects ("s_barrier")}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
      llvm.inline_asm has_side_effects "s_barrier", "" : () -> ()
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // A cache invalidation in inline assembly, which acts as an acquire fence.
  // CHECK-LABEL: gpu.func @fence_asm
  // CHECK:      llvm.inline_asm has_side_effects "buffer_inv sc1"
  // CHECK-NEXT: } {loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
  gpu.func @fence_asm(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{it contains inline assembly with side effects ("buffer_inv sc1")}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
      llvm.inline_asm has_side_effects "buffer_inv sc1", "" : () -> ()
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // CHECK-LABEL: gpu.func @lds_copy
  // CHECK:      rocdl.load.to.lds
  gpu.func @lds_copy(%src: !llvm.ptr<7>, %dst: !llvm.ptr<3>, %in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{it contains a copy into or out of LDS, or an asynchronous copy}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      rocdl.load.to.lds %src, %dst, 4, 0, 0 : <7>
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %pout : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // CHECK-LABEL: gpu.func @printf
  // CHECK:      gpu.printf
  gpu.func @printf(%in: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{it contains a print or an assert}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      gpu.printf "%f", %v : f32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // CHECK-LABEL: gpu.func @seq_cst_load
  // CHECK:      llvm.load {{.*}} atomic seq_cst
  gpu.func @seq_cst_load(%flag: !llvm.ptr<1>, %in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{it contains a fence, a volatile access or an atomic with acquire or stronger ordering}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %f = llvm.load %flag atomic seq_cst {alignment = 4 : i64} : !llvm.ptr<1> -> i32
      %p = llvm.getelementptr %in[%f] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // CHECK-LABEL: gpu.func @volatile_load
  // CHECK:      llvm.load volatile
  gpu.func @volatile_load(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    // expected-remark @below {{it contains a fence, a volatile access or an atomic with acquire or stronger ordering}}
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load volatile %p : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // A buffer access with the volatile bit (bit 31) of its aux operand set.
  // CHECK-LABEL: gpu.func @volatile_buffer_load
  // CHECK:      scf.for
  // CHECK-NEXT:   rocdl.raw.ptr.buffer.load {{.*}}, 2147483648 : f32
  gpu.func @volatile_buffer_load(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    %rout = rocdl.make.buffer.rsrc %out, %c0_i16, %num_records, %flags : <1> to <8>
    // expected-remark @below {{it contains a fence, a volatile access or an atomic with acquire or stronger ordering}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 2147483648 : f32
      rocdl.raw.ptr.buffer.store %v, %rout, %i, %c0, 0 : f32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // A scalar buffer load with the volatile bit (bit 31) of its cache policy set.
  // CHECK-LABEL: gpu.func @volatile_scalar_buffer_load
  // CHECK:      scf.for
  // CHECK:        llvm.call_intrinsic "llvm.amdgcn.s.buffer.load.i32"
  gpu.func @volatile_scalar_buffer_load(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %volatile = arith.constant -2147483648 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    %rout = rocdl.make.buffer.rsrc %out, %c0_i16, %num_records, %flags : <1> to <8>
    %rin_int = llvm.ptrtoint %rin : !llvm.ptr<8> to i128
    %rin_vec = llvm.bitcast %rin_int : i128 to vector<4xi32>
    // expected-remark @below {{it contains a fence, a volatile access or an atomic with acquire or stronger ordering}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = llvm.call_intrinsic "llvm.amdgcn.s.buffer.load.i32"(%rin_vec, %i, %volatile) : (vector<4xi32>, i32, i32) -> i32
      rocdl.raw.ptr.buffer.store %v, %rout, %i, %c0, 0 : i32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // A scalar buffer load from the same address every iteration, of a buffer the loop writes.
  // CHECK-LABEL: gpu.func @invariant_scalar_buffer_load
  // CHECK:      scf.for
  // CHECK:        llvm.call_intrinsic "llvm.amdgcn.s.buffer.load.i32"
  gpu.func @invariant_scalar_buffer_load(%buf: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %r = rocdl.make.buffer.rsrc %buf, %c0_i16, %num_records, %flags : <1> to <8>
    %r_int = llvm.ptrtoint %r : !llvm.ptr<8> to i128
    %r_vec = llvm.bitcast %r_int : i128 to vector<4xi32>
    // expected-remark @below {{a load reads the same address in every iteration from a buffer the loop writes}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = llvm.call_intrinsic "llvm.amdgcn.s.buffer.load.i32"(%r_vec, %c0, %c0) : (vector<4xi32>, i32, i32) -> i32
      rocdl.raw.ptr.buffer.store %v, %r, %i, %c0, 0 : i32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 2 : i32>>}
    gpu.return
  }

  // CHECK-LABEL: gpu.func @private_memory
  // CHECK:      llvm.load %{{.*}} : !llvm.ptr<5>
  gpu.func @private_memory(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %slot = llvm.alloca %c1 x f32 : (i32) -> !llvm.ptr<5>
    // expected-remark @below {{it accesses private memory}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %old = llvm.load %slot : !llvm.ptr<5> -> f32
      %sum = arith.addf %old, %v : f32
      llvm.store %sum, %slot : f32, !llvm.ptr<5>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    %final = llvm.load %slot : !llvm.ptr<5> -> f32
    llvm.store %final, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // A store then a reload of what it stored, in the same iteration.
  // CHECK-LABEL: gpu.func @load_after_store
  // CHECK:      scf.for
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.load
  // CHECK-NEXT:   llvm.getelementptr
  // CHECK-NEXT:   llvm.store
  // CHECK-NEXT:   llvm.load
  gpu.func @load_after_store(%in: !llvm.ptr<1>, %tmp: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    // expected-remark @below {{a load follows a possible store in the same iteration and would move above it}}
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %tmp[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
      %w = llvm.load %q : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %w : f32
      scf.yield %s : f32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // An accumulator kept in memory: every iteration reads the address the previous one wrote.
  // CHECK-LABEL: gpu.func @invariant_accumulator
  // CHECK:      scf.for
  // CHECK-NEXT:   rocdl.raw.ptr.buffer.load
  // CHECK-NEXT:   arith.muli
  // CHECK-NEXT:   rocdl.raw.ptr.buffer.load
  gpu.func @invariant_accumulator(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32, %lane: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c0_i16 = arith.constant 0 : i16
    %num_records = arith.constant 4294967295 : i64
    %flags = arith.constant 159744 : i32
    %rin = rocdl.make.buffer.rsrc %in, %c0_i16, %num_records, %flags : <1> to <8>
    %rout = rocdl.make.buffer.rsrc %out, %c0_i16, %num_records, %flags : <1> to <8>
    // expected-remark @below {{a load reads the same address in every iteration from a buffer the loop writes}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %v = rocdl.raw.ptr.buffer.load %rin, %i, %c0, 0 : f32
      %off = arith.muli %lane, %c4 : i32
      %old = rocdl.raw.ptr.buffer.load %rout, %off, %c0, 0 : f32
      %sum = arith.addf %old, %v : f32
      rocdl.raw.ptr.buffer.store %sum, %rout, %off, %c0, 0 : f32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // The next iteration's offset comes from an atomic, which cannot move above the rest of the body.
  // CHECK-LABEL: gpu.func @address_from_atomic
  // CHECK:      llvm.atomicrmw
  gpu.func @address_from_atomic(%in: !llvm.ptr<1>, %counter: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %f0 = arith.constant 0.0 : f32
    // expected-remark @below {{a load's address depends on a value the loop computes after its loads (in the same iteration or carried from the previous one)}}
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%off = %c0, %acc = %f0) -> (i32, f32) : i32 {
      %p = llvm.getelementptr %in[%off] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %v : f32
      %next = llvm.atomicrmw add %counter, %c1 monotonic : !llvm.ptr<1>, i32
      scf.yield %next, %s : i32, f32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    llvm.store %r#1, %in : f32, !llvm.ptr<1>
    gpu.return
  }

  // CHECK-LABEL: gpu.func @no_load
  // CHECK:      } {loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
  gpu.func @no_load(%out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{its body has no load at the top level}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
      llvm.store %i, %q : i32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // A shuffle with lane-dependent bounds: the unrolled loop and its remainder would split lanes'
  // iterations differently, and the shuffle would see other lanes active.
  // CHECK-LABEL: gpu.func @shuffle_lane_bounds
  // CHECK:      gpu.shuffle
  // CHECK:      } {loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
  gpu.func @shuffle_lane_bounds(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %tid = gpu.thread_id x
    %lo = arith.index_cast %tid : index to i32
    // expected-remark @below {{it exchanges values across lanes (or calls an unknown op) and its bounds are not proven wave-uniform}}
    scf.for %i = %lo to %n step %c64 : i32 {
      %pin = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %pin : !llvm.ptr<1> -> f32
      %s, %valid = gpu.shuffle xor %v, %c1, %c64 : f32
      %pout = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %s, %pout : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // An MFMA combines the registers of all lanes, like a shuffle.
  // CHECK-LABEL: gpu.func @mfma_lane_bounds
  // CHECK:      rocdl.mfma
  gpu.func @mfma_lane_bounds(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c64 = arith.constant 64 : i32
    %zero = arith.constant dense<0.0> : vector<4xf32>
    %tid = gpu.thread_id x
    %lane = arith.index_cast %tid : index to i32
    // expected-remark @below {{it exchanges values across lanes}}
    %result = scf.for %i = %lane to %n step %c64 iter_args(%acc = %zero) -> vector<4xf32> : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, vector<4xf16>
      %a = llvm.load %p : !llvm.ptr<1> -> vector<4xf16>
      %d = rocdl.mfma.f32.16x16x16f16 %a, %a, %acc, 0, 0, none : (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
      scf.yield %d : vector<4xf32>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    llvm.store %result, %out : vector<4xf32>, !llvm.ptr<1>
    gpu.return
  }

  // A Python range counting down lowers to a negative step, which scf.for does not allow.
  // CHECK-LABEL: gpu.func @negative_step
  // CHECK:      } {loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
  gpu.func @negative_step(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %cm1 = arith.constant -1 : i32
    %f0 = arith.constant 0.0 : f32
    // expected-remark @below {{its step is not positive, which scf.for does not allow}}
    %r = scf.for %i = %n to %c0 step %cm1 iter_args(%acc = %f0) -> f32 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %s = arith.addf %acc, %v : f32
      scf.yield %s : f32
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    llvm.store %r, %out : f32, !llvm.ptr<1>
    gpu.return
  }

  // CHECK-LABEL: gpu.func @overflowing_step
  // CHECK:      } {loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
  gpu.func @overflowing_step(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i64) kernel {
    %c0 = arith.constant 0 : i64
    %step = arith.constant 4611686018427387904 : i64
    // expected-remark @below {{its step times N may overflow a signed 64-bit integer}}
    scf.for %i = %c0 to %n step %step : i64 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // The frontend never emits an unsigned loop.
  // CHECK-LABEL: gpu.func @unsigned
  // CHECK:      scf.for unsigned
  gpu.func @unsigned(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{it compares its bounds unsigned}}
    scf.for unsigned %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32>>}
    gpu.return
  }

  // The count must be from 2 to 32; unroll=1 is a disabling hint, and a missing hint has no count.
  // CHECK-LABEL: gpu.func @bad_counts
  // CHECK:      } {loop_annotation = #llvm.loop_annotation<unroll = <count = 33 : i32>>}
  // CHECK:      } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true, count = 1 : i32>>}
  // CHECK:      }
  // CHECK-NEXT: gpu.return
  gpu.func @bad_counts(%in: !llvm.ptr<1>, %out: !llvm.ptr<1>, %n: i32) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-remark @below {{assume_independent ignored (unroll=33), the loop is left to LLVM: it needs an unroll count from 2 to 32}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <count = 33 : i32>>}
    // expected-remark @below {{assume_independent ignored (unroll=1), the loop is left to LLVM: it needs an unroll count from 2 to 32}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.assume_independent, loop_annotation = #llvm.loop_annotation<unroll = <disable = true, count = 1 : i32>>}
    // expected-remark @below {{assume_independent ignored (unroll=0), the loop is left to LLVM: it needs an unroll count from 2 to 32}}
    scf.for %i = %c0 to %n step %c1 : i32 {
      %p = llvm.getelementptr %in[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      %v = llvm.load %p : !llvm.ptr<1> -> f32
      %q = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
      llvm.store %v, %q : f32, !llvm.ptr<1>
    } {fly.assume_independent}
    gpu.return
  }
}
