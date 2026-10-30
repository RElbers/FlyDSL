# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Arch-dependent ``fx.rocdl`` ops follow the compile target (``ARCH``), not the installed device.

``FLYDSL_GPU_ARCH`` stands in for the installed device, so these run the same on any host.
"""

import pytest

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.jit_function import _create_mlir_context

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

# s_waitcnt(vmcnt=0) with every other counter left unwaited, per encoding.
RDNA_VMCNT0 = 1015
CDNA_VMCNT0 = 3952


@pytest.fixture
def cdna_device_compiling_for(monkeypatch):
    def target(arch):
        monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")
        monkeypatch.setenv("ARCH", arch)

    return target


def _s_waitcnt_vmcnt0():
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            fx.rocdl.s_waitcnt(vmcnt=0)
        return str(module.body.operations[0]).split()[-1]


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1201"])
def test_s_waitcnt_uses_the_target_encoding(cdna_device_compiling_for, arch):
    cdna_device_compiling_for(arch)
    assert _s_waitcnt_vmcnt0() == str(RDNA_VMCNT0)


def test_s_waitcnt_on_the_device_arch_without_arch_override(monkeypatch):
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx950")
    monkeypatch.delenv("ARCH", raising=False)
    assert _s_waitcnt_vmcnt0() == str(CDNA_VMCNT0)


@pytest.mark.parametrize("arch, family", [("gfx1100", "gfx11.wmma"), ("gfx1201", "gfx120x.wmma")])
def test_wmma_uses_the_target_arch(cdna_device_compiling_for, arch, family):
    cdna_device_compiling_for(arch)
    with _create_mlir_context():
        assert family in str(fx.rocdl.WMMA(16, 16, 16, fx.Float16))
