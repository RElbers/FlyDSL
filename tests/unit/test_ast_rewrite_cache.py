# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``ASTRewriter.transform`` rewrites a code object once and reuses it for every function built from it."""

import pytest

from flydsl.compiler.ast_rewriter import ASTRewriter

pytestmark = [pytest.mark.l0_backend_agnostic]


def _factory(scale):
    def scaled(x):
        return x * scale

    return scaled


def test_functions_from_one_factory_share_the_rewritten_code():
    double, triple = ASTRewriter.transform(_factory(2)), ASTRewriter.transform(_factory(3))

    assert double.__code__ is triple.__code__
    assert double.__code__ is not _factory(2).__code__
    assert (double(5), triple(5)) == (10, 15)
