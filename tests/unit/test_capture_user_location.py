# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``capture_user_location`` records the file, line and column of each user frame an op is built under."""

import importlib.util
import sys

import pytest

from flydsl._mlir import ir
from flydsl.expr.meta import capture_user_location

pytestmark = [pytest.mark.l0_backend_agnostic]


def _source_position(marker):
    """Line and column of the capture call on the line ending in ``# <marker>`` (column 0 before 3.11)."""
    with open(_SOURCE) as f:
        for line, text in enumerate(f, 1):
            if text.rstrip().endswith(f"# {marker}"):
                col = text.index("capture_user_location") if sys.version_info >= (3, 11) else 0
                return line, col
    raise AssertionError(f"no line ends in # {marker}")


def _capture_two():
    first = capture_user_location()  # first
    second = [capture_user_location()][0]  # second
    return first, second


# The file the captures are reported in; it can differ from ``__file__`` when the checkout is reached
# through another path and the cached bytecode remembers the first one.
_SOURCE = _capture_two.__code__.co_filename


@pytest.fixture(autouse=True)
def _default_loc_depth(monkeypatch):
    # A depth of 1 drops the call-site chain these tests match on.
    monkeypatch.delenv("FLYDSL_DEBUG_MAX_LOC_DEPTH", raising=False)


def test_each_capture_records_its_own_line_and_column():
    with ir.Context(), ir.Location.unknown():
        locs = _capture_two()

    for loc, marker in zip(locs, ("first", "second")):
        line, col = _source_position(marker)
        assert str(loc).startswith(f'loc(callsite("{_SOURCE}":{line}:{col} at ')


def test_identical_code_in_two_files_keeps_each_file(tmp_path):
    # Code objects with the same body and line numbers compare equal across files.
    helpers = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.py"
        path.write_text("def helper(fn):\n    return fn()\n")
        spec = importlib.util.spec_from_file_location(f"_capture_location_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        helpers.append((module.helper, path))
    assert helpers[0][0].__code__ == helpers[1][0].__code__

    with ir.Context(), ir.Location.unknown():
        for helper, path in helpers:
            assert str(helper(capture_user_location)).startswith(f'loc(callsite("{path}":2:')


def _capture_in_a_loop_and_once_more():
    looped = []
    for _ in range(3):
        looped.append(capture_user_location())  # looped
    once = capture_user_location()  # once more
    return looped, once


def test_cached_captures_in_a_trace_keep_each_call_site():
    import flydsl.expr.meta as meta

    with ir.Context(), ir.Location.unknown():
        with meta.tracing_context(_capture_in_a_loop_and_once_more):
            looped, once = _capture_in_a_loop_and_once_more()
            assert meta._tls.loc_cache is not None

        assert meta._tls.loc_cache is None
        assert len({str(loc) for loc in looped}) == 1
        for loc, marker in ((looped[0], "looped"), (once, "once more")):
            line, col = _source_position(marker)
            assert str(loc).startswith(f'loc("{_SOURCE}":{line}:{col})')
