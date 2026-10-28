# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for the DSL compile-error formatting and excepthook filtering.

These cover message + caret rendering, ``sys.excepthook`` frame filtering,
the ``FLYDSL_DEBUG_SHOW_STACKTRACE`` escape hatch, and MLIR diagnostic forwarding.
No GPU execution is involved.
"""

import sys
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import flydsl.compiler.diagnostics as diagnostics
from flydsl._mlir import ir
from flydsl.compiler.diagnostics import DiagRecord, DSLCompileError, SourceFrame, install_excepthook

pytestmark = [pytest.mark.l0_backend_agnostic]


# --------------------------------------------------------------------------- #
# DSLCompileError message / caret formatting
# --------------------------------------------------------------------------- #
def test_plain_message_without_diagnostics():
    err = DSLCompileError("pipeline failed")
    assert str(err) == "pipeline failed"


def test_diagnostics_without_message_fall_back_to_plain():
    # A record carrying no message must not produce an empty "DSL Traceback" block.
    err = DSLCompileError("verification failed", diagnostics=[DiagRecord(message="", chain=[])])
    assert str(err) == "verification failed"


def test_message_renders_source_snippet_and_caret(tmp_path):
    src_file = tmp_path / "user_kernel.py"
    # Indented by 4 spaces; the offending span is ``compute(a, b)``.
    src_file.write_text("def k():\n    x = compute(a, b)\n")

    # col=8 points at the start of ``compute`` (0-based within the raw line),
    # end_col=21 at the closing paren -> a 13-char span.
    frame = SourceFrame(filename=str(src_file), line=2, col=8, end_col=21)
    err = DSLCompileError("op failed", diagnostics=[DiagRecord(message="bad op", chain=[frame])])
    text = str(err)

    assert "bad op" in text
    assert "DSL Traceback (most recent operation last):" in text
    assert f'File "{src_file}", line 2' in text
    # The snippet is stripped of its leading indentation.
    assert "    x = compute(a, b)" in text
    # Caret column is offset by the stripped indentation (8 - 4 = 4) and the
    # caret width matches the span (21 - 8 = 13).
    assert "    " + " " * 4 + "^" * 13 in text


def test_chain_is_printed_outermost_first(tmp_path):
    src_file = tmp_path / "chain.py"
    src_file.write_text("outer_call()\ninner_call()\n")

    # chain is innermost-first; the rendered traceback must end with the innermost.
    inner = SourceFrame(filename=str(src_file), line=2, col=0, end_col=1)
    outer = SourceFrame(filename=str(src_file), line=1, col=0, end_col=1)
    err = DSLCompileError("x", diagnostics=[DiagRecord(message="m", chain=[inner, outer])])
    text = str(err)

    assert text.index("line 1") < text.index("line 2")


# --------------------------------------------------------------------------- #
# install_excepthook: frame filtering + escape hatch
# --------------------------------------------------------------------------- #
@pytest.fixture
def fresh_excepthook(monkeypatch):
    """Reset the idempotency guard and restore ``sys.excepthook`` afterwards."""
    saved = sys.excepthook
    monkeypatch.setattr(diagnostics, "_dsl_excepthook_installed", False)
    yield
    sys.excepthook = saved


def _make_traceback(err):
    """Build a real traceback that passes through a 'framework' then a 'user' file."""
    g = {"err": err}
    exec(compile("def framework_call():\n    raise err\n", "/fake/framework/lib.py", "exec"), g)
    exec(compile("def user_entry():\n    framework_call()\n", "/fake/user/app.py", "exec"), g)
    try:
        g["user_entry"]()
    except DSLCompileError as caught:
        return caught.__traceback__


def test_excepthook_filters_framework_frames(monkeypatch, capsys, fresh_excepthook):
    # Treat only the synthetic framework path as DSL-internal.
    monkeypatch.setattr(diagnostics, "_is_framework_file", lambda fn: "/fake/framework/" in fn)
    monkeypatch.setenv("FLYDSL_DEBUG_SHOW_STACKTRACE", "0")

    previous_called = []
    sys.excepthook = lambda *a: previous_called.append(a)
    install_excepthook()
    hook = sys.excepthook

    err = DSLCompileError("boom")
    tb = _make_traceback(err)
    hook(DSLCompileError, err, tb)

    out = capsys.readouterr().err
    assert "/fake/user/app.py" in out  # user frame kept
    assert "/fake/framework/lib.py" not in out  # framework frame filtered out
    assert "DSLCompileError: boom" in out
    assert "-" * 40 in out  # rule separating the user stack from the DSL error
    assert not previous_called  # custom rendering, not the raw traceback


def test_show_stacktrace_uses_raw_traceback(monkeypatch, capsys, fresh_excepthook):
    monkeypatch.setenv("FLYDSL_DEBUG_SHOW_STACKTRACE", "1")

    previous_called = []
    sys.excepthook = lambda *a: previous_called.append(a)
    install_excepthook()
    hook = sys.excepthook

    err = DSLCompileError("boom")
    tb = _make_traceback(err)
    hook(DSLCompileError, err, tb)

    assert previous_called  # escape hatch delegates to the original hook
    assert capsys.readouterr().err == ""  # no custom rendering emitted


def test_excepthook_passes_through_non_dsl_errors(monkeypatch, capsys, fresh_excepthook):
    monkeypatch.setenv("FLYDSL_DEBUG_SHOW_STACKTRACE", "0")

    previous_called = []
    sys.excepthook = lambda *a: previous_called.append(a)
    install_excepthook()
    hook = sys.excepthook

    err = ValueError("not a dsl error")
    hook(ValueError, err, None)

    assert previous_called  # non-DSL errors delegate to the original hook
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------- #
# location_chain: MLIR locations -> source frames
# --------------------------------------------------------------------------- #
def test_location_chain_resolves_callsite_name_and_fused_locations():
    with ir.Context():
        user = ir.Location.file("user_kernel.py", 7, 4)
        caller = ir.Location.file("caller.py", 3, 1)
        user_frame = SourceFrame("user_kernel.py", 7, 4, 4)
        caller_frame = SourceFrame("caller.py", 3, 1, 1)

        assert diagnostics.location_chain(ir.Location.callsite(user, [caller])) == [user_frame, caller_frame]
        assert diagnostics.location_chain(ir.Location.name("op", user)) == [user_frame]
        assert diagnostics.location_chain(ir.Location.fused([user, caller])) == [user_frame, caller_frame]


def test_location_chain_skips_synthetic_sources():
    with ir.Context():
        assert diagnostics.location_chain(ir.Location.file("<string>", 20, 1)) == []


def test_location_chain_skips_unknown_locations():
    with ir.Context():
        assert diagnostics.location_chain(ir.Location.unknown()) == []


def test_verifier_error_keeps_the_source_chain():
    # An MLIRError's diagnostics carry plain Locations, not the typed subclasses the bindings build.
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse("func.func @f() { return }")
        loc = ir.Location.callsite(ir.Location.file("a.py", 7, 4), [ir.Location.file("b.py", 3, 1)])
        with ir.InsertionPoint.at_block_begin(module.body.operations[0].regions[0].blocks[0]):
            ir.Operation.create("arith.addi", results=[ir.IntegerType.get_signless(32)], loc=loc)
        with pytest.raises(ir.MLIRError) as raised:
            module.operation.verify()

    records = diagnostics.diag_records_from_mlir_error(raised.value)
    assert records[0].chain == [SourceFrame("a.py", 7, 4, 4), SourceFrame("b.py", 3, 1, 1)]


def _report(severity, location):
    """Pass a diagnostic to the handler dsl_ir_diagnostics attaches, as MLIR would."""
    ctx = Mock()
    with diagnostics.dsl_ir_diagnostics(ctx):
        callback = ctx.attach_diagnostic_handler.call_args.args[0]
        assert callback(SimpleNamespace(severity=severity, message="not pipelined", location=location))


@pytest.mark.parametrize("severity", [ir.DiagnosticSeverity.WARNING, ir.DiagnosticSeverity.REMARK])
def test_ir_warning_is_deferred_until_the_block_exits(severity):
    ctx = Mock()
    handler = ctx.attach_diagnostic_handler.return_value
    with ir.Context(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with diagnostics.dsl_ir_diagnostics(ctx) as errors:
            callback = ctx.attach_diagnostic_handler.call_args.args[0]
            location = ir.Location.file("user_kernel.py", 12, 8)
            assert callback(SimpleNamespace(severity=severity, message="not pipelined", location=location))
            assert caught == []
            assert errors == []
            handler.detach.assert_not_called()
    handler.detach.assert_called_once()
    assert len(caught) == 1


def test_ir_warning_points_at_the_source_line():
    with ir.Context(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _report(ir.DiagnosticSeverity.REMARK, ir.Location.file("user_kernel.py", 12, 8))
    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert str(caught[0].message) == "not pipelined"
    assert (caught[0].filename, caught[0].lineno) == ("user_kernel.py", 12)


def test_ir_warning_without_source_suggests_debug_info():
    with ir.Context(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _report(ir.DiagnosticSeverity.REMARK, ir.Location.unknown())
    assert len(caught) == 1
    assert str(caught[0].message) == "not pipelined (set FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 to see the source line)"
    assert caught[0].filename == __file__


def test_ir_warning_filtered_as_error_raises_after_detach():
    # A warning raised inside the MLIR callback would be swallowed by the binding.
    ctx = Mock()
    handler = ctx.attach_diagnostic_handler.return_value
    with ir.Context(), warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="not pipelined"):
            with diagnostics.dsl_ir_diagnostics(ctx):
                callback = ctx.attach_diagnostic_handler.call_args.args[0]
                location = ir.Location.file("user_kernel.py", 12, 8)
                assert callback(
                    SimpleNamespace(severity=ir.DiagnosticSeverity.REMARK, message="not pipelined", location=location)
                )
                handler.detach.assert_not_called()
    handler.detach.assert_called_once()


@pytest.mark.parametrize("action", ["always", "error"])
def test_ir_warning_does_not_replace_compile_error(action):
    ctx = Mock()
    with ir.Context(), warnings.catch_warnings(record=True):
        warnings.simplefilter(action)
        location = ir.Location.file("user_kernel.py", 12, 8)
        with pytest.raises(DSLCompileError, match="bad op") as raised:
            with diagnostics.dsl_ir_diagnostics(ctx) as errors:
                callback = ctx.attach_diagnostic_handler.call_args.args[0]
                assert callback(
                    SimpleNamespace(severity=ir.DiagnosticSeverity.REMARK, message="not pipelined", location=location)
                )
                assert callback(
                    SimpleNamespace(severity=ir.DiagnosticSeverity.ERROR, message="bad op", location=location)
                )
                assert not callback(SimpleNamespace(severity=ir.DiagnosticSeverity.NOTE))
                raise DSLCompileError("pipeline failed", diagnostics=errors)

        assert raised.value.diagnostics == [DiagRecord("bad op", [SourceFrame("user_kernel.py", 12, 8, 8)])]
        ctx.attach_diagnostic_handler.return_value.detach.assert_called_once()
