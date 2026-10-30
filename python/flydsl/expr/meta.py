# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import contextlib
import inspect
import os
import threading
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, overload

from .._mlir import ir
from ..utils import env

if TYPE_CHECKING:
    from typing import Any, Callable, ParamSpec, TypeVar

    _P = ParamSpec("_P")
    _R = TypeVar("_R")

__all__ = [
    "capture_user_location",
    "dsl_loc_tracing",
    "dsl_wrap_result",
    "tracing_context",
    "tracing_option",
]

# Package root for the ``flydsl`` Python package: ``.../python/flydsl``.
# Any frame whose file lives under this prefix is treated as DSL library code
# and skipped when locating the user's source position.
_FLYDSL_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@lru_cache(maxsize=1024)
def _is_framework_file(filename: str) -> bool:
    """True if *filename* belongs to DSL library code (or is not locatable)."""
    if not filename or filename[0] == "<":
        # ``<string>``, ``<frozen ...>`` and similar synthetic names are not
        # user source we can point at.
        return True
    return os.path.abspath(filename).startswith(_FLYDSL_PKG_ROOT)


_code_source_infos: dict = {}


def _code_source_info(code) -> tuple:
    """Absolute source path and per-code-unit positions of *code*, as ``inspect.getframeinfo`` finds them.

    ``getframeinfo`` resolves the path again and scans ``co_positions()`` up to the current code unit
    on every call, so capturing every op of a trace grows with the square of the traced kernel body's
    size. Python 3.10 has no ``co_positions``, so it gets no positions.

    Keyed by identity, not equality: code objects from different files compare equal when their
    bodies and line numbers match. The entry keeps its code object alive, so the id is not reused.
    """
    entry = _code_source_infos.get(id(code))
    if entry is None or entry[0] is not code:
        if len(_code_source_infos) >= 1024:
            _code_source_infos.clear()
        path = os.path.abspath(inspect.getsourcefile(code) or inspect.getfile(code))
        positions = tuple(code.co_positions()) if hasattr(code, "co_positions") else ()
        entry = _code_source_infos[id(code)] = (code, path, positions)
    return entry[1], entry[2]


# --------------------------------------------------------------------------- #
# Tracing Variable (thread-local)
# --------------------------------------------------------------------------- #
_tls = threading.local()


def _stack():
    s = getattr(_tls, "stack", None)
    if s is None:
        s = _tls.stack = []
    return s


@contextlib.contextmanager
def tracing_context(func=None, **options):
    """Push a tracing frame for the duration of the ``with`` block.

    *func*, when given, marks the call-site boundary used by
    ``capture_user_location``; a frame entered without one inherits the
    enclosing boundary.  Keyword *options* become ambient settings readable
    with ``tracing_option(key)``.  Every key passed is set, ``None`` included
    -- an explicit ``None`` shadows an enclosing setting.
    """
    stack = _stack()
    boundary = getattr(func, "__code__", None)
    if boundary is None and stack:
        boundary = stack[-1][0]
    stack.append((boundary, options))
    try:
        yield
    finally:
        stack = _stack()
        if stack:
            stack.pop()
        if not stack:
            # Release the cached locations, and with them the trace's MLIR context.
            _tls.loc_cache = None


def tracing_option(key, default=None):
    """Return the ambient value of *key*, innermost tracing frame first."""
    stack = getattr(_tls, "stack", None)
    if stack:
        for _, options in reversed(stack):
            if key in options:
                return options[key]
    return default


def file_location(filename: str, line: int, col: int = 0, context=None) -> ir.Location:
    ctx = context or ir.Context.current
    if filename and not filename.startswith("<"):
        filename = os.path.abspath(filename)
    return ir.Location.file(filename, line, col, context=ctx)


def capture_user_location() -> ir.Location:
    """Build a ``CallSiteLoc`` chain over the *user* frames.

    Walks up from the op-building site, skips DSL-library frames, and records
    every user frame from the innermost (where the op is written) up to the
    tracing boundary.
    """
    stack = getattr(_tls, "stack", None)
    boundary = stack[-1][0] if stack else None
    max_depth = env.debug.max_loc_depth
    ctx = ir.Context.current
    kept = []  # the recorded user frames, innermost first
    dropped = 0

    frame = inspect.currentframe().f_back
    try:
        while frame is not None:
            code = frame.f_code
            is_boundary = boundary is not None and code is boundary
            if not _is_framework_file(code.co_filename):
                if len(kept) < max_depth or is_boundary:
                    kept.append(frame)
            else:
                dropped += 1
            if is_boundary:
                break
            if len(kept) == max_depth:
                # The depth is full, so only the kernel frame can still be recorded: it is the top of the stack.
                frame = frame.f_back
                while frame is not None and frame.f_code is not boundary:
                    frame = frame.f_back
                if frame is not None and not _is_framework_file(boundary.co_filename):
                    kept.append(frame)
                break
            frame = frame.f_back

        if not stack:
            return _location_from_frames(kept, ctx)

        # Unrolled loops and shared helpers build ops at the same call sites again and again.
        cache = getattr(_tls, "loc_cache", None)
        if cache is None or cache[0] is not ctx:
            cache = _tls.loc_cache = (ctx, {}, [])
        key = []
        for kept_frame in kept:
            key.append(id(kept_frame.f_code))
            key.append(kept_frame.f_lasti)
        key = tuple(key)
        loc = cache[1].get(key)
        if loc is None:
            loc = cache[1][key] = _location_from_frames(kept, ctx)
            # Keep the keyed code objects alive, so their ids are not reused while the entry exists.
            cache[2].extend(kept_frame.f_code for kept_frame in kept)
        return loc
    finally:
        del frame, kept


def _location_from_frames(frames, ctx) -> ir.Location:
    locs = []
    for frame in frames:
        path, positions = _code_source_info(frame.f_code)
        # One position per 2-byte code unit; without one, fall back to f_lineno / col 0.
        pos = positions[frame.f_lasti // 2] if positions and frame.f_lasti >= 0 else None
        line = pos[0] if pos is not None and pos[0] is not None else frame.f_lineno
        col = pos[2] if pos is not None and pos[2] is not None else 0
        locs.append(ir.Location.file(path, line, col, context=ctx))
    if not locs:
        return ir.Location.unknown()
    callee, callers = locs[0], locs[1:]
    if not callers:
        return callee
    return ir.Location.callsite(callee, callers)


def dsl_loc_tracing(fn: "Callable[_P, _R]") -> "Callable[_P, _R]":
    """Attach a source ``Location`` to the op(s) a primitive builds.

    Location policy (single source of truth for the whole ``expr`` layer):

    * The location is the **full user call-site chain** -- a ``CallSiteLoc``
      from the innermost user frame (where the op is written) up to the
      boundary, each frame a ``FileLineColLoc`` (line + column).  A lone user
      frame collapses to a plain ``FileLineColLoc``.
    * It is captured **once** and entered as a dynamic ``with loc:`` scope, so
      every op the decorated function builds inherits it via
      ``Location.current``.
    """

    @wraps(fn)
    def wrapper(*args: "_P.args", **kwargs: "_P.kwargs") -> "_R":
        if getattr(_tls, "active_loc", None) is not None:
            # Already inside a captured scope
            return fn(*args, **kwargs)
        loc = capture_user_location()
        _tls.active_loc = loc
        try:
            with loc:
                return fn(*args, **kwargs)
        finally:
            _tls.active_loc = None

    return wrapper


@overload
def dsl_wrap_result(target: "Callable[_P, Any]") -> "Callable[_P, Any]": ...
@overload
def dsl_wrap_result(target: "type | None" = None) -> "Callable[[Callable[_P, Any]], Callable[_P, Any]]": ...
def dsl_wrap_result(target=None):
    """Wrap the op result(s) back into DslType values.

    - ``target=None`` (default): dispatch by the result's ``ir.Type``.
    - ``target=SomeClass``: force ``SomeClass(value)`` — useful when the result
      type cannot be uniquely determined from the ``ir.Type`` (vectors, …).

    Multi-value returns (tuples / lists) are wrapped element-wise.
    """

    def decorator(op, target):
        @wraps(op)
        def wrapper(*args, **kwargs):
            from .typing import as_dsl_value

            return as_dsl_value(op(*args, **kwargs), target)

        return wrapper

    if inspect.isfunction(target):
        return decorator(target, None)
    return lambda op: decorator(op, target)
