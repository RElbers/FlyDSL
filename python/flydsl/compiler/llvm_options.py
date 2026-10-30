# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Scoped LLVM cl::opt control via context manager."""

from contextlib import contextmanager
from typing import Dict, Union

_fly_module = None


def _get_fly_module():
    global _fly_module
    if _fly_module is None:
        from .._mlir._mlir_libs import _mlirDialectsFly

        _fly_module = _mlirDialectsFly
    return _fly_module


# The values that open llvm_options scopes have set, per option, innermost last.
_scope_values: Dict[str, list] = {}


def _set_llvm_option(fly, name, value, *, return_old):
    if isinstance(value, bool):
        return fly.set_llvm_option_bool(name, value, return_old=return_old)
    if isinstance(value, int):
        return fly.set_llvm_option_int(name, value, return_old=return_old)
    if isinstance(value, str):
        return fly.set_llvm_option_str(name, value, return_old=return_old)
    raise TypeError(f"Unsupported type {type(value).__name__} for LLVM option '{name}'; use bool, int, or str")


@contextmanager
def llvm_options(opts: Dict[str, Union[bool, int, str]]):
    """Temporarily set LLVM cl::opt values, restoring originals on exit.

    On exit an option goes back to the value its enclosing ``llvm_options`` scope set. Without one, an
    option that was not set before is reset to its default and unset state rather than set to its old
    value: some options only take effect when set, and some default to unset. The values are tracked
    here instead of read back from LLVM, which can only read them as bool, int or string.
    """
    _fly = _get_fly_module()
    entered = []  # (name, value it had when set outside any scope, else None)
    try:
        for name, value in opts.items():
            set_outside = name not in _scope_values and _fly.llvm_option_occurrences(name) > 0
            old = _set_llvm_option(_fly, name, value, return_old=set_outside)
            _scope_values.setdefault(name, []).append(value)
            entered.append((name, old))
        yield
    finally:
        for name, old in reversed(entered):
            values = _scope_values[name]
            values.pop()
            if values:
                _set_llvm_option(_fly, name, values[-1], return_old=False)
                continue
            del _scope_values[name]
            if old is None:
                _fly.reset_llvm_option(name)
            else:
                _set_llvm_option(_fly, name, old, return_old=False)
