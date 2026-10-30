# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""An inline struct type gets the same name in every process.

The name reaches the JIT disk-cache key through the closures that capture the type, so a name that changed
with the process's hash seed made a kernel miss the cache in every new process.
"""

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_PRINT_NAME = (
    "import flydsl.expr as fx; " "print(fx.Struct['count': fx.Int32, 'slots': fx.Array[fx.Float32, 4]].__name__)"
)


def _name_with_hash_seed(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-c", _PRINT_NAME], env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_inline_struct_name_does_not_depend_on_the_hash_seed():
    names = {_name_with_hash_seed(seed) for seed in (1, 2, 3)}

    assert len(names) == 1
    assert next(iter(names)).startswith("_DslStruct_")
