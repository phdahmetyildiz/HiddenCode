"""
Reset kernel backend so tests don't leak numba/cuda state.

Author: Cursor Grok 4.6 High Fast
"""

import pytest

from src.kernels import set_backend


@pytest.fixture(autouse=True)
def _numpy_backend():
    set_backend("numpy")
    yield
    set_backend("numpy")
