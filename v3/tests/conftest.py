"""Reset kernel backend so tests don't leak numba/cuda state."""

import pytest

from src.kernels import set_backend


@pytest.fixture(autouse=True)
def _numpy_backend():
    set_backend("numpy")
    yield
    set_backend("numpy")
