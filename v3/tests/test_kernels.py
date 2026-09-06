"""
Kernel backends: numpy vs numba parity; CUDA skipped if no GPU.

Author: Cursor Grok 4.6 High Fast
"""

import numpy as np
import pytest

from src import kernels
from src.engine import popcount32


def test_resolve_unknown_is_numpy():
    assert kernels.resolve_backend("nope") == "numpy"


def test_popcount32_known():
    x = np.array([0, 1, 0xFFFFFFFF], dtype=np.uint32)
    kernels.set_backend("numpy")
    got = kernels.popcount32(x)
    assert list(got) == [0, 1, 32]


def test_nearest_food_empty():
    ax = np.array([1, 2], dtype=np.int32)
    ay = np.array([1, 2], dtype=np.int32)
    fx = np.array([], dtype=np.int32)
    fy = np.array([], dtype=np.int32)
    in_r, tx, ty = kernels.nearest_food(ax, ay, fx, fy, 5, 20, 20)
    assert not np.any(in_r)


@pytest.mark.skipif(not kernels.numba_available(), reason="numba not installed")
def test_numba_popcount_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2**32, size=64, dtype=np.uint32)
    kernels.set_backend("numpy")
    a = kernels.popcount32_numpy(x)
    kernels.set_backend("numba")
    b = kernels.popcount32(x)
    assert np.array_equal(a, b)


@pytest.mark.skipif(not kernels.numba_available(), reason="numba not installed")
def test_numba_nearest_food_matches_numpy():
    rng = np.random.default_rng(1)
    ax = rng.integers(0, 40, size=30, dtype=np.int32)
    ay = rng.integers(0, 40, size=30, dtype=np.int32)
    fx = rng.integers(0, 40, size=12, dtype=np.int32)
    fy = rng.integers(0, 40, size=12, dtype=np.int32)
    kernels.set_backend("numpy")
    a = kernels.nearest_food_numpy(ax, ay, fx, fy, 8, 40, 40)
    kernels.set_backend("numba")
    b = kernels.nearest_food(ax, ay, fx, fy, 8, 40, 40)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])


@pytest.mark.skipif(not kernels.cuda_available(), reason="no CUDA GPU")
def test_cuda_popcount_matches_numpy():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 2**32, size=300, dtype=np.uint32)
    kernels.set_backend("numpy")
    a = kernels.popcount32_numpy(x)
    kernels.set_backend("cuda")
    b = kernels.popcount32(x)
    assert np.array_equal(a, b)


@pytest.mark.skipif(not kernels.cuda_available(), reason="no CUDA GPU")
def test_cuda_nearest_food_matches_numpy():
    rng = np.random.default_rng(3)
    ax = rng.integers(0, 40, size=300, dtype=np.int32)
    ay = rng.integers(0, 40, size=300, dtype=np.int32)
    fx = rng.integers(0, 40, size=20, dtype=np.int32)
    fy = rng.integers(0, 40, size=20, dtype=np.int32)
    kernels.set_backend("numpy")
    a = kernels.nearest_food_numpy(ax, ay, fx, fy, 8, 40, 40)
    kernels.set_backend("cuda")
    b = kernels.nearest_food(ax, ay, fx, fy, 8, 40, 40)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert np.array_equal(a[2], b[2])


def test_cuda_request_falls_back_without_gpu():
    if kernels.cuda_available():
        pytest.skip("GPU present")
    assert kernels.resolve_backend("cuda") in ("numba", "numpy")
    assert kernels.resolve_backend("numba_cuda") in ("numba", "numpy")


def test_apply_drain_clips():
    energy = np.array([1.0, 0.01, 0.5], dtype=np.float32)
    weight = np.ones(3, dtype=np.float32)
    speed = np.ones(3, dtype=np.float32)
    kernels.set_backend("numpy")
    kernels.apply_drain(energy, weight, speed, 0.5, 0.0)
    assert energy[0] == pytest.approx(0.5)
    assert energy[1] == pytest.approx(0.0)


def test_engine_popcount_wrapper():
    x = np.array([0xF], dtype=np.uint32)
    assert int(popcount32(x)[0]) == 4
