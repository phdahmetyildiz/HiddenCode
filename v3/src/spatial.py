"""Toroidal grid math (vectorized where it matters)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def wrap(x: NDArray, width: int) -> NDArray:
    return np.mod(x, width)


def toroidal_delta(a: NDArray, b: NDArray, size: int) -> NDArray:
    raw = np.mod(b - a, size)
    return np.where(raw > size // 2, raw - size, raw)


def toroidal_distance_sq(
    x1: NDArray, y1: NDArray,
    x2: NDArray, y2: NDArray,
    width: int, height: int,
) -> NDArray:
    dx = toroidal_delta(x1, x2, width)
    dy = toroidal_delta(y1, y2, height)
    return dx * dx + dy * dy


def move_toward(
    cx: NDArray[np.int32],
    cy: NDArray[np.int32],
    tx: NDArray[np.int32],
    ty: NDArray[np.int32],
    width: int,
    height: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    dx = toroidal_delta(cx, tx, width)
    dy = toroidal_delta(cy, ty, height)
    sx = np.sign(dx).astype(np.int32)
    sy = np.sign(dy).astype(np.int32)
    nx = wrap(cx + sx, width).astype(np.int32)
    ny = wrap(cy + sy, height).astype(np.int32)
    return nx, ny


# Eight neighbor offsets excluding (0,0)
_DIRS = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int32,
)


def random_steps(
    n: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    idx = rng.integers(0, 8, size=n)
    d = _DIRS[idx]
    return d[:, 0], d[:, 1]


def nearest_food(
    ax: NDArray[np.int32],
    ay: NDArray[np.int32],
    fx: NDArray[np.int32],
    fy: NDArray[np.int32],
    radius: int,
    width: int,
    height: int,
) -> tuple[NDArray[np.bool_], NDArray[np.int32], NDArray[np.int32]]:
    from src import kernels
    return kernels.nearest_food(ax, ay, fx, fy, radius, width, height)
