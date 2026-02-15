"""
Spatial utilities for the Evolution Simulator.

Provides toroidal (wrap-around) grid math: distance calculations,
coordinate wrapping, neighbor enumeration, and movement helpers.

All functions assume a 2D grid with dimensions (width, height) where
coordinates wrap: x % width, y % height.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def toroidal_wrap(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    """
    Wrap (x, y) coordinates to stay within grid bounds.

    Args:
        x, y: Raw coordinates (may be negative or >= dimensions).
        width, height: Grid dimensions.

    Returns:
        Wrapped (x, y) tuple within [0, width) and [0, height).
    """
    return x % width, y % height


def toroidal_delta(a: int, b: int, size: int) -> int:
    """
    Compute the signed shortest displacement from a to b on a toroidal axis.

    Returns a value in (-size/2, size/2].

    Args:
        a: Source coordinate.
        b: Target coordinate.
        size: Axis length.

    Returns:
        Signed shortest displacement (negative = go backwards).
    """
    raw = (b - a) % size
    if raw > size // 2:
        raw -= size
    return raw


def toroidal_distance_sq(
    x1: int, y1: int,
    x2: int, y2: int,
    width: int, height: int,
) -> int:
    """
    Compute squared Euclidean distance on a toroidal grid.

    Using squared distance avoids a sqrt and is sufficient for comparisons.

    Args:
        x1, y1: First position.
        x2, y2: Second position.
        width, height: Grid dimensions.

    Returns:
        Squared distance (int).
    """
    dx = toroidal_delta(x1, x2, width)
    dy = toroidal_delta(y1, y2, height)
    return dx * dx + dy * dy


def toroidal_distance(
    x1: int, y1: int,
    x2: int, y2: int,
    width: int, height: int,
) -> float:
    """
    Compute Euclidean distance on a toroidal grid.

    Args:
        x1, y1: First position.
        x2, y2: Second position.
        width, height: Grid dimensions.

    Returns:
        Distance (float).
    """
    return toroidal_distance_sq(x1, y1, x2, y2, width, height) ** 0.5


def toroidal_manhattan(
    x1: int, y1: int,
    x2: int, y2: int,
    width: int, height: int,
) -> int:
    """
    Compute Manhattan distance on a toroidal grid.

    Args:
        x1, y1: First position.
        x2, y2: Second position.
        width, height: Grid dimensions.

    Returns:
        Manhattan distance (int).
    """
    dx = abs(toroidal_delta(x1, x2, width))
    dy = abs(toroidal_delta(y1, y2, height))
    return dx + dy


def move_toward(
    cx: int, cy: int,
    tx: int, ty: int,
    width: int, height: int,
) -> tuple[int, int]:
    """
    Compute one step from (cx, cy) toward (tx, ty) on a toroidal grid.

    Moves in one of 8 cardinal directions (including diagonals).
    Chooses the direction that minimizes toroidal distance.

    Args:
        cx, cy: Current position.
        tx, ty: Target position.
        width, height: Grid dimensions.

    Returns:
        New (x, y) position after one step, wrapped.
    """
    dx = toroidal_delta(cx, tx, width)
    dy = toroidal_delta(cy, ty, height)

    # Clamp to -1, 0, +1
    step_x = (1 if dx > 0 else -1) if dx != 0 else 0
    step_y = (1 if dy > 0 else -1) if dy != 0 else 0

    return toroidal_wrap(cx + step_x, cy + step_y, width, height)


def random_move(
    cx: int, cy: int,
    width: int, height: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """
    Move one step in a random direction (8 cardinal + stay).

    Args:
        cx, cy: Current position.
        width, height: Grid dimensions.
        rng: Random generator.

    Returns:
        New (x, y) position, wrapped.
    """
    dx = rng.integers(-1, 2)  # -1, 0, or 1
    dy = rng.integers(-1, 2)
    return toroidal_wrap(cx + dx, cy + dy, width, height)


def cells_in_radius(
    cx: int, cy: int,
    radius: int,
    width: int, height: int,
) -> list[tuple[int, int]]:
    """
    Enumerate all grid cells within a Euclidean radius (circle) on a torus.

    Includes the center cell. Uses squared radius to avoid sqrt.

    Args:
        cx, cy: Center position.
        radius: Euclidean radius (integer).
        width, height: Grid dimensions.

    Returns:
        List of (x, y) tuples within the radius.
    """
    r_sq = radius * radius
    cells = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy <= r_sq:
                x, y = toroidal_wrap(cx + dx, cy + dy, width, height)
                cells.append((x, y))
    return cells


def cells_in_radius_set(
    cx: int, cy: int,
    radius: int,
    width: int, height: int,
) -> set[tuple[int, int]]:
    """Same as cells_in_radius but returns a set for O(1) lookups."""
    return set(cells_in_radius(cx, cy, radius, width, height))
