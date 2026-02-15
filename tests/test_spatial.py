"""
Unit tests for spatial utilities (toroidal grid math).

Tests cover:
- Toroidal wrapping
- Toroidal delta (signed shortest displacement)
- Distance calculations (Euclidean, squared, Manhattan)
- move_toward on torus (prefers shorter path)
- random_move
- cells_in_radius enumeration
- Edge cases (grid boundaries, wrapping)
"""

import numpy as np
import pytest

from src.utils.spatial import (
    toroidal_wrap,
    toroidal_delta,
    toroidal_distance_sq,
    toroidal_distance,
    toroidal_manhattan,
    move_toward,
    random_move,
    cells_in_radius,
    cells_in_radius_set,
)


# ---------------------------------------------------------------------------
# Toroidal Wrap
# ---------------------------------------------------------------------------

class TestToroidalWrap:
    def test_within_bounds(self):
        assert toroidal_wrap(5, 10, 500, 500) == (5, 10)

    def test_at_boundary(self):
        assert toroidal_wrap(500, 500, 500, 500) == (0, 0)

    def test_negative_x(self):
        assert toroidal_wrap(-1, 0, 500, 500) == (499, 0)

    def test_negative_y(self):
        assert toroidal_wrap(0, -1, 500, 500) == (0, 499)

    def test_large_negative(self):
        assert toroidal_wrap(-501, -501, 500, 500) == (499, 499)

    def test_large_positive(self):
        assert toroidal_wrap(1000, 1000, 500, 500) == (0, 0)

    def test_small_grid(self):
        assert toroidal_wrap(5, 5, 3, 3) == (2, 2)
        assert toroidal_wrap(-1, -1, 3, 3) == (2, 2)


# ---------------------------------------------------------------------------
# Toroidal Delta
# ---------------------------------------------------------------------------

class TestToroidalDelta:
    def test_same_position(self):
        assert toroidal_delta(5, 5, 500) == 0

    def test_forward_short(self):
        assert toroidal_delta(10, 15, 500) == 5

    def test_backward_short(self):
        assert toroidal_delta(15, 10, 500) == -5

    def test_wraps_forward(self):
        """Going from 499 to 1 is shorter by wrapping (delta = +2)."""
        assert toroidal_delta(499, 1, 500) == 2

    def test_wraps_backward(self):
        """Going from 1 to 499 is shorter by wrapping (delta = -2)."""
        assert toroidal_delta(1, 499, 500) == -2

    def test_exact_halfway(self):
        """At exactly half the grid, delta = half (convention: positive)."""
        delta = toroidal_delta(0, 250, 500)
        assert abs(delta) == 250

    def test_small_grid_wrap(self):
        assert toroidal_delta(0, 9, 10) == -1  # Shorter to wrap


# ---------------------------------------------------------------------------
# Distance Calculations
# ---------------------------------------------------------------------------

class TestToroidalDistance:
    def test_same_position(self):
        assert toroidal_distance_sq(5, 5, 5, 5, 500, 500) == 0
        assert toroidal_distance(5, 5, 5, 5, 500, 500) == 0.0

    def test_adjacent(self):
        assert toroidal_distance_sq(5, 5, 6, 5, 500, 500) == 1

    def test_diagonal(self):
        assert toroidal_distance_sq(5, 5, 6, 6, 500, 500) == 2

    def test_wrapping_shorter(self):
        """(0,0) to (499,0) on 500-wide: distance = 1 via wrapping."""
        assert toroidal_distance_sq(0, 0, 499, 0, 500, 500) == 1
        assert toroidal_distance(0, 0, 499, 0, 500, 500) == 1.0

    def test_wrapping_2d(self):
        """(0,0) to (499,499): distance² = 1+1 = 2 via wrapping."""
        assert toroidal_distance_sq(0, 0, 499, 499, 500, 500) == 2

    def test_symmetric(self):
        d1 = toroidal_distance(10, 20, 30, 40, 500, 500)
        d2 = toroidal_distance(30, 40, 10, 20, 500, 500)
        assert abs(d1 - d2) < 1e-10

    def test_manhattan_basic(self):
        assert toroidal_manhattan(5, 5, 8, 10, 500, 500) == 8

    def test_manhattan_wrapping(self):
        assert toroidal_manhattan(0, 0, 499, 499, 500, 500) == 2

    def test_manhattan_symmetric(self):
        d1 = toroidal_manhattan(10, 20, 30, 40, 500, 500)
        d2 = toroidal_manhattan(30, 40, 10, 20, 500, 500)
        assert d1 == d2


# ---------------------------------------------------------------------------
# move_toward
# ---------------------------------------------------------------------------

class TestMoveToward:
    def test_move_right(self):
        new = move_toward(5, 5, 10, 5, 500, 500)
        assert new == (6, 5)

    def test_move_left(self):
        new = move_toward(5, 5, 0, 5, 500, 500)
        assert new == (4, 5)

    def test_move_up(self):
        new = move_toward(5, 5, 5, 0, 500, 500)
        assert new == (5, 4)

    def test_move_down(self):
        new = move_toward(5, 5, 5, 10, 500, 500)
        assert new == (5, 6)

    def test_move_diagonal(self):
        new = move_toward(5, 5, 10, 10, 500, 500)
        assert new == (6, 6)

    def test_already_at_target(self):
        new = move_toward(5, 5, 5, 5, 500, 500)
        assert new == (5, 5)

    def test_wrapping_prefers_shorter_path(self):
        """From (0,0) toward (499,0): should move left (wrap) not right."""
        new = move_toward(0, 0, 499, 0, 500, 500)
        assert new == (499, 0)

    def test_wrapping_prefers_shorter_path_2d(self):
        """From (0,0) toward (499,499): move diagonally via wrap."""
        new = move_toward(0, 0, 499, 499, 500, 500)
        assert new == (499, 499)

    def test_move_reduces_distance(self):
        """Each step should reduce or maintain distance to target."""
        cx, cy = 100, 200
        tx, ty = 300, 400
        old_dist = toroidal_distance_sq(cx, cy, tx, ty, 500, 500)
        nx, ny = move_toward(cx, cy, tx, ty, 500, 500)
        new_dist = toroidal_distance_sq(nx, ny, tx, ty, 500, 500)
        assert new_dist < old_dist


# ---------------------------------------------------------------------------
# random_move
# ---------------------------------------------------------------------------

class TestRandomMove:
    def test_stays_in_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            x, y = random_move(250, 250, 500, 500, rng)
            assert 0 <= x < 500
            assert 0 <= y < 500

    def test_wraps_at_boundary(self):
        rng = np.random.default_rng(42)
        positions = set()
        for _ in range(200):
            x, y = random_move(0, 0, 10, 10, rng)
            positions.add((x, y))
        # Should include wrapped positions like (9, 9) and (0, 0) and (1, 1)
        assert (9, 9) in positions or (0, 0) in positions or (1, 1) in positions

    def test_moves_within_one_step(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            x, y = random_move(250, 250, 500, 500, rng)
            assert abs(x - 250) <= 1
            assert abs(y - 250) <= 1


# ---------------------------------------------------------------------------
# cells_in_radius
# ---------------------------------------------------------------------------

class TestCellsInRadius:
    def test_radius_0(self):
        cells = cells_in_radius(5, 5, 0, 500, 500)
        assert cells == [(5, 5)]

    def test_radius_1_count(self):
        """Radius 1 circle: center + 4 cardinal = 5 cells."""
        cells = cells_in_radius(5, 5, 1, 500, 500)
        assert len(cells) == 5
        assert (5, 5) in cells
        assert (6, 5) in cells
        assert (4, 5) in cells
        assert (5, 6) in cells
        assert (5, 4) in cells

    def test_radius_2_count(self):
        """Radius 2: all cells where dx²+dy² ≤ 4."""
        cells = cells_in_radius(50, 50, 2, 500, 500)
        expected = set()
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx * dx + dy * dy <= 4:
                    expected.add((50 + dx, 50 + dy))
        assert set(cells) == expected

    def test_wrapping_at_corner(self):
        """Cells near (0,0) should wrap around."""
        cells = cells_in_radius_set(0, 0, 2, 10, 10)
        # Should include (9, 0) — wrapping left
        assert (9, 0) in cells
        # Should include (0, 9) — wrapping up
        assert (0, 9) in cells

    def test_all_cells_in_bounds(self):
        cells = cells_in_radius(0, 0, 5, 20, 20)
        for x, y in cells:
            assert 0 <= x < 20
            assert 0 <= y < 20

    def test_center_always_included(self):
        for r in range(5):
            cells = cells_in_radius(10, 10, r, 100, 100)
            assert (10, 10) in cells

    def test_set_version_matches(self):
        cells_list = cells_in_radius(5, 5, 3, 50, 50)
        cells_set = cells_in_radius_set(5, 5, 3, 50, 50)
        assert set(cells_list) == cells_set

    def test_large_radius(self):
        """Radius larger than grid dimension — should include many wrapped cells."""
        cells = cells_in_radius_set(5, 5, 6, 10, 10)
        # With radius=6 on a 10x10 grid, most cells are within reach
        assert len(cells) > 50  # Most of 100 cells
