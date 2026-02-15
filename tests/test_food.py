"""
Unit tests for the Food resource.

Tests cover:
- Creation and initial state
- Lifespan ticking and expiry
- Consumption mechanics
- Edge cases (double consume, expired consume)
- Properties (position, active, expired)
"""

import pytest

from src.core.food import Food


# ---------------------------------------------------------------------------
# Creation Tests
# ---------------------------------------------------------------------------

class TestFoodCreation:
    """Tests for Food instantiation."""

    def test_basic_creation(self):
        food = Food(x=10, y=20, remaining_lifespan=50, energy_value=0.2)
        assert food.x == 10
        assert food.y == 20
        assert food.remaining_lifespan == 50
        assert food.energy_value == 0.2
        assert food.consumed is False

    def test_position_property(self):
        food = Food(x=5, y=15, remaining_lifespan=30, energy_value=0.3)
        assert food.position == (5, 15)

    def test_initial_state_active(self):
        food = Food(x=0, y=0, remaining_lifespan=10, energy_value=0.2)
        assert food.active is True
        assert food.expired is False
        assert food.consumed is False

    def test_custom_energy_value(self):
        food = Food(x=0, y=0, remaining_lifespan=10, energy_value=0.5)
        assert food.energy_value == 0.5


# ---------------------------------------------------------------------------
# Lifespan / Tick Tests
# ---------------------------------------------------------------------------

class TestFoodLifespan:
    """Tests for food lifespan and ticking."""

    def test_tick_decrements_lifespan(self):
        food = Food(x=0, y=0, remaining_lifespan=10, energy_value=0.2)
        food.tick()
        assert food.remaining_lifespan == 9

    def test_tick_returns_false_when_alive(self):
        food = Food(x=0, y=0, remaining_lifespan=10, energy_value=0.2)
        assert food.tick() is False

    def test_tick_returns_true_when_expired(self):
        food = Food(x=0, y=0, remaining_lifespan=1, energy_value=0.2)
        assert food.tick() is True

    def test_expires_at_zero(self):
        food = Food(x=0, y=0, remaining_lifespan=1, energy_value=0.2)
        food.tick()
        assert food.expired is True
        assert food.active is False

    def test_full_lifespan_countdown(self):
        lifespan = 50
        food = Food(x=0, y=0, remaining_lifespan=lifespan, energy_value=0.2)
        for i in range(lifespan - 1):
            assert food.tick() is False
            assert food.active is True
        # Final tick
        assert food.tick() is True
        assert food.expired is True
        assert food.active is False

    def test_negative_lifespan_stays_expired(self):
        food = Food(x=0, y=0, remaining_lifespan=0, energy_value=0.2)
        assert food.expired is True
        food.tick()
        assert food.remaining_lifespan == -1
        assert food.expired is True


# ---------------------------------------------------------------------------
# Consumption Tests
# ---------------------------------------------------------------------------

class TestFoodConsumption:
    """Tests for food consumption mechanics."""

    def test_consume_returns_energy(self):
        food = Food(x=0, y=0, remaining_lifespan=10, energy_value=0.2)
        energy = food.consume()
        assert energy == 0.2

    def test_consume_marks_consumed(self):
        food = Food(x=0, y=0, remaining_lifespan=10, energy_value=0.2)
        food.consume()
        assert food.consumed is True
        assert food.active is False

    def test_double_consume_raises(self):
        food = Food(x=0, y=0, remaining_lifespan=10, energy_value=0.2)
        food.consume()
        with pytest.raises(RuntimeError, match="already consumed"):
            food.consume()

    def test_consume_expired_raises(self):
        food = Food(x=0, y=0, remaining_lifespan=0, energy_value=0.2)
        with pytest.raises(RuntimeError, match="has expired"):
            food.consume()

    def test_consumed_food_not_active(self):
        food = Food(x=0, y=0, remaining_lifespan=10, energy_value=0.2)
        food.consume()
        assert food.active is False

    def test_consume_different_energy_values(self):
        for energy_val in [0.1, 0.2, 0.5, 1.0]:
            food = Food(x=0, y=0, remaining_lifespan=10, energy_value=energy_val)
            assert food.consume() == energy_val


# ---------------------------------------------------------------------------
# Repr Test
# ---------------------------------------------------------------------------

class TestFoodRepr:
    def test_repr_active(self):
        food = Food(x=5, y=10, remaining_lifespan=20, energy_value=0.2)
        r = repr(food)
        assert "active" in r
        assert "(5,10)" in r

    def test_repr_consumed(self):
        food = Food(x=5, y=10, remaining_lifespan=20, energy_value=0.2)
        food.consume()
        assert "consumed" in repr(food)

    def test_repr_expired(self):
        food = Food(x=5, y=10, remaining_lifespan=0, energy_value=0.2)
        assert "expired" in repr(food)
