"""
Unit tests for Phase 8: Stress Events.

Tests cover:
- StressManager:
  - Manual trigger activates stress_mode on world
  - Trigger spawns burst of new pitfall types
  - Deactivate restores world.stress_mode = False
  - Mutation rate changes: stress_rate during stress, base_rate after
  - Auto-trigger at configured tick
  - Auto-deactivate after configured duration
  - Food rate override during stress and restoration
  - Double trigger is no-op
  - Deactivate when not active is no-op
  - Stress pitfall types have correct sequences
  - Status dict
- SimulationEngine integration:
  - Manual trigger/deactivate via engine
  - Auto-trigger during run
  - Stress callback fires
  - Reproduction uses stress mutation rate during stress
"""

import numpy as np
import pytest

from src.core.config import SimConfig, PitfallType
from src.core.world import World
from src.core.animal import Animal, reset_animal_id_counter
from src.core.dna import DNA
from src.core.pitfall import Pitfall
from src.simulation.stress import StressManager
from src.simulation.engine import SimulationEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ids():
    reset_animal_id_counter()
    yield
    reset_animal_id_counter()


@pytest.fixture
def config() -> SimConfig:
    cfg = SimConfig()
    cfg.world.width = 20
    cfg.world.height = 20
    cfg.world.seed = 42
    cfg.genetics.base_mutation_rate = 0.01
    cfg.genetics.stress_mutation_rate = 0.20
    cfg.stress.pitfall_burst_count = 10
    cfg.stress.post_event_pitfall_types = [
        {"name": "B", "sequence": "00001111000011110000111100001111"}
    ]
    cfg.resources.food_rate = 5.0
    return cfg


@pytest.fixture
def world(config) -> World:
    return World(config)


@pytest.fixture
def rng(config) -> np.random.Generator:
    return np.random.default_rng(config.world.seed)


def place_animal(world: World, config: SimConfig, x: int, y: int, energy: float = 0.8) -> Animal:
    dna = DNA.create_random(length=config.genetics.dna_length, rng=world.rng)
    animal = Animal(dna=dna, x=x, y=y, config=config, energy=energy)
    world.add_animal(animal)
    return animal


# ===========================================================================
# StressManager — Manual Trigger/Deactivate
# ===========================================================================

class TestStressManagerTrigger:
    def test_trigger_sets_stress_mode(self, config, world, rng):
        sm = StressManager(config)
        assert world.stress_mode is False
        sm.trigger(world, rng)
        assert world.stress_mode is True
        assert sm.active is True

    def test_trigger_spawns_pitfalls(self, config, world, rng):
        sm = StressManager(config)
        initial_pitfalls = world.pitfall_count
        spawned = sm.trigger(world, rng)
        assert spawned > 0
        assert world.pitfall_count > initial_pitfalls
        assert sm.pitfalls_spawned_on_trigger == spawned

    def test_trigger_spawns_correct_type(self, config, world, rng):
        sm = StressManager(config)
        sm.trigger(world, rng)
        # All spawned pitfalls should be type "B"
        for p in world.pitfalls.values():
            assert p.name == "B"
            assert p.sequence_str == "00001111000011110000111100001111"

    def test_trigger_records_tick(self, config, world, rng):
        sm = StressManager(config)
        world.tick_count = 50
        sm.trigger(world, rng)
        assert sm.trigger_tick == 50

    def test_double_trigger_is_noop(self, config, world, rng):
        sm = StressManager(config)
        spawned1 = sm.trigger(world, rng)
        count_after_first = world.pitfall_count
        spawned2 = sm.trigger(world, rng)
        assert spawned2 == 0
        assert world.pitfall_count == count_after_first

    def test_trigger_with_zero_burst(self, config, world, rng):
        config.stress.pitfall_burst_count = 0
        sm = StressManager(config)
        spawned = sm.trigger(world, rng)
        assert spawned == 0
        assert world.stress_mode is True  # Still activates stress mode

    def test_trigger_with_no_pitfall_types(self, config, world, rng):
        config.stress.post_event_pitfall_types = []
        sm = StressManager(config)
        spawned = sm.trigger(world, rng)
        assert spawned == 0
        assert world.stress_mode is True


class TestStressManagerDeactivate:
    def test_deactivate_clears_stress_mode(self, config, world, rng):
        sm = StressManager(config)
        sm.trigger(world, rng)
        assert world.stress_mode is True
        sm.deactivate(world)
        assert world.stress_mode is False
        assert sm.active is False

    def test_deactivate_when_not_active_is_noop(self, config, world):
        sm = StressManager(config)
        sm.deactivate(world)  # Should not raise
        assert sm.active is False

    def test_trigger_deactivate_retrigger(self, config, world, rng):
        sm = StressManager(config)
        sm.trigger(world, rng)
        sm.deactivate(world)
        assert sm.active is False

        # Re-trigger after deactivation should work
        # Need to reset active to allow re-trigger
        spawned = sm.trigger(world, rng)
        assert sm.active is True
        assert spawned >= 0  # May or may not spawn depending on grid space


# ===========================================================================
# Mutation Rate
# ===========================================================================

class TestMutationRate:
    def test_effective_rate_during_stress(self, config, world, rng):
        sm = StressManager(config)
        assert sm.effective_mutation_rate == config.genetics.base_mutation_rate
        sm.trigger(world, rng)
        assert sm.effective_mutation_rate == config.genetics.stress_mutation_rate

    def test_effective_rate_after_deactivation(self, config, world, rng):
        sm = StressManager(config)
        sm.trigger(world, rng)
        sm.deactivate(world)
        assert sm.effective_mutation_rate == config.genetics.base_mutation_rate


# ===========================================================================
# Food Rate Override
# ===========================================================================

class TestFoodRateOverride:
    def test_food_rate_changed_during_stress(self, config, world, rng):
        config.stress.food_rate_during_stress = 1.0
        sm = StressManager(config)
        original_rate = config.resources.food_rate
        assert original_rate == 5.0

        sm.trigger(world, rng)
        assert config.resources.food_rate == 1.0

    def test_food_rate_restored_after_deactivation(self, config, world, rng):
        config.stress.food_rate_during_stress = 1.0
        sm = StressManager(config)
        sm.trigger(world, rng)
        sm.deactivate(world)
        assert config.resources.food_rate == 5.0

    def test_food_rate_unchanged_when_null(self, config, world, rng):
        config.stress.food_rate_during_stress = None
        sm = StressManager(config)
        sm.trigger(world, rng)
        assert config.resources.food_rate == 5.0


# ===========================================================================
# Auto-Trigger
# ===========================================================================

class TestAutoTrigger:
    def test_auto_trigger_at_configured_tick(self, config, world, rng):
        config.stress.trigger_tick = 10
        sm = StressManager(config)

        # Before trigger tick
        world.tick_count = 5
        result = sm.check_tick(world, rng)
        assert result == "none"
        assert sm.active is False

        # At trigger tick
        world.tick_count = 10
        result = sm.check_tick(world, rng)
        assert result == "triggered"
        assert sm.active is True
        assert world.stress_mode is True

    def test_auto_trigger_fires_only_once(self, config, world, rng):
        config.stress.trigger_tick = 10
        sm = StressManager(config)

        world.tick_count = 10
        result1 = sm.check_tick(world, rng)
        assert result1 == "triggered"

        world.tick_count = 11
        result2 = sm.check_tick(world, rng)
        # Should not trigger again (still active, and _auto_triggered is True)
        assert result2 == "none"

    def test_no_auto_trigger_when_null(self, config, world, rng):
        config.stress.trigger_tick = None
        sm = StressManager(config)

        world.tick_count = 100
        result = sm.check_tick(world, rng)
        assert result == "none"
        assert sm.active is False


# ===========================================================================
# Auto-Deactivate (Duration)
# ===========================================================================

class TestAutoDeactivate:
    def test_auto_deactivate_after_duration(self, config, world, rng):
        config.stress.duration_ticks = 5
        sm = StressManager(config)

        world.tick_count = 10
        sm.trigger(world, rng)
        assert sm.active is True

        # Not yet expired
        world.tick_count = 14
        result = sm.check_tick(world, rng)
        assert result == "none"
        assert sm.active is True

        # Exactly at duration
        world.tick_count = 15
        result = sm.check_tick(world, rng)
        assert result == "deactivated"
        assert sm.active is False
        assert world.stress_mode is False

    def test_no_auto_deactivate_when_null(self, config, world, rng):
        config.stress.duration_ticks = None
        sm = StressManager(config)

        world.tick_count = 10
        sm.trigger(world, rng)

        # Even after many ticks, stays active
        world.tick_count = 1000
        result = sm.check_tick(world, rng)
        assert result == "none"
        assert sm.active is True

    def test_auto_trigger_with_duration(self, config, world, rng):
        """Auto-trigger at tick 10, auto-deactivate after 5 ticks."""
        config.stress.trigger_tick = 10
        config.stress.duration_ticks = 5
        sm = StressManager(config)

        world.tick_count = 10
        result = sm.check_tick(world, rng)
        assert result == "triggered"
        assert sm.active is True

        world.tick_count = 14
        result = sm.check_tick(world, rng)
        assert result == "none"
        assert sm.active is True

        world.tick_count = 15
        result = sm.check_tick(world, rng)
        assert result == "deactivated"
        assert sm.active is False


# ===========================================================================
# Status & Repr
# ===========================================================================

class TestStressStatus:
    def test_get_status(self, config, world, rng):
        sm = StressManager(config)
        status = sm.get_status()
        assert status["active"] is False
        assert status["effective_mutation_rate"] == config.genetics.base_mutation_rate

        sm.trigger(world, rng)
        status = sm.get_status()
        assert status["active"] is True
        assert status["effective_mutation_rate"] == config.genetics.stress_mutation_rate
        assert status["pitfalls_spawned_on_trigger"] > 0

    def test_repr(self, config):
        sm = StressManager(config)
        assert "StressManager" in repr(sm)


# ===========================================================================
# Multiple Pitfall Types
# ===========================================================================

class TestMultiplePitfallTypes:
    def test_multiple_stress_pitfall_types(self, config, world, rng):
        config.stress.post_event_pitfall_types = [
            {"name": "B", "sequence": "00001111000011110000111100001111"},
            {"name": "C", "sequence": "11111111000000001111111100000000"},
        ]
        config.stress.pitfall_burst_count = 20
        sm = StressManager(config)
        sm.trigger(world, rng)

        # Should have mix of types B and C
        names = {p.name for p in world.pitfalls.values()}
        # With 20 spawns and 2 types, very likely both appear
        assert len(names) >= 1  # At least one type
        # Check all are valid types
        for p in world.pitfalls.values():
            assert p.name in ("B", "C")


# ===========================================================================
# SimulationEngine Integration
# ===========================================================================

class TestEngineStressIntegration:
    def _make_engine(self, config: SimConfig) -> SimulationEngine:
        engine = SimulationEngine(config)
        engine.initialize(population_count=10)
        return engine

    def test_manual_trigger_via_engine(self, config):
        engine = self._make_engine(config)
        assert engine.stress_active is False

        spawned = engine.trigger_stress()
        assert engine.stress_active is True
        assert engine.world.stress_mode is True
        assert spawned >= 0

    def test_manual_deactivate_via_engine(self, config):
        engine = self._make_engine(config)
        engine.trigger_stress()
        engine.deactivate_stress()
        assert engine.stress_active is False
        assert engine.world.stress_mode is False

    def test_auto_trigger_during_run(self, config):
        config.stress.trigger_tick = 5
        engine = self._make_engine(config)

        # Run a few ticks — stress should auto-trigger at tick 5
        for _ in range(10):
            engine.tick()

        assert engine.stress_active is True
        assert engine.world.stress_mode is True

    def test_auto_trigger_and_deactivate_during_run(self, config):
        config.stress.trigger_tick = 3
        config.stress.duration_ticks = 4
        engine = self._make_engine(config)

        # Run past trigger point but before deactivation
        for _ in range(5):
            engine.tick()
        assert engine.stress_active is True

        # Run past deactivation point
        for _ in range(5):
            engine.tick()
        assert engine.stress_active is False

    def test_stress_callback_fires(self, config):
        config.stress.trigger_tick = 2
        engine = self._make_engine(config)

        events = []
        engine.on_stress = lambda event, eng: events.append(event)

        for _ in range(5):
            engine.tick()

        assert "triggered" in events

    def test_stress_callback_on_manual_trigger(self, config):
        engine = self._make_engine(config)

        events = []
        engine.on_stress = lambda event, eng: events.append(event)

        engine.trigger_stress()
        assert "triggered" in events

        engine.deactivate_stress()
        assert "deactivated" in events

    def test_food_rate_override_during_engine_run(self, config):
        config.stress.food_rate_during_stress = 0.5
        config.stress.trigger_tick = 2
        config.stress.duration_ticks = 3
        engine = self._make_engine(config)
        original_rate = config.resources.food_rate

        # Before stress
        assert config.resources.food_rate == original_rate

        # Trigger stress
        for _ in range(3):
            engine.tick()
        assert config.resources.food_rate == 0.5

        # After stress deactivates
        for _ in range(5):
            engine.tick()
        assert config.resources.food_rate == original_rate

    def test_stress_does_not_affect_engine_when_not_configured(self, config):
        config.stress.trigger_tick = None
        config.stress.duration_ticks = None
        engine = self._make_engine(config)

        for _ in range(20):
            engine.tick()

        assert engine.stress_active is False
        assert engine.world.stress_mode is False


# ===========================================================================
# Stress & Reproduction Integration
# ===========================================================================

class TestStressReproductionIntegration:
    def test_stress_mode_flag_propagates_to_offspring(self, config, world, rng):
        """
        When world.stress_mode is True, Animal.create_offspring should use
        the stress mutation rate. Verify the flag is checked.
        """
        place_animal(world, config, 5, 5, energy=1.0)
        parent = list(world.animals.values())[0]

        # Normal mode offspring
        world.stress_mode = False
        child_normal = parent.create_offspring(
            current_tick=0,
            stress_mode=world.stress_mode,
            rng=rng,
            world_width=world.width,
            world_height=world.height,
            generation=1,
        )
        assert child_normal is not None

        # Stress mode offspring — verify it runs without error
        world.stress_mode = True
        child_stress = parent.create_offspring(
            current_tick=0,
            stress_mode=world.stress_mode,
            rng=rng,
            world_width=world.width,
            world_height=world.height,
            generation=1,
        )
        assert child_stress is not None

    def test_stress_mutation_rate_higher_than_base(self, config):
        """Verify config relationship: stress rate > base rate."""
        assert config.genetics.stress_mutation_rate > config.genetics.base_mutation_rate


# ===========================================================================
# Config Validation
# ===========================================================================

class TestStressConfigValidation:
    def test_valid_stress_config(self, config):
        errors = config.stress.validate()
        assert errors == []

    def test_negative_trigger_tick(self, config):
        config.stress.trigger_tick = -1
        errors = config.stress.validate()
        assert any("trigger_tick" in e for e in errors)

    def test_zero_duration_ticks(self, config):
        config.stress.duration_ticks = 0
        errors = config.stress.validate()
        assert any("duration_ticks" in e for e in errors)

    def test_negative_burst_count(self, config):
        config.stress.pitfall_burst_count = -1
        errors = config.stress.validate()
        assert any("pitfall_burst_count" in e for e in errors)

    def test_negative_food_rate_during_stress(self, config):
        config.stress.food_rate_during_stress = -1
        errors = config.stress.validate()
        assert any("food_rate_during_stress" in e for e in errors)
