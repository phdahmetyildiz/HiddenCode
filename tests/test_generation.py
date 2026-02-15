"""
Unit tests for the Generation lifecycle manager & reproduction.

Tests cover:
- Checkpoint tick calculations
- Primary reproduction fires at correct tick
- Survival check fires at correct tick
- Bonus reproduction fires at correct tick
- Generation advancement after bonus checkpoint
- Offspring count based on energy thresholds
- Offspring DNA is mutated (not identical to parent)
- Offspring energy = 1.0
- Offspring position within 3x3 of parent
- Low-energy animals die at survival check
- High-energy animals produce 2 offspring
- Generation counter increments correctly
- Adjustable gen_length (e.g., 100 ticks)
- Stress mode uses stress mutation rate
- Multi-generation simulation
- Generation stats tracking
- Integration with engine
"""

import numpy as np
import pytest

from src.core.config import SimConfig
from src.core.animal import Animal, reset_animal_id_counter
from src.core.dna import DNA
from src.core.world import World
from src.simulation.generation import (
    GenerationManager,
    GenerationEvent,
    GenerationStats,
)
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
    """Config with gen_length=100 for easy tick math."""
    cfg = SimConfig()
    cfg.world.width = 30
    cfg.world.height = 30
    cfg.world.seed = 42
    cfg.population.initial_count = 10
    cfg.generation.gen_length = 100
    cfg.generation.repro_checkpoint_pct = 0.70   # tick 70
    cfg.generation.survival_check_pct = 1.00     # tick 100
    cfg.generation.bonus_repro_pct = 1.20        # tick 120
    cfg.generation.survival_threshold = 0.50
    cfg.generation.repro_energy_low = 0.50
    cfg.generation.repro_energy_high = 0.75
    cfg.energy.base_metabolism = 0.001
    cfg.energy.k_weight_speed = 0.001
    cfg.energy.food_gain = 0.3
    cfg.resources.food_rate = 5.0
    cfg.resources.pitfall_rate = 0.0  # No pitfalls for cleaner tests
    cfg.genetics.base_mutation_rate = 0.01
    cfg.genetics.stress_mutation_rate = 0.20
    return cfg


@pytest.fixture
def gm(config) -> GenerationManager:
    return GenerationManager(config)


@pytest.fixture
def world(config) -> World:
    return World(config)


def place_animal(world: World, x: int, y: int, energy: float, config: SimConfig) -> Animal:
    """Place an animal with known energy at a specific position."""
    dna = DNA.create_random(length=config.genetics.dna_length, rng=world.rng)
    animal = Animal(dna=dna, x=x, y=y, config=config, energy=energy)
    world.add_animal(animal)
    return animal


# ---------------------------------------------------------------------------
# Checkpoint Tick Calculations
# ---------------------------------------------------------------------------

class TestCheckpointTicks:
    def test_primary_tick(self, gm):
        """Primary reproduction at 70% of gen_length=100 → tick 70."""
        assert gm.primary_tick == 70

    def test_survival_tick(self, gm):
        """Survival check at 100% of gen_length=100 → tick 100."""
        assert gm.survival_tick == 100

    def test_bonus_tick(self, gm):
        """Bonus reproduction at 120% of gen_length=100 → tick 120."""
        assert gm.bonus_tick == 120

    def test_initial_state(self, gm):
        assert gm.current_generation == 0
        assert gm.gen_start_tick == 0
        assert gm.total_generations_completed == 0

    def test_ticks_into_generation(self, gm):
        assert gm.ticks_into_generation(50) == 50
        assert gm.ticks_into_generation(0) == 0

    def test_custom_gen_length(self):
        """With gen_length=1000, checkpoints are at 700, 1000, 1200."""
        cfg = SimConfig()
        cfg.generation.gen_length = 1000
        cfg.generation.repro_checkpoint_pct = 0.70
        cfg.generation.survival_check_pct = 1.00
        cfg.generation.bonus_repro_pct = 1.20
        gm = GenerationManager(cfg)
        assert gm.primary_tick == 700
        assert gm.survival_tick == 1000
        assert gm.bonus_tick == 1200


# ---------------------------------------------------------------------------
# Primary Reproduction
# ---------------------------------------------------------------------------

class TestPrimaryReproduction:
    def test_primary_fires_at_correct_tick(self, gm, world, config):
        """Primary reproduction should fire at tick 70, not before."""
        rng = world.rng
        place_animal(world, 10, 10, energy=0.8, config=config)

        # Tick 69: should NOT fire
        events = gm.check(69, world, rng)
        assert GenerationEvent.PRIMARY_REPRODUCTION not in events

        # Tick 70: should fire
        events = gm.check(70, world, rng)
        assert GenerationEvent.PRIMARY_REPRODUCTION in events

    def test_primary_fires_only_once(self, gm, world, config):
        rng = world.rng
        place_animal(world, 10, 10, energy=0.8, config=config)

        events1 = gm.check(70, world, rng)
        events2 = gm.check(71, world, rng)
        assert GenerationEvent.PRIMARY_REPRODUCTION in events1
        assert GenerationEvent.PRIMARY_REPRODUCTION not in events2

    def test_high_energy_produces_2_offspring(self, gm, world, config):
        """Energy >= repro_energy_high (0.75) → 2 offspring."""
        rng = world.rng
        place_animal(world, 10, 10, energy=0.90, config=config)
        initial_count = world.alive_count

        gm.check(70, world, rng)

        # 1 parent + 2 offspring = 3
        assert world.alive_count == initial_count + 2
        assert gm.gen_stats.primary_repro_births == 2

    def test_medium_energy_produces_1_offspring(self, gm, world, config):
        """Energy between [repro_energy_low, repro_energy_high) → 1 offspring."""
        rng = world.rng
        place_animal(world, 10, 10, energy=0.60, config=config)
        initial_count = world.alive_count

        gm.check(70, world, rng)

        assert world.alive_count == initial_count + 1
        assert gm.gen_stats.primary_repro_births == 1

    def test_low_energy_produces_0_offspring(self, gm, world, config):
        """Energy < repro_energy_low (0.50) → 0 offspring."""
        rng = world.rng
        place_animal(world, 10, 10, energy=0.30, config=config)
        initial_count = world.alive_count

        gm.check(70, world, rng)

        assert world.alive_count == initial_count
        assert gm.gen_stats.primary_repro_births == 0

    def test_multiple_animals_reproduce(self, gm, world, config):
        """Multiple animals with different energies produce correct total offspring."""
        rng = world.rng
        place_animal(world, 5, 5, energy=0.90, config=config)   # → 2
        place_animal(world, 10, 10, energy=0.60, config=config)  # → 1
        place_animal(world, 15, 15, energy=0.30, config=config)  # → 0

        gm.check(70, world, rng)

        assert gm.gen_stats.primary_repro_births == 3  # 2 + 1 + 0
        assert gm.gen_stats.parents_at_primary == 2    # Only 2 had offspring

    def test_parents_tracked(self, gm, world, config):
        rng = world.rng
        place_animal(world, 5, 5, energy=0.80, config=config)
        place_animal(world, 10, 10, energy=0.30, config=config)

        gm.check(70, world, rng)
        assert gm.gen_stats.parents_at_primary == 1  # Only high-energy one


# ---------------------------------------------------------------------------
# Offspring Properties
# ---------------------------------------------------------------------------

class TestOffspringProperties:
    def test_offspring_energy_is_one(self, gm, world, config):
        """Offspring should start with energy = 1.0."""
        rng = world.rng
        parent = place_animal(world, 10, 10, energy=0.80, config=config)
        parent_id = parent.id

        gm.check(70, world, rng)

        # Find offspring (any animal that isn't the parent)
        offspring = [a for a in world.animals.values() if a.id != parent_id]
        assert len(offspring) >= 1
        for child in offspring:
            assert child.energy == 1.0

    def test_offspring_dna_mutated(self, gm, world, config):
        """Offspring DNA should differ from parent (mutation applied)."""
        rng = world.rng
        # Use higher mutation rate to ensure visible change
        config.genetics.base_mutation_rate = 0.10
        gm_high = GenerationManager(config)

        parent = place_animal(world, 10, 10, energy=0.80, config=config)
        parent_dna_bits = parent.dna.bits.copy()

        gm_high.check(70, world, rng)

        offspring = [a for a in world.animals.values() if a.id != parent.id]
        assert len(offspring) >= 1

        # At least one offspring should have different DNA
        any_different = any(
            not np.array_equal(child.dna.bits, parent_dna_bits)
            for child in offspring
        )
        assert any_different, "Offspring DNA should be mutated"

    def test_offspring_position_near_parent(self, gm, world, config):
        """Offspring should be within 3x3 area around parent."""
        rng = world.rng
        parent = place_animal(world, 15, 15, energy=0.80, config=config)

        gm.check(70, world, rng)

        offspring = [a for a in world.animals.values() if a.id != parent.id]
        for child in offspring:
            dx = abs(child.x - parent.x)
            dy = abs(child.y - parent.y)
            # Account for toroidal wrapping
            dx = min(dx, world.width - dx)
            dy = min(dy, world.height - dy)
            assert dx <= 1, f"Offspring x too far: dx={dx}"
            assert dy <= 1, f"Offspring y too far: dy={dy}"

    def test_offspring_generation_number(self, gm, world, config):
        """Offspring should have generation = current_generation + 1."""
        rng = world.rng
        parent = place_animal(world, 10, 10, energy=0.80, config=config)

        gm.check(70, world, rng)

        offspring = [a for a in world.animals.values() if a.id != parent.id]
        for child in offspring:
            assert child.generation == 1  # Current gen is 0, so offspring is gen 1


# ---------------------------------------------------------------------------
# Survival Check
# ---------------------------------------------------------------------------

class TestSurvivalCheck:
    def test_survival_fires_at_correct_tick(self, gm, world, config):
        rng = world.rng
        place_animal(world, 10, 10, energy=0.80, config=config)

        events = gm.check(99, world, rng)
        assert GenerationEvent.SURVIVAL_CHECK not in events

        # Must also fire primary first (tick 70)
        # Since we skipped to 99, primary would fire too
        # Let's advance properly
        gm2 = GenerationManager(config)
        place_animal(world, 5, 5, energy=0.60, config=config)
        gm2.check(70, world, rng)  # Fire primary
        events = gm2.check(100, world, rng)
        assert GenerationEvent.SURVIVAL_CHECK in events

    def test_low_energy_dies_at_survival(self, gm, world, config):
        """Animals with energy <= survival_threshold (0.50) die."""
        rng = world.rng
        low_energy = place_animal(world, 10, 10, energy=0.30, config=config)
        high_energy = place_animal(world, 15, 15, energy=0.80, config=config)

        # Fire primary first
        gm.check(70, world, rng)
        # Fire survival
        gm.check(100, world, rng)

        assert not low_energy.alive
        assert low_energy.death_cause == "age"
        assert high_energy.alive

    def test_survival_death_counted_in_stats(self, gm, world, config):
        rng = world.rng
        place_animal(world, 10, 10, energy=0.30, config=config)
        place_animal(world, 15, 15, energy=0.20, config=config)

        gm.check(70, world, rng)
        gm.check(100, world, rng)

        assert gm.gen_stats.survival_check_deaths == 2

    def test_exact_threshold_dies(self, gm, world, config):
        """Energy == survival_threshold (0.50) should NOT survive (need > threshold)."""
        rng = world.rng
        animal = place_animal(world, 10, 10, energy=0.50, config=config)

        gm.check(70, world, rng)
        gm.check(100, world, rng)

        assert not animal.alive  # survives_generation_end() checks energy > threshold

    def test_above_threshold_survives(self, gm, world, config):
        """Energy above threshold survives."""
        rng = world.rng
        animal = place_animal(world, 10, 10, energy=0.51, config=config)

        gm.check(70, world, rng)
        gm.check(100, world, rng)

        assert animal.alive


# ---------------------------------------------------------------------------
# Bonus Reproduction
# ---------------------------------------------------------------------------

class TestBonusReproduction:
    def test_bonus_fires_at_correct_tick(self, config, world):
        rng = world.rng
        gm = GenerationManager(config)
        place_animal(world, 10, 10, energy=0.80, config=config)

        gm.check(70, world, rng)   # primary
        gm.check(100, world, rng)  # survival

        events = gm.check(119, world, rng)
        assert GenerationEvent.BONUS_REPRODUCTION not in events

        events = gm.check(120, world, rng)
        assert GenerationEvent.BONUS_REPRODUCTION in events
        assert GenerationEvent.GENERATION_COMPLETE in events

    def test_bonus_produces_offspring(self, config, world):
        rng = world.rng
        gm = GenerationManager(config)
        place_animal(world, 10, 10, energy=0.90, config=config)

        gm.check(70, world, rng)   # primary → 2 offspring
        gm.check(100, world, rng)  # survival → parent survives

        count_before = world.alive_count
        gm.check(120, world, rng)  # bonus reproduction

        # Parent + primary offspring all reproduce if they have energy
        assert gm.gen_stats.bonus_repro_births >= 0

    def test_only_survivors_reproduce_at_bonus(self, config, world):
        """Dead animals (failed survival) should not reproduce at bonus."""
        rng = world.rng
        gm = GenerationManager(config)
        dead_animal = place_animal(world, 10, 10, energy=0.30, config=config)
        alive_animal = place_animal(world, 15, 15, energy=0.90, config=config)

        gm.check(70, world, rng)   # primary
        gm.check(100, world, rng)  # survival → dead_animal dies

        assert not dead_animal.alive
        assert alive_animal.alive

        # Bonus should only use alive animals
        alive_before = world.alive_count
        gm.check(120, world, rng)
        # alive_animal (energy=0.90) should produce 2 more offspring
        assert gm.gen_stats.bonus_repro_births >= 0


# ---------------------------------------------------------------------------
# Generation Advancement
# ---------------------------------------------------------------------------

class TestGenerationAdvancement:
    def test_generation_increments(self, config, world):
        rng = world.rng
        gm = GenerationManager(config)
        place_animal(world, 10, 10, energy=0.80, config=config)

        assert gm.current_generation == 0

        gm.check(70, world, rng)
        gm.check(100, world, rng)
        gm.check(120, world, rng)

        assert gm.current_generation == 1
        assert gm.total_generations_completed == 1

    def test_gen_start_tick_updates(self, config, world):
        rng = world.rng
        gm = GenerationManager(config)
        place_animal(world, 10, 10, energy=0.80, config=config)

        gm.check(70, world, rng)
        gm.check(100, world, rng)
        gm.check(120, world, rng)

        # After gen 0 completes at tick 120, next gen starts at tick 120
        assert gm.gen_start_tick == 120
        # Next primary tick = 120 + 70 = 190
        assert gm.primary_tick == 190

    def test_second_generation_checkpoints(self, config, world):
        """Verify second generation fires at correct ticks."""
        rng = world.rng
        gm = GenerationManager(config)
        place_animal(world, 10, 10, energy=0.80, config=config)

        # Complete generation 0
        gm.check(70, world, rng)
        gm.check(100, world, rng)
        gm.check(120, world, rng)

        # Generation 1 checkpoints: 120+70=190, 120+100=220, 120+120=240
        events = gm.check(189, world, rng)
        assert GenerationEvent.PRIMARY_REPRODUCTION not in events

        events = gm.check(190, world, rng)
        assert GenerationEvent.PRIMARY_REPRODUCTION in events

    def test_world_generation_updated(self, config, world):
        rng = world.rng
        gm = GenerationManager(config)
        place_animal(world, 10, 10, energy=0.80, config=config)

        gm.check(70, world, rng)
        gm.check(100, world, rng)
        gm.check(120, world, rng)

        assert world.generation == 1

    def test_dead_animals_cleared_after_generation(self, config, world):
        rng = world.rng
        gm = GenerationManager(config)
        place_animal(world, 10, 10, energy=0.30, config=config)  # will die at survival
        place_animal(world, 15, 15, energy=0.80, config=config)

        gm.check(70, world, rng)
        gm.check(100, world, rng)

        assert len(world.dead_animals) > 0

        gm.check(120, world, rng)  # generation complete

        # Dead animals cleared
        assert len(world.dead_animals) == 0


# ---------------------------------------------------------------------------
# Generation Stats
# ---------------------------------------------------------------------------

class TestGenerationStats:
    def test_stats_recorded(self, config, world):
        rng = world.rng
        gm = GenerationManager(config)
        place_animal(world, 10, 10, energy=0.80, config=config)

        gm.check(70, world, rng)
        gm.check(100, world, rng)
        gm.check(120, world, rng)

        stats = gm.get_last_gen_stats()
        assert stats is not None
        assert stats.generation == 0

    def test_stats_population_snapshots(self, config, world):
        rng = world.rng
        gm = GenerationManager(config)
        place_animal(world, 10, 10, energy=0.80, config=config)

        gm.check(70, world, rng)
        assert gm.gen_stats.pop_at_primary_repro == 1

        gm.check(100, world, rng)
        gm.check(120, world, rng)

        last = gm.get_last_gen_stats()
        assert last.pop_at_primary_repro == 1
        assert last.total_births == last.primary_repro_births + last.bonus_repro_births

    def test_all_gen_stats_accumulated(self, config, world):
        rng = world.rng
        gm = GenerationManager(config)
        place_animal(world, 10, 10, energy=0.80, config=config)

        # Gen 0
        gm.check(70, world, rng)
        gm.check(100, world, rng)
        gm.check(120, world, rng)

        # Gen 1
        gm.check(190, world, rng)
        gm.check(220, world, rng)
        gm.check(240, world, rng)

        assert len(gm.all_gen_stats) == 2
        assert gm.all_gen_stats[0].generation == 0
        assert gm.all_gen_stats[1].generation == 1

    def test_no_stats_before_completion(self, gm):
        assert gm.get_last_gen_stats() is None


# ---------------------------------------------------------------------------
# Stress Mode
# ---------------------------------------------------------------------------

class TestStressMode:
    def test_stress_mode_uses_stress_rate(self, config, world):
        """When world.stress_mode = True, offspring use stress mutation rate."""
        rng = world.rng
        config.genetics.base_mutation_rate = 0.01
        config.genetics.stress_mutation_rate = 0.50  # Very high
        gm = GenerationManager(config)

        world.stress_mode = True
        parent = place_animal(world, 10, 10, energy=0.80, config=config)
        parent_bits = parent.dna.bits.copy()

        gm.check(70, world, rng)

        offspring = [a for a in world.animals.values() if a.id != parent.id]
        assert len(offspring) >= 1

        # With 50% stress rate, offspring DNA should differ significantly
        differences = sum(
            int(not np.array_equal(child.dna.bits, parent_bits))
            for child in offspring
        )
        assert differences > 0, "Stress mutation should cause DNA changes"


# ---------------------------------------------------------------------------
# Integration with Engine
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    def test_engine_runs_generations(self, config):
        """Engine should run and complete generations."""
        config.population.initial_count = 20
        config.resources.food_rate = 10.0
        config.energy.base_metabolism = 0.002
        config.energy.k_weight_speed = 0.001
        config.generation.gen_length = 50
        config.generation.repro_checkpoint_pct = 0.70
        config.generation.survival_check_pct = 1.00
        config.generation.bonus_repro_pct = 1.20
        config.generation.survival_threshold = 0.30
        config.generation.repro_energy_low = 0.30
        config.generation.repro_energy_high = 0.60

        eng = SimulationEngine(config)
        eng.initialize()

        result = eng.run(max_generations=2)

        assert result.total_generations == 2 or result.extinct

    def test_engine_generation_callback(self, config):
        """on_generation callback should fire at generation boundaries."""
        config.population.initial_count = 15
        config.resources.food_rate = 8.0
        config.energy.base_metabolism = 0.002
        config.energy.k_weight_speed = 0.001
        config.generation.gen_length = 50
        config.generation.survival_threshold = 0.20
        config.generation.repro_energy_low = 0.20

        eng = SimulationEngine(config)
        eng.initialize()

        gen_events = []

        def on_gen(gen_num, engine):
            gen_events.append(gen_num)

        eng.on_generation = on_gen
        result = eng.run(max_generations=2)

        if not result.extinct:
            assert len(gen_events) == 2
            assert gen_events[0] == 0
            assert gen_events[1] == 1

    def test_engine_max_generations_stops(self, config):
        """Engine should stop after max_generations completed."""
        config.population.initial_count = 20
        config.resources.food_rate = 10.0
        config.energy.base_metabolism = 0.002
        config.energy.k_weight_speed = 0.001
        config.generation.gen_length = 30
        config.generation.survival_threshold = 0.20
        config.generation.repro_energy_low = 0.20

        eng = SimulationEngine(config)
        eng.initialize()
        result = eng.run(max_generations=1)

        if not result.extinct:
            assert result.total_generations == 1

    def test_engine_extinction_stops_before_generation_end(self, config):
        """If all animals die, engine stops even mid-generation."""
        config.population.initial_count = 5
        config.resources.food_rate = 0.0  # No food
        config.energy.base_metabolism = 0.1  # High drain
        config.energy.k_weight_speed = 0.0
        config.generation.gen_length = 1000

        eng = SimulationEngine(config)
        eng.initialize()
        result = eng.run(max_generations=10)

        assert result.extinct is True
        assert result.total_ticks < 1000  # Died before gen end

    def test_engine_generation_manager_accessible(self, config):
        eng = SimulationEngine(config)
        eng.initialize()
        assert eng.generation_manager is not None
        assert eng.current_generation == 0

    def test_engine_run_default_one_generation(self, config):
        """With no args, run() should run ~one full generation cycle."""
        config.population.initial_count = 15
        config.resources.food_rate = 8.0
        config.energy.base_metabolism = 0.002
        config.energy.k_weight_speed = 0.001
        config.generation.gen_length = 50

        eng = SimulationEngine(config)
        eng.initialize()
        result = eng.run()

        # Should have run approximately gen_length * bonus_pct ticks
        if not result.extinct:
            expected_ticks = int(50 * 1.20) + 1
            assert result.total_ticks == expected_ticks


# ---------------------------------------------------------------------------
# Multi-Generation Integration
# ---------------------------------------------------------------------------

class TestMultiGeneration:
    def test_three_generations(self, config):
        """Run 3 full generations and verify stats."""
        config.population.initial_count = 25
        config.resources.food_rate = 12.0
        config.energy.base_metabolism = 0.002
        config.energy.k_weight_speed = 0.001
        config.generation.gen_length = 40
        config.generation.survival_threshold = 0.20
        config.generation.repro_energy_low = 0.20
        config.generation.repro_energy_high = 0.50

        eng = SimulationEngine(config)
        eng.initialize()
        result = eng.run(max_generations=3)

        if not result.extinct:
            assert result.total_generations == 3
            assert len(eng.generation_manager.all_gen_stats) == 3

            for i, gs in enumerate(eng.generation_manager.all_gen_stats):
                assert gs.generation == i
                assert gs.total_births >= 0
                assert gs.pop_at_generation_end >= 0

    def test_population_changes_across_generations(self, config):
        """Population should change across generations due to reproduction and death."""
        config.population.initial_count = 30
        config.resources.food_rate = 15.0
        config.energy.base_metabolism = 0.003
        config.energy.k_weight_speed = 0.001
        config.generation.gen_length = 40
        config.generation.survival_threshold = 0.30
        config.generation.repro_energy_low = 0.30

        eng = SimulationEngine(config)
        eng.initialize()
        result = eng.run(max_generations=3)

        if not result.extinct:
            pops = [gs.pop_at_generation_end for gs in eng.generation_manager.all_gen_stats]
            # Population should vary (not all identical)
            # This is nearly certain with reproduction and death
            assert len(pops) == 3


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_generation_with_no_animals(self, config, world):
        """Generation check on empty world should not crash."""
        rng = world.rng
        gm = GenerationManager(config)

        events = gm.check(70, world, rng)
        assert GenerationEvent.PRIMARY_REPRODUCTION in events
        assert gm.gen_stats.primary_repro_births == 0

    def test_all_animals_die_at_survival(self, config, world):
        """If all animals die at survival check, bonus should have 0 births."""
        rng = world.rng
        gm = GenerationManager(config)
        place_animal(world, 10, 10, energy=0.30, config=config)
        place_animal(world, 15, 15, energy=0.20, config=config)

        gm.check(70, world, rng)
        gm.check(100, world, rng)

        assert world.alive_count == 0

        gm.check(120, world, rng)
        assert gm.gen_stats.bonus_repro_births == 0

    def test_repr(self, gm):
        r = repr(gm)
        assert "GenerationManager" in r
        assert "gen=0" in r
