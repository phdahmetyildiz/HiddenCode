"""
Unit tests for the World (simulation environment).

Tests cover:
- Initialization and population creation
- Animal management (add, remove, kill, move, spatial index)
- Food management (spawn, decay, query)
- Pitfall management (spawn, decay, query)
- Spatial queries (nearest food, food in range, heaviest animal at cell)
- Resource spawning rates (statistical)
- Edge cases (empty world, extinction)
"""

import numpy as np
import pytest

from src.core.config import SimConfig, PitfallType
from src.core.world import World
from src.core.animal import Animal, reset_animal_id_counter
from src.core.dna import DNA
from src.core.food import Food
from src.core.pitfall import Pitfall


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
    cfg.world.width = 50
    cfg.world.height = 50
    cfg.world.seed = 42
    cfg.population.initial_count = 20
    return cfg


@pytest.fixture
def world(config) -> World:
    return World(config)


@pytest.fixture
def populated_world(config) -> World:
    w = World(config)
    w.initialize_population()
    return w


def make_animal_at(config: SimConfig, x: int, y: int, energy: float = 1.0) -> Animal:
    """Helper: create an animal at a specific position."""
    dna = DNA.create_random(length=config.genetics.dna_length, rng=np.random.default_rng(42))
    return Animal(dna=dna, x=x, y=y, config=config, energy=energy)


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------

class TestWorldInit:
    def test_empty_world(self, world):
        assert world.alive_count == 0
        assert world.food_count == 0
        assert world.pitfall_count == 0
        assert world.tick_count == 0
        assert world.generation == 0
        assert world.stress_mode is False

    def test_dimensions(self, world, config):
        assert world.width == config.world.width
        assert world.height == config.world.height

    def test_initialize_population(self, populated_world, config):
        assert populated_world.alive_count == config.population.initial_count

    def test_initial_animals_alive(self, populated_world):
        for animal in populated_world.animals.values():
            assert animal.alive is True
            assert animal.energy == 1.0

    def test_initial_animals_in_bounds(self, populated_world):
        for animal in populated_world.animals.values():
            assert 0 <= animal.x < populated_world.width
            assert 0 <= animal.y < populated_world.height

    def test_custom_population_count(self, config):
        w = World(config)
        w.initialize_population(count=5)
        assert w.alive_count == 5

    def test_is_extinct_empty(self, world):
        assert world.is_extinct is True

    def test_not_extinct_with_animals(self, populated_world):
        assert populated_world.is_extinct is False

    def test_repr(self, populated_world):
        r = repr(populated_world)
        assert "World" in r
        assert "50x50" in r


# ---------------------------------------------------------------------------
# Animal Management Tests
# ---------------------------------------------------------------------------

class TestAnimalManagement:
    def test_add_animal(self, world, config):
        animal = make_animal_at(config, 10, 20)
        world.add_animal(animal)
        assert world.alive_count == 1
        assert animal.id in world.animals

    def test_animals_at(self, world, config):
        a1 = make_animal_at(config, 10, 20)
        a2 = make_animal_at(config, 10, 20)
        a3 = make_animal_at(config, 15, 25)
        world.add_animal(a1)
        world.add_animal(a2)
        world.add_animal(a3)

        at_10_20 = world.animals_at(10, 20)
        assert len(at_10_20) == 2
        at_15_25 = world.animals_at(15, 25)
        assert len(at_15_25) == 1
        at_empty = world.animals_at(0, 0)
        assert len(at_empty) == 0

    def test_remove_animal(self, world, config):
        animal = make_animal_at(config, 10, 20)
        world.add_animal(animal)
        world.remove_animal(animal)
        assert world.alive_count == 0
        assert len(world.animals_at(10, 20)) == 0

    def test_kill_animal(self, world, config):
        animal = make_animal_at(config, 10, 20)
        world.add_animal(animal)
        world.kill_animal(animal, cause="starvation")
        assert world.alive_count == 0
        assert animal.alive is False
        assert animal.death_cause == "starvation"
        assert len(world.dead_animals) == 1

    def test_move_animal(self, world, config):
        animal = make_animal_at(config, 10, 20)
        world.add_animal(animal)
        world.move_animal(animal, 15, 25)
        assert animal.x == 15
        assert animal.y == 25
        assert len(world.animals_at(10, 20)) == 0
        assert len(world.animals_at(15, 25)) == 1

    def test_move_animal_wraps(self, world, config):
        animal = make_animal_at(config, 49, 49)
        world.add_animal(animal)
        world.move_animal(animal, 50, 50)
        assert animal.x == 0
        assert animal.y == 0
        assert len(world.animals_at(0, 0)) == 1

    def test_get_alive_animals(self, populated_world):
        alive = populated_world.get_alive_animals()
        assert len(alive) == populated_world.alive_count
        # Should be a copy (safe to iterate)
        assert alive is not populated_world.animals

    def test_get_shuffled_animals(self, populated_world):
        shuffled = populated_world.get_shuffled_animals()
        assert len(shuffled) == populated_world.alive_count
        # Not guaranteed different order but should be a valid permutation
        assert set(a.id for a in shuffled) == set(populated_world.animals.keys())

    def test_clear_dead_animals(self, world, config):
        animal = make_animal_at(config, 10, 20)
        world.add_animal(animal)
        world.kill_animal(animal, "test")
        dead = world.clear_dead_animals()
        assert len(dead) == 1
        assert len(world.dead_animals) == 0


# ---------------------------------------------------------------------------
# Food Management Tests
# ---------------------------------------------------------------------------

class TestFoodManagement:
    def test_add_food(self, world):
        food = Food(x=5, y=10, remaining_lifespan=50, energy_value=0.2)
        world.add_food(food)
        assert world.food_count == 1

    def test_food_at(self, world):
        food = Food(x=5, y=10, remaining_lifespan=50, energy_value=0.2)
        world.add_food(food)
        result = world.food_at(5, 10)
        assert result is food
        assert world.food_at(0, 0) is None

    def test_remove_food(self, world):
        food = Food(x=5, y=10, remaining_lifespan=50, energy_value=0.2)
        world.add_food(food)
        removed = world.remove_food((5, 10))
        assert removed is food
        assert world.food_count == 0

    def test_spawn_food(self, world):
        spawned = world.spawn_food(rate=10.0)
        assert spawned > 0
        assert world.food_count == spawned

    def test_spawn_food_zero_rate(self, world):
        spawned = world.spawn_food(rate=0.0)
        assert spawned == 0

    def test_spawn_food_statistical(self, world):
        """Over many ticks, average food spawned ≈ rate."""
        rate = 5.0
        total_spawned = 0
        world_fresh = World(world.config)
        for _ in range(1000):
            total_spawned += world_fresh.spawn_food(rate=rate)
            # Decay to avoid position conflicts
            world_fresh.food.clear()
        avg = total_spawned / 1000
        assert abs(avg - rate) < 1.0, f"Average spawned {avg}, expected ≈{rate}"

    def test_decay_food(self, world):
        food = Food(x=5, y=10, remaining_lifespan=2, energy_value=0.2)
        world.add_food(food)
        expired = world.decay_food()
        assert expired == 0
        assert world.food_count == 1
        expired = world.decay_food()
        assert expired == 1
        assert world.food_count == 0

    def test_spawn_no_overwrite(self, world):
        """Spawning shouldn't overwrite existing food."""
        food = Food(x=0, y=0, remaining_lifespan=999, energy_value=0.5)
        world.add_food(food)
        # Spawn many — none should overwrite (0,0)
        world.spawn_food(rate=100.0)
        assert world.food_at(0, 0).energy_value == 0.5

    def test_get_all_food_positions(self, world):
        world.add_food(Food(x=1, y=2, remaining_lifespan=10, energy_value=0.2))
        world.add_food(Food(x=3, y=4, remaining_lifespan=10, energy_value=0.2))
        positions = world.get_all_food_positions()
        assert (1, 2) in positions
        assert (3, 4) in positions
        assert len(positions) == 2


# ---------------------------------------------------------------------------
# Pitfall Management Tests
# ---------------------------------------------------------------------------

class TestPitfallManagement:
    def test_add_pitfall(self, world):
        p = Pitfall.from_string(x=5, y=10, name="A", sequence_str="1" * 32, lifespan=100)
        world.add_pitfall(p)
        assert world.pitfall_count == 1

    def test_pitfall_at(self, world):
        p = Pitfall.from_string(x=5, y=10, name="A", sequence_str="1" * 32, lifespan=100)
        world.add_pitfall(p)
        assert world.pitfall_at(5, 10) is p
        assert world.pitfall_at(0, 0) is None

    def test_spawn_pitfalls(self, world):
        spawned = world.spawn_pitfalls(rate=5.0)
        assert spawned > 0

    def test_spawn_pitfalls_zero_rate(self, world):
        assert world.spawn_pitfalls(rate=0.0) == 0

    def test_decay_pitfalls(self, world):
        p = Pitfall.from_string(x=5, y=10, name="A", sequence_str="1" * 32, lifespan=2)
        world.add_pitfall(p)
        expired = world.decay_pitfalls()
        assert expired == 0
        expired = world.decay_pitfalls()
        assert expired == 1
        assert world.pitfall_count == 0

    def test_spawn_pitfalls_batch(self, world):
        types = [PitfallType(name="X", sequence="10101010" * 4)]
        spawned = world.spawn_pitfalls_batch(count=10, pitfall_types=types)
        assert spawned > 0
        assert world.pitfall_count == spawned

    def test_get_all_pitfall_positions(self, world):
        world.add_pitfall(
            Pitfall.from_string(x=1, y=2, name="A", sequence_str="0" * 32, lifespan=10)
        )
        world.add_pitfall(
            Pitfall.from_string(x=3, y=4, name="B", sequence_str="1" * 32, lifespan=10)
        )
        positions = world.get_all_pitfall_positions()
        assert (1, 2) in positions
        assert (3, 4) in positions


# ---------------------------------------------------------------------------
# Spatial Query Tests
# ---------------------------------------------------------------------------

class TestSpatialQueries:
    def test_nearest_food_in_range_found(self, world):
        """Place food within range → should be found."""
        world.add_food(Food(x=10, y=10, remaining_lifespan=50, energy_value=0.2))
        food = world.nearest_food_in_range(10, 10, radius=5)
        assert food is not None
        assert food.position == (10, 10)

    def test_nearest_food_in_range_not_found(self, world):
        """Place food far away → should not be found."""
        world.add_food(Food(x=40, y=40, remaining_lifespan=50, energy_value=0.2))
        food = world.nearest_food_in_range(0, 0, radius=5)
        assert food is None

    def test_nearest_food_picks_closest(self, world):
        """When multiple foods are in range, return the closest one."""
        world.add_food(Food(x=12, y=10, remaining_lifespan=50, energy_value=0.2))
        world.add_food(Food(x=11, y=10, remaining_lifespan=50, energy_value=0.3))
        food = world.nearest_food_in_range(10, 10, radius=5)
        assert food is not None
        assert food.position == (11, 10)  # Closer

    def test_nearest_food_wrapping(self, world):
        """Food at wrapped position should be found."""
        world.add_food(Food(x=49, y=0, remaining_lifespan=50, energy_value=0.2))
        food = world.nearest_food_in_range(0, 0, radius=5)
        assert food is not None
        assert food.position == (49, 0)

    def test_food_in_range_bool(self, world):
        world.add_food(Food(x=10, y=10, remaining_lifespan=50, energy_value=0.2))
        assert world.food_in_range(10, 10, radius=5) is True
        assert world.food_in_range(40, 40, radius=5) is False

    def test_food_in_range_wrapping(self, world):
        world.add_food(Food(x=49, y=0, remaining_lifespan=50, energy_value=0.2))
        assert world.food_in_range(0, 0, radius=2) is True

    def test_nearest_food_ignores_consumed(self, world):
        food = Food(x=10, y=10, remaining_lifespan=50, energy_value=0.2)
        food.consumed = True
        world.food[(10, 10)] = food
        result = world.nearest_food_in_range(10, 10, radius=5)
        assert result is None

    def test_heaviest_animal_at(self, world, config):
        """Heaviest animal at a cell should be returned."""
        # Create animals with known weights by using known DNA
        dna_heavy = DNA(length=2048, bits=np.ones(2048, dtype=np.uint8))
        dna_light = DNA(length=2048, bits=np.zeros(2048, dtype=np.uint8))

        heavy = Animal(dna=dna_heavy, x=10, y=10, config=config)
        light = Animal(dna=dna_light, x=10, y=10, config=config)

        world.add_animal(heavy)
        world.add_animal(light)

        winner = world.heaviest_animal_at(10, 10)
        assert winner is not None
        assert winner.weight == heavy.weight

    def test_heaviest_animal_empty_cell(self, world):
        assert world.heaviest_animal_at(0, 0) is None

    def test_heaviest_animal_single(self, world, config):
        animal = make_animal_at(config, 10, 10)
        world.add_animal(animal)
        assert world.heaviest_animal_at(10, 10) is animal

    def test_heaviest_animal_tie_breaking(self, world, config):
        """Two animals with same weight → one should be returned (randomly)."""
        dna = DNA(length=2048, bits=np.ones(2048, dtype=np.uint8))
        a1 = Animal(dna=dna, x=5, y=5, config=config)
        # Need a second animal with same DNA (same weight)
        dna2 = DNA(length=2048, bits=np.ones(2048, dtype=np.uint8))
        a2 = Animal(dna=dna2, x=5, y=5, config=config)

        world.add_animal(a1)
        world.add_animal(a2)

        winner = world.heaviest_animal_at(5, 5)
        assert winner in (a1, a2)


# ---------------------------------------------------------------------------
# Decay All Resources
# ---------------------------------------------------------------------------

class TestDecayAllResources:
    def test_decay_both(self, world):
        world.add_food(Food(x=1, y=1, remaining_lifespan=1, energy_value=0.2))
        world.add_pitfall(
            Pitfall.from_string(x=2, y=2, name="A", sequence_str="1" * 32, lifespan=1)
        )
        food_exp, pit_exp = world.decay_all_resources()
        assert food_exp == 1
        assert pit_exp == 1
        assert world.food_count == 0
        assert world.pitfall_count == 0

    def test_decay_none_expired(self, world):
        world.add_food(Food(x=1, y=1, remaining_lifespan=100, energy_value=0.2))
        food_exp, pit_exp = world.decay_all_resources()
        assert food_exp == 0
        assert pit_exp == 0
        assert world.food_count == 1


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestWorldEdgeCases:
    def test_empty_world_queries(self, world):
        """All queries on empty world should return None/empty/False."""
        assert world.nearest_food_in_range(0, 0) is None
        assert world.food_in_range(0, 0) is False
        assert world.pitfall_at(0, 0) is None
        assert world.heaviest_animal_at(0, 0) is None
        assert world.get_alive_animals() == []
        assert world.get_all_food_positions() == []
        assert world.get_all_pitfall_positions() == []

    def test_many_animals_same_cell(self, world, config):
        """Multiple animals on the same cell should work."""
        for _ in range(10):
            world.add_animal(make_animal_at(config, 5, 5))
        assert len(world.animals_at(5, 5)) == 10
        assert world.alive_count == 10

    def test_remove_nonexistent_animal(self, world, config):
        """Removing an animal not in the world should not crash."""
        animal = make_animal_at(config, 5, 5)
        world.remove_animal(animal)  # Not added — should be silent

    def test_food_at_returns_none_for_expired(self, world):
        """food_at should return None for expired food."""
        food = Food(x=5, y=5, remaining_lifespan=0, energy_value=0.2)
        world.food[(5, 5)] = food
        assert world.food_at(5, 5) is None

    def test_pitfall_at_returns_none_for_expired(self, world):
        p = Pitfall.from_string(x=5, y=5, name="A", sequence_str="1" * 32, lifespan=0)
        world.pitfalls[(5, 5)] = p
        assert world.pitfall_at(5, 5) is None
