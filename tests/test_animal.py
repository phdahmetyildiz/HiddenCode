"""
Unit tests for the Animal agent.

Tests cover:
- Creation and property extraction from DNA
- Energy drain formula
- Energy gain and clamping
- Pitfall damage application
- Death checks (starvation, emergency)
- Reproduction (offspring count, offspring creation)
- Offspring position (3x3, toroidal)
- Defense bits caching
- Serialization (to_dict)
"""

import numpy as np
import pytest

from src.core.animal import Animal, reset_animal_id_counter
from src.core.config import SimConfig
from src.core.dna import DNA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ids():
    """Reset the animal ID counter before each test."""
    reset_animal_id_counter()
    yield
    reset_animal_id_counter()


@pytest.fixture
def config() -> SimConfig:
    """Default simulation config."""
    return SimConfig()


@pytest.fixture
def small_config() -> SimConfig:
    """Config for a small grid (useful for toroidal tests)."""
    cfg = SimConfig()
    cfg.world.width = 20
    cfg.world.height = 20
    return cfg


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def known_dna_all_ones() -> DNA:
    """DNA with all 1-bits → max weight, max speed, max defense."""
    return DNA(length=2048, bits=np.ones(2048, dtype=np.uint8))


@pytest.fixture
def known_dna_all_zeros() -> DNA:
    """DNA with all 0-bits → min weight, min speed, no defense."""
    return DNA(length=2048, bits=np.zeros(2048, dtype=np.uint8))


def make_animal(
    config: SimConfig,
    dna: DNA | None = None,
    x: int = 0,
    y: int = 0,
    energy: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Animal:
    """Helper to create an animal with known or random DNA."""
    if dna is None:
        if rng is None:
            rng = np.random.default_rng(42)
        dna = DNA(length=config.genetics.dna_length, rng=rng)
    return Animal(dna=dna, x=x, y=y, config=config, energy=energy)


# ---------------------------------------------------------------------------
# Creation Tests
# ---------------------------------------------------------------------------

class TestAnimalCreation:
    """Tests for Animal instantiation and property extraction."""

    def test_basic_creation(self, config, rng):
        animal = make_animal(config, rng=rng)
        assert animal.alive is True
        assert animal.energy == 1.0
        assert animal.birth_tick == 0
        assert animal.death_tick is None
        assert animal.death_cause is None

    def test_unique_ids(self, config, rng):
        a1 = make_animal(config, rng=np.random.default_rng(1))
        a2 = make_animal(config, rng=np.random.default_rng(2))
        assert a1.id != a2.id

    def test_weight_from_all_ones_dna(self, config, known_dna_all_ones):
        """All-ones DNA → normalized=1.0 → weight = weight_limits[1]."""
        animal = Animal(dna=known_dna_all_ones, x=0, y=0, config=config)
        assert abs(animal.weight - config.properties.weight_limits[1]) < 1e-10

    def test_weight_from_all_zeros_dna(self, config, known_dna_all_zeros):
        """All-zeros DNA → normalized=0.0 → weight = weight_limits[0]."""
        animal = Animal(dna=known_dna_all_zeros, x=0, y=0, config=config)
        assert abs(animal.weight - config.properties.weight_limits[0]) < 1e-10

    def test_speed_from_all_ones_dna(self, config, known_dna_all_ones):
        animal = Animal(dna=known_dna_all_ones, x=0, y=0, config=config)
        assert abs(animal.speed - config.properties.speed_limits[1]) < 1e-10

    def test_speed_from_all_zeros_dna(self, config, known_dna_all_zeros):
        animal = Animal(dna=known_dna_all_zeros, x=0, y=0, config=config)
        assert abs(animal.speed - config.properties.speed_limits[0]) < 1e-10

    def test_weight_within_limits(self, config, rng):
        """Any random DNA should produce weight within limits."""
        limits = config.properties.weight_limits
        for seed in range(50):
            dna = DNA(length=2048, rng=np.random.default_rng(seed))
            animal = Animal(dna=dna, x=0, y=0, config=config)
            assert limits[0] <= animal.weight <= limits[1], (
                f"Weight {animal.weight} out of range for seed {seed}"
            )

    def test_speed_within_limits(self, config, rng):
        limits = config.properties.speed_limits
        for seed in range(50):
            dna = DNA(length=2048, rng=np.random.default_rng(seed))
            animal = Animal(dna=dna, x=0, y=0, config=config)
            assert limits[0] <= animal.speed <= limits[1]

    def test_position(self, config):
        animal = make_animal(config, x=42, y=99)
        assert animal.x == 42
        assert animal.y == 99
        assert animal.position == (42, 99)

    def test_custom_energy(self, config):
        animal = make_animal(config, energy=0.5)
        assert animal.energy == 0.5

    def test_energy_clamped_above_1(self, config):
        animal = make_animal(config, energy=1.5)
        assert animal.energy == 1.0

    def test_energy_clamped_below_0(self, config):
        animal = make_animal(config, energy=-0.5)
        assert animal.energy == 0.0

    def test_eyesight_from_config(self, config):
        animal = make_animal(config)
        assert animal.eyesight_radius == config.properties.eyesight_radius


# ---------------------------------------------------------------------------
# Defense Bits Tests
# ---------------------------------------------------------------------------

class TestAnimalDefense:
    """Tests for defense bits extraction."""

    def test_defense_bits_all_ones(self, config, known_dna_all_ones):
        animal = Animal(dna=known_dna_all_ones, x=0, y=0, config=config)
        assert animal.defense_ones_count == 32
        assert len(animal.defense_bits) == 32
        assert all(b == 1 for b in animal.defense_bits)

    def test_defense_bits_all_zeros(self, config, known_dna_all_zeros):
        animal = Animal(dna=known_dna_all_zeros, x=0, y=0, config=config)
        assert animal.defense_ones_count == 0

    def test_defense_bits_cached(self, config, known_dna_all_ones):
        """Accessing defense_bits twice should return the same cached array."""
        animal = Animal(dna=known_dna_all_ones, x=0, y=0, config=config)
        bits1 = animal.defense_bits
        bits2 = animal.defense_bits
        assert bits1 is bits2  # Same object (cached)


# ---------------------------------------------------------------------------
# Energy Drain Tests
# ---------------------------------------------------------------------------

class TestEnergyDrain:
    """Tests for per-tick energy drain formula."""

    def test_drain_formula_basic(self, config, known_dna_all_zeros):
        """Minimum weight (0.1) and speed (0.1) → known drain value."""
        animal = Animal(dna=known_dna_all_zeros, x=0, y=0, config=config)
        expected = config.energy.base_metabolism + config.energy.k_weight_speed * 0.1 * 0.1
        drain = animal.calculate_energy_drain()
        assert abs(drain - expected) < 1e-10

    def test_drain_formula_max(self, config, known_dna_all_ones):
        """Maximum weight (1.0) and speed (1.0) → known drain value."""
        animal = Animal(dna=known_dna_all_ones, x=0, y=0, config=config)
        expected = config.energy.base_metabolism + config.energy.k_weight_speed * 1.0 * 1.0
        drain = animal.calculate_energy_drain()
        assert abs(drain - expected) < 1e-10

    def test_drain_increases_with_weight_speed(self, config):
        """Heavier and faster animals drain more energy."""
        dna_low = DNA(length=2048, bits=np.zeros(2048, dtype=np.uint8))
        dna_high = DNA(length=2048, bits=np.ones(2048, dtype=np.uint8))
        animal_low = Animal(dna=dna_low, x=0, y=0, config=config)
        animal_high = Animal(dna=dna_high, x=0, y=0, config=config)
        assert animal_high.calculate_energy_drain() > animal_low.calculate_energy_drain()

    def test_drain_with_defense_cost(self, config, known_dna_all_ones):
        """When defense_cost_enabled, defense 1-bits add to drain."""
        config.energy.defense_cost_enabled = True
        animal = Animal(dna=known_dna_all_ones, x=0, y=0, config=config)
        expected = (
            config.energy.base_metabolism
            + config.energy.k_weight_speed * 1.0 * 1.0
            + config.energy.k_defense_cost * 32
        )
        drain = animal.calculate_energy_drain()
        assert abs(drain - expected) < 1e-10

    def test_drain_without_defense_cost(self, config, known_dna_all_ones):
        """When defense_cost_enabled=False, defense bits don't affect drain."""
        config.energy.defense_cost_enabled = False
        animal = Animal(dna=known_dna_all_ones, x=0, y=0, config=config)
        expected = config.energy.base_metabolism + config.energy.k_weight_speed * 1.0 * 1.0
        drain = animal.calculate_energy_drain()
        assert abs(drain - expected) < 1e-10

    def test_apply_energy_drain(self, config):
        animal = make_animal(config, energy=1.0)
        drain = animal.apply_energy_drain()
        assert drain > 0
        assert animal.energy < 1.0
        assert animal.energy == 1.0 - drain

    def test_apply_energy_drain_clamps_at_zero(self, config):
        animal = make_animal(config, energy=0.0001)
        animal.apply_energy_drain()
        assert animal.energy >= 0.0

    def test_drain_always_positive(self, config):
        for seed in range(50):
            dna = DNA(length=2048, rng=np.random.default_rng(seed))
            animal = Animal(dna=dna, x=0, y=0, config=config)
            assert animal.calculate_energy_drain() > 0


# ---------------------------------------------------------------------------
# Energy Gain Tests
# ---------------------------------------------------------------------------

class TestEnergyGain:
    """Tests for energy gain from food."""

    def test_gain_energy(self, config):
        animal = make_animal(config, energy=0.5)
        gained = animal.gain_energy(0.2)
        assert abs(gained - 0.2) < 1e-10
        assert abs(animal.energy - 0.7) < 1e-10

    def test_gain_energy_capped_at_1(self, config):
        animal = make_animal(config, energy=0.9)
        gained = animal.gain_energy(0.5)
        assert abs(animal.energy - 1.0) < 1e-10
        assert abs(gained - 0.1) < 1e-10  # Only 0.1 was actually gained

    def test_gain_zero(self, config):
        animal = make_animal(config, energy=0.5)
        gained = animal.gain_energy(0.0)
        assert gained == 0.0
        assert animal.energy == 0.5


# ---------------------------------------------------------------------------
# Pitfall Damage Application Tests
# ---------------------------------------------------------------------------

class TestPitfallDamageApplication:
    """Tests for applying pitfall energy loss."""

    def test_apply_pitfall_damage(self, config):
        animal = make_animal(config, energy=0.8)
        lost = animal.apply_pitfall_damage(0.25)
        assert abs(lost - 0.25) < 1e-10
        assert abs(animal.energy - 0.55) < 1e-10

    def test_pitfall_damage_clamps_at_zero(self, config):
        animal = make_animal(config, energy=0.1)
        lost = animal.apply_pitfall_damage(0.5)
        assert animal.energy == 0.0
        assert abs(lost - 0.1) < 1e-10  # Only lost what was available

    def test_zero_damage(self, config):
        animal = make_animal(config, energy=0.8)
        lost = animal.apply_pitfall_damage(0.0)
        assert lost == 0.0
        assert animal.energy == 0.8


# ---------------------------------------------------------------------------
# Death Check Tests
# ---------------------------------------------------------------------------

class TestDeathChecks:
    """Tests for starvation and emergency death checks."""

    def test_is_starved_at_zero(self, config):
        animal = make_animal(config, energy=0.0)
        assert animal.is_starved() is True

    def test_not_starved_above_zero(self, config):
        animal = make_animal(config, energy=0.01)
        assert animal.is_starved() is False

    def test_emergency_low_energy_no_food(self, config):
        """Energy below threshold and no food → emergency death."""
        animal = make_animal(config, energy=0.05)
        assert animal.is_emergency(food_in_range=False) is True

    def test_no_emergency_low_energy_with_food(self, config):
        """Energy below threshold but food nearby → no emergency."""
        animal = make_animal(config, energy=0.05)
        assert animal.is_emergency(food_in_range=True) is False

    def test_no_emergency_high_energy(self, config):
        """Energy above threshold → no emergency regardless of food."""
        animal = make_animal(config, energy=0.5)
        assert animal.is_emergency(food_in_range=False) is False

    def test_emergency_exactly_at_threshold(self, config):
        """Energy exactly at threshold → NOT emergency (< not <=)."""
        threshold = config.energy.low_energy_death_threshold
        animal = make_animal(config, energy=threshold)
        assert animal.is_emergency(food_in_range=False) is False

    def test_die_marks_dead(self, config):
        animal = make_animal(config, energy=0.5)
        animal.die(cause="starvation", tick=500)
        assert animal.alive is False
        assert animal.death_tick == 500
        assert animal.death_cause == "starvation"

    def test_die_different_causes(self, config):
        for cause in ["starvation", "emergency", "age", "pitfall"]:
            animal = make_animal(config)
            animal.die(cause=cause, tick=100)
            assert animal.death_cause == cause


# ---------------------------------------------------------------------------
# Reproduction Tests
# ---------------------------------------------------------------------------

class TestReproduction:
    """Tests for offspring count and creation."""

    def test_offspring_count_low_energy(self, config):
        """Energy below repro_energy_low → 0 offspring."""
        animal = make_animal(config, energy=0.3)
        assert animal.offspring_count() == 0

    def test_offspring_count_mid_energy(self, config):
        """Energy between low and high → 1 offspring."""
        animal = make_animal(config, energy=0.6)
        assert animal.offspring_count() == 1

    def test_offspring_count_high_energy(self, config):
        """Energy >= repro_energy_high → 2 offspring."""
        animal = make_animal(config, energy=0.8)
        assert animal.offspring_count() == 2

    def test_offspring_count_exactly_at_low(self, config):
        """Energy exactly at repro_energy_low → 1 offspring (>= low, < high)."""
        animal = make_animal(config, energy=config.generation.repro_energy_low)
        assert animal.offspring_count() == 1

    def test_offspring_count_exactly_at_high(self, config):
        """Energy exactly at repro_energy_high → 2 offspring."""
        animal = make_animal(config, energy=config.generation.repro_energy_high)
        assert animal.offspring_count() == 2

    def test_survives_generation_above_threshold(self, config):
        animal = make_animal(config, energy=0.6)
        assert animal.survives_generation_end() is True

    def test_dies_at_generation_below_threshold(self, config):
        animal = make_animal(config, energy=0.3)
        assert animal.survives_generation_end() is False

    def test_survives_exactly_at_threshold(self, config):
        """Exactly at threshold → does NOT survive (> not >=)."""
        animal = make_animal(config, energy=config.generation.survival_threshold)
        assert animal.survives_generation_end() is False


# ---------------------------------------------------------------------------
# Offspring Creation Tests
# ---------------------------------------------------------------------------

class TestOffspringCreation:
    """Tests for the create_offspring method."""

    def test_offspring_has_energy_1(self, config, rng):
        parent = make_animal(config, energy=0.8, rng=rng)
        child = parent.create_offspring(
            current_tick=100, stress_mode=False, rng=rng,
            world_width=500, world_height=500, generation=1,
        )
        assert child.energy == 1.0

    def test_offspring_is_alive(self, config, rng):
        parent = make_animal(config, rng=rng)
        child = parent.create_offspring(
            current_tick=100, stress_mode=False, rng=rng,
            world_width=500, world_height=500, generation=1,
        )
        assert child.alive is True

    def test_offspring_birth_tick(self, config, rng):
        parent = make_animal(config, rng=rng)
        child = parent.create_offspring(
            current_tick=777, stress_mode=False, rng=rng,
            world_width=500, world_height=500, generation=3,
        )
        assert child.birth_tick == 777
        assert child.generation == 3

    def test_offspring_position_near_parent(self, config, rng):
        """Offspring should be within 3x3 area around parent."""
        parent = make_animal(config, x=250, y=250, rng=rng)
        for seed in range(50):
            r = np.random.default_rng(seed)
            child = parent.create_offspring(
                current_tick=100, stress_mode=False, rng=r,
                world_width=500, world_height=500, generation=1,
            )
            dx = abs(child.x - parent.x)
            dy = abs(child.y - parent.y)
            assert dx <= 1, f"dx={dx} for seed {seed}"
            assert dy <= 1, f"dy={dy} for seed {seed}"

    def test_offspring_position_toroidal(self, small_config):
        """Offspring near grid edge should wrap toroidally."""
        rng = np.random.default_rng(42)
        parent = make_animal(small_config, x=0, y=0, rng=rng)
        # Run many times — some offspring should wrap to 19 (width-1)
        positions = set()
        for seed in range(100):
            r = np.random.default_rng(seed)
            child = parent.create_offspring(
                current_tick=0, stress_mode=False, rng=r,
                world_width=20, world_height=20, generation=0,
            )
            positions.add((child.x, child.y))
            assert 0 <= child.x < 20
            assert 0 <= child.y < 20

        # Should include wrapped positions (19 = -1 mod 20)
        assert any(pos[0] == 19 or pos[1] == 19 for pos in positions), (
            "Expected some offspring to wrap to position 19"
        )

    def test_offspring_dna_differs(self, config, rng):
        """Offspring DNA should be a mutated copy of parent DNA."""
        parent = make_animal(config, rng=rng)
        # Use high mutation to guarantee difference
        config.genetics.base_mutation_rate = 0.5
        child = parent.create_offspring(
            current_tick=100, stress_mode=False, rng=np.random.default_rng(99),
            world_width=500, world_height=500, generation=1,
        )
        assert parent.dna != child.dna

    def test_offspring_dna_does_not_change_parent(self, config, rng):
        """Creating offspring should not modify parent's DNA."""
        parent = make_animal(config, rng=rng)
        original_bits = parent.dna.bits.copy()
        parent.create_offspring(
            current_tick=100, stress_mode=False, rng=np.random.default_rng(42),
            world_width=500, world_height=500, generation=1,
        )
        np.testing.assert_array_equal(parent.dna.bits, original_bits)

    def test_stress_mode_uses_stress_rate(self, config, rng):
        """In stress mode, offspring DNA should mutate at the higher stress rate."""
        config.genetics.base_mutation_rate = 0.01
        config.genetics.stress_mutation_rate = 0.50

        parent = make_animal(config, rng=rng)

        # Stress=False
        base_distances = []
        for seed in range(20):
            child = parent.create_offspring(
                current_tick=0, stress_mode=False, rng=np.random.default_rng(seed),
                world_width=500, world_height=500, generation=0,
            )
            base_distances.append(parent.dna.hamming_distance(child.dna))

        # Stress=True
        stress_distances = []
        for seed in range(20):
            child = parent.create_offspring(
                current_tick=0, stress_mode=True, rng=np.random.default_rng(seed),
                world_width=500, world_height=500, generation=0,
            )
            stress_distances.append(parent.dna.hamming_distance(child.dna))

        assert np.mean(stress_distances) > np.mean(base_distances) * 2

    def test_offspring_has_valid_properties(self, config, rng):
        """Offspring should have weight and speed within config limits."""
        parent = make_animal(config, rng=rng)
        child = parent.create_offspring(
            current_tick=100, stress_mode=False, rng=np.random.default_rng(99),
            world_width=500, world_height=500, generation=1,
        )
        assert config.properties.weight_limits[0] <= child.weight <= config.properties.weight_limits[1]
        assert config.properties.speed_limits[0] <= child.speed <= config.properties.speed_limits[1]


# ---------------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------------

class TestAnimalSerialization:
    """Tests for to_dict serialization."""

    def test_to_dict_has_all_fields(self, config, rng):
        animal = make_animal(config, rng=rng)
        d = animal.to_dict()
        required_keys = {
            "id", "x", "y", "energy", "weight", "speed",
            "defense_ones", "alive", "birth_tick", "death_tick",
            "death_cause", "generation",
        }
        assert required_keys.issubset(d.keys())

    def test_to_dict_values(self, config, rng):
        animal = make_animal(config, x=10, y=20, energy=0.75, rng=rng)
        d = animal.to_dict()
        assert d["x"] == 10
        assert d["y"] == 20
        assert abs(d["energy"] - 0.75) < 1e-5
        assert d["alive"] is True
        assert d["death_cause"] is None

    def test_to_dict_after_death(self, config, rng):
        animal = make_animal(config, rng=rng)
        animal.die(cause="starvation", tick=500)
        d = animal.to_dict()
        assert d["alive"] is False
        assert d["death_cause"] == "starvation"
        assert d["death_tick"] == 500


# ---------------------------------------------------------------------------
# Repr Test
# ---------------------------------------------------------------------------

class TestAnimalRepr:
    def test_repr_alive(self, config, rng):
        animal = make_animal(config, rng=rng)
        r = repr(animal)
        assert "Animal" in r
        assert "alive" in r

    def test_repr_dead(self, config, rng):
        animal = make_animal(config, rng=rng)
        animal.die("starvation", 100)
        r = repr(animal)
        assert "dead" in r
        assert "starvation" in r
