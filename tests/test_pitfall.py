"""
Unit tests for the Pitfall resource.

Tests cover:
- Creation (from array, from string)
- Lifespan ticking and expiry
- Damage calculation (bitwise comparison)
- Energy loss calculation
- Edge cases (perfect defense, no defense, partial match)
- Properties
"""

import numpy as np
import pytest

from src.core.pitfall import Pitfall


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pitfall_all_ones() -> Pitfall:
    """Pitfall with all-ones sequence (maximum danger)."""
    return Pitfall.from_string(
        x=10, y=20, name="AllDanger",
        sequence_str="1" * 32,
        lifespan=100,
    )


@pytest.fixture
def pitfall_all_zeros() -> Pitfall:
    """Pitfall with all-zeros sequence (no danger)."""
    return Pitfall.from_string(
        x=5, y=5, name="NoDanger",
        sequence_str="0" * 32,
        lifespan=50,
    )


@pytest.fixture
def pitfall_type_a() -> Pitfall:
    """Standard pitfall type A from the spec."""
    return Pitfall.from_string(
        x=0, y=0, name="A",
        sequence_str="11110000111100001111000011110000",
        lifespan=100,
    )


@pytest.fixture
def defense_all_ones() -> np.ndarray:
    return np.ones(32, dtype=np.uint8)


@pytest.fixture
def defense_all_zeros() -> np.ndarray:
    return np.zeros(32, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Creation Tests
# ---------------------------------------------------------------------------

class TestPitfallCreation:
    """Tests for Pitfall instantiation."""

    def test_from_string(self):
        p = Pitfall.from_string(
            x=10, y=20, name="A",
            sequence_str="11110000111100001111000011110000",
            lifespan=100,
        )
        assert p.x == 10
        assert p.y == 20
        assert p.name == "A"
        assert len(p.sequence) == 32
        assert p.remaining_lifespan == 100

    def test_position_property(self):
        p = Pitfall.from_string(x=3, y=7, name="B", sequence_str="0" * 32, lifespan=50)
        assert p.position == (3, 7)

    def test_sequence_str_property(self):
        seq = "11110000111100001111000011110000"
        p = Pitfall.from_string(x=0, y=0, name="A", sequence_str=seq, lifespan=50)
        assert p.sequence_str == seq

    def test_initial_state_active(self):
        p = Pitfall.from_string(x=0, y=0, name="A", sequence_str="0" * 32, lifespan=10)
        assert p.active is True
        assert p.expired is False

    def test_from_numpy_array(self):
        seq = np.array([1, 0] * 16, dtype=np.uint8)
        p = Pitfall(x=0, y=0, name="Alt", sequence=seq, remaining_lifespan=50)
        assert len(p.sequence) == 32
        assert p.sequence[0] == 1
        assert p.sequence[1] == 0

    def test_num_danger_bits_all_ones(self, pitfall_all_ones):
        assert pitfall_all_ones.num_danger_bits == 32

    def test_num_danger_bits_all_zeros(self, pitfall_all_zeros):
        assert pitfall_all_zeros.num_danger_bits == 0

    def test_num_danger_bits_type_a(self, pitfall_type_a):
        # "11110000" repeated 4 times = 16 ones
        assert pitfall_type_a.num_danger_bits == 16


# ---------------------------------------------------------------------------
# Lifespan / Tick Tests
# ---------------------------------------------------------------------------

class TestPitfallLifespan:
    """Tests for pitfall lifespan and ticking."""

    def test_tick_decrements(self):
        p = Pitfall.from_string(x=0, y=0, name="A", sequence_str="0" * 32, lifespan=10)
        p.tick()
        assert p.remaining_lifespan == 9

    def test_tick_returns_false_when_active(self):
        p = Pitfall.from_string(x=0, y=0, name="A", sequence_str="0" * 32, lifespan=10)
        assert p.tick() is False

    def test_tick_returns_true_when_expired(self):
        p = Pitfall.from_string(x=0, y=0, name="A", sequence_str="0" * 32, lifespan=1)
        assert p.tick() is True

    def test_full_lifespan(self):
        lifespan = 100
        p = Pitfall.from_string(x=0, y=0, name="A", sequence_str="1" * 32, lifespan=lifespan)
        for _ in range(lifespan - 1):
            assert p.tick() is False
        assert p.tick() is True
        assert p.expired is True


# ---------------------------------------------------------------------------
# Damage Calculation Tests
# ---------------------------------------------------------------------------

class TestPitfallDamage:
    """Tests for the bitwise damage calculation."""

    def test_no_defense_vs_all_danger(self, pitfall_all_ones, defense_all_zeros):
        """No defense against all-ones pitfall → 32 damage."""
        assert pitfall_all_ones.calculate_damage(defense_all_zeros) == 32

    def test_full_defense_vs_all_danger(self, pitfall_all_ones, defense_all_ones):
        """Perfect defense against all-ones pitfall → 0 damage."""
        assert pitfall_all_ones.calculate_damage(defense_all_ones) == 0

    def test_any_defense_vs_no_danger(self, pitfall_all_zeros, defense_all_ones):
        """Any defense against all-zeros pitfall → 0 damage."""
        assert pitfall_all_zeros.calculate_damage(defense_all_ones) == 0

    def test_no_defense_vs_no_danger(self, pitfall_all_zeros, defense_all_zeros):
        """No defense against all-zeros pitfall → 0 damage."""
        assert pitfall_all_zeros.calculate_damage(defense_all_zeros) == 0

    def test_type_a_no_defense(self, pitfall_type_a, defense_all_zeros):
        """Type A has 16 danger bits → 16 damage with no defense."""
        assert pitfall_type_a.calculate_damage(defense_all_zeros) == 16

    def test_type_a_full_defense(self, pitfall_type_a, defense_all_ones):
        """Type A → 0 damage with full defense."""
        assert pitfall_type_a.calculate_damage(defense_all_ones) == 0

    def test_partial_defense(self, pitfall_type_a):
        """Partial defense: match first 4 bits, miss next 4."""
        # Type A = 11110000 11110000 11110000 11110000
        # Defense = 11110000 00000000 00000000 00000000
        defense = np.zeros(32, dtype=np.uint8)
        defense[0:4] = 1  # Match first 4 danger bits
        # Damage = 16 total danger bits - 4 matched = 12
        assert pitfall_type_a.calculate_damage(defense) == 12

    def test_exact_match_defense(self, pitfall_type_a):
        """Defense bits exactly match pitfall → 0 damage."""
        defense = np.array([int(c) for c in "11110000111100001111000011110000"], dtype=np.uint8)
        assert pitfall_type_a.calculate_damage(defense) == 0

    def test_inverted_defense(self, pitfall_type_a):
        """Defense is inverted of pitfall → maximum damage for this type."""
        defense = np.array([int(c) for c in "00001111000011110000111100001111"], dtype=np.uint8)
        # Pitfall 1s at positions where defense has 0 → all 16 hit
        assert pitfall_type_a.calculate_damage(defense) == 16

    def test_single_bit_damage(self):
        """Pitfall with single danger bit, no defense at that bit."""
        seq = "0" * 31 + "1"
        p = Pitfall.from_string(x=0, y=0, name="T", sequence_str=seq, lifespan=10)
        defense = np.zeros(32, dtype=np.uint8)
        assert p.calculate_damage(defense) == 1

    def test_single_bit_immune(self):
        """Pitfall with single danger bit, defense at that bit."""
        seq = "0" * 31 + "1"
        p = Pitfall.from_string(x=0, y=0, name="T", sequence_str=seq, lifespan=10)
        defense = np.zeros(32, dtype=np.uint8)
        defense[31] = 1
        assert p.calculate_damage(defense) == 0

    def test_wrong_defense_length_raises(self, pitfall_type_a):
        """Defense bits of wrong length should raise ValueError."""
        defense_short = np.zeros(16, dtype=np.uint8)
        with pytest.raises(ValueError, match="Defense bits length"):
            pitfall_type_a.calculate_damage(defense_short)

    def test_damage_range(self, pitfall_type_a):
        """Damage should always be in [0, 32]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            defense = rng.integers(0, 2, size=32, dtype=np.uint8)
            damage = pitfall_type_a.calculate_damage(defense)
            assert 0 <= damage <= 32


# ---------------------------------------------------------------------------
# Energy Loss Calculation Tests
# ---------------------------------------------------------------------------

class TestPitfallEnergyLoss:
    """Tests for energy loss calculation from pitfall encounters."""

    def test_max_damage_max_loss(self, pitfall_all_ones, defense_all_zeros):
        """32/32 damage at 0.5 max → 0.5 energy loss."""
        loss = pitfall_all_ones.calculate_energy_loss(defense_all_zeros, max_pitfall_loss_pct=0.5)
        assert abs(loss - 0.5) < 1e-10

    def test_zero_damage_zero_loss(self, pitfall_all_ones, defense_all_ones):
        """0/32 damage → 0.0 energy loss."""
        loss = pitfall_all_ones.calculate_energy_loss(defense_all_ones, max_pitfall_loss_pct=0.5)
        assert abs(loss - 0.0) < 1e-10

    def test_half_damage_half_loss(self, pitfall_type_a, defense_all_zeros):
        """16/32 damage at max 0.5 → 0.25 energy loss."""
        loss = pitfall_type_a.calculate_energy_loss(defense_all_zeros, max_pitfall_loss_pct=0.5)
        assert abs(loss - 0.25) < 1e-10

    def test_custom_max_loss(self, pitfall_all_ones, defense_all_zeros):
        """Full damage at max 1.0 → 1.0 energy loss."""
        loss = pitfall_all_ones.calculate_energy_loss(defense_all_zeros, max_pitfall_loss_pct=1.0)
        assert abs(loss - 1.0) < 1e-10

    def test_zero_max_loss(self, pitfall_all_ones, defense_all_zeros):
        """Even with full damage, max_loss=0 → 0 energy loss."""
        loss = pitfall_all_ones.calculate_energy_loss(defense_all_zeros, max_pitfall_loss_pct=0.0)
        assert abs(loss - 0.0) < 1e-10

    def test_loss_proportional_to_damage(self, pitfall_type_a):
        """More damage → more energy loss, linearly."""
        defense_none = np.zeros(32, dtype=np.uint8)
        defense_half = np.array([int(c) for c in "11110000111100001111000011110000"], dtype=np.uint8)

        loss_none = pitfall_type_a.calculate_energy_loss(defense_none, 0.5)
        loss_half = pitfall_type_a.calculate_energy_loss(defense_half, 0.5)

        assert loss_none > loss_half
        assert loss_half == 0.0


# ---------------------------------------------------------------------------
# Repr Test
# ---------------------------------------------------------------------------

class TestPitfallRepr:
    def test_repr_active(self):
        p = Pitfall.from_string(x=5, y=10, name="A", sequence_str="1" * 32, lifespan=50)
        r = repr(p)
        assert "active" in r
        assert "A" in r
        assert "danger_bits=32" in r

    def test_repr_expired(self):
        p = Pitfall.from_string(x=5, y=10, name="A", sequence_str="1" * 32, lifespan=0)
        assert "expired" in repr(p)
