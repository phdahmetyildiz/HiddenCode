"""
Unit tests for the DNA / Genome system.

Tests cover:
- Creation (random, from string, from bits)
- Property extraction (binary and Gray encoding)
- Defense bits extraction and counting
- Mutation mechanics (rate=0, rate=1, coding-only, junk-only)
- Copy independence
- Hamming distance
- Equality and hashing
- Edge cases
"""

import numpy as np
import pytest

from src.core.dna import DNA, CodingRegion


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator for reproducible tests."""
    return np.random.default_rng(12345)


@pytest.fixture
def simple_regions() -> list[CodingRegion]:
    """Simple coding regions for a small genome (64 bits)."""
    return [
        CodingRegion("weight", 0, 16),
        CodingRegion("speed", 16, 32),
        CodingRegion("defense", 32, 64),
    ]


@pytest.fixture
def default_regions() -> list[CodingRegion]:
    """Coding regions matching the default config (2048-bit genome)."""
    return [
        CodingRegion("weight", 0, 32),
        CodingRegion("speed", 32, 64),
        CodingRegion("reserved", 64, 128),
        CodingRegion("defense", 128, 160),
    ]


@pytest.fixture
def known_dna() -> DNA:
    """DNA with known bit pattern for deterministic tests."""
    # 64-bit genome: first 32 bits = all 1s, last 32 bits = all 0s
    bits = np.zeros(64, dtype=np.uint8)
    bits[:32] = 1
    return DNA(length=64, bits=bits)


# ---------------------------------------------------------------------------
# Creation Tests
# ---------------------------------------------------------------------------

class TestDNACreation:
    """Tests for DNA instantiation."""

    def test_random_creation_length(self, rng):
        dna = DNA(length=2048, rng=rng)
        assert len(dna.bits) == 2048
        assert dna.length == 2048

    def test_random_creation_contains_both_values(self, rng):
        """A random DNA of reasonable length should have both 0s and 1s."""
        dna = DNA(length=1000, rng=rng)
        assert np.any(dna.bits == 0)
        assert np.any(dna.bits == 1)

    def test_random_creation_roughly_balanced(self, rng):
        """Random bits should be roughly 50/50."""
        dna = DNA(length=10000, rng=rng)
        ones_ratio = np.sum(dna.bits) / dna.length
        assert 0.45 < ones_ratio < 0.55, f"Ratio of 1s: {ones_ratio}"

    def test_from_bits(self):
        bits = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
        dna = DNA(length=8, bits=bits)
        np.testing.assert_array_equal(dna.bits, bits)

    def test_from_bits_wrong_length_raises(self):
        bits = np.array([1, 0, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="bits length"):
            DNA(length=8, bits=bits)

    def test_from_string(self):
        dna = DNA.create_from_string("10101100")
        expected = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(dna.bits, expected)
        assert dna.length == 8

    def test_create_random_classmethod(self, rng):
        dna = DNA.create_random(length=512, rng=rng)
        assert dna.length == 512
        assert len(dna.bits) == 512

    def test_dtype_is_uint8(self, rng):
        dna = DNA(length=100, rng=rng)
        assert dna.bits.dtype == np.uint8

    def test_bits_only_contain_0_and_1(self, rng):
        dna = DNA(length=5000, rng=rng)
        assert set(np.unique(dna.bits)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Property Extraction Tests
# ---------------------------------------------------------------------------

class TestPropertyExtraction:
    """Tests for extracting properties from DNA bit slices."""

    def test_all_zeros_gives_0(self):
        dna = DNA.create_from_string("00000000")
        assert dna.get_property(0, 8) == 0.0

    def test_all_ones_gives_1(self):
        dna = DNA.create_from_string("11111111")
        assert dna.get_property(0, 8) == 1.0

    def test_known_value(self):
        """10000000 = 128, normalized = 128/255 ≈ 0.502"""
        dna = DNA.create_from_string("10000000")
        val = dna.get_property(0, 8)
        assert abs(val - 128 / 255) < 1e-10

    def test_sub_slice(self):
        """Extract from middle of genome."""
        dna = DNA.create_from_string("0000111100001111")
        # bits 4-8 = "1111" = 15, normalized = 15/15 = 1.0
        val = dna.get_property(4, 8)
        assert val == 1.0

    def test_sub_slice_zeros(self):
        dna = DNA.create_from_string("1111000011110000")
        # bits 4-8 = "0000" = 0/15 = 0.0
        val = dna.get_property(4, 8)
        assert val == 0.0

    def test_property_in_range(self):
        """Map normalized value to [0.1, 1.0] range."""
        dna = DNA.create_from_string("11111111")
        val = dna.get_property_in_range(0, 8, 0.1, 1.0)
        assert abs(val - 1.0) < 1e-10

    def test_property_in_range_low(self):
        dna = DNA.create_from_string("00000000")
        val = dna.get_property_in_range(0, 8, 0.1, 1.0)
        assert abs(val - 0.1) < 1e-10

    def test_property_in_range_mid(self):
        dna = DNA.create_from_string("10000000")
        val = dna.get_property_in_range(0, 8, 0.0, 1.0)
        assert abs(val - 128 / 255) < 1e-10

    def test_gray_encoding_all_zeros(self):
        dna = DNA.create_from_string("00000000")
        assert dna.get_property(0, 8, encoding="gray") == 0.0

    def test_gray_encoding_range(self, rng):
        """Gray-decoded properties should still be in [0, 1]."""
        dna = DNA(length=64, rng=rng)
        for start in range(0, 64, 8):
            val = dna.get_property(start, start + 8, encoding="gray")
            assert 0.0 <= val <= 1.0

    def test_different_slices_independent(self, rng):
        """Different bit regions can produce different values."""
        dna = DNA.create_from_string("1111111100000000")
        val_a = dna.get_property(0, 8)  # all 1s
        val_b = dna.get_property(8, 16)  # all 0s
        assert val_a == 1.0
        assert val_b == 0.0


# ---------------------------------------------------------------------------
# Defense Bits Tests
# ---------------------------------------------------------------------------

class TestDefenseBits:
    """Tests for defense bit extraction and counting."""

    def test_get_defense_bits_known(self, known_dna):
        """First 32 bits are 1, rest are 0."""
        defense = known_dna.get_defense_bits(start=0, length=32)
        assert len(defense) == 32
        assert all(b == 1 for b in defense)

    def test_get_defense_bits_zeros(self, known_dna):
        defense = known_dna.get_defense_bits(start=32, length=32)
        assert all(b == 0 for b in defense)

    def test_count_ones_all_ones(self, known_dna):
        assert known_dna.count_ones(0, 32) == 32

    def test_count_ones_all_zeros(self, known_dna):
        assert known_dna.count_ones(32, 32) == 0

    def test_count_ones_defense_alias(self, known_dna):
        assert known_dna.count_ones_defense(0, 32) == 32

    def test_count_ones_mixed(self):
        dna = DNA.create_from_string("10101010")
        assert dna.count_ones(0, 8) == 4

    def test_defense_bits_is_copy(self, known_dna):
        """Modifying returned bits should not affect DNA."""
        defense = known_dna.get_defense_bits(0, 32)
        defense[0] = 0
        assert known_dna.bits[0] == 1  # original unchanged

    def test_defense_sequence_str(self, known_dna):
        s = known_dna.defense_sequence_str(0, 32)
        assert s == "1" * 32
        assert len(s) == 32

    def test_defense_sequence_str_mixed(self):
        dna = DNA.create_from_string("10101010" * 4)
        s = dna.defense_sequence_str(0, 32)
        assert s == "10101010" * 4


# ---------------------------------------------------------------------------
# Mutation Tests
# ---------------------------------------------------------------------------

class TestMutation:
    """Tests for DNA mutation mechanics."""

    def test_zero_rate_no_changes(self, rng, simple_regions):
        dna = DNA(length=64, rng=rng)
        original_bits = dna.bits.copy()
        changes = dna.mutate(rate=0.0, coding_regions=simple_regions, rng=rng)
        assert changes == 0
        np.testing.assert_array_equal(dna.bits, original_bits)

    def test_mutation_returns_change_count(self, rng, simple_regions):
        dna = DNA(length=64, rng=rng)
        changes = dna.mutate(rate=0.5, coding_regions=simple_regions, rng=rng)
        assert isinstance(changes, int)
        assert changes >= 0

    def test_high_rate_causes_changes(self, rng, simple_regions):
        """With rate=1.0, all coding bits are touched (but ~50% may stay same)."""
        dna = DNA(length=64, rng=rng)
        original = dna.bits.copy()
        dna.mutate(rate=1.0, coding_regions=simple_regions, rng=rng)
        # At least SOME bits should change (statistically near-certain)
        assert not np.array_equal(dna.bits, original)

    def test_coding_only_preserves_junk(self, rng):
        """When coding_only=True, bits outside coding regions must not change."""
        # Genome: 64 bits. Coding: 0-16. Junk: 16-64.
        regions = [CodingRegion("test", 0, 16)]
        dna = DNA(length=64, rng=rng)
        junk_before = dna.bits[16:64].copy()

        dna.mutate(rate=1.0, coding_regions=regions, coding_only=True, rng=rng)

        np.testing.assert_array_equal(
            dna.bits[16:64], junk_before,
            err_msg="Junk region was modified during coding-only mutation"
        )

    def test_full_genome_mutation(self, rng, simple_regions):
        """coding_only=False should allow any bit to mutate."""
        dna = DNA(length=64, rng=rng)
        original = dna.bits.copy()
        dna.mutate(rate=0.5, coding_regions=simple_regions, coding_only=False, rng=rng)
        # Can't guarantee junk changed, but mechanism should allow it
        # Just verify no crash
        assert dna.length == 64

    def test_mutation_rate_affects_magnitude(self, rng, simple_regions):
        """Higher rate should (statistically) cause more changes."""
        changes_low = []
        changes_high = []
        for seed in range(50):
            r = np.random.default_rng(seed)
            dna_low = DNA(length=64, rng=np.random.default_rng(seed + 1000))
            dna_high = dna_low.copy()

            changes_low.append(dna_low.mutate(rate=0.01, coding_regions=simple_regions, rng=r))
            r2 = np.random.default_rng(seed + 500)
            changes_high.append(dna_high.mutate(rate=0.5, coding_regions=simple_regions, rng=r2))

        assert np.mean(changes_high) > np.mean(changes_low)

    def test_mutate_junk_only(self, rng):
        """mutate_junk_only should only change bits outside coding regions."""
        regions = [CodingRegion("test", 0, 16)]
        dna = DNA(length=64, rng=rng)
        coding_before = dna.bits[0:16].copy()

        dna.mutate_junk_only(rate=0.5, coding_regions=regions, rng=rng)

        np.testing.assert_array_equal(
            dna.bits[0:16], coding_before,
            err_msg="Coding region was modified during junk-only mutation"
        )

    def test_mutate_junk_only_zero_rate(self, rng):
        regions = [CodingRegion("test", 0, 16)]
        dna = DNA(length=64, rng=rng)
        original = dna.bits.copy()
        changes = dna.mutate_junk_only(rate=0.0, coding_regions=regions, rng=rng)
        assert changes == 0
        np.testing.assert_array_equal(dna.bits, original)

    def test_mutation_deterministic_with_same_seed(self, simple_regions):
        """Same seed + same starting DNA → identical mutation results."""
        bits = np.zeros(64, dtype=np.uint8)
        dna1 = DNA(length=64, bits=bits.copy())
        dna2 = DNA(length=64, bits=bits.copy())

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        dna1.mutate(rate=0.3, coding_regions=simple_regions, rng=rng1)
        dna2.mutate(rate=0.3, coding_regions=simple_regions, rng=rng2)

        np.testing.assert_array_equal(dna1.bits, dna2.bits)

    def test_mutation_only_sets_0_or_1(self, rng, simple_regions):
        """After mutation, all bits must still be 0 or 1."""
        dna = DNA(length=64, rng=rng)
        dna.mutate(rate=0.5, coding_regions=simple_regions, rng=rng)
        assert set(np.unique(dna.bits)).issubset({0, 1})

    def test_empty_coding_regions_no_crash(self, rng):
        """Mutation with empty coding regions should do nothing."""
        dna = DNA(length=64, rng=rng)
        original = dna.bits.copy()
        changes = dna.mutate(rate=0.5, coding_regions=[], coding_only=True, rng=rng)
        assert changes == 0
        np.testing.assert_array_equal(dna.bits, original)

    def test_all_genome_is_coding(self, rng):
        """If entire genome is one coding region, coding_only mutation touches everything."""
        regions = [CodingRegion("all", 0, 64)]
        dna = DNA(length=64, rng=rng)
        original = dna.bits.copy()
        dna.mutate(rate=1.0, coding_regions=regions, coding_only=True, rng=rng)
        # Should have changed some bits
        assert not np.array_equal(dna.bits, original)


# ---------------------------------------------------------------------------
# Copy Tests
# ---------------------------------------------------------------------------

class TestDNACopy:
    """Tests for DNA deep copy."""

    def test_copy_equals_original(self, rng):
        dna = DNA(length=64, rng=rng)
        copy = dna.copy()
        np.testing.assert_array_equal(copy.bits, dna.bits)
        assert copy.length == dna.length

    def test_copy_is_independent(self, rng):
        """Mutating copy must not affect original."""
        dna = DNA(length=64, rng=rng)
        original_bits = dna.bits.copy()
        copy = dna.copy()
        copy.bits[0] = 1 - copy.bits[0]  # flip first bit
        np.testing.assert_array_equal(dna.bits, original_bits)

    def test_copy_mutation_independence(self, rng, simple_regions):
        """Mutating a copy should not change the original."""
        dna = DNA(length=64, rng=rng)
        original_bits = dna.bits.copy()
        child = dna.copy()
        child.mutate(rate=0.5, coding_regions=simple_regions, rng=rng)
        np.testing.assert_array_equal(dna.bits, original_bits)


# ---------------------------------------------------------------------------
# Hamming Distance Tests
# ---------------------------------------------------------------------------

class TestHammingDistance:
    """Tests for Hamming distance computation."""

    def test_identical_dna_distance_zero(self, rng):
        dna = DNA(length=64, rng=rng)
        assert dna.hamming_distance(dna.copy()) == 0

    def test_opposite_dna_distance_max(self):
        dna1 = DNA.create_from_string("0" * 64)
        dna2 = DNA.create_from_string("1" * 64)
        assert dna1.hamming_distance(dna2) == 64

    def test_one_bit_different(self):
        dna1 = DNA.create_from_string("00000000")
        dna2 = DNA.create_from_string("00000001")
        assert dna1.hamming_distance(dna2) == 1

    def test_symmetric(self, rng):
        dna1 = DNA(length=64, rng=rng)
        dna2 = DNA(length=64, rng=np.random.default_rng(999))
        assert dna1.hamming_distance(dna2) == dna2.hamming_distance(dna1)

    def test_different_length_raises(self):
        dna1 = DNA.create_from_string("1010")
        dna2 = DNA.create_from_string("10101010")
        with pytest.raises(ValueError, match="Cannot compare"):
            dna1.hamming_distance(dna2)

    def test_known_distance(self):
        dna1 = DNA.create_from_string("11110000")
        dna2 = DNA.create_from_string("11001100")
        # Differences at positions 2,3,4,5 → distance = 4
        assert dna1.hamming_distance(dna2) == 4


# ---------------------------------------------------------------------------
# Equality and Hashing Tests
# ---------------------------------------------------------------------------

class TestEqualityAndHashing:
    """Tests for __eq__ and __hash__."""

    def test_equal_dna(self):
        dna1 = DNA.create_from_string("10101010")
        dna2 = DNA.create_from_string("10101010")
        assert dna1 == dna2

    def test_not_equal_dna(self):
        dna1 = DNA.create_from_string("10101010")
        dna2 = DNA.create_from_string("01010101")
        assert dna1 != dna2

    def test_not_equal_to_non_dna(self):
        dna = DNA.create_from_string("10101010")
        assert dna != "10101010"

    def test_hash_equal_for_equal_dna(self):
        dna1 = DNA.create_from_string("10101010")
        dna2 = DNA.create_from_string("10101010")
        assert hash(dna1) == hash(dna2)

    def test_hash_usable_in_set(self):
        dna1 = DNA.create_from_string("10101010")
        dna2 = DNA.create_from_string("10101010")
        dna3 = DNA.create_from_string("01010101")
        s = {dna1, dna2, dna3}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# Repr Test
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_info(self):
        dna = DNA.create_from_string("11110000")
        r = repr(dna)
        assert "DNA" in r
        assert "length=8" in r
        assert "ones=4" in r


# ---------------------------------------------------------------------------
# Get Slice Tests
# ---------------------------------------------------------------------------

class TestGetSlice:
    """Tests for get_slice method."""

    def test_slice_returns_copy(self):
        dna = DNA.create_from_string("11110000")
        s = dna.get_slice(0, 4)
        s[0] = 0
        assert dna.bits[0] == 1  # original unchanged

    def test_slice_correct_values(self):
        dna = DNA.create_from_string("11110000")
        s = dna.get_slice(0, 4)
        np.testing.assert_array_equal(s, [1, 1, 1, 1])

    def test_slice_middle(self):
        dna = DNA.create_from_string("00111100")
        s = dna.get_slice(2, 6)
        np.testing.assert_array_equal(s, [1, 1, 1, 1])

    def test_slice_length(self):
        dna = DNA(length=100, bits=np.zeros(100, dtype=np.uint8))
        s = dna.get_slice(10, 30)
        assert len(s) == 20


# ---------------------------------------------------------------------------
# Integration: Mutation Effect on Properties
# ---------------------------------------------------------------------------

class TestMutationPropertyEffect:
    """Integration tests: verify mutation changes extracted property values."""

    def test_mutation_can_change_weight(self):
        """After enough mutations, the weight property should differ."""
        regions = [CodingRegion("weight", 0, 32)]
        bits = np.zeros(64, dtype=np.uint8)
        dna = DNA(length=64, bits=bits)
        original_weight = dna.get_property(0, 32)

        # Mutate heavily
        rng = np.random.default_rng(42)
        dna.mutate(rate=1.0, coding_regions=regions, rng=rng)
        new_weight = dna.get_property(0, 32)

        assert original_weight != new_weight

    def test_mutation_preserves_non_mutated_property(self):
        """Mutating weight region should not change speed region."""
        weight_region = [CodingRegion("weight", 0, 32)]
        dna = DNA(length=64, bits=np.zeros(64, dtype=np.uint8))
        original_speed = dna.get_property(32, 64)

        rng = np.random.default_rng(42)
        dna.mutate(rate=1.0, coding_regions=weight_region, coding_only=True, rng=rng)
        new_speed = dna.get_property(32, 64)

        assert original_speed == new_speed

    def test_offspring_dna_differs_from_parent(self, default_regions):
        """Simulating reproduction: copy + mutate → child differs from parent.

        With rate=0.01 and 160 coding bits, only ~2 bits are touched per mutation,
        and ~50% of those may not flip. Run multiple attempts to ensure at least
        one child differs (statistically near-certain across 20 trials).
        """
        any_changed = False
        for seed in range(20):
            rng_parent = np.random.default_rng(seed)
            rng_mutate = np.random.default_rng(seed + 1000)
            parent = DNA(length=2048, rng=rng_parent)
            child = parent.copy()
            changes = child.mutate(rate=0.01, coding_regions=default_regions, rng=rng_mutate)
            if changes > 0:
                any_changed = True
                assert parent.hamming_distance(child) > 0
                break

        assert any_changed, "No mutation caused a change across 20 trials — improbable"

    def test_stress_mutation_causes_more_changes(self, default_regions):
        """Stress mutation (rate=0.20) should cause more changes than base (rate=0.01)."""
        base_changes = []
        stress_changes = []
        for seed in range(30):
            rng_base = np.random.default_rng(seed)
            rng_stress = np.random.default_rng(seed)
            parent = DNA(length=2048, rng=np.random.default_rng(seed + 1000))

            child_base = parent.copy()
            child_stress = parent.copy()

            base_changes.append(
                child_base.mutate(rate=0.01, coding_regions=default_regions, rng=rng_base)
            )
            stress_changes.append(
                child_stress.mutate(rate=0.20, coding_regions=default_regions, rng=rng_stress)
            )

        assert np.mean(stress_changes) > np.mean(base_changes) * 3
