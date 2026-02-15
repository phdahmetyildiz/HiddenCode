"""
Unit tests for encoding utilities (binary ↔ Gray code, normalization).

Tests cover:
- bits_to_int / int_to_bits roundtrip
- Binary ↔ Gray code conversion
- Gray code adjacency property (adjacent ints differ by 1 bit)
- bits_to_normalized range and edge cases
- normalized_to_range mapping
- clamp function
"""

import numpy as np
import pytest

from src.utils.encoding import (
    bits_to_int,
    int_to_bits,
    binary_to_gray,
    gray_to_binary,
    bits_to_normalized,
    normalized_to_range,
    clamp,
)


# ---------------------------------------------------------------------------
# bits_to_int / int_to_bits
# ---------------------------------------------------------------------------

class TestBitsToInt:
    """Tests for bit array → integer conversion."""

    def test_all_zeros(self):
        bits = np.array([0, 0, 0, 0], dtype=np.uint8)
        assert bits_to_int(bits) == 0

    def test_all_ones_4bit(self):
        bits = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert bits_to_int(bits) == 15

    def test_known_value_1010(self):
        bits = np.array([1, 0, 1, 0], dtype=np.uint8)
        assert bits_to_int(bits) == 10

    def test_known_value_0110(self):
        bits = np.array([0, 1, 1, 0], dtype=np.uint8)
        assert bits_to_int(bits) == 6

    def test_single_bit_1(self):
        bits = np.array([1], dtype=np.uint8)
        assert bits_to_int(bits) == 1

    def test_single_bit_0(self):
        bits = np.array([0], dtype=np.uint8)
        assert bits_to_int(bits) == 0

    def test_empty_array(self):
        bits = np.array([], dtype=np.uint8)
        assert bits_to_int(bits) == 0

    def test_8bit_max(self):
        bits = np.ones(8, dtype=np.uint8)
        assert bits_to_int(bits) == 255

    def test_16bit_value(self):
        # 256 in binary = 0000000100000000
        bits = int_to_bits(256, 16)
        assert bits_to_int(bits) == 256

    def test_32bit_value(self):
        # Large value
        val = 2_147_483_648  # 2^31
        bits = int_to_bits(val, 32)
        assert bits_to_int(bits) == val


class TestIntToBits:
    """Tests for integer → bit array conversion."""

    def test_zero_4bit(self):
        result = int_to_bits(0, 4)
        expected = np.array([0, 0, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_max_4bit(self):
        result = int_to_bits(15, 4)
        expected = np.array([1, 1, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_ten_4bit(self):
        result = int_to_bits(10, 4)
        expected = np.array([1, 0, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_length_zero(self):
        result = int_to_bits(0, 0)
        assert len(result) == 0

    def test_roundtrip_many_values(self):
        """int → bits → int should be identity for values that fit."""
        for val in [0, 1, 2, 7, 15, 100, 255, 1000, 65535]:
            bits = int_to_bits(val, 16)
            assert bits_to_int(bits) == val, f"Roundtrip failed for {val}"

    def test_output_dtype(self):
        result = int_to_bits(5, 8)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Binary ↔ Gray code
# ---------------------------------------------------------------------------

class TestGrayCode:
    """Tests for Gray code conversion."""

    def test_binary_to_gray_0000(self):
        binary = np.array([0, 0, 0, 0], dtype=np.uint8)
        gray = binary_to_gray(binary)
        np.testing.assert_array_equal(gray, [0, 0, 0, 0])

    def test_binary_to_gray_0001(self):
        binary = np.array([0, 0, 0, 1], dtype=np.uint8)
        gray = binary_to_gray(binary)
        np.testing.assert_array_equal(gray, [0, 0, 0, 1])

    def test_binary_to_gray_0010(self):
        # binary 2 = 0010; gray 2 = 0011
        binary = np.array([0, 0, 1, 0], dtype=np.uint8)
        gray = binary_to_gray(binary)
        np.testing.assert_array_equal(gray, [0, 0, 1, 1])

    def test_binary_to_gray_0011(self):
        # binary 3 = 0011; gray 3 = 0010
        binary = np.array([0, 0, 1, 1], dtype=np.uint8)
        gray = binary_to_gray(binary)
        np.testing.assert_array_equal(gray, [0, 0, 1, 0])

    def test_binary_to_gray_1111(self):
        # binary 15 = 1111; gray 15 = 1000
        binary = np.array([1, 1, 1, 1], dtype=np.uint8)
        gray = binary_to_gray(binary)
        np.testing.assert_array_equal(gray, [1, 0, 0, 0])

    def test_roundtrip_binary_gray_binary(self):
        """binary → gray → binary should be identity."""
        for val in range(16):
            binary = int_to_bits(val, 4)
            gray = binary_to_gray(binary)
            recovered = gray_to_binary(gray)
            np.testing.assert_array_equal(recovered, binary, err_msg=f"Failed for value {val}")

    def test_roundtrip_8bit_all_values(self):
        """Test roundtrip for all 8-bit values."""
        for val in range(256):
            binary = int_to_bits(val, 8)
            gray = binary_to_gray(binary)
            recovered = gray_to_binary(gray)
            np.testing.assert_array_equal(recovered, binary, err_msg=f"Failed for value {val}")

    def test_gray_adjacency_property(self):
        """
        Adjacent integers in Gray code must differ by exactly 1 bit.
        This is the key property of Gray code.
        """
        for val in range(255):
            gray_a = binary_to_gray(int_to_bits(val, 8))
            gray_b = binary_to_gray(int_to_bits(val + 1, 8))
            diff = int(np.sum(gray_a != gray_b))
            assert diff == 1, (
                f"Gray codes for {val} and {val+1} differ by {diff} bits, expected 1"
            )

    def test_empty_array(self):
        empty = np.array([], dtype=np.uint8)
        np.testing.assert_array_equal(binary_to_gray(empty), empty)
        np.testing.assert_array_equal(gray_to_binary(empty), empty)

    def test_single_bit(self):
        for b in [0, 1]:
            bits = np.array([b], dtype=np.uint8)
            np.testing.assert_array_equal(binary_to_gray(bits), bits)
            np.testing.assert_array_equal(gray_to_binary(bits), bits)


# ---------------------------------------------------------------------------
# bits_to_normalized
# ---------------------------------------------------------------------------

class TestBitsToNormalized:
    """Tests for normalizing bit arrays to [0, 1] floats."""

    def test_all_zeros(self):
        bits = np.zeros(8, dtype=np.uint8)
        assert bits_to_normalized(bits) == 0.0

    def test_all_ones(self):
        bits = np.ones(8, dtype=np.uint8)
        assert bits_to_normalized(bits) == 1.0

    def test_midpoint_8bit(self):
        """128 / 255 ≈ 0.502"""
        bits = int_to_bits(128, 8)
        result = bits_to_normalized(bits)
        assert abs(result - 128 / 255) < 1e-10

    def test_range_always_0_to_1(self):
        """Any bit pattern should produce a value in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            bits = rng.integers(0, 2, size=16, dtype=np.uint8)
            val = bits_to_normalized(bits)
            assert 0.0 <= val <= 1.0, f"Out of range: {val}"

    def test_gray_encoding_all_zeros(self):
        bits = np.zeros(8, dtype=np.uint8)
        assert bits_to_normalized(bits, encoding="gray") == 0.0

    def test_gray_encoding_produces_valid_range(self):
        """Gray-decoded values should also be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            bits = rng.integers(0, 2, size=16, dtype=np.uint8)
            val = bits_to_normalized(bits, encoding="gray")
            assert 0.0 <= val <= 1.0, f"Out of range (gray): {val}"

    def test_gray_all_ones(self):
        """All-ones in Gray code: gray→binary→int, then normalize."""
        bits = np.ones(4, dtype=np.uint8)
        # Gray 1111 → binary: 1010 → int: 10 → normalized: 10/15
        result = bits_to_normalized(bits, encoding="gray")
        assert abs(result - 10 / 15) < 1e-10

    def test_empty_bits(self):
        bits = np.array([], dtype=np.uint8)
        assert bits_to_normalized(bits) == 0.0

    def test_single_bit_0(self):
        bits = np.array([0], dtype=np.uint8)
        assert bits_to_normalized(bits) == 0.0

    def test_single_bit_1(self):
        bits = np.array([1], dtype=np.uint8)
        assert bits_to_normalized(bits) == 1.0

    def test_32bit_precision(self):
        """32-bit all-ones should give exactly 1.0."""
        bits = np.ones(32, dtype=np.uint8)
        assert bits_to_normalized(bits) == 1.0

    def test_32bit_all_zeros(self):
        bits = np.zeros(32, dtype=np.uint8)
        assert bits_to_normalized(bits) == 0.0


# ---------------------------------------------------------------------------
# normalized_to_range
# ---------------------------------------------------------------------------

class TestNormalizedToRange:
    """Tests for mapping normalized values to a range."""

    def test_zero_maps_to_low(self):
        assert normalized_to_range(0.0, 0.1, 1.0) == 0.1

    def test_one_maps_to_high(self):
        assert normalized_to_range(1.0, 0.1, 1.0) == 1.0

    def test_half_maps_to_midpoint(self):
        result = normalized_to_range(0.5, 0.0, 1.0)
        assert abs(result - 0.5) < 1e-10

    def test_custom_range(self):
        result = normalized_to_range(0.5, 0.2, 0.8)
        assert abs(result - 0.5) < 1e-10

    def test_zero_width_range(self):
        assert normalized_to_range(0.5, 0.5, 0.5) == 0.5


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

class TestClamp:
    """Tests for value clamping."""

    def test_within_range(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_low(self):
        assert clamp(-0.5, 0.0, 1.0) == 0.0

    def test_above_high(self):
        assert clamp(1.5, 0.0, 1.0) == 1.0

    def test_at_low(self):
        assert clamp(0.0, 0.0, 1.0) == 0.0

    def test_at_high(self):
        assert clamp(1.0, 0.0, 1.0) == 1.0
