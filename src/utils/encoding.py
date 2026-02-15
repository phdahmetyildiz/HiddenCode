"""
Binary and Gray code encoding utilities.

Provides conversions between binary bit arrays, integers, and Gray code,
plus normalization to [0, 1] range for property extraction from DNA.

Standard binary: flipping a high-order bit causes a large value change;
    flipping a low-order bit causes a tiny change.

Gray code: flipping ANY single bit causes the integer value to change
    by a small amount (adjacent codes differ by exactly 1 bit).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bits_to_int(bits: NDArray[np.uint8]) -> int:
    """
    Convert a bit array (MSB first) to a non-negative integer.

    Args:
        bits: 1-D array of 0/1 values, MSB at index 0.

    Returns:
        Integer value.

    Examples:
        >>> bits_to_int(np.array([1, 0, 1, 0], dtype=np.uint8))
        10
        >>> bits_to_int(np.array([0, 0, 0, 0], dtype=np.uint8))
        0
    """
    if len(bits) == 0:
        return 0
    # Use NumPy dot with powers-of-2 vector for speed
    n = len(bits)
    powers = np.left_shift(np.uint64(1), np.arange(n - 1, -1, -1, dtype=np.uint64))
    return int(np.dot(bits.astype(np.uint64), powers))


def int_to_bits(value: int, length: int) -> NDArray[np.uint8]:
    """
    Convert a non-negative integer to a bit array of fixed length (MSB first).

    Args:
        value: Non-negative integer.
        length: Number of bits in output.

    Returns:
        1-D uint8 array of 0/1 values, MSB at index 0.

    Examples:
        >>> int_to_bits(10, 4)
        array([1, 0, 1, 0], dtype=uint8)
    """
    if length == 0:
        return np.array([], dtype=np.uint8)
    bits = np.zeros(length, dtype=np.uint8)
    for i in range(length - 1, -1, -1):
        bits[length - 1 - i] = (value >> i) & 1
    return bits


def binary_to_gray(bits: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Convert standard binary bit array to Gray code bit array.

    Gray code property: adjacent integer values differ by exactly 1 bit.

    Algorithm: gray[0] = binary[0]; gray[i] = binary[i-1] XOR binary[i]

    Args:
        bits: 1-D uint8 array of 0/1 (MSB first, standard binary).

    Returns:
        1-D uint8 array of 0/1 (MSB first, Gray code).
    """
    if len(bits) == 0:
        return np.array([], dtype=np.uint8)
    gray = np.empty_like(bits)
    gray[0] = bits[0]
    gray[1:] = bits[:-1] ^ bits[1:]
    return gray


def gray_to_binary(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Convert Gray code bit array to standard binary bit array.

    Algorithm: binary[0] = gray[0]; binary[i] = binary[i-1] XOR gray[i]

    Args:
        gray: 1-D uint8 array of 0/1 (MSB first, Gray code).

    Returns:
        1-D uint8 array of 0/1 (MSB first, standard binary).
    """
    if len(gray) == 0:
        return np.array([], dtype=np.uint8)
    binary = np.empty_like(gray)
    binary[0] = gray[0]
    for i in range(1, len(gray)):
        binary[i] = binary[i - 1] ^ gray[i]
    return binary


def bits_to_normalized(bits: NDArray[np.uint8], encoding: str = "binary") -> float:
    """
    Convert a bit array to a normalized float in [0.0, 1.0].

    Steps:
        1. If encoding="gray", convert Gray→binary first.
        2. Interpret bits as integer (MSB first).
        3. Normalize: value / (2^n - 1), where n = len(bits).

    Special case: 0-length bits → 0.0; 1-bit → 0.0 or 1.0.

    Args:
        bits: 1-D uint8 array of 0/1 values.
        encoding: "binary" or "gray".

    Returns:
        Float in [0.0, 1.0].
    """
    if len(bits) == 0:
        return 0.0

    if encoding == "gray":
        bits = gray_to_binary(bits)

    int_val = bits_to_int(bits)
    max_val = (1 << len(bits)) - 1  # 2^n - 1
    if max_val == 0:
        return 0.0
    return int_val / max_val


def normalized_to_range(normalized: float, low: float, high: float) -> float:
    """
    Map a normalized [0, 1] value to a [low, high] range.

    Args:
        normalized: Value in [0, 1].
        low: Lower bound of target range.
        high: Upper bound of target range.

    Returns:
        Value in [low, high].
    """
    return low + normalized * (high - low)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to [low, high]."""
    return max(low, min(high, value))
