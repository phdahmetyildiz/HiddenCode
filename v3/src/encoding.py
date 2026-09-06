"""
Binary / Gray encoding helpers for packed and unpacked bit slices.

Author: Cursor Grok 4.6 High Fast
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bits_to_int(bits: NDArray[np.uint8]) -> int:
    if len(bits) == 0:
        return 0
    n = len(bits)
    powers = np.left_shift(np.uint64(1), np.arange(n - 1, -1, -1, dtype=np.uint64))
    return int(np.dot(bits.astype(np.uint64), powers))


def int_to_bits(value: int, length: int) -> NDArray[np.uint8]:
    if length == 0:
        return np.array([], dtype=np.uint8)
    bits = np.zeros(length, dtype=np.uint8)
    for i in range(length - 1, -1, -1):
        bits[length - 1 - i] = (value >> i) & 1
    return bits


def binary_to_gray(bits: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if len(bits) == 0:
        return np.array([], dtype=np.uint8)
    gray = np.empty_like(bits)
    gray[0] = bits[0]
    gray[1:] = bits[:-1] ^ bits[1:]
    return gray


def gray_to_binary(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if len(gray) == 0:
        return np.array([], dtype=np.uint8)
    binary = np.empty_like(gray)
    binary[0] = gray[0]
    for i in range(1, len(gray)):
        binary[i] = binary[i - 1] ^ gray[i]
    return binary


def bits_to_normalized(bits: NDArray[np.uint8], encoding: str = "binary") -> float:
    if len(bits) == 0:
        return 0.0
    if encoding == "gray":
        bits = gray_to_binary(bits)
    int_val = bits_to_int(bits)
    max_val = (1 << len(bits)) - 1
    if max_val == 0:
        return 0.0
    return int_val / max_val


def normalized_to_range(normalized: float, low: float, high: float) -> float:
    return low + float(normalized) * (high - low)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
