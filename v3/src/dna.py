"""Packed binary genomes: uint64 words, bit index 0 = LSB of word 0."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.encoding import bits_to_normalized


def n_words(dna_length: int) -> int:
    return (int(dna_length) + 63) // 64


def allocate(n: int, dna_length: int) -> NDArray[np.uint64]:
    return np.zeros((n, n_words(dna_length)), dtype=np.uint64)


def random_genomes(
    n: int,
    dna_length: int,
    rng: np.random.Generator,
) -> NDArray[np.uint64]:
    bits = rng.integers(0, 2, size=(n, dna_length), dtype=np.uint8)
    return pack_bits(bits)


def pack_bits(bits: NDArray[np.uint8]) -> NDArray[np.uint64]:
    """Pack (n, dna_length) bits into (n, n_words) uint64. bit 0 → LSB of word 0."""
    if bits.ndim == 1:
        bits = bits[None, :]
    from src.kernels import numba_available, pack_bits_njit
    if numba_available() and bits.shape[0] * bits.shape[1] >= 256:
        return pack_bits_njit(np.ascontiguousarray(bits, dtype=np.uint8))
    n, length = bits.shape
    words = n_words(length)
    out = np.zeros((n, words), dtype=np.uint64)
    for i in range(length):
        w, b = divmod(i, 64)
        out[:, w] |= bits[:, i].astype(np.uint64) << np.uint64(b)
    return out


def unpack_bits(dna: NDArray[np.uint64], dna_length: int) -> NDArray[np.uint8]:
    if dna.ndim == 1:
        dna = dna[None, :]
    n = dna.shape[0]
    bits = np.zeros((n, dna_length), dtype=np.uint8)
    for i in range(dna_length):
        w, b = divmod(i, 64)
        bits[:, i] = ((dna[:, w] >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)
    return bits


def get_bit(dna: NDArray[np.uint64], index: int) -> NDArray[np.uint8]:
    w, b = divmod(int(index), 64)
    return ((dna[:, w] >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)


def extract_slice_bits(
    dna: NDArray[np.uint64],
    start: int,
    end: int,
) -> NDArray[np.uint8]:
    """Return (n, end-start) bits, genome order (index 0 = start)."""
    n = dna.shape[0]
    length = end - start
    out = np.empty((n, length), dtype=np.uint8)
    for j, i in enumerate(range(start, end)):
        w, b = divmod(i, 64)
        out[:, j] = ((dna[:, w] >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)
    return out


def extract_normalized(
    dna: NDArray[np.uint64],
    start: int,
    end: int,
    encoding: str = "binary",
) -> NDArray[np.float64]:
    bits = extract_slice_bits(dna, start, end)
    n = bits.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = bits_to_normalized(bits[i], encoding=encoding)
    return out


def extract_uint32_slice(dna: NDArray[np.uint64], start: int, end: int) -> NDArray[np.uint32]:
    """Pack slice bits with slice[0] as bit 0 (LSB). Length must be <= 32."""
    length = end - start
    if length > 32:
        raise ValueError("slice longer than 32 bits")
    bits = extract_slice_bits(dna, start, end)
    n = bits.shape[0]
    out = np.zeros(n, dtype=np.uint32)
    for j in range(length):
        out |= bits[:, j].astype(np.uint32) << np.uint32(j)
    return out


def coding_bit_indices(coding_regions: list[list[int]]) -> NDArray[np.intp]:
    idxs: list[int] = []
    for start, end in coding_regions:
        idxs.extend(range(int(start), int(end)))
    return np.asarray(idxs, dtype=np.intp)


def mutate_coding(
    dna: NDArray[np.uint64],
    rate: float,
    coding_regions: list[list[int]],
    rng: np.random.Generator,
    dna_length: int,
    coding_only: bool = True,
) -> None:
    """In-place: pick N=round(region_len*rate) bits, set each to random 0/1."""
    if rate <= 0.0:
        return
    n = dna.shape[0]
    if coding_only:
        pool = coding_bit_indices(coding_regions)
    else:
        pool = np.arange(dna_length, dtype=np.intp)
    if pool.size == 0:
        return
    n_mut = int(round(pool.size * float(rate)))
    if n_mut <= 0:
        return
    n_mut = min(n_mut, pool.size)
    for i in range(n):
        chosen = rng.choice(pool, size=n_mut, replace=False)
        new_vals = rng.integers(0, 2, size=n_mut, dtype=np.uint8)
        for idx, val in zip(chosen, new_vals):
            w, b = divmod(int(idx), 64)
            mask = np.uint64(1) << np.uint64(b)
            dna[i, w] &= ~mask
            if val:
                dna[i, w] |= mask


def hamming_distance_pair(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> int:
    x = np.bitwise_xor(a, b)
    # popcount words
    total = 0
    for word in x:
        total += int(word).bit_count()
    return total


def hamming_sample(
    dna: NDArray[np.uint64],
    rng: np.random.Generator,
    max_animals: int = 100,
) -> float:
    """Mean pairwise Hamming distance (sampled)."""
    n = dna.shape[0]
    if n < 2:
        return 0.0
    if n > max_animals:
        pick = rng.choice(n, size=max_animals, replace=False)
        dna = dna[pick]
        n = max_animals
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += hamming_distance_pair(dna[i], dna[j])
            count += 1
    return total / count if count else 0.0
