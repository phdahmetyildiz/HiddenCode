"""Encoding + packed DNA."""

import numpy as np

from src import dna as dnalib
from src.encoding import binary_to_gray, bits_to_normalized, gray_to_binary, int_to_bits


def test_normalize_zeros_ones():
    z = np.zeros(8, dtype=np.uint8)
    o = np.ones(8, dtype=np.uint8)
    assert bits_to_normalized(z) == 0.0
    assert bits_to_normalized(o) == 1.0


def test_gray_adjacent_differ_one_bit():
    for i in range(15):
        b0 = int_to_bits(i, 4)
        b1 = int_to_bits(i + 1, 4)
        g0 = binary_to_gray(b0)
        g1 = binary_to_gray(b1)
        assert int(np.sum(g0 != g1)) == 1


def test_gray_roundtrip():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    assert np.array_equal(gray_to_binary(binary_to_gray(bits)), bits)


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(5, 128), dtype=np.uint8)
    packed = dnalib.pack_bits(bits)
    back = dnalib.unpack_bits(packed, 128)
    assert np.array_equal(bits, back)


def test_mutation_rate_zero():
    rng = np.random.default_rng(1)
    dna = dnalib.random_genomes(3, 256, rng)
    copy = dna.copy()
    dnalib.mutate_coding(dna, 0.0, [[0, 64]], rng, 256, coding_only=True)
    assert np.array_equal(dna, copy)


def test_mutation_coding_only_leaves_junk():
    rng = np.random.default_rng(2)
    dna = dnalib.random_genomes(1, 256, rng)
    before = dna.copy()
    dnalib.mutate_coding(dna, 1.0, [[0, 32]], rng, 256, coding_only=True)
    junk_before = dnalib.unpack_bits(before, 256)[0, 32:]
    junk_after = dnalib.unpack_bits(dna, 256)[0, 32:]
    assert np.array_equal(junk_before, junk_after)


def test_fertility_bits_map_min_max():
    dna_len = 2048
    zeros = dnalib.allocate(1, dna_len)
    ones_bits = np.ones((1, dna_len), dtype=np.uint8)
    ones = dnalib.pack_bits(ones_bits)
    raw0 = dnalib.extract_normalized(zeros, 64, 96)[0]
    raw1 = dnalib.extract_normalized(ones, 64, 96)[0]
    assert raw0 == 0.0
    assert raw1 == 1.0
    rmin, rmax = 700, 1100
    assert int(round(raw0 * (rmax - rmin)) + rmin) == rmin
    assert int(round(raw1 * (rmax - rmin)) + rmin) == rmax


def test_packed_matches_unpacked_reference():
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=(20, 64), dtype=np.uint8)
    packed = dnalib.pack_bits(bits)
    got = dnalib.extract_normalized(packed, 0, 32)
    for i in range(20):
        ref = bits_to_normalized(bits[i, 0:32])
        assert abs(got[i] - ref) < 1e-12


def test_hamming_identical_zero():
    dna = dnalib.random_genomes(4, 128, np.random.default_rng(4))
    dna[:] = dna[0]
    assert dnalib.hamming_sample(dna, np.random.default_rng(0), max_animals=4) == 0.0
