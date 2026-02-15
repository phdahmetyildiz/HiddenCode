"""
DNA / Genome system for the Evolution Simulator.

Each animal has a fixed-length binary genome (array of 0/1 bits).
The genome is divided into:
  - Coding regions: bits that map to phenotypic properties (weight, speed, defense)
  - Junk regions: remaining bits (neutral; only mutate at base rate)

Properties are extracted by slicing a bit region, converting to an integer,
and normalizing to [0, 1]. Supports both standard binary and Gray code encoding.

Mutation: select N random bits in a region and set each to a random 0 or 1.
This means ~50% of "mutations" are silent (bit unchanged), which is intentional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.utils.encoding import (
    bits_to_int,
    bits_to_normalized,
    normalized_to_range,
    clamp,
)


@dataclass
class CodingRegion:
    """
    Defines a named region of the genome that encodes a property.

    Attributes:
        name: Human-readable name (e.g., "weight", "speed", "defense").
        start: Start bit index (inclusive).
        end: End bit index (exclusive).
    """
    name: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class DNA:
    """
    Binary genome for an animal agent.

    Attributes:
        bits: 1-D NumPy uint8 array of 0/1 values.
        length: Total number of bits.
    """

    __slots__ = ("bits", "length")

    def __init__(
        self,
        length: int = 2048,
        bits: Optional[NDArray[np.uint8]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Create a DNA instance.

        Args:
            length: Number of bits in the genome.
            bits: Pre-set bit array. If None, random bits are generated.
            rng: NumPy random generator. If None, uses default.
        """
        self.length = length
        if bits is not None:
            if len(bits) != length:
                raise ValueError(f"bits length {len(bits)} != expected {length}")
            self.bits = np.asarray(bits, dtype=np.uint8)
        else:
            if rng is None:
                rng = np.random.default_rng()
            self.bits = rng.integers(0, 2, size=length, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def copy(self) -> DNA:
        """Create an independent deep copy of this DNA."""
        return DNA(length=self.length, bits=self.bits.copy())

    def get_slice(self, start: int, end: int) -> NDArray[np.uint8]:
        """
        Extract a contiguous bit region.

        Args:
            start: Start index (inclusive).
            end: End index (exclusive).

        Returns:
            Copy of the bit sub-array.
        """
        return self.bits[start:end].copy()

    def get_property(
        self,
        start: int,
        end: int,
        encoding: str = "binary",
    ) -> float:
        """
        Extract a property value from a bit region, normalized to [0.0, 1.0].

        Args:
            start: Start bit index (inclusive).
            end: End bit index (exclusive).
            encoding: "binary" or "gray".

        Returns:
            Normalized float in [0.0, 1.0].
        """
        bits = self.bits[start:end]
        return bits_to_normalized(bits, encoding=encoding)

    def get_property_in_range(
        self,
        start: int,
        end: int,
        low: float,
        high: float,
        encoding: str = "binary",
    ) -> float:
        """
        Extract a property and map it to a [low, high] range.

        Args:
            start, end: Bit region.
            low, high: Target value range.
            encoding: "binary" or "gray".

        Returns:
            Float in [low, high].
        """
        normalized = self.get_property(start, end, encoding)
        return normalized_to_range(normalized, low, high)

    def get_defense_bits(self, start: int, length: int = 32) -> NDArray[np.uint8]:
        """
        Extract the defense bit sequence.

        Args:
            start: Start index of defense bits.
            length: Number of defense bits (default 32).

        Returns:
            Copy of defense bit array.
        """
        return self.bits[start:start + length].copy()

    def count_ones(self, start: int, length: int) -> int:
        """
        Count the number of 1-bits (Hamming weight) in a region.

        Args:
            start: Start index.
            length: Number of bits.

        Returns:
            Count of 1-bits.
        """
        return int(np.sum(self.bits[start:start + length]))

    def count_ones_defense(self, start: int, length: int = 32) -> int:
        """Count 1-bits in the defense region. Convenience alias."""
        return self.count_ones(start, length)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(
        self,
        rate: float,
        coding_regions: list[CodingRegion],
        coding_only: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """
        Apply mutation: select N random bits in the target region(s) and
        set each to a random 0 or 1 (may or may not flip the bit).

        N = round(target_length * rate)

        Args:
            rate: Mutation rate in [0, 1]. Fraction of target bits to touch.
            coding_regions: List of CodingRegion defining coding areas.
            coding_only: If True, only mutate bits within coding regions.
                         If False, mutate any bit in the genome.
            rng: Random generator. Uses default if None.

        Returns:
            Number of bits that actually changed value.
        """
        if rng is None:
            rng = np.random.default_rng()

        if rate <= 0.0:
            return 0

        if coding_only:
            # Collect all coding bit indices
            coding_indices = []
            for region in coding_regions:
                coding_indices.extend(range(region.start, region.end))
            if not coding_indices:
                return 0
            target_indices = np.array(coding_indices, dtype=np.int64)
        else:
            target_indices = np.arange(self.length, dtype=np.int64)

        # Number of bits to touch
        n_mutations = max(1, round(len(target_indices) * rate))
        n_mutations = min(n_mutations, len(target_indices))

        # Select random positions (without replacement)
        selected = rng.choice(target_indices, size=n_mutations, replace=False)

        # Record old values to count actual changes
        old_values = self.bits[selected].copy()

        # Set to random 0/1
        new_values = rng.integers(0, 2, size=n_mutations, dtype=np.uint8)
        self.bits[selected] = new_values

        # Count bits that actually changed
        actual_changes = int(np.sum(old_values != new_values))
        return actual_changes

    def mutate_junk_only(
        self,
        rate: float,
        coding_regions: list[CodingRegion],
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """
        Mutate only junk (non-coding) regions at a given rate.

        Useful for applying base-rate drift to non-functional DNA.

        Args:
            rate: Mutation rate.
            coding_regions: Coding regions to exclude.
            rng: Random generator.

        Returns:
            Number of bits that actually changed.
        """
        if rng is None:
            rng = np.random.default_rng()

        if rate <= 0.0:
            return 0

        # Compute junk indices (all bits NOT in any coding region)
        coding_set = set()
        for region in coding_regions:
            coding_set.update(range(region.start, region.end))
        junk_indices = np.array(
            [i for i in range(self.length) if i not in coding_set],
            dtype=np.int64,
        )
        if len(junk_indices) == 0:
            return 0

        n_mutations = max(1, round(len(junk_indices) * rate))
        n_mutations = min(n_mutations, len(junk_indices))

        selected = rng.choice(junk_indices, size=n_mutations, replace=False)
        old_values = self.bits[selected].copy()
        new_values = rng.integers(0, 2, size=n_mutations, dtype=np.uint8)
        self.bits[selected] = new_values

        return int(np.sum(old_values != new_values))

    # ------------------------------------------------------------------
    # Comparison / Diversity
    # ------------------------------------------------------------------

    def hamming_distance(self, other: DNA) -> int:
        """
        Compute Hamming distance (number of differing bits) between two genomes.

        Args:
            other: Another DNA instance of the same length.

        Returns:
            Number of positions where bits differ.

        Raises:
            ValueError: If lengths don't match.
        """
        if self.length != other.length:
            raise ValueError(
                f"Cannot compare DNA of length {self.length} with {other.length}"
            )
        return int(np.sum(self.bits != other.bits))

    def defense_sequence_str(self, start: int, length: int = 32) -> str:
        """Return defense bits as a '0'/'1' string."""
        return "".join(str(b) for b in self.bits[start:start + length])

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def create_random(
        cls,
        length: int = 2048,
        rng: Optional[np.random.Generator] = None,
    ) -> DNA:
        """Create a DNA with uniformly random bits."""
        return cls(length=length, rng=rng)

    @classmethod
    def create_from_string(cls, bit_string: str) -> DNA:
        """
        Create DNA from a '0'/'1' string (useful for testing).

        Args:
            bit_string: String of '0' and '1' characters.

        Returns:
            DNA instance.
        """
        bits = np.array([int(c) for c in bit_string], dtype=np.uint8)
        return cls(length=len(bits), bits=bits)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ones = int(np.sum(self.bits))
        return f"DNA(length={self.length}, ones={ones}/{self.length})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DNA):
            return NotImplemented
        return self.length == other.length and np.array_equal(self.bits, other.bits)

    def __hash__(self) -> int:
        return hash(self.bits.tobytes())
