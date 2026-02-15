"""
Pitfall resource for the Evolution Simulator.

Pitfalls are environmental hazards on the grid. Each pitfall has a 32-bit
binary sequence. When an animal steps on a pitfall, damage is computed by
comparing the pitfall's sequence with the animal's defense bits:

  - Pitfall bit = 0  →  no effect
  - Pitfall bit = 1 AND animal defense bit = 1  →  immune (no effect)
  - Pitfall bit = 1 AND animal defense bit = 0  →  +1 damage point

Total damage ranges from 0 (fully immune) to 32 (no defense).
Energy loss is scaled: loss = (damage / 32) * max_pitfall_loss_pct.

Pitfalls are NOT consumed by interaction — they persist until lifespan expires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Pitfall:
    """
    A pitfall hazard on the grid.

    Attributes:
        x: Grid x-coordinate.
        y: Grid y-coordinate.
        name: Human-readable type identifier (e.g., "A", "B").
        sequence: 32-element uint8 array of 0/1 bits defining the pitfall's pattern.
        remaining_lifespan: Ticks before this pitfall expires.
    """
    x: int
    y: int
    name: str
    sequence: NDArray[np.uint8]  # shape (32,)
    remaining_lifespan: int

    @property
    def position(self) -> tuple[int, int]:
        """Grid position as (x, y) tuple."""
        return (self.x, self.y)

    @property
    def expired(self) -> bool:
        """True if this pitfall has run out of lifespan."""
        return self.remaining_lifespan <= 0

    @property
    def active(self) -> bool:
        """True if this pitfall is still on the grid."""
        return not self.expired

    @property
    def sequence_str(self) -> str:
        """Return the 32-bit sequence as a '0'/'1' string."""
        return "".join(str(b) for b in self.sequence)

    @property
    def num_danger_bits(self) -> int:
        """Count of 1-bits in the pitfall sequence (maximum possible damage)."""
        return int(np.sum(self.sequence))

    def tick(self) -> bool:
        """
        Advance one simulation tick.

        Decrements remaining lifespan by 1.

        Returns:
            True if the pitfall has expired (should be removed), False otherwise.
        """
        self.remaining_lifespan -= 1
        return self.expired

    def calculate_damage(self, defense_bits: NDArray[np.uint8]) -> int:
        """
        Calculate damage points from a pitfall encounter.

        Rules per bit position:
          - pitfall=0              →  0 damage (no threat)
          - pitfall=1, defense=1   →  0 damage (immune)
          - pitfall=1, defense=0   →  1 damage point

        Vectorized: damage = sum(pitfall & ~defense) = sum(pitfall * (1 - defense))

        Args:
            defense_bits: Animal's 32-bit defense array (uint8, shape (32,)).

        Returns:
            Total damage points (0 to 32).
        """
        if len(defense_bits) != len(self.sequence):
            raise ValueError(
                f"Defense bits length {len(defense_bits)} != pitfall sequence length {len(self.sequence)}"
            )
        # pitfall=1 AND defense=0 → damage
        damage = np.sum(self.sequence & (1 - defense_bits))
        return int(damage)

    def calculate_energy_loss(
        self,
        defense_bits: NDArray[np.uint8],
        max_pitfall_loss_pct: float = 0.5,
    ) -> float:
        """
        Calculate energy loss fraction from a pitfall encounter.

        loss = (damage / 32) * max_pitfall_loss_pct

        Args:
            defense_bits: Animal's 32-bit defense array.
            max_pitfall_loss_pct: Maximum energy fraction lost at full damage (0 to 1).

        Returns:
            Energy loss as a fraction of total energy (0.0 to max_pitfall_loss_pct).
        """
        damage = self.calculate_damage(defense_bits)
        return (damage / 32.0) * max_pitfall_loss_pct

    @classmethod
    def from_string(
        cls,
        x: int,
        y: int,
        name: str,
        sequence_str: str,
        lifespan: int,
    ) -> Pitfall:
        """
        Create a Pitfall from a '0'/'1' string sequence.

        Args:
            x, y: Grid coordinates.
            name: Type identifier.
            sequence_str: 32-character binary string.
            lifespan: Initial lifespan in ticks.

        Returns:
            Pitfall instance.
        """
        sequence = np.array([int(c) for c in sequence_str], dtype=np.uint8)
        return cls(x=x, y=y, name=name, sequence=sequence, remaining_lifespan=lifespan)

    def __repr__(self) -> str:
        status = "expired" if self.expired else "active"
        return (
            f"Pitfall(pos=({self.x},{self.y}), name='{self.name}', "
            f"lifespan={self.remaining_lifespan}, danger_bits={self.num_danger_bits}, "
            f"status={status})"
        )
