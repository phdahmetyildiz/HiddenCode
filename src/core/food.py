"""
Food resource for the Evolution Simulator.

Food items spawn on the grid, persist for a limited number of ticks,
and grant a fixed energy boost to the animal that consumes them.
Only the heaviest animal at a food cell eats it (ties broken randomly).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Food:
    """
    A food item on the grid.

    Attributes:
        x: Grid x-coordinate.
        y: Grid y-coordinate.
        remaining_lifespan: Ticks before this food expires. Decremented each tick.
        energy_value: Energy granted to the animal that consumes it (normalized 0-1).
        consumed: Whether this food has been eaten.
    """
    x: int
    y: int
    remaining_lifespan: int
    energy_value: float
    consumed: bool = False

    @property
    def position(self) -> tuple[int, int]:
        """Grid position as (x, y) tuple."""
        return (self.x, self.y)

    @property
    def expired(self) -> bool:
        """True if this food has run out of lifespan."""
        return self.remaining_lifespan <= 0

    @property
    def active(self) -> bool:
        """True if this food is still on the grid (not consumed, not expired)."""
        return not self.consumed and not self.expired

    def tick(self) -> bool:
        """
        Advance one simulation tick.

        Decrements remaining lifespan by 1.

        Returns:
            True if the food has expired (should be removed), False otherwise.
        """
        self.remaining_lifespan -= 1
        return self.expired

    def consume(self) -> float:
        """
        Mark this food as consumed and return its energy value.

        Returns:
            Energy value of the food.

        Raises:
            RuntimeError: If food is already consumed or expired.
        """
        if self.consumed:
            raise RuntimeError(f"Food at ({self.x}, {self.y}) already consumed")
        if self.expired:
            raise RuntimeError(f"Food at ({self.x}, {self.y}) has expired")
        self.consumed = True
        return self.energy_value

    def __repr__(self) -> str:
        status = "consumed" if self.consumed else ("expired" if self.expired else "active")
        return (
            f"Food(pos=({self.x},{self.y}), lifespan={self.remaining_lifespan}, "
            f"energy={self.energy_value}, status={status})"
        )
