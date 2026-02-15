"""
Animal (Agent) for the Evolution Simulator.

Each animal has a binary DNA genome from which phenotypic properties are
derived (weight, speed, defense). Animals live on a 2D toroidal grid,
consume food for energy, take damage from pitfalls, and reproduce at
generation checkpoints.

Energy is normalized [0, 1]. Animals die when energy reaches 0 or
when the emergency death rule triggers (low energy + no food in sight).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.core.config import SimConfig
from src.core.dna import DNA, CodingRegion
from src.utils.encoding import clamp


# Unique ID counter for animals
_next_animal_id: int = 0


def _get_next_id() -> int:
    """Generate a globally unique animal ID."""
    global _next_animal_id
    aid = _next_animal_id
    _next_animal_id += 1
    return aid


def reset_animal_id_counter() -> None:
    """Reset the ID counter (useful for tests)."""
    global _next_animal_id
    _next_animal_id = 0


class Animal:
    """
    An animal agent on the simulation grid.

    Properties (weight, speed) are extracted from DNA on creation and cached.
    Defense bits are extracted from DNA as needed.

    Attributes:
        id: Unique identifier.
        dna: Binary genome.
        x: Current x-coordinate on the grid.
        y: Current y-coordinate on the grid.
        energy: Current energy level [0.0, 1.0].
        weight: Phenotypic weight [weight_limits], derived from DNA.
        speed: Phenotypic speed [speed_limits], derived from DNA.
        alive: Whether the animal is currently alive.
        birth_tick: The simulation tick when this animal was born.
        death_tick: The simulation tick when this animal died (None if alive).
        death_cause: Cause of death string (None if alive).
        generation: Which generation this animal belongs to.
    """

    __slots__ = (
        "id", "dna", "x", "y", "energy", "weight", "speed",
        "alive", "birth_tick", "death_tick", "death_cause", "generation",
        "_defense_bits_cache", "_config",
    )

    def __init__(
        self,
        dna: DNA,
        x: int,
        y: int,
        config: SimConfig,
        birth_tick: int = 0,
        energy: float = 1.0,
        generation: int = 0,
    ):
        """
        Create an animal.

        Args:
            dna: Binary genome for this animal.
            x, y: Initial grid position.
            config: Simulation config (for property extraction parameters).
            birth_tick: Tick at which this animal was born.
            energy: Starting energy (default 1.0 for newborns).
            generation: Generation number this animal belongs to.
        """
        self.id = _get_next_id()
        self.dna = dna
        self.x = x
        self.y = y
        self.energy = clamp(energy, 0.0, 1.0)
        self.alive = True
        self.birth_tick = birth_tick
        self.death_tick: Optional[int] = None
        self.death_cause: Optional[str] = None
        self.generation = generation
        self._config = config
        self._defense_bits_cache: Optional[NDArray[np.uint8]] = None

        # Extract phenotypic properties from DNA
        encoding = config.genetics.encoding
        g = config.genetics
        p = config.properties

        # Weight: normalize from DNA bits, then map to weight_limits
        weight_raw = dna.get_property(g.weight_bits[0], g.weight_bits[1], encoding)
        self.weight = clamp(
            weight_raw * (p.weight_limits[1] - p.weight_limits[0]) + p.weight_limits[0],
            p.weight_limits[0],
            p.weight_limits[1],
        )

        # Speed: normalize from DNA bits, then map to speed_limits
        speed_raw = dna.get_property(g.speed_bits[0], g.speed_bits[1], encoding)
        self.speed = clamp(
            speed_raw * (p.speed_limits[1] - p.speed_limits[0]) + p.speed_limits[0],
            p.speed_limits[0],
            p.speed_limits[1],
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> tuple[int, int]:
        """Current grid position."""
        return (self.x, self.y)

    @property
    def defense_bits(self) -> NDArray[np.uint8]:
        """
        32-bit defense sequence extracted from DNA.
        Cached on first access.
        """
        if self._defense_bits_cache is None:
            g = self._config.genetics
            self._defense_bits_cache = self.dna.get_defense_bits(
                g.defense_bits[0],
                g.defense_bits[1] - g.defense_bits[0],
            )
        return self._defense_bits_cache

    @property
    def defense_ones_count(self) -> int:
        """Number of 1-bits in the defense sequence."""
        return int(np.sum(self.defense_bits))

    @property
    def eyesight_radius(self) -> int:
        """Eyesight radius from config (fixed, not evolved)."""
        return self._config.properties.eyesight_radius

    @property
    def age(self) -> int:
        """Age in ticks since birth. Only meaningful if birth_tick is tracked."""
        # Will be computed externally as current_tick - birth_tick
        return 0  # Placeholder; actual age computed by engine

    # ------------------------------------------------------------------
    # Energy mechanics
    # ------------------------------------------------------------------

    def calculate_energy_drain(self) -> float:
        """
        Calculate per-tick energy drain.

        Formula: base_metabolism + k_weight_speed * weight * speed
        Optionally: + k_defense_cost * defense_ones (if defense_cost_enabled)

        Returns:
            Energy drain amount (positive float).
        """
        e = self._config.energy
        drain = e.base_metabolism + e.k_weight_speed * self.weight * self.speed

        if e.defense_cost_enabled:
            drain += e.k_defense_cost * self.defense_ones_count

        return drain

    def apply_energy_drain(self) -> float:
        """
        Apply per-tick energy drain. Clamps energy to [0, 1].

        Returns:
            The amount of energy actually drained.
        """
        drain = self.calculate_energy_drain()
        old_energy = self.energy
        self.energy = clamp(self.energy - drain, 0.0, 1.0)
        return old_energy - self.energy

    def gain_energy(self, amount: float) -> float:
        """
        Add energy (e.g., from eating food). Clamps to [0, 1].

        Args:
            amount: Energy to add.

        Returns:
            Actual energy gained (may be less if capped at 1.0).
        """
        old_energy = self.energy
        self.energy = clamp(self.energy + amount, 0.0, 1.0)
        return self.energy - old_energy

    def apply_pitfall_damage(self, energy_loss: float) -> float:
        """
        Apply energy loss from a pitfall encounter. Clamps to [0, 1].

        Args:
            energy_loss: Fractional energy to lose.

        Returns:
            Actual energy lost.
        """
        old_energy = self.energy
        self.energy = clamp(self.energy - energy_loss, 0.0, 1.0)
        return old_energy - self.energy

    # ------------------------------------------------------------------
    # Death checks
    # ------------------------------------------------------------------

    def is_starved(self) -> bool:
        """Check if the animal has run out of energy."""
        return self.energy <= 0.0

    def is_emergency(self, food_in_range: bool) -> bool:
        """
        Check if the emergency death rule applies.

        Dies if energy is below the low threshold AND no food is within eyesight.

        Args:
            food_in_range: Whether any food is within this animal's eyesight radius.

        Returns:
            True if the animal should die from emergency.
        """
        threshold = self._config.energy.low_energy_death_threshold
        return self.energy < threshold and not food_in_range

    def die(self, cause: str, tick: int) -> None:
        """
        Mark this animal as dead.

        Args:
            cause: Reason for death (e.g., "starvation", "emergency", "age", "pitfall").
            tick: The tick at which death occurred.
        """
        self.alive = False
        self.death_tick = tick
        self.death_cause = cause

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def offspring_count(self) -> int:
        """
        Determine how many offspring this animal produces at a reproduction checkpoint.

        Based on current energy:
          - energy < repro_energy_low  →  0 offspring
          - energy < repro_energy_high →  1 offspring
          - energy >= repro_energy_high → 2 offspring

        Returns:
            Number of offspring (0, 1, or 2).
        """
        gen = self._config.generation
        if self.energy < gen.repro_energy_low:
            return 0
        elif self.energy < gen.repro_energy_high:
            return 1
        else:
            return 2

    def survives_generation_end(self) -> bool:
        """
        Check if this animal survives past the 100% generation checkpoint.

        Returns:
            True if energy > survival_threshold.
        """
        return self.energy > self._config.generation.survival_threshold

    def create_offspring(
        self,
        current_tick: int,
        stress_mode: bool,
        rng: np.random.Generator,
        world_width: int,
        world_height: int,
        generation: int,
    ) -> Animal:
        """
        Create a single offspring from this parent.

        Offspring:
          - DNA: copy of parent → mutated (base or stress rate)
          - Energy: 1.0
          - Position: random cell within 3x3 area around parent (toroidal)

        Args:
            current_tick: Current simulation tick.
            stress_mode: Whether stress mode is active (affects mutation rate).
            rng: Random number generator.
            world_width: Grid width for toroidal wrapping.
            world_height: Grid height for toroidal wrapping.
            generation: Generation number for the offspring.

        Returns:
            A new Animal instance.
        """
        g = self._config.genetics

        # Copy and mutate DNA
        child_dna = self.dna.copy()

        # Select mutation rate
        if stress_mode:
            rate = g.stress_mutation_rate
        else:
            rate = g.base_mutation_rate

        # Build coding regions list
        coding_regions = [
            CodingRegion(f"region_{i}", r[0], r[1])
            for i, r in enumerate(g.coding_regions)
        ]

        # Mutate coding regions
        child_dna.mutate(
            rate=rate,
            coding_regions=coding_regions,
            coding_only=g.stress_mode_coding_only if stress_mode else True,
            rng=rng,
        )

        # Position: random in 3x3 around parent (toroidal)
        dx = rng.integers(-1, 2)  # -1, 0, or 1
        dy = rng.integers(-1, 2)
        child_x = (self.x + dx) % world_width
        child_y = (self.y + dy) % world_height

        return Animal(
            dna=child_dna,
            x=child_x,
            y=child_y,
            config=self._config,
            birth_tick=current_tick,
            energy=1.0,
            generation=generation,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "alive" if self.alive else f"dead({self.death_cause})"
        return (
            f"Animal(id={self.id}, pos=({self.x},{self.y}), "
            f"energy={self.energy:.3f}, weight={self.weight:.3f}, "
            f"speed={self.speed:.3f}, defense_1s={self.defense_ones_count}, "
            f"status={status})"
        )

    def to_dict(self) -> dict:
        """Serialize animal state for snapshots/logging."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "energy": round(self.energy, 6),
            "weight": round(self.weight, 6),
            "speed": round(self.speed, 6),
            "defense_ones": self.defense_ones_count,
            "alive": self.alive,
            "birth_tick": self.birth_tick,
            "death_tick": self.death_tick,
            "death_cause": self.death_cause,
            "generation": self.generation,
        }
