"""
Generation lifecycle manager for the Evolution Simulator.

Handles the multi-phase generation cycle:
  - Primary reproduction checkpoint (default 70% of gen_length)
  - Survival check (default 100% of gen_length)
  - Secondary (bonus) reproduction checkpoint (default 120% of gen_length)

Each generation spans gen_length * bonus_repro_pct ticks total.
After the bonus checkpoint, the generation counter advances and the
lifecycle resets for the next generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from src.core.config import SimConfig
from src.core.world import World
from src.core.animal import Animal


# ---------------------------------------------------------------------------
# Generation events
# ---------------------------------------------------------------------------

class GenerationEvent(Enum):
    """Events that can fire during a generation lifecycle."""
    NONE = auto()
    PRIMARY_REPRODUCTION = auto()
    SURVIVAL_CHECK = auto()
    BONUS_REPRODUCTION = auto()
    GENERATION_COMPLETE = auto()


# ---------------------------------------------------------------------------
# Generation statistics
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    """Statistics collected during one full generation."""
    generation: int = 0

    # Reproduction
    primary_repro_births: int = 0
    bonus_repro_births: int = 0
    total_births: int = 0

    # Deaths at checkpoints
    survival_check_deaths: int = 0

    # Population snapshots
    pop_at_primary_repro: int = 0
    pop_at_survival_check: int = 0
    pop_at_bonus_repro: int = 0
    pop_at_generation_end: int = 0

    # Parents who reproduced
    parents_at_primary: int = 0
    parents_at_bonus: int = 0


# ---------------------------------------------------------------------------
# Generation Manager
# ---------------------------------------------------------------------------

class GenerationManager:
    """
    Manages the generation lifecycle.

    A generation spans from gen_start_tick to gen_start_tick + full_gen_ticks,
    where full_gen_ticks = int(gen_length * bonus_repro_pct).

    Checkpoint ticks (relative to gen start):
      - primary_tick  = int(gen_length * repro_checkpoint_pct)
      - survival_tick = int(gen_length * survival_check_pct)
      - bonus_tick    = int(gen_length * bonus_repro_pct)

    Attributes:
        config: Simulation configuration.
        current_generation: Current generation number (0-indexed).
        gen_start_tick: The tick at which the current generation started.
        gen_stats: Statistics for the current generation.
        all_gen_stats: Statistics for all completed generations.
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.current_generation: int = 0
        self.gen_start_tick: int = 0

        # Pre-compute checkpoint tick offsets
        gen = config.generation
        self._gen_length = gen.gen_length
        self._primary_offset = int(self._gen_length * gen.repro_checkpoint_pct)
        self._survival_offset = int(self._gen_length * gen.survival_check_pct)
        self._bonus_offset = int(self._gen_length * gen.bonus_repro_pct)

        # Track which checkpoints have fired this generation
        self._primary_fired: bool = False
        self._survival_fired: bool = False
        self._bonus_fired: bool = False

        # Statistics
        self.gen_stats = GenerationStats(generation=0)
        self.all_gen_stats: list[GenerationStats] = []

    # ------------------------------------------------------------------
    # Tick offsets / thresholds
    # ------------------------------------------------------------------

    @property
    def primary_tick(self) -> int:
        """Absolute tick for primary reproduction checkpoint."""
        return self.gen_start_tick + self._primary_offset

    @property
    def survival_tick(self) -> int:
        """Absolute tick for survival check."""
        return self.gen_start_tick + self._survival_offset

    @property
    def bonus_tick(self) -> int:
        """Absolute tick for bonus reproduction checkpoint."""
        return self.gen_start_tick + self._bonus_offset

    @property
    def gen_progress(self) -> float:
        """
        Progress through the current generation as a fraction.
        0.0 = start, 1.0 = survival check, 1.2 = bonus checkpoint (for default config).
        """
        if self._gen_length == 0:
            return 0.0
        return 0.0  # Computed externally from tick

    def ticks_into_generation(self, current_tick: int) -> int:
        """How many ticks into the current generation."""
        return current_tick - self.gen_start_tick

    # ------------------------------------------------------------------
    # Main check — call once per tick after animal processing
    # ------------------------------------------------------------------

    def check(
        self,
        current_tick: int,
        world: World,
        rng: np.random.Generator,
    ) -> list[GenerationEvent]:
        """
        Check if any generation checkpoint should fire at the current tick.

        This should be called once per tick, AFTER the main animal processing.
        Multiple events can fire in a single tick (though typically only one).

        Args:
            current_tick: The current simulation tick (after increment).
            world: The simulation world.
            rng: Random generator for offspring placement.

        Returns:
            List of events that fired this tick (may be empty).
        """
        events: list[GenerationEvent] = []

        # Primary reproduction at repro_checkpoint_pct
        if not self._primary_fired and current_tick >= self.primary_tick:
            self._primary_fired = True
            self._do_primary_reproduction(current_tick, world, rng)
            events.append(GenerationEvent.PRIMARY_REPRODUCTION)

        # Survival check at survival_check_pct
        if not self._survival_fired and current_tick >= self.survival_tick:
            self._survival_fired = True
            self._do_survival_check(current_tick, world)
            events.append(GenerationEvent.SURVIVAL_CHECK)

        # Bonus reproduction at bonus_repro_pct
        if not self._bonus_fired and current_tick >= self.bonus_tick:
            self._bonus_fired = True
            self._do_bonus_reproduction(current_tick, world, rng)
            events.append(GenerationEvent.BONUS_REPRODUCTION)

            # After bonus checkpoint → advance generation
            self._advance_generation(current_tick, world)
            events.append(GenerationEvent.GENERATION_COMPLETE)

        return events

    # ------------------------------------------------------------------
    # Checkpoint implementations
    # ------------------------------------------------------------------

    def _do_primary_reproduction(
        self,
        current_tick: int,
        world: World,
        rng: np.random.Generator,
    ) -> None:
        """
        Primary reproduction checkpoint.

        For each alive animal, determine offspring count based on energy:
          - energy < repro_energy_low  → 0
          - energy < repro_energy_high → 1
          - energy >= repro_energy_high → 2

        Create offspring and add them to the world.
        """
        self.gen_stats.pop_at_primary_repro = world.alive_count

        births = 0
        parents = 0
        new_animals: list[Animal] = []

        for animal in world.get_alive_animals():
            n_offspring = animal.offspring_count()
            if n_offspring > 0:
                parents += 1
            for _ in range(n_offspring):
                child = animal.create_offspring(
                    current_tick=current_tick,
                    stress_mode=world.stress_mode,
                    rng=rng,
                    world_width=world.width,
                    world_height=world.height,
                    generation=self.current_generation + 1,
                )
                new_animals.append(child)
                births += 1

        # Add all new animals to world (after iteration to avoid modifying during loop)
        for child in new_animals:
            world.add_animal(child)

        self.gen_stats.primary_repro_births = births
        self.gen_stats.parents_at_primary = parents

    def _do_survival_check(
        self,
        current_tick: int,
        world: World,
    ) -> None:
        """
        Survival check at 100% of gen_length.

        Animals with energy <= survival_threshold die ("age" death).
        Survivors continue to the bonus reproduction phase.
        """
        self.gen_stats.pop_at_survival_check = world.alive_count
        deaths = 0

        for animal in world.get_alive_animals():
            if not animal.survives_generation_end():
                world.kill_animal(animal, cause="age")
                deaths += 1

        self.gen_stats.survival_check_deaths = deaths

    def _do_bonus_reproduction(
        self,
        current_tick: int,
        world: World,
        rng: np.random.Generator,
    ) -> None:
        """
        Bonus (secondary) reproduction at 120% of gen_length.

        Same logic as primary reproduction, but only for animals that
        survived the 100% survival check.
        """
        self.gen_stats.pop_at_bonus_repro = world.alive_count

        births = 0
        parents = 0
        new_animals: list[Animal] = []

        for animal in world.get_alive_animals():
            n_offspring = animal.offspring_count()
            if n_offspring > 0:
                parents += 1
            for _ in range(n_offspring):
                child = animal.create_offspring(
                    current_tick=current_tick,
                    stress_mode=world.stress_mode,
                    rng=rng,
                    world_width=world.width,
                    world_height=world.height,
                    generation=self.current_generation + 1,
                )
                new_animals.append(child)
                births += 1

        for child in new_animals:
            world.add_animal(child)

        self.gen_stats.bonus_repro_births = births
        self.gen_stats.parents_at_bonus = parents

    # ------------------------------------------------------------------
    # Generation advancement
    # ------------------------------------------------------------------

    def _advance_generation(self, current_tick: int, world: World) -> None:
        """
        Finalize the current generation and start the next one.

        - Record final population
        - Store generation stats
        - Reset checkpoint flags
        - Increment generation counter
        - Update gen_start_tick
        """
        # Finalize stats
        self.gen_stats.total_births = (
            self.gen_stats.primary_repro_births + self.gen_stats.bonus_repro_births
        )
        self.gen_stats.pop_at_generation_end = world.alive_count

        # Archive stats
        self.all_gen_stats.append(self.gen_stats)

        # Increment generation
        self.current_generation += 1
        world.generation = self.current_generation

        # Reset for next generation
        self.gen_start_tick = current_tick
        self._primary_fired = False
        self._survival_fired = False
        self._bonus_fired = False
        self.gen_stats = GenerationStats(generation=self.current_generation)

        # Clear dead animals list (metrics already collected)
        world.clear_dead_animals()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def total_generations_completed(self) -> int:
        """Number of fully completed generations."""
        return len(self.all_gen_stats)

    def get_last_gen_stats(self) -> Optional[GenerationStats]:
        """Get stats for the last completed generation, or None."""
        if self.all_gen_stats:
            return self.all_gen_stats[-1]
        return None

    def __repr__(self) -> str:
        return (
            f"GenerationManager(gen={self.current_generation}, "
            f"start_tick={self.gen_start_tick}, "
            f"primary={'done' if self._primary_fired else 'pending'}, "
            f"survival={'done' if self._survival_fired else 'pending'}, "
            f"bonus={'done' if self._bonus_fired else 'pending'})"
        )
