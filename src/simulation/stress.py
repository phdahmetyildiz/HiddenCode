"""
Stress Event Manager for the Evolution Simulator.

Handles user-triggered and auto-triggered environmental stress events:
  - Activates stress_mode on the world
  - Spawns a burst of new pitfall types
  - Optionally modifies food spawning rate
  - Increases mutation rate (applied automatically during reproduction via stress_mode flag)
  - Supports auto-trigger at a configured tick and auto-deactivation after a duration

Stress effects on reproduction (higher mutation rate, coding-only mutations) are
handled by Animal.create_offspring() reading world.stress_mode — no changes needed here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.core.config import SimConfig, PitfallType
from src.core.world import World


class StressManager:
    """
    Manages stress events during a simulation.

    Responsibilities:
      - Trigger stress (manual or auto at configured tick)
      - Spawn burst of new pitfall types on the grid
      - Track stress duration and auto-deactivate
      - Optionally override food spawning rate during stress
      - Store original food rate for restoration

    Attributes:
        config: Simulation configuration.
        active: Whether stress mode is currently active.
        trigger_tick: Tick at which stress was activated (None if never).
        pitfalls_spawned_on_trigger: Number of pitfalls spawned when triggered.
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.active: bool = False
        self.trigger_tick: Optional[int] = None
        self.pitfalls_spawned_on_trigger: int = 0

        # Saved state for restoration
        self._original_food_rate: Optional[float] = None

        # Auto-trigger config
        self._auto_trigger_tick = config.stress.trigger_tick
        self._auto_triggered: bool = False

    # ------------------------------------------------------------------
    # Trigger / Deactivate
    # ------------------------------------------------------------------

    def trigger(self, world: World, rng: np.random.Generator) -> int:
        """
        Activate stress mode.

        Effects:
          1. Set world.stress_mode = True
          2. Spawn a burst of new pitfall types
          3. Optionally override food rate

        Args:
            world: The simulation world.
            rng: Random generator for pitfall placement.

        Returns:
            Number of pitfalls spawned in the burst.
        """
        if self.active:
            return 0  # Already active, no-op

        self.active = True
        self.trigger_tick = world.tick_count
        world.stress_mode = True

        # Save original food rate
        self._original_food_rate = self.config.resources.food_rate

        # Override food rate if configured
        if self.config.stress.food_rate_during_stress is not None:
            self.config.resources.food_rate = self.config.stress.food_rate_during_stress

        # Get stress pitfall types
        stress_pitfall_types = self._get_stress_pitfall_types()

        # Spawn burst of new pitfalls
        burst_count = self.config.stress.pitfall_burst_count
        if burst_count > 0 and stress_pitfall_types:
            self.pitfalls_spawned_on_trigger = world.spawn_pitfalls_batch(
                count=burst_count,
                pitfall_types=stress_pitfall_types,
            )
        else:
            self.pitfalls_spawned_on_trigger = 0

        return self.pitfalls_spawned_on_trigger

    def deactivate(self, world: World) -> None:
        """
        Deactivate stress mode and restore original settings.

        Effects:
          1. Set world.stress_mode = False
          2. Restore original food rate

        Args:
            world: The simulation world.
        """
        if not self.active:
            return  # Not active, no-op

        self.active = False
        world.stress_mode = False

        # Restore food rate
        if self._original_food_rate is not None:
            self.config.resources.food_rate = self._original_food_rate
            self._original_food_rate = None

    # ------------------------------------------------------------------
    # Per-tick check (auto-trigger / auto-deactivate)
    # ------------------------------------------------------------------

    def check_tick(self, world: World, rng: np.random.Generator) -> str:
        """
        Check if any stress events should fire at the current tick.

        Should be called once per tick (after the main simulation tick).

        Actions:
          - Auto-trigger stress at configured tick
          - Auto-deactivate stress after configured duration

        Args:
            world: The simulation world.
            rng: Random generator.

        Returns:
            Event string: "triggered", "deactivated", or "none".
        """
        current_tick = world.tick_count

        # Auto-trigger
        if (
            not self._auto_triggered
            and self._auto_trigger_tick is not None
            and current_tick >= self._auto_trigger_tick
            and not self.active
        ):
            self._auto_triggered = True
            self.trigger(world, rng)
            return "triggered"

        # Auto-deactivate (duration-based)
        if (
            self.active
            and self.trigger_tick is not None
            and self.config.stress.duration_ticks is not None
        ):
            elapsed = current_tick - self.trigger_tick
            if elapsed >= self.config.stress.duration_ticks:
                self.deactivate(world)
                return "deactivated"

        return "none"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_stress_pitfall_types(self) -> list[PitfallType]:
        """Convert config stress pitfall type dicts to PitfallType objects."""
        types = []
        for pt_dict in self.config.stress.post_event_pitfall_types:
            types.append(PitfallType(**pt_dict))
        return types

    @property
    def effective_mutation_rate(self) -> float:
        """
        Return the currently effective mutation rate.

        When stress is active: stress_mutation_rate
        Otherwise: base_mutation_rate
        """
        if self.active:
            return self.config.genetics.stress_mutation_rate
        return self.config.genetics.base_mutation_rate

    @property
    def stress_duration_elapsed(self) -> Optional[int]:
        """
        How many ticks stress has been active.

        Returns None if stress is not active.
        """
        if not self.active or self.trigger_tick is None:
            return None
        return 0  # Computed externally from world tick

    def get_status(self) -> dict:
        """Return a status dict for logging/UI."""
        return {
            "active": self.active,
            "trigger_tick": self.trigger_tick,
            "pitfalls_spawned_on_trigger": self.pitfalls_spawned_on_trigger,
            "effective_mutation_rate": self.effective_mutation_rate,
            "auto_trigger_tick": self._auto_trigger_tick,
            "auto_triggered": self._auto_triggered,
            "duration_ticks": self.config.stress.duration_ticks,
        }

    def __repr__(self) -> str:
        return (
            f"StressManager(active={self.active}, "
            f"trigger_tick={self.trigger_tick}, "
            f"burst_spawned={self.pitfalls_spawned_on_trigger})"
        )
