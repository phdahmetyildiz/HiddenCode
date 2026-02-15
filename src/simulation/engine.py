"""
Simulation Engine — Main tick loop for the Evolution Simulator.

Manages the core simulation step: resource spawning/decay, agent energy drain,
movement, feeding, pitfall interactions, and death checks.

Generation lifecycle (reproduction, survival checkpoints) is handled by
the integrated `GenerationManager` (Phase 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from src.core.config import SimConfig
from src.core.world import World
from src.core.animal import Animal, reset_animal_id_counter
from src.core.food import Food
from src.core.pitfall import Pitfall
from src.simulation.generation import GenerationManager, GenerationEvent, GenerationStats
from src.simulation.stress import StressManager
from src.utils.spatial import (
    toroidal_wrap,
    toroidal_distance_sq,
    move_toward,
    random_move,
)


# ---------------------------------------------------------------------------
# Tick statistics — lightweight counters for one tick
# ---------------------------------------------------------------------------

@dataclass
class TickStats:
    """Statistics collected during a single tick."""
    food_spawned: int = 0
    food_expired: int = 0
    food_eaten: int = 0
    pitfalls_spawned: int = 0
    pitfalls_expired: int = 0
    pitfall_encounters: int = 0
    pitfall_total_damage: int = 0
    pitfall_zero_damage_encounters: int = 0
    deaths_starvation: int = 0
    deaths_emergency: int = 0
    deaths_pitfall: int = 0
    moves_toward_food: int = 0
    moves_random: int = 0


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a complete simulation run."""
    config: SimConfig
    seed: int
    total_ticks: int = 0
    total_generations: int = 0
    final_alive_count: int = 0
    extinct: bool = False
    extinction_tick: Optional[int] = None
    tick_stats_history: list[TickStats] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Core simulation engine.

    Manages the tick loop: resource spawning/decay, agent processing
    (energy drain, death checks, movement, feeding, pitfall interaction).

    Attributes:
        config: Simulation configuration.
        world: The simulation world.
        generation_manager: Manages generation lifecycle & reproduction.
        stress_manager: Manages stress events.
        rng: Master random generator (seeded).
        tick_stats: Statistics for the current tick.
        on_tick: Optional callback invoked after each tick(tick_number, engine).
        on_generation: Optional callback invoked after generation boundary(gen_number, engine).
        on_stress: Optional callback invoked when stress is triggered/deactivated(event, engine).
    """

    def __init__(self, config: SimConfig, seed: Optional[int] = None):
        """
        Create a simulation engine.

        Args:
            config: Simulation configuration.
            seed: Random seed override. None = use config.world.seed.
        """
        self.config = config

        if seed is not None:
            self.config.world.seed = seed

        self.world = World(self.config)
        self.rng = self.world.rng  # share the world's seeded RNG

        self.generation_manager = GenerationManager(self.config)
        self.stress_manager = StressManager(self.config)

        self.tick_stats = TickStats()
        self._accumulated_tick_stats: list[TickStats] = []

        # Callbacks
        self.on_tick: Optional[Callable[[int, "SimulationEngine"], None]] = None
        self.on_generation: Optional[Callable[[int, "SimulationEngine"], None]] = None
        self.on_stress: Optional[Callable[[str, "SimulationEngine"], None]] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, population_count: Optional[int] = None) -> None:
        """
        Set up the world: create initial population and resources.

        Args:
            population_count: Override initial animal count. None = use config.
        """
        reset_animal_id_counter()
        self.world.initialize_population(count=population_count)

        # Spawn some initial food so animals don't all starve immediately
        initial_food = max(
            1,
            int(self.config.resources.food_rate * self.config.resources.food_lifespan * 0.5),
        )
        for _ in range(initial_food):
            self.world.spawn_food(rate=1.0)

    # ------------------------------------------------------------------
    # Core tick
    # ------------------------------------------------------------------

    def tick(self) -> TickStats:
        """
        Execute one simulation tick.

        Processing order:
          1. Spawn food and pitfalls
          2. Decay resources (remove expired)
          3. Process each alive animal (shuffled order):
             a. Energy drain
             b. Starvation death check
             c. Emergency death check (low energy + no food in sight)
             d. Movement (toward food or random)
             e. Food interaction (heaviest at cell eats)
             f. Pitfall interaction (damage calculation)
          4. Increment tick counter
          5. Fire callbacks

        Returns:
            TickStats for this tick.
        """
        stats = TickStats()
        world = self.world
        config = self.config

        # --- 1. Spawn resources ---
        stats.food_spawned = world.spawn_food()
        stats.pitfalls_spawned = world.spawn_pitfalls()

        # --- 2. Decay resources ---
        food_expired, pitfalls_expired = world.decay_all_resources()
        stats.food_expired = food_expired
        stats.pitfalls_expired = pitfalls_expired

        # --- 3. Process each animal ---
        # Get shuffled list for fair processing order
        animals = world.get_shuffled_animals()

        # Track which food positions have been eaten this tick
        # (to avoid double-eating at the same cell)
        eaten_this_tick: set[tuple[int, int]] = set()

        for animal in animals:
            if not animal.alive:
                continue  # May have been killed earlier in this tick

            # --- 3a. Energy drain ---
            animal.apply_energy_drain()

            # --- 3b. Starvation check ---
            if animal.is_starved():
                world.kill_animal(animal, cause="starvation")
                stats.deaths_starvation += 1
                continue

            # --- 3c. Emergency death check ---
            has_food_nearby = world.food_in_range(
                animal.x, animal.y,
                radius=animal.eyesight_radius,
            )
            if animal.is_emergency(food_in_range=has_food_nearby):
                world.kill_animal(animal, cause="emergency")
                stats.deaths_emergency += 1
                continue

            # --- 3d. Movement ---
            nearest = world.nearest_food_in_range(
                animal.x, animal.y,
                radius=animal.eyesight_radius,
            )

            if nearest is not None:
                # Move toward nearest food
                new_x, new_y = move_toward(
                    animal.x, animal.y,
                    nearest.x, nearest.y,
                    world.width, world.height,
                )
                world.move_animal(animal, new_x, new_y)
                stats.moves_toward_food += 1
            else:
                # Random movement
                new_x, new_y = random_move(
                    animal.x, animal.y,
                    world.width, world.height,
                    self.rng,
                )
                world.move_animal(animal, new_x, new_y)
                stats.moves_random += 1

            # --- 3e. Food interaction ---
            food_here = world.food_at(animal.x, animal.y)
            if food_here is not None and (animal.x, animal.y) not in eaten_this_tick:
                # Only the heaviest animal at this cell eats
                heaviest = world.heaviest_animal_at(animal.x, animal.y, rng=self.rng)
                if heaviest is not None and heaviest.id == animal.id:
                    energy_gained = food_here.consume()
                    animal.gain_energy(energy_gained)
                    world.remove_food((animal.x, animal.y))
                    eaten_this_tick.add((animal.x, animal.y))
                    stats.food_eaten += 1

            # --- 3f. Pitfall interaction ---
            pitfall_here = world.pitfall_at(animal.x, animal.y)
            if pitfall_here is not None:
                damage = pitfall_here.calculate_damage(animal.defense_bits)
                stats.pitfall_encounters += 1
                stats.pitfall_total_damage += damage

                if damage == 0:
                    stats.pitfall_zero_damage_encounters += 1
                else:
                    energy_loss = pitfall_here.calculate_energy_loss(
                        animal.defense_bits,
                        max_pitfall_loss_pct=config.energy.max_pitfall_loss_pct,
                    )
                    animal.apply_pitfall_damage(energy_loss)

                    # Check if pitfall killed the animal
                    if animal.is_starved():
                        world.kill_animal(animal, cause="pitfall")
                        stats.deaths_pitfall += 1

        # --- 4. Increment tick counter ---
        world.tick_count += 1

        # --- 5. Generation lifecycle checks ---
        gen_events = self.generation_manager.check(
            current_tick=world.tick_count,
            world=world,
            rng=self.rng,
        )

        # Fire generation callback if generation completed
        if GenerationEvent.GENERATION_COMPLETE in gen_events:
            if self.on_generation is not None:
                self.on_generation(self.generation_manager.current_generation - 1, self)

        # --- 6. Stress event checks (auto-trigger / auto-deactivate) ---
        stress_event = self.stress_manager.check_tick(world, self.rng)
        if stress_event != "none" and self.on_stress is not None:
            self.on_stress(stress_event, self)

        # --- 7. Store stats and fire tick callback ---
        self.tick_stats = stats
        self._accumulated_tick_stats.append(stats)

        if self.on_tick is not None:
            self.on_tick(world.tick_count, self)

        return stats

    # ------------------------------------------------------------------
    # Multi-tick run
    # ------------------------------------------------------------------

    def run(
        self,
        max_ticks: Optional[int] = None,
        max_generations: Optional[int] = None,
    ) -> RunResult:
        """
        Run the simulation for a specified number of ticks or generations.

        Stops when ANY of these conditions is met:
          - max_ticks reached
          - max_generations completed (full generations including bonus repro)
          - extinction (all animals dead)

        If neither max_ticks nor max_generations is specified, runs for one
        full generation cycle.

        Args:
            max_ticks: Maximum number of ticks to simulate.
            max_generations: Maximum number of full generations to simulate.

        Returns:
            RunResult with summary statistics.
        """
        if max_ticks is None and max_generations is None:
            # Default: run one full generation cycle (including bonus)
            gen = self.config.generation
            max_ticks = int(gen.gen_length * gen.bonus_repro_pct) + 1

        result = RunResult(
            config=self.config,
            seed=self.config.world.seed,
        )

        ticks_run = 0
        while True:
            # Check tick limit
            if max_ticks is not None and ticks_run >= max_ticks:
                break

            # Check generation limit
            if (max_generations is not None
                    and self.generation_manager.total_generations_completed >= max_generations):
                break

            self.tick()
            ticks_run += 1

            # Check extinction
            if self.world.is_extinct:
                result.extinct = True
                result.extinction_tick = self.world.tick_count
                break

        result.total_ticks = ticks_run
        result.total_generations = self.generation_manager.total_generations_completed
        result.final_alive_count = self.world.alive_count
        result.tick_stats_history = list(self._accumulated_tick_stats)

        return result

    # ------------------------------------------------------------------
    # Accumulated statistics helpers
    # ------------------------------------------------------------------

    def get_accumulated_stats(self) -> dict[str, int]:
        """
        Sum all tick stats from the current accumulation period.

        Returns:
            Dict of stat_name → total_value.
        """
        totals: dict[str, int] = {}
        for stats in self._accumulated_tick_stats:
            for attr in (
                "food_spawned", "food_expired", "food_eaten",
                "pitfalls_spawned", "pitfalls_expired",
                "pitfall_encounters", "pitfall_total_damage",
                "pitfall_zero_damage_encounters",
                "deaths_starvation", "deaths_emergency", "deaths_pitfall",
                "moves_toward_food", "moves_random",
            ):
                totals[attr] = totals.get(attr, 0) + getattr(stats, attr)
        return totals

    def reset_accumulated_stats(self) -> list[TickStats]:
        """
        Reset and return the accumulated tick stats (e.g., at generation boundary).

        Returns:
            The accumulated stats before reset.
        """
        old = self._accumulated_tick_stats
        self._accumulated_tick_stats = []
        return old

    # ------------------------------------------------------------------
    # Stress control (manual trigger / deactivate from UI or script)
    # ------------------------------------------------------------------

    def trigger_stress(self) -> int:
        """
        Manually trigger a stress event.

        Returns:
            Number of pitfalls spawned in the burst.
        """
        count = self.stress_manager.trigger(self.world, self.rng)
        if self.on_stress is not None:
            self.on_stress("triggered", self)
        return count

    def deactivate_stress(self) -> None:
        """Manually deactivate stress mode."""
        self.stress_manager.deactivate(self.world)
        if self.on_stress is not None:
            self.on_stress("deactivated", self)

    @property
    def stress_active(self) -> bool:
        """Whether stress mode is currently active."""
        return self.stress_manager.active

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_extinct(self) -> bool:
        """Check if the population is extinct."""
        return self.world.is_extinct

    @property
    def alive_count(self) -> int:
        """Number of alive animals."""
        return self.world.alive_count

    @property
    def current_tick(self) -> int:
        """Current tick number."""
        return self.world.tick_count

    @property
    def current_generation(self) -> int:
        """Current generation number."""
        return self.generation_manager.current_generation

    @property
    def generations_completed(self) -> int:
        """Number of fully completed generations."""
        return self.generation_manager.total_generations_completed

    def __repr__(self) -> str:
        return (
            f"SimulationEngine(tick={self.current_tick}, "
            f"alive={self.alive_count}, "
            f"food={self.world.food_count}, "
            f"pitfalls={self.world.pitfall_count})"
        )
