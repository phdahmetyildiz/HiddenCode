"""
World (Simulation Environment) for the Evolution Simulator.

Manages the 2D toroidal grid, all animals, food items, and pitfall hazards.
Provides spatial queries (nearest food in range, pitfall at position, etc.)
and resource spawning/decay.

Uses grid-based spatial indexing for O(1) lookups of entities at a given cell,
making eyesight queries efficient even on large grids.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.core.config import SimConfig, PitfallType
from src.core.dna import DNA
from src.core.animal import Animal, reset_animal_id_counter
from src.core.food import Food
from src.core.pitfall import Pitfall
from src.utils.spatial import (
    toroidal_wrap,
    toroidal_distance_sq,
    cells_in_radius,
    random_move,
)


class World:
    """
    The simulation world: a 2D toroidal grid with animals, food, and pitfalls.

    Spatial indexing:
      - _animal_grid[x][y] → set of animal IDs at that cell
      - _food_grid[(x, y)] → Food instance
      - _pitfall_grid[(x, y)] → Pitfall instance

    Attributes:
        width: Grid width.
        height: Grid height.
        config: Simulation configuration.
        tick_count: Current simulation tick.
        generation: Current generation number.
        stress_mode: Whether stress mode is active.
        animals: Dict of animal_id → Animal (alive only).
        dead_animals: List of animals that died (for metrics).
        food: Dict of (x, y) → Food (active only).
        pitfalls: Dict of (x, y) → Pitfall (active only).
        rng: Seeded random generator.
    """

    def __init__(self, config: SimConfig):
        """
        Initialize the world from a configuration.

        Args:
            config: Simulation configuration.
        """
        self.config = config
        self.width = config.world.width
        self.height = config.world.height
        self.rng = np.random.default_rng(config.world.seed)

        self.tick_count: int = 0
        self.generation: int = 0
        self.stress_mode: bool = False

        # Entity storage
        self.animals: dict[int, Animal] = {}
        self.dead_animals: list[Animal] = []
        self.food: dict[tuple[int, int], Food] = {}
        self.pitfalls: dict[tuple[int, int], Pitfall] = {}

        # Spatial index for animals (grid cell → set of animal IDs)
        self._animal_grid: dict[tuple[int, int], set[int]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def initialize_population(self, count: Optional[int] = None) -> None:
        """
        Create the initial population of animals with random DNA and positions.

        Args:
            count: Number of animals. If None, uses config.population.initial_count.
        """
        if count is None:
            count = self.config.population.initial_count

        for _ in range(count):
            dna = DNA.create_random(
                length=self.config.genetics.dna_length,
                rng=self.rng,
            )
            x = int(self.rng.integers(0, self.width))
            y = int(self.rng.integers(0, self.height))
            animal = Animal(
                dna=dna,
                x=x,
                y=y,
                config=self.config,
                birth_tick=0,
                energy=1.0,
                generation=0,
            )
            self.add_animal(animal)

    # ------------------------------------------------------------------
    # Animal management
    # ------------------------------------------------------------------

    def add_animal(self, animal: Animal) -> None:
        """Add an animal to the world."""
        self.animals[animal.id] = animal
        self._animal_grid[(animal.x, animal.y)].add(animal.id)

    def remove_animal(self, animal: Animal) -> None:
        """Remove an animal from the world (move to dead list)."""
        if animal.id in self.animals:
            del self.animals[animal.id]
        self._animal_grid[(animal.x, animal.y)].discard(animal.id)

    def kill_animal(self, animal: Animal, cause: str) -> None:
        """Mark an animal as dead and move it to the dead list."""
        animal.die(cause=cause, tick=self.tick_count)
        self.remove_animal(animal)
        self.dead_animals.append(animal)

    def move_animal(self, animal: Animal, new_x: int, new_y: int) -> None:
        """
        Update an animal's position, maintaining the spatial index.

        Args:
            animal: The animal to move.
            new_x, new_y: New grid coordinates (will be toroidally wrapped).
        """
        # Remove from old cell
        self._animal_grid[(animal.x, animal.y)].discard(animal.id)
        # Update position
        animal.x, animal.y = toroidal_wrap(new_x, new_y, self.width, self.height)
        # Add to new cell
        self._animal_grid[(animal.x, animal.y)].add(animal.id)

    def animals_at(self, x: int, y: int) -> list[Animal]:
        """Get all alive animals at a given grid cell."""
        ids = self._animal_grid.get((x, y), set())
        return [self.animals[aid] for aid in ids if aid in self.animals]

    @property
    def alive_count(self) -> int:
        """Number of alive animals."""
        return len(self.animals)

    @property
    def is_extinct(self) -> bool:
        """True if all animals are dead."""
        return len(self.animals) == 0

    def get_alive_animals(self) -> list[Animal]:
        """Get a list of all alive animals (snapshot — safe to iterate while modifying)."""
        return list(self.animals.values())

    def get_shuffled_animals(self) -> list[Animal]:
        """Get alive animals in a random shuffled order (for fair tick processing)."""
        animals = self.get_alive_animals()
        self.rng.shuffle(animals)
        return animals

    # ------------------------------------------------------------------
    # Food management
    # ------------------------------------------------------------------

    def add_food(self, food: Food) -> None:
        """
        Add a food item to the world.

        If a food item already exists at the same position, the new one replaces it.
        """
        self.food[(food.x, food.y)] = food

    def remove_food(self, pos: tuple[int, int]) -> Optional[Food]:
        """Remove and return food at a position, or None if no food there."""
        return self.food.pop(pos, None)

    def food_at(self, x: int, y: int) -> Optional[Food]:
        """Get food at a position (if active), or None."""
        food = self.food.get((x, y))
        if food is not None and food.active:
            return food
        return None

    def spawn_food(self, rate: Optional[float] = None) -> int:
        """
        Spawn food items at random positions.

        Number of items is drawn from a Poisson distribution with mean=rate.
        Food is placed at uniformly random grid positions.

        Args:
            rate: Expected food items per tick. None = use config value.

        Returns:
            Number of food items actually spawned.
        """
        if rate is None:
            rate = self.config.resources.food_rate

        if rate <= 0:
            return 0

        count = int(self.rng.poisson(rate))
        spawned = 0

        for _ in range(count):
            x = int(self.rng.integers(0, self.width))
            y = int(self.rng.integers(0, self.height))
            # Don't overwrite existing food
            if (x, y) not in self.food:
                food = Food(
                    x=x,
                    y=y,
                    remaining_lifespan=self.config.resources.food_lifespan,
                    energy_value=self.config.energy.food_gain,
                )
                self.food[(x, y)] = food
                spawned += 1

        return spawned

    def decay_food(self) -> int:
        """
        Tick all food items and remove expired ones.

        Returns:
            Number of food items that expired.
        """
        expired_keys = []
        for pos, food in self.food.items():
            if food.tick():
                expired_keys.append(pos)

        for pos in expired_keys:
            del self.food[pos]

        return len(expired_keys)

    @property
    def food_count(self) -> int:
        """Number of active food items on the grid."""
        return len(self.food)

    # ------------------------------------------------------------------
    # Pitfall management
    # ------------------------------------------------------------------

    def add_pitfall(self, pitfall: Pitfall) -> None:
        """Add a pitfall to the world. Replaces existing at same position."""
        self.pitfalls[(pitfall.x, pitfall.y)] = pitfall

    def remove_pitfall(self, pos: tuple[int, int]) -> Optional[Pitfall]:
        """Remove and return pitfall at a position, or None."""
        return self.pitfalls.pop(pos, None)

    def pitfall_at(self, x: int, y: int) -> Optional[Pitfall]:
        """Get pitfall at a position (if active), or None."""
        pitfall = self.pitfalls.get((x, y))
        if pitfall is not None and pitfall.active:
            return pitfall
        return None

    def spawn_pitfalls(
        self,
        rate: Optional[float] = None,
        pitfall_types: Optional[list[PitfallType]] = None,
    ) -> int:
        """
        Spawn pitfall items at random positions.

        Number drawn from Poisson(rate). Type chosen uniformly at random
        from available pitfall types.

        Args:
            rate: Expected pitfalls per tick. None = use config value.
            pitfall_types: List of PitfallType to choose from. None = use config.

        Returns:
            Number of pitfalls actually spawned.
        """
        if rate is None:
            rate = self.config.resources.pitfall_rate

        if pitfall_types is None:
            pitfall_types = self.config.resources.get_pitfall_types()

        if rate <= 0 or not pitfall_types:
            return 0

        count = int(self.rng.poisson(rate))
        spawned = 0

        for _ in range(count):
            x = int(self.rng.integers(0, self.width))
            y = int(self.rng.integers(0, self.height))
            if (x, y) not in self.pitfalls:
                # Choose random pitfall type
                pt = pitfall_types[int(self.rng.integers(0, len(pitfall_types)))]
                pitfall = Pitfall.from_string(
                    x=x, y=y,
                    name=pt.name,
                    sequence_str=pt.sequence,
                    lifespan=self.config.resources.pitfall_lifespan,
                )
                self.pitfalls[(x, y)] = pitfall
                spawned += 1

        return spawned

    def spawn_pitfalls_batch(
        self,
        count: int,
        pitfall_types: list[PitfallType],
    ) -> int:
        """
        Spawn a fixed number of pitfalls (used for stress event bursts).

        Args:
            count: Exact number to try to spawn.
            pitfall_types: Available pitfall types.

        Returns:
            Number actually spawned.
        """
        spawned = 0
        for _ in range(count):
            x = int(self.rng.integers(0, self.width))
            y = int(self.rng.integers(0, self.height))
            if (x, y) not in self.pitfalls:
                pt = pitfall_types[int(self.rng.integers(0, len(pitfall_types)))]
                pitfall = Pitfall.from_string(
                    x=x, y=y,
                    name=pt.name,
                    sequence_str=pt.sequence,
                    lifespan=self.config.resources.pitfall_lifespan,
                )
                self.pitfalls[(x, y)] = pitfall
                spawned += 1
        return spawned

    def decay_pitfalls(self) -> int:
        """
        Tick all pitfalls and remove expired ones.

        Returns:
            Number of pitfalls that expired.
        """
        expired_keys = []
        for pos, pitfall in self.pitfalls.items():
            if pitfall.tick():
                expired_keys.append(pos)

        for pos in expired_keys:
            del self.pitfalls[pos]

        return len(expired_keys)

    @property
    def pitfall_count(self) -> int:
        """Number of active pitfalls on the grid."""
        return len(self.pitfalls)

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def nearest_food_in_range(
        self,
        x: int, y: int,
        radius: Optional[int] = None,
    ) -> Optional[Food]:
        """
        Find the nearest active food item within a given radius.

        Uses squared distance for efficiency. On ties, returns any of the closest.

        Args:
            x, y: Center position.
            radius: Search radius (Euclidean). None = config eyesight radius.

        Returns:
            Nearest Food instance, or None if none in range.
        """
        if radius is None:
            radius = self.config.properties.eyesight_radius

        r_sq = radius * radius
        best_food: Optional[Food] = None
        best_dist_sq = r_sq + 1  # sentinel

        # Iterate over all food — for large grids, this can be optimized
        # with spatial bucketing, but for typical food counts this is fast enough.
        # For VERY large grids (Phase 11), we can switch to grid-cell scanning.
        for pos, food in self.food.items():
            if not food.active:
                continue
            dist_sq = toroidal_distance_sq(x, y, food.x, food.y, self.width, self.height)
            if dist_sq <= r_sq and dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_food = food

        return best_food

    def food_in_range(
        self,
        x: int, y: int,
        radius: Optional[int] = None,
    ) -> bool:
        """
        Check if any active food exists within radius.

        Faster than nearest_food_in_range when you only need a boolean.

        Args:
            x, y: Center position.
            radius: Search radius. None = config eyesight radius.

        Returns:
            True if at least one food item is within range.
        """
        if radius is None:
            radius = self.config.properties.eyesight_radius

        r_sq = radius * radius

        for pos, food in self.food.items():
            if not food.active:
                continue
            dist_sq = toroidal_distance_sq(x, y, food.x, food.y, self.width, self.height)
            if dist_sq <= r_sq:
                return True
        return False

    def heaviest_animal_at(self, x: int, y: int, rng: Optional[np.random.Generator] = None) -> Optional[Animal]:
        """
        Get the heaviest animal at a given cell. Ties broken randomly.

        Args:
            x, y: Grid cell.
            rng: Random generator for tie-breaking. Uses world rng if None.

        Returns:
            The heaviest Animal at the cell, or None if empty.
        """
        animals = self.animals_at(x, y)
        if not animals:
            return None
        if len(animals) == 1:
            return animals[0]

        if rng is None:
            rng = self.rng

        max_weight = max(a.weight for a in animals)
        heaviest = [a for a in animals if a.weight == max_weight]

        if len(heaviest) == 1:
            return heaviest[0]
        # Tie: pick randomly
        return heaviest[int(rng.integers(0, len(heaviest)))]

    # ------------------------------------------------------------------
    # Resource decay (both food and pitfalls)
    # ------------------------------------------------------------------

    def decay_all_resources(self) -> tuple[int, int]:
        """
        Decay both food and pitfalls.

        Returns:
            (food_expired, pitfalls_expired) counts.
        """
        food_expired = self.decay_food()
        pitfalls_expired = self.decay_pitfalls()
        return food_expired, pitfalls_expired

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_all_animal_positions(self) -> list[tuple[int, int]]:
        """Get positions of all alive animals."""
        return [(a.x, a.y) for a in self.animals.values()]

    def get_all_food_positions(self) -> list[tuple[int, int]]:
        """Get positions of all active food."""
        return [pos for pos, f in self.food.items() if f.active]

    def get_all_pitfall_positions(self) -> list[tuple[int, int]]:
        """Get positions of all active pitfalls."""
        return [pos for pos, p in self.pitfalls.items() if p.active]

    def clear_dead_animals(self) -> list[Animal]:
        """
        Clear and return the dead animals list (e.g., after metrics collection).

        Returns:
            List of dead animals from this period.
        """
        dead = self.dead_animals
        self.dead_animals = []
        return dead

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"World(size={self.width}x{self.height}, "
            f"tick={self.tick_count}, gen={self.generation}, "
            f"animals={self.alive_count}, food={self.food_count}, "
            f"pitfalls={self.pitfall_count}, stress={self.stress_mode})"
        )
