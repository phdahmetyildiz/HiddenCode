"""
Snapshot manager for the Evolution Simulator.

Saves and loads full world state snapshots (JSON) per generation.
Snapshots capture all alive animals, food, pitfalls, and world state
for later replay or analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.core.world import World
from src.core.animal import Animal
from src.core.food import Food
from src.core.pitfall import Pitfall


class SnapshotManager:
    """
    Saves and loads world state snapshots as JSON files.

    Each snapshot is saved to: {output_dir}/snapshots/gen_{N:04d}.json

    Attributes:
        output_dir: Base output directory for the run.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.snapshot_dir = self.output_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save(self, world: World, generation: int) -> Path:
        """
        Save a snapshot of the current world state.

        Args:
            world: The simulation world to snapshot.
            generation: Current generation number (for filename).

        Returns:
            Path to the saved snapshot file.
        """
        snapshot = self._world_to_dict(world, generation)
        file_path = self.snapshot_dir / f"gen_{generation:04d}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=_json_default)

        return file_path

    def load(self, generation: int) -> dict:
        """
        Load a snapshot for a specific generation.

        Args:
            generation: Generation number.

        Returns:
            Snapshot dict.

        Raises:
            FileNotFoundError: If snapshot doesn't exist.
        """
        file_path = self.snapshot_dir / f"gen_{generation:04d}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_snapshots(self) -> list[int]:
        """
        List all available snapshot generation numbers.

        Returns:
            Sorted list of generation numbers.
        """
        generations = []
        for p in self.snapshot_dir.glob("gen_*.json"):
            try:
                gen_num = int(p.stem.split("_")[1])
                generations.append(gen_num)
            except (IndexError, ValueError):
                continue
        return sorted(generations)

    def _world_to_dict(self, world: World, generation: int) -> dict:
        """Convert world state to a serializable dict."""
        return {
            "generation": generation,
            "tick": world.tick_count,
            "width": world.width,
            "height": world.height,
            "stress_mode": world.stress_mode,
            "alive_count": world.alive_count,
            "food_count": world.food_count,
            "pitfall_count": world.pitfall_count,
            "animals": [
                self._animal_to_dict(a) for a in world.animals.values()
            ],
            "food": [
                self._food_to_dict(f) for f in world.food.values() if f.active
            ],
            "pitfalls": [
                self._pitfall_to_dict(p) for p in world.pitfalls.values() if p.active
            ],
        }

    @staticmethod
    def _animal_to_dict(animal: Animal) -> dict:
        return animal.to_dict()

    @staticmethod
    def _food_to_dict(food: Food) -> dict:
        return {
            "x": food.x,
            "y": food.y,
            "remaining_lifespan": food.remaining_lifespan,
            "energy_value": round(food.energy_value, 6),
            "consumed": food.consumed,
        }

    @staticmethod
    def _pitfall_to_dict(pitfall: Pitfall) -> dict:
        return {
            "x": pitfall.x,
            "y": pitfall.y,
            "name": pitfall.name,
            "sequence": pitfall.sequence_str,
            "remaining_lifespan": pitfall.remaining_lifespan,
        }


def _json_default(obj: Any) -> Any:
    """JSON serialization fallback for NumPy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
