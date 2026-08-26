"""Run directory, CSV metrics, optional snapshots."""

from __future__ import annotations

import csv
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import SimConfig, save_config
from src.metrics import EpochMetrics
from src.world import World


class RunManager:
    def __init__(self, config: SimConfig, base_dir: str | Path | None = None):
        base = Path(base_dir) if base_dir is not None else Path(config.viz.output_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = base / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, self.run_dir / "config.json")
        self.csv_path = self.run_dir / "metrics.csv"
        self._writer: Optional[csv.DictWriter] = None
        self._csv_file = None
        self.config = config

    def log_epoch(self, metrics: EpochMetrics) -> None:
        row = metrics.to_dict()
        if self._writer is None:
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._csv_file.flush()

    def save_snapshot(self, world: World, epoch: int) -> Path:
        snap_dir = self.run_dir / "snapshots"
        snap_dir.mkdir(exist_ok=True)
        path = snap_dir / f"epoch_{epoch:04d}.pkl"
        payload = {
            "tick": world.tick,
            "n": world.n,
            "x": world.x[: world.n].copy(),
            "y": world.y[: world.n].copy(),
            "energy": world.energy[: world.n].copy(),
            "weight": world.weight[: world.n].copy(),
            "speed": world.speed[: world.n].copy(),
            "dna": world.dna[: world.n].copy(),
            "food_life": world.food_life.copy(),
            "pitfall_life": world.pitfall_life.copy(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

    def finalize(self, summary: dict[str, Any]) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
