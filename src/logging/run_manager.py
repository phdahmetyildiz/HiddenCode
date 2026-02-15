"""
Run Manager for the Evolution Simulator.

Manages output directories for simulation runs:
  - Creates timestamped run directories under a base output path
  - Copies the config used for the run
  - Provides paths for metrics CSV, snapshots, etc.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.core.config import SimConfig, save_config
from src.logging.csv_logger import CSVLogger
from src.logging.snapshot import SnapshotManager


class RunManager:
    """
    Manages a single simulation run's output directory.

    Directory structure:
        {base_dir}/{run_name}/
            config.json          — copy of the simulation config
            metrics.csv          — per-generation KPIs
            snapshots/           — world state snapshots (JSON)
                gen_0000.json
                gen_0001.json
                ...

    Attributes:
        run_dir: Path to this run's output directory.
        csv_logger: CSVLogger instance for metrics.
        snapshot_manager: SnapshotManager instance for world snapshots.
    """

    def __init__(
        self,
        config: SimConfig,
        base_dir: Optional[str | Path] = None,
        run_name: Optional[str] = None,
    ):
        """
        Initialize a run manager and create the output directory.

        Args:
            config: Simulation configuration (will be saved as config.json).
            base_dir: Base output directory. None = use config.viz.output_dir.
            run_name: Name for this run's subdirectory. None = timestamp.
        """
        if base_dir is None:
            base_dir = config.viz.output_dir

        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_dir = Path(base_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = self.run_dir / "config.json"
        save_config(config, config_path)
        self._config_path = config_path

        # Initialize loggers
        self.csv_logger = CSVLogger(self.run_dir / "metrics.csv")
        self.snapshot_manager = SnapshotManager(self.run_dir)

    @property
    def config_path(self) -> Path:
        """Path to the saved config file."""
        return self._config_path

    @property
    def metrics_path(self) -> Path:
        """Path to the metrics CSV file."""
        return self.csv_logger.file_path

    @property
    def snapshots_dir(self) -> Path:
        """Path to the snapshots directory."""
        return self.snapshot_manager.snapshot_dir

    def log_generation(self, kpi_dict: dict) -> None:
        """Log a generation's KPIs to CSV."""
        self.csv_logger.log_row(kpi_dict)

    def save_snapshot(self, world, generation: int) -> Path:
        """Save a world state snapshot."""
        return self.snapshot_manager.save(world, generation)

    def finalize(self, summary: Optional[dict] = None) -> None:
        """
        Finalize the run (write summary file if provided).

        Args:
            summary: Optional summary dict to save as summary.json.
        """
        if summary is not None:
            summary_path = self.run_dir / "summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

    @staticmethod
    def list_runs(base_dir: str | Path) -> list[str]:
        """
        List all run directories under the base directory.

        Args:
            base_dir: Base output directory.

        Returns:
            Sorted list of run directory names.
        """
        base = Path(base_dir)
        if not base.exists():
            return []
        return sorted(
            d.name for d in base.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        )

    def __repr__(self) -> str:
        return f"RunManager(run_dir='{self.run_dir}')"
