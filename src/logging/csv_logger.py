"""
CSV Logger for the Evolution Simulator.

Writes one row per generation to a CSV file with all KPI columns.
Supports incremental appending (writes header on first row, then appends).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

from src.simulation.metrics import MetricsCollector


class CSVLogger:
    """
    Logs generation KPIs to a CSV file.

    Usage:
        logger = CSVLogger("runs/my_run/metrics.csv")
        logger.log_row(kpi_dict)          # append one row
        logger.log_all(metrics.history)   # write all rows at once

    Attributes:
        file_path: Path to the CSV file.
        columns: Ordered list of column names.
    """

    def __init__(
        self,
        file_path: str | Path,
        columns: Optional[list[str]] = None,
    ):
        """
        Initialize the CSV logger.

        Args:
            file_path: Path to the output CSV file. Directory is created if needed.
            columns: Ordered column names. None = use MetricsCollector.kpi_names().
        """
        self.file_path = Path(file_path)
        self.columns = columns or MetricsCollector.kpi_names()
        self._header_written = False

        # Create parent directory
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_header(self) -> None:
        """Write header row if not yet written and file doesn't exist or is empty."""
        if self._header_written:
            return

        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            self._header_written = True
            return

        with open(self.file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            writer.writeheader()

        self._header_written = True

    def log_row(self, kpi_dict: dict) -> None:
        """
        Append a single KPI row to the CSV file.

        Args:
            kpi_dict: Dict of KPI_name → value.
        """
        self._ensure_header()

        with open(self.file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            writer.writerow(kpi_dict)

    def log_all(self, kpi_list: list[dict]) -> None:
        """
        Write all KPI rows at once (overwrites existing file).

        Args:
            kpi_list: List of KPI dicts (one per generation).
        """
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            writer.writeheader()
            for row in kpi_list:
                writer.writerow(row)

        self._header_written = True

    def read_back(self) -> list[dict]:
        """
        Read back all rows from the CSV file.

        Returns:
            List of dicts (one per row).
        """
        if not self.file_path.exists():
            return []

        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
