"""
Run directory, CSV metrics, optional snapshots.

Author: Cursor Grok 4.6 High Fast
"""

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


def export_epochs(path: str | Path, epochs: list[EpochMetrics]) -> Path:
    """Write all epoch KPIs to CSV (Excel-friendly UTF-8 BOM) or .xlsx."""
    path = Path(path)
    rows = [m.to_dict() for m in epochs]
    if not rows:
        raise ValueError("No completed epochs to export yet.")
    fieldnames = list(rows[0].keys())
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        path = path.with_suffix(".xlsx")
        _write_xlsx(path, fieldnames, rows)
    else:
        if suffix != ".csv":
            path = path.with_suffix(".csv")
        _write_csv(path, fieldnames, rows)
    return path


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _col_letter(index: int) -> str:
    """0-based column index → A, B, ..., Z, AA, ..."""
    n = index + 1
    letters = []
    while n:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(65 + rem))
    return "".join(reversed(letters))


def _write_xlsx(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    cells = []
    for c, name in enumerate(fieldnames):
        ref = f"{_col_letter(c)}1"
        cells.append(
            f'<c r="{ref}" t="inlineStr"><is><t>{_xml_escape(str(name))}</t></is></c>'
        )
    header = f'<row r="1">{"".join(cells)}</row>'
    body = []
    for r, row in enumerate(rows, start=2):
        parts = []
        for c, name in enumerate(fieldnames):
            ref = f"{_col_letter(c)}{r}"
            val = row.get(name, "")
            if isinstance(val, bool):
                parts.append(f'<c r="{ref}"><v>{1 if val else 0}</v></c>')
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                parts.append(f'<c r="{ref}"><v>{val}</v></c>')
            else:
                parts.append(
                    f'<c r="{ref}" t="inlineStr"><is><t>{_xml_escape(str(val))}</t></is></c>'
                )
        body.append(f'<row r="{r}">{"".join(parts)}</row>')
    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{header}{"".join(body)}</sheetData></worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="epochs" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    pkg_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", pkg_rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/worksheets/sheet1.xml", sheet)


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
