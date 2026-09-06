"""
Save and restore a full simulation (world arrays, DNA, RNG, engine stats).

Author: Cursor Grok 4.6 High Fast
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import PitfallType, SimConfig, load_config, save_config
from src.engine import SimulationEngine, TickStats
from src.metrics import EpochMetrics


def default_saves_dir() -> Path:
    return Path("saves")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _slug(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip(), flags=re.UNICODE)
    s = s.strip("._") or "checkpoint"
    return s[:80]


def _tick_stats_from_dict(data: dict | None) -> TickStats:
    if not data:
        return TickStats()
    allowed = {f.name for f in fields(TickStats)}
    return TickStats(**{k: v for k, v in data.items() if k in allowed})


def _epoch_from_dict(data: dict) -> EpochMetrics:
    allowed = {f.name for f in fields(EpochMetrics)}
    return EpochMetrics(**{k: v for k, v in data.items() if k in allowed})


def _types_to_json(types: list) -> list[dict]:
    out = []
    for pt in types:
        if isinstance(pt, PitfallType):
            out.append({"name": pt.name, "sequence": pt.sequence})
        elif isinstance(pt, dict):
            out.append({"name": pt["name"], "sequence": pt["sequence"]})
    return out


def _types_from_json(raw: list) -> list[PitfallType]:
    return [PitfallType(name=d["name"], sequence=d["sequence"]) for d in raw]


def save_checkpoint(
    engine: SimulationEngine,
    saves_dir: str | Path,
    name: str,
    notes: str = "",
    parent: str | None = None,
) -> Path:
    """Write a resume-able checkpoint directory. Returns the folder path."""
    w = engine.world
    n = w.n
    root = Path(saves_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = root / f"{stamp}_{_slug(name)}"
    folder.mkdir(parents=True, exist_ok=False)

    save_config(engine.config, folder / "config.json")
    np.savez_compressed(
        folder / "arrays.npz",
        id=w.id[:n].copy(),
        x=w.x[:n].copy(),
        y=w.y[:n].copy(),
        energy=w.energy[:n].copy(),
        weight=w.weight[:n].copy(),
        speed=w.speed[:n].copy(),
        birth_tick=w.birth_tick[:n].copy(),
        repro_age=w.repro_age[:n].copy(),
        has_reproduced=w.has_reproduced[:n].copy(),
        cohort_of=w.cohort_of[:n].copy(),
        defense=w.defense[:n].copy(),
        dna=w.dna[:n].copy(),
        food_life=w.food_life.copy(),
        pitfall_life=w.pitfall_life.copy(),
        pitfall_seq=w.pitfall_seq.copy(),
        pitfall_type_id=w.pitfall_type_id.copy(),
    )
    engine_blob = {
        "n": n,
        "next_id": w.next_id,
        "tick": w.tick,
        "cohort": w.cohort,
        "stress_mode": bool(w.stress_mode),
        "births_skipped": w.births_skipped,
        "epochs_completed": engine.epochs_completed,
        "lifetime": asdict(engine.lifetime),
        "epoch_counters": asdict(engine.epoch_counters),
        "tick_stats": asdict(engine.tick_stats),
        "adaptation_series": [None if v != v else float(v) for v in engine.adaptation_series],
        "epoch_history": [m.to_dict() for m in engine.epoch_history],
        "stress": {
            "active": engine.stress.state.active,
            "started_tick": engine.stress.state.started_tick,
        },
        "rng": _jsonable(engine.rng.bit_generator.state),
        "active_pitfall_types": _types_to_json(w.active_pitfall_types),
        "pitfall_type_registry": _types_to_json(w.pitfall_type_registry),
    }
    with open(folder / "engine.json", "w", encoding="utf-8") as f:
        json.dump(engine_blob, f)
    meta = {
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "tick": w.tick,
        "alive": n,
        "epochs": engine.epochs_completed,
        "seed": engine.config.world.seed,
        "parent": parent,
        "notes": notes,
        "folder": folder.name,
    }
    with open(folder / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return folder


def list_checkpoints(saves_dir: str | Path) -> list[dict]:
    root = Path(saves_dir)
    if not root.exists():
        return []
    rows = []
    for child in sorted(root.iterdir(), reverse=True):
        meta_path = child / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        meta["path"] = str(child)
        rows.append(meta)
    return rows


def delete_checkpoint(path: str | Path) -> None:
    folder = Path(path)
    if not (folder / "meta.json").is_file():
        raise FileNotFoundError(f"Not a checkpoint: {folder}")
    shutil.rmtree(folder)


def load_checkpoint(
    path: str | Path,
    config_override: Optional[SimConfig] = None,
) -> SimulationEngine:
    """Restore an engine. Optional config_override forks rules (not grid/DNA layout)."""
    folder = Path(path)
    saved_cfg = load_config(folder / "config.json")
    with open(folder / "engine.json", encoding="utf-8") as f:
        blob = json.load(f)
    arrays = np.load(folder / "arrays.npz")
    n = int(blob["n"])
    if n != int(arrays["x"].shape[0]):
        raise ValueError("Checkpoint array length does not match engine.json n")

    config = config_override.copy() if config_override is not None else saved_cfg.copy()
    if config_override is not None:
        _assert_fork_compatible(saved_cfg, config)
        config.world.width = saved_cfg.world.width
        config.world.height = saved_cfg.world.height
        config.genetics.dna_length = saved_cfg.genetics.dna_length
        config.world.seed = saved_cfg.world.seed
    if config.perf.max_animals < n:
        config.perf.max_animals = int(n * 2) if n >= 2 else 16

    engine = SimulationEngine(config)
    w = engine.world
    w.n = n
    w.next_id = int(blob["next_id"])
    w.tick = int(blob["tick"])
    w.cohort = int(blob["cohort"])
    w.stress_mode = bool(blob["stress_mode"])
    w.births_skipped = int(blob.get("births_skipped", 0))
    w.id[:n] = arrays["id"]
    w.x[:n] = arrays["x"]
    w.y[:n] = arrays["y"]
    w.energy[:n] = arrays["energy"]
    w.weight[:n] = arrays["weight"]
    w.speed[:n] = arrays["speed"]
    w.birth_tick[:n] = arrays["birth_tick"]
    w.repro_age[:n] = arrays["repro_age"]
    w.has_reproduced[:n] = arrays["has_reproduced"]
    w.cohort_of[:n] = arrays["cohort_of"]
    w.defense[:n] = arrays["defense"]
    w.dna[:n] = arrays["dna"]
    if arrays["food_life"].shape != (w.width, w.height):
        raise ValueError("Checkpoint grid size does not match config world size")
    w.food_life[:] = arrays["food_life"]
    w.pitfall_life[:] = arrays["pitfall_life"]
    w.pitfall_seq[:] = arrays["pitfall_seq"]
    w.pitfall_type_id[:] = arrays["pitfall_type_id"]
    w.active_pitfall_types = _types_from_json(blob.get("active_pitfall_types") or [])
    w.pitfall_type_registry = _types_from_json(blob.get("pitfall_type_registry") or [])
    if not w.active_pitfall_types:
        w.active_pitfall_types = config.resources.get_pitfall_types()
    if not w.pitfall_type_registry:
        w.pitfall_type_registry = list(w.active_pitfall_types)

    engine.epochs_completed = int(blob.get("epochs_completed", 0))
    engine.lifetime = _tick_stats_from_dict(blob.get("lifetime"))
    engine.epoch_counters = _tick_stats_from_dict(blob.get("epoch_counters"))
    engine.tick_stats = _tick_stats_from_dict(blob.get("tick_stats"))
    series = blob.get("adaptation_series") or []
    engine.adaptation_series = [float("nan") if v is None else float(v) for v in series]
    engine.epoch_history = [_epoch_from_dict(m) for m in blob.get("epoch_history") or []]
    st = blob.get("stress") or {}
    engine.stress.state.active = bool(st.get("active", False))
    engine.stress.state.started_tick = st.get("started_tick")
    engine.stress.config = config
    rng_state = blob.get("rng")
    if rng_state:
        engine.rng.bit_generator.state = rng_state
    w.rng = engine.rng
    return engine


def _assert_fork_compatible(saved: SimConfig, new: SimConfig) -> None:
    if saved.world.width != new.world.width or saved.world.height != new.world.height:
        raise ValueError(
            f"Cannot fork onto a different grid "
            f"({saved.world.width}×{saved.world.height} vs {new.world.width}×{new.world.height})"
        )
    if saved.genetics.dna_length != new.genetics.dna_length:
        raise ValueError("Cannot fork onto a different DNA length")
