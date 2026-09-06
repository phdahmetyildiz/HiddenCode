"""
Checkpoint save/load and optional config fork.

Author: Cursor Grok 4.6 High Fast
"""

from pathlib import Path

import numpy as np
import pytest

from src.checkpoint import delete_checkpoint, list_checkpoints, load_checkpoint, save_checkpoint
from src.config import get_default_config
from src.engine import SimulationEngine
from src.logging_io import export_epochs
from src.metrics import EpochMetrics
from src.render import world_rgb


def _tiny():
    cfg = get_default_config()
    cfg.world.width = 24
    cfg.world.height = 24
    cfg.population.initial_count = 10
    cfg.perf.max_animals = 200
    cfg.metrics.interval = 20
    cfg.resources.food_rate = 6.0
    cfg.resources.pitfall_rate = 1.0
    cfg.aging.onset = 80
    cfg.aging.max_age = 200
    cfg.reproduction.repro_age_min = 20
    cfg.reproduction.repro_age_max = 40
    cfg.viz.snapshot_every_epoch = False
    return cfg


def test_checkpoint_roundtrip_rng_and_dna(tmp_path: Path):
    cfg = _tiny()
    a = SimulationEngine(cfg, seed=7)
    a.initialize()
    for _ in range(35):
        a.tick()
    folder = save_checkpoint(a, tmp_path, "mid")
    assert (folder / "arrays.npz").exists()
    assert (folder / "meta.json").exists()

    b = load_checkpoint(folder)
    n = a.world.n
    assert b.world.n == n
    assert b.world.tick == a.world.tick
    assert np.array_equal(a.world.dna[:n], b.world.dna[:n])
    assert np.array_equal(a.world.x[:n], b.world.x[:n])
    assert np.allclose(a.world.energy[:n], b.world.energy[:n])
    assert b.lifetime.births == a.lifetime.births
    assert b.epochs_completed == a.epochs_completed

    a.tick()
    b.tick()
    n2 = a.world.n
    assert b.world.n == n2
    assert np.array_equal(a.world.x[:n2], b.world.x[:n2])
    assert np.allclose(a.world.energy[:n2], b.world.energy[:n2])


def test_list_and_delete(tmp_path: Path):
    cfg = _tiny()
    eng = SimulationEngine(cfg, seed=1)
    eng.initialize()
    save_checkpoint(eng, tmp_path, "one")
    rows = list_checkpoints(tmp_path)
    assert len(rows) == 1
    assert rows[0]["name"] == "one"
    delete_checkpoint(rows[0]["path"])
    assert list_checkpoints(tmp_path) == []


def test_fork_keeps_animals_changes_rules(tmp_path: Path):
    cfg = _tiny()
    eng = SimulationEngine(cfg, seed=3)
    eng.initialize()
    for _ in range(10):
        eng.tick()
    folder = save_checkpoint(eng, tmp_path, "base")
    other = cfg.copy()
    other.energy.max_pitfall_loss_pct = 1.0
    other.resources.pitfall_rate = 4.0
    forked = load_checkpoint(folder, config_override=other)
    assert forked.config.energy.max_pitfall_loss_pct == pytest.approx(1.0)
    assert forked.config.resources.pitfall_rate == pytest.approx(4.0)
    assert forked.world.n == eng.world.n
    assert np.array_equal(forked.world.dna[: eng.world.n], eng.world.dna[: eng.world.n])


def test_fork_rejects_different_grid(tmp_path: Path):
    cfg = _tiny()
    eng = SimulationEngine(cfg, seed=1)
    eng.initialize()
    folder = save_checkpoint(eng, tmp_path, "g")
    other = cfg.copy()
    other.world.width = 40
    with pytest.raises(ValueError, match="different grid"):
        load_checkpoint(folder, config_override=other)


def test_world_rgb_shape():
    cfg = _tiny()
    eng = SimulationEngine(cfg, seed=0)
    eng.initialize()
    img = world_rgb(eng.world)
    assert img.shape == (24, 24, 3)
    assert img.dtype == np.uint8


def test_export_epochs_csv_and_xlsx(tmp_path: Path):
    epochs = [
        EpochMetrics(epoch=0, tick=1000, alive_count=80, births_count=3, adaptation_score=0.4),
        EpochMetrics(epoch=1, tick=2000, alive_count=90, births_count=5, adaptation_score=0.5),
    ]
    csv_path = export_epochs(tmp_path / "epochs.csv", epochs)
    text = csv_path.read_text(encoding="utf-8-sig")
    assert "alive_count" in text
    assert "0.4" in text
    xlsx_path = export_epochs(tmp_path / "epochs.xlsx", epochs)
    assert xlsx_path.suffix == ".xlsx"
    assert xlsx_path.stat().st_size > 100
