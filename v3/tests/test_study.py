"""
Scientific study core: seed determinism, aggregation, extinction alignment,
overrides, output nesting, report/manifest, and extend-study.

Author: Cursor Claude Opus 4.8 High
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.checkpoint import save_checkpoint
from src.config import get_default_config
from src.engine import SimulationEngine
from src.study import (
    Arm,
    ReplicateResult,
    Study,
    StudySpec,
    _build_engine,
    _mean_ci,
    aggregate_arm,
    aggregate_existing,
    build_report,
    cohens_d,
    export_results,
    load_replicate_results,
    replicate_seed,
    study_output_dir,
)


def _tiny_config():
    cfg = get_default_config()
    cfg.world.width = 24
    cfg.world.height = 24
    cfg.population.initial_count = 12
    cfg.perf.max_animals = 300
    cfg.metrics.interval = 20
    cfg.resources.food_rate = 6.0
    cfg.resources.pitfall_rate = 1.0
    cfg.aging.onset = 80
    cfg.aging.max_age = 200
    cfg.reproduction.repro_age_min = 20
    cfg.reproduction.repro_age_max = 40
    cfg.viz.snapshot_every_epoch = False
    return cfg


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    cfg = _tiny_config()
    eng = SimulationEngine(cfg, seed=7)
    eng.initialize()
    for _ in range(45):
        eng.tick()
    folder = save_checkpoint(eng, tmp_path / "saves", "start")
    return folder


def _spec(checkpoint_dir: Path, **kw) -> StudySpec:
    defaults = dict(
        name="unit",
        origin_path=str(checkpoint_dir),
        origin_kind="checkpoint",
        arms=[Arm("Baseline", {})],
        replicates_per_arm=3,
        max_epochs=2,
        base_seed=1000,
        workers=1,          # in-process, deterministic, no spawning in tests
        bootstrap=False,
    )
    defaults.update(kw)
    return StudySpec(**defaults)


# --------------------------------------------------------------- seeds
def test_replicate_seed_is_deterministic_and_unique():
    seeds = {replicate_seed(1000, arm, rep) for arm in range(3) for rep in range(5)}
    assert len(seeds) == 15
    assert replicate_seed(1000, 0, 0) == 1000
    assert replicate_seed(1000, 1, 0) == 11000


def test_build_jobs_span_arms_and_replicates(checkpoint_dir):
    study = Study(_spec(checkpoint_dir, arms=[Arm("A", {}), Arm("B", {"resources.food_rate": 9.0})]))
    jobs = study.build_jobs()
    assert len(jobs) == 6
    assert {j["arm_label"] for j in jobs} == {"A", "B"}
    assert len({j["seed"] for j in jobs}) == 6


# ----------------------------------------------------------- run + reseed
def test_study_runs_from_checkpoint(checkpoint_dir):
    result = Study(_spec(checkpoint_dir)).run()
    assert len(result.replicates) == 3
    assert all(r.error is None for r in result.replicates)
    assert all(len(r.epoch_kpis) >= 1 for r in result.replicates)
    arm = result.arms[0]
    assert arm.n_replicates == 3
    assert "adaptation_score" in arm.final
    assert 0.0 <= arm.survival_prob <= 1.0


def test_same_base_seed_reproduces_results(checkpoint_dir):
    r1 = Study(_spec(checkpoint_dir)).run()
    r2 = Study(_spec(checkpoint_dir)).run()
    a1 = [r.epoch_kpis[-1]["alive_count"] for r in r1.replicates]
    a2 = [r.epoch_kpis[-1]["alive_count"] for r in r2.replicates]
    assert a1 == a2


def test_reseed_makes_replicates_diverge(checkpoint_dir):
    # Different seeds per replicate should not all be identical trajectories.
    result = Study(_spec(checkpoint_dir, replicates_per_arm=4, max_epochs=3)).run()
    finals = [r.epoch_kpis[-1]["avg_energy"] for r in result.replicates]
    assert len(set(round(f, 6) for f in finals)) > 1


# --------------------------------------------------------------- overrides
def test_override_is_applied_to_engine(checkpoint_dir):
    eng = _build_engine(str(checkpoint_dir), "checkpoint",
                        {"resources.food_rate": 42.0}, seed=5)
    assert eng.config.resources.food_rate == 42.0
    # reseed is in effect: world.rng is the engine.rng
    assert eng.world.rng is eng.rng


# --------------------------------------------------------------- stats math
def test_mean_ci_known_values():
    s = _mean_ci(np.array([2.0, 4.0, 6.0]))
    assert s["mean"] == pytest.approx(4.0)
    assert s["std"] == pytest.approx(2.0)          # ddof=1
    assert s["n"] == 3
    assert s["ci_low"] < 4.0 < s["ci_high"]


def test_cohens_d_sign_and_scale():
    a = np.array([2.0, 3.0, 4.0])
    b = np.array([0.0, 1.0, 2.0])
    assert cohens_d(a, b) > 0


def test_extinction_alignment_fills_zero_population():
    spec = _spec(Path("."), replicates_per_arm=2)  # spec only used for quantiles/bootstrap flags
    reps = [
        ReplicateResult(0, "A", 0, 1, epoch_kpis=[
            {"epoch": 0, "tick": 10, "alive_count": 10, "avg_energy": 0.5},
            {"epoch": 1, "tick": 20, "alive_count": 8, "avg_energy": 0.4},
        ]),
        ReplicateResult(0, "A", 1, 2, extinct=True, extinction_tick=15, epoch_kpis=[
            {"epoch": 0, "tick": 10, "alive_count": 6, "avg_energy": 0.3},
        ]),
    ]
    agg = aggregate_arm(0, "A", {}, reps, spec, np.random.default_rng(0))
    # epoch index 1: alive filled 0 for the extinct replicate -> mean (8+0)/2 = 4
    assert agg.metrics["alive_count"]["mean"][1] == pytest.approx(4.0)
    # survival at epoch index 1: only 1 of 2 replicates still running
    assert agg.survival_curve[1] == pytest.approx(0.5)
    assert agg.extinction_rate == pytest.approx(0.5)


# --------------------------------------------------- output tree + export
def test_output_nests_under_checkpoint(checkpoint_dir):
    spec = _spec(checkpoint_dir)
    out = study_output_dir(spec)
    assert out.parent == checkpoint_dir / "studies"


def test_export_writes_report_and_manifest(checkpoint_dir, tmp_path):
    result = Study(_spec(checkpoint_dir, arms=[Arm("A", {}), Arm("B", {"resources.food_rate": 9.0})])).run()
    dest = tmp_path / "study_out"
    paths = export_results(result, dest)
    for key in ("report_json", "manifest", "arm_summary", "replicates_csv", "comparison"):
        assert paths[key].exists()

    report = json.loads((dest / "report.json").read_text(encoding="utf-8"))
    assert len(report["arms"]) == 2
    assert report["comparisons"]                 # A vs B present
    assert "verdict" in report

    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["origin"]["kind"] == "checkpoint"
    assert manifest["origin"]["hash"]
    all_seeds = [s for arm in manifest["arms"] for s in arm["seeds"]]
    assert len(all_seeds) == 6


def test_report_has_verdict_string(checkpoint_dir):
    result = Study(_spec(checkpoint_dir)).run()
    report, text = build_report(result)
    assert isinstance(text, str) and "Verdict:" in text


# --------------------------------------------------------------- extend
def test_extend_via_aggregate_existing(checkpoint_dir, tmp_path):
    spec = _spec(checkpoint_dir, replicates_per_arm=2)
    r1 = Study(spec).run()
    export_results(r1, tmp_path / "s")
    reloaded = load_replicate_results(tmp_path / "s")
    assert len(reloaded) == 2

    # add 2 more replicates (indices 2,3) and re-aggregate
    more = Study(_spec(checkpoint_dir, replicates_per_arm=2)).run(start_index=2)
    combined = list(reloaded) + list(more.replicates)
    spec.replicates_per_arm = 4
    merged = aggregate_existing(spec, combined)
    assert merged.arms[0].n_replicates == 4
