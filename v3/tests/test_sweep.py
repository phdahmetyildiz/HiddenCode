"""Parameter sweep tests."""

import json
from pathlib import Path

import pytest

from src.config import get_default_config
from src.sweep import (
    CombinationResult,
    ParameterSweep,
    SingleRunResult,
    SweepSettings,
    classify_stability,
    generate_combinations,
    run_single_job,
)


def test_cartesian_2x2():
    combos = generate_combinations({
        "population.initial_count": [10, 20],
        "resources.food_rate": [1.0, 2.0],
    })
    assert len(combos) == 4
    assert {"population.initial_count": 10, "resources.food_rate": 1.0} in combos
    assert {"population.initial_count": 20, "resources.food_rate": 2.0} in combos


def test_cartesian_empty():
    assert generate_combinations({}) == [{}]


def test_settings_from_dict_epochs():
    settings = SweepSettings.from_dict({
        "fixed_params": {"world.width": 40},
        "variable_params": {"population.initial_count": [40, 80]},
        "sweep_settings": {
            "runs_per_set": 3,
            "max_epochs": 10,
            "base_seed": 7,
            "stability_band": {
                "min_population_pct": 0.2,
                "max_population_pct": 4.0,
                "check_after_epoch": 3,
            },
            "parallel_workers": 2,
        },
    })
    assert settings.max_epochs == 10
    assert settings.check_after_epoch == 3
    assert settings.runs_per_set == 3
    assert settings.fixed_params["world.width"] == 40


def test_settings_legacy_max_generations_key():
    settings = SweepSettings.from_dict({
        "variable_params": {"a": [1]},
        "sweep_settings": {
            "max_generations": 12,
            "stability_band": {"check_after_generation": 4},
        },
    })
    assert settings.max_epochs == 12
    assert settings.check_after_epoch == 4


def test_settings_from_file(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({
        "variable_params": {"population.initial_count": [10]},
        "sweep_settings": {"runs_per_set": 2, "max_epochs": 3},
    }), encoding="utf-8")
    s = SweepSettings.from_file(p)
    assert s.runs_per_set == 2
    assert s.max_epochs == 3


def test_settings_validate_empty_variable():
    s = SweepSettings(fixed_params={}, variable_params={})
    assert any("variable_params" in e for e in s.validate())


def test_aggregate_math():
    combo = CombinationResult(combination_id=0, params={"x": 1})
    combo.runs = [
        SingleRunResult(0, {"x": 1}, seed=1, run_index=0, final_alive_count=10, extinct=False, stable=True),
        SingleRunResult(0, {"x": 1}, seed=2, run_index=1, final_alive_count=20, extinct=True, stable=False),
        SingleRunResult(0, {"x": 1}, seed=3, run_index=2, final_alive_count=30, extinct=False, stable=True),
    ]
    combo.aggregate()
    assert combo.total_runs == 3
    assert combo.extinction_count == 1
    assert combo.survival_rate == pytest.approx(2 / 3)
    assert combo.stable_count == 2
    assert combo.stability_rate == pytest.approx(2 / 3)
    assert combo.avg_final_alive == pytest.approx(20.0)


def test_stability_band_synthetic():
    # initial 100, band 20–500, check after epoch 1
    stable, bad = classify_stability([100, 80, 90], 100, 0.20, 5.0, 1)
    assert stable and bad is None
    unstable, bad2 = classify_stability([100, 80, 5], 100, 0.20, 5.0, 1)
    assert not unstable
    assert bad2 == 2
    # epoch 0 ignored
    ok, _ = classify_stability([1, 50], 100, 0.20, 5.0, 1)
    assert ok


def _fast_base():
    cfg = get_default_config()
    cfg.world.width = 24
    cfg.world.height = 24
    cfg.population.initial_count = 12
    cfg.perf.max_animals = 200
    cfg.metrics.interval = 25
    cfg.resources.food_rate = 6.0
    cfg.resources.pitfall_rate = 0.0
    cfg.energy.low_energy_death_threshold = 0.02
    cfg.energy.base_metabolism = 0.0005
    cfg.energy.k_weight_speed = 0.002
    cfg.aging.onset = 80
    cfg.aging.max_age = 200
    cfg.reproduction.repro_age_min = 20
    cfg.reproduction.repro_age_max = 40
    cfg.viz.snapshot_every_epoch = False
    return cfg


def _fast_settings(**over) -> SweepSettings:
    data = {
        "fixed_params": {
            "world.width": 24,
            "world.height": 24,
            "metrics.interval": 25,
            "resources.pitfall_rate": 0.0,
            "viz.snapshot_every_epoch": False,
            "aging.onset": 80,
            "aging.max_age": 200,
            "reproduction.repro_age_min": 20,
            "reproduction.repro_age_max": 40,
            "perf.max_animals": 400,
        },
        "variable_params": {
            "population.initial_count": [8, 12],
            "resources.food_rate": [4.0, 8.0],
        },
        "sweep_settings": {
            "runs_per_set": 2,
            "max_epochs": 2,
            "base_seed": 42,
            "parallel_workers": 1,
            "stability_band": {
                "min_population_pct": 0.10,
                "max_population_pct": 20.0,
                "check_after_epoch": 0,
            },
        },
    }
    s = SweepSettings.from_dict(data)
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_mini_sweep_job_count():
    sweep = ParameterSweep(_fast_settings(), base_config=_fast_base())
    assert sweep.total_combinations == 4
    assert sweep.total_runs == 8


def test_mini_sweep_sequential(tmp_path: Path):
    sweep = ParameterSweep(_fast_settings(parallel_workers=1), base_config=_fast_base())
    result = sweep.run(parallel=False)
    assert result.total_runs == 8
    assert result.total_combinations == 4
    for combo in result.combinations:
        assert combo.total_runs == 2
        assert len(combo.runs) == 2
        assert combo.runs[0].seed != combo.runs[1].seed
    paths = sweep.export_results(result, tmp_path)
    assert paths["summary"].exists()
    assert paths["detailed"].exists()
    report = json.loads(paths["stability_report"].read_text(encoding="utf-8"))
    assert report["total_runs"] == 8
    assert "combinations" in report


def test_single_job_applies_params():
    cfg = _fast_base()
    job = {
        "base_config_dict": cfg.to_dict(),
        "combination_id": 0,
        "combination_params": {"population.initial_count": 9},
        "fixed_params": {"world.width": 24, "metrics.interval": 20, "resources.pitfall_rate": 0.0},
        "seed": 99,
        "run_index": 0,
        "max_epochs": 1,
        "stability_band_min_pct": 0.0,
        "stability_band_max_pct": 50.0,
        "check_after_epoch": 0,
        "early_termination_on_extinction": True,
    }
    r = run_single_job(job)
    assert r.initial_count == 9
    assert r.seed == 99
    assert r.total_epochs >= 1 or r.extinct


def test_parallel_matches_sequential_summary():
    settings = _fast_settings(parallel_workers=2)
    base = _fast_base()
    seq = ParameterSweep(settings, base_config=base.copy()).run(parallel=False)
    par = ParameterSweep(settings, base_config=base.copy()).run(parallel=True)
    assert seq.total_runs == par.total_runs == 8
    for a, b in zip(seq.combinations, par.combinations):
        assert a.survival_rate == pytest.approx(b.survival_rate)
        assert a.avg_final_alive == pytest.approx(b.avg_final_alive)
        assert [r.seed for r in a.runs] == [r.seed for r in b.runs]
        assert [r.final_alive_count for r in a.runs] == [r.final_alive_count for r in b.runs]
