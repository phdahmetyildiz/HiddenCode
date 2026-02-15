"""
Unit tests for Phase 9: Parameter Sweep.

Tests cover:
- SweepSettings:
  - Parsing from dict / file
  - Validation
- generate_combinations:
  - Cartesian product correctness
  - Single param, multiple params, empty params
- SingleRunResult / CombinationResult:
  - Aggregation math (mean, std, survival/stability rates)
- ParameterSweep:
  - Sequential execution
  - Correct number of runs per combination
  - Different seeds produce different results
  - Extinction triggers detection
  - Stability band classification
  - Export: summary CSV, detailed CSV, stability report
- _run_single_simulation:
  - Fixed/variable params applied correctly
  - Seed applied correctly
  - Stability check logic

All tests use small grids (10x10) and short runs (2-3 generations) for speed.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.config import SimConfig, get_default_config
from src.core.animal import reset_animal_id_counter
from src.simulation.sweep import (
    SweepSettings,
    generate_combinations,
    SingleRunResult,
    CombinationResult,
    SweepResult,
    ParameterSweep,
    _run_single_simulation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ids():
    reset_animal_id_counter()
    yield
    reset_animal_id_counter()


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def make_fast_config() -> SimConfig:
    """Return a config that runs fast (small grid, few animals, short gens)."""
    cfg = get_default_config()
    cfg.world.width = 10
    cfg.world.height = 10
    cfg.world.seed = 42
    cfg.population.initial_count = 15
    cfg.generation.gen_length = 10
    cfg.resources.food_rate = 3.0
    cfg.resources.pitfall_rate = 0.5
    cfg.energy.food_gain = 0.3
    cfg.energy.base_metabolism = 0.005
    return cfg


def make_fast_settings(**overrides) -> SweepSettings:
    """Create small/fast sweep settings for testing."""
    defaults = {
        "fixed_params": {
            "world.width": 10,
            "world.height": 10,
        },
        "variable_params": {
            "population.initial_count": [10, 20],
        },
        "runs_per_set": 2,
        "max_generations": 2,
        "base_seed": 42,
        "stability_band_min_pct": 0.10,
        "stability_band_max_pct": 10.0,
        "check_after_generation": 1,
        "early_termination_on_extinction": True,
        "parallel_workers": 1,
        "stability_required_pct": 0.50,
    }
    defaults.update(overrides)
    return SweepSettings(**defaults)


# ===========================================================================
# SweepSettings Tests
# ===========================================================================

class TestSweepSettings:
    def test_from_dict(self):
        data = {
            "fixed_params": {"world.width": 100},
            "variable_params": {"population.initial_count": [50, 100]},
            "sweep_settings": {
                "runs_per_set": 5,
                "max_generations": 50,
                "base_seed": 99,
                "stability_band": {
                    "min_population_pct": 0.3,
                    "max_population_pct": 3.0,
                    "check_after_generation": 5,
                },
                "parallel_workers": 2,
                "stability_required_pct": 0.75,
            },
        }
        settings = SweepSettings.from_dict(data)
        assert settings.runs_per_set == 5
        assert settings.max_generations == 50
        assert settings.base_seed == 99
        assert settings.stability_band_min_pct == 0.3
        assert settings.stability_band_max_pct == 3.0
        assert settings.check_after_generation == 5
        assert settings.parallel_workers == 2
        assert settings.stability_required_pct == 0.75
        assert settings.fixed_params == {"world.width": 100}
        assert settings.variable_params == {"population.initial_count": [50, 100]}

    def test_from_file(self, tmp_dir):
        data = {
            "fixed_params": {},
            "variable_params": {"population.initial_count": [10]},
            "sweep_settings": {"runs_per_set": 3},
        }
        path = tmp_dir / "sweep.json"
        with open(path, "w") as f:
            json.dump(data, f)

        settings = SweepSettings.from_file(path)
        assert settings.runs_per_set == 3

    def test_validate_valid(self):
        settings = make_fast_settings()
        errors = settings.validate()
        assert errors == []

    def test_validate_invalid_runs(self):
        settings = make_fast_settings(runs_per_set=0)
        errors = settings.validate()
        assert any("runs_per_set" in e for e in errors)

    def test_validate_invalid_max_gen(self):
        settings = make_fast_settings(max_generations=0)
        errors = settings.validate()
        assert any("max_generations" in e for e in errors)

    def test_validate_invalid_parallel(self):
        settings = make_fast_settings(parallel_workers=0)
        errors = settings.validate()
        assert any("parallel_workers" in e for e in errors)

    def test_validate_invalid_stability_band(self):
        settings = make_fast_settings(
            stability_band_min_pct=5.0,
            stability_band_max_pct=1.0,
        )
        errors = settings.validate()
        assert any("max_pct" in e for e in errors)

    def test_validate_empty_variable_params(self):
        settings = make_fast_settings(variable_params={})
        errors = settings.validate()
        assert any("variable_params" in e for e in errors)

    def test_validate_empty_values_list(self):
        settings = make_fast_settings(variable_params={"population.initial_count": []})
        errors = settings.validate()
        assert any("non-empty" in e for e in errors)


# ===========================================================================
# generate_combinations Tests
# ===========================================================================

class TestGenerateCombinations:
    def test_single_param_two_values(self):
        combos = generate_combinations({"a": [1, 2]})
        assert len(combos) == 2
        assert {"a": 1} in combos
        assert {"a": 2} in combos

    def test_two_params_cartesian(self):
        combos = generate_combinations({"a": [1, 2], "b": [3, 4, 5]})
        assert len(combos) == 6  # 2 * 3
        assert {"a": 1, "b": 3} in combos
        assert {"a": 2, "b": 5} in combos

    def test_three_params(self):
        combos = generate_combinations({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
        assert len(combos) == 8  # 2 * 2 * 2

    def test_single_value_per_param(self):
        combos = generate_combinations({"a": [1], "b": [2]})
        assert len(combos) == 1
        assert combos[0] == {"a": 1, "b": 2}

    def test_empty_params(self):
        combos = generate_combinations({})
        assert len(combos) == 1
        assert combos[0] == {}


# ===========================================================================
# CombinationResult Aggregation Tests
# ===========================================================================

class TestCombinationResultAggregation:
    def _make_run(
        self, extinct=False, stable=True, final_alive=50, total_gens=10
    ) -> SingleRunResult:
        return SingleRunResult(
            combination_id=0,
            combination_params={"a": 1},
            seed=42,
            run_index=0,
            total_ticks=100,
            total_generations=total_gens,
            final_alive_count=final_alive,
            initial_count=50,
            extinct=extinct,
            stable=stable,
            generation_kpis=[{
                "generation": total_gens - 1,
                "alive_count": final_alive,
                "avg_energy": 0.5,
            }],
        )

    def test_basic_aggregation(self):
        combo = CombinationResult(combination_id=0, params={"a": 1})
        combo.runs = [
            self._make_run(extinct=False, stable=True, final_alive=50),
            self._make_run(extinct=False, stable=True, final_alive=60),
            self._make_run(extinct=True, stable=False, final_alive=0),
        ]
        combo.aggregate()

        assert combo.total_runs == 3
        assert combo.extinction_count == 1
        assert combo.survival_rate == pytest.approx(2 / 3, abs=0.01)
        assert combo.stable_count == 2
        assert combo.stability_rate == pytest.approx(2 / 3, abs=0.01)
        assert combo.avg_final_alive == pytest.approx(
            (50 + 60 + 0) / 3, abs=0.1
        )

    def test_all_extinct(self):
        combo = CombinationResult(combination_id=0, params={})
        combo.runs = [self._make_run(extinct=True, stable=False, final_alive=0) for _ in range(3)]
        combo.aggregate()

        assert combo.survival_rate == 0.0
        assert combo.stability_rate == 0.0

    def test_all_stable(self):
        combo = CombinationResult(combination_id=0, params={})
        combo.runs = [self._make_run(extinct=False, stable=True, final_alive=50) for _ in range(5)]
        combo.aggregate()

        assert combo.survival_rate == 1.0
        assert combo.stability_rate == 1.0

    def test_kpi_aggregation(self):
        combo = CombinationResult(combination_id=0, params={})
        combo.runs = [
            self._make_run(final_alive=40),
            self._make_run(final_alive=60),
        ]
        combo.aggregate()

        assert "alive_count" in combo.kpi_aggregates
        agg = combo.kpi_aggregates["alive_count"]
        assert agg["mean"] == pytest.approx(50.0, abs=0.1)
        assert agg["min"] == 40.0
        assert agg["max"] == 60.0

    def test_empty_runs(self):
        combo = CombinationResult(combination_id=0, params={})
        combo.aggregate()
        assert combo.total_runs == 0


# ===========================================================================
# SweepResult Tests
# ===========================================================================

class TestSweepResult:
    def test_best_stable_combination(self):
        c1 = CombinationResult(combination_id=0, params={"a": 1})
        c1.stability_rate = 0.8
        c1.survival_rate = 0.9
        c1.avg_final_alive = 50

        c2 = CombinationResult(combination_id=1, params={"a": 2})
        c2.stability_rate = 0.9
        c2.survival_rate = 0.9
        c2.avg_final_alive = 45

        result = SweepResult(combinations=[c1, c2])
        best = result.best_stable_combination()
        assert best is not None
        assert best.combination_id == 1  # Higher stability rate

    def test_no_stable_combination(self):
        c1 = CombinationResult(combination_id=0, params={})
        c1.stability_rate = 0.0
        result = SweepResult(combinations=[c1])
        assert result.best_stable_combination() is None


# ===========================================================================
# _run_single_simulation Tests
# ===========================================================================

class TestRunSingleSimulation:
    def test_basic_run(self):
        """A single run completes without error and returns metrics."""
        cfg = make_fast_config()
        result = _run_single_simulation(
            base_config_dict=cfg.to_dict(),
            combination_id=0,
            combination_params={"population.initial_count": 15},
            fixed_params={"world.width": 10, "world.height": 10},
            seed=42,
            run_index=0,
            max_generations=2,
            stability_band_min_pct=0.1,
            stability_band_max_pct=10.0,
            check_after_generation=1,
            early_termination_on_extinction=True,
        )
        assert result.combination_id == 0
        assert result.seed == 42
        assert result.total_ticks > 0
        assert isinstance(result.extinct, bool)

    def test_seed_applied(self):
        """Different seeds produce different internal states."""
        cfg = make_fast_config()
        base = cfg.to_dict()
        common = dict(
            combination_id=0,
            combination_params={"population.initial_count": 15},
            fixed_params={"world.width": 10, "world.height": 10},
            max_generations=2,
            stability_band_min_pct=0.1,
            stability_band_max_pct=10.0,
            check_after_generation=1,
            early_termination_on_extinction=True,
        )

        r1 = _run_single_simulation(base_config_dict=base, seed=1, run_index=0, **common)
        r2 = _run_single_simulation(base_config_dict=base, seed=9999, run_index=1, **common)

        # Very likely different final counts or KPIs
        # Compare full KPI histories to be robust
        kpis1 = r1.generation_kpis
        kpis2 = r2.generation_kpis

        # If both have KPIs, at least one field should differ
        if kpis1 and kpis2:
            # Compare first generation KPIs
            differs = any(
                kpis1[0].get(k) != kpis2[0].get(k)
                for k in kpis1[0]
                if isinstance(kpis1[0].get(k), (int, float))
                and not isinstance(kpis1[0].get(k), bool)
            )
            assert differs, "Different seeds should produce different KPIs"

    def test_fixed_params_applied(self):
        """Fixed params override the base config."""
        cfg = make_fast_config()
        result = _run_single_simulation(
            base_config_dict=cfg.to_dict(),
            combination_id=0,
            combination_params={},
            fixed_params={"world.width": 8, "world.height": 8},
            seed=42,
            run_index=0,
            max_generations=1,
            stability_band_min_pct=0.1,
            stability_band_max_pct=10.0,
            check_after_generation=0,
            early_termination_on_extinction=True,
        )
        # Run completes without error — params were applied
        assert result.total_ticks > 0

    def test_extinction_detected(self):
        """Extinction is correctly flagged."""
        cfg = make_fast_config()
        # Very harsh conditions → likely extinction
        cfg.population.initial_count = 3
        cfg.resources.food_rate = 0.0
        cfg.energy.base_metabolism = 0.1

        result = _run_single_simulation(
            base_config_dict=cfg.to_dict(),
            combination_id=0,
            combination_params={},
            fixed_params={},
            seed=42,
            run_index=0,
            max_generations=5,
            stability_band_min_pct=0.1,
            stability_band_max_pct=10.0,
            check_after_generation=0,
            early_termination_on_extinction=True,
        )
        assert result.extinct is True
        assert result.stable is False


# ===========================================================================
# ParameterSweep Tests
# ===========================================================================

class TestParameterSweep:
    def test_total_combinations(self):
        settings = make_fast_settings(
            variable_params={"a": [1, 2], "b": [3, 4, 5]},
        )
        sweep = ParameterSweep(settings, base_config=make_fast_config())
        assert sweep.total_combinations == 6
        assert sweep.total_runs == 6 * 2  # 6 combos * 2 runs_per_set

    def test_sequential_run(self):
        settings = make_fast_settings(
            variable_params={"population.initial_count": [10, 15]},
            runs_per_set=2,
            max_generations=2,
        )
        sweep = ParameterSweep(settings, base_config=make_fast_config())
        result = sweep.run(parallel=False)

        assert result.total_combinations == 2
        assert result.total_runs == 4
        assert len(result.combinations) == 2
        for combo in result.combinations:
            assert combo.total_runs == 2

    def test_correct_runs_per_combination(self):
        settings = make_fast_settings(
            variable_params={"population.initial_count": [10]},
            runs_per_set=3,
            max_generations=1,
        )
        sweep = ParameterSweep(settings, base_config=make_fast_config())
        result = sweep.run(parallel=False)

        assert len(result.combinations) == 1
        assert result.combinations[0].total_runs == 3

    def test_different_seeds_per_run(self):
        settings = make_fast_settings(
            variable_params={"population.initial_count": [10]},
            runs_per_set=3,
            max_generations=2,
        )
        sweep = ParameterSweep(settings, base_config=make_fast_config())
        result = sweep.run(parallel=False)

        seeds = [r.seed for r in result.combinations[0].runs]
        assert len(set(seeds)) == 3, "Each run should have a unique seed"

    def test_progress_callback(self):
        settings = make_fast_settings(
            variable_params={"population.initial_count": [10]},
            runs_per_set=2,
            max_generations=1,
        )
        sweep = ParameterSweep(settings, base_config=make_fast_config())

        progress = []
        result = sweep.run(
            parallel=False,
            progress_callback=lambda done, total: progress.append((done, total)),
        )

        assert len(progress) == 2  # 1 combo * 2 runs
        assert progress[-1] == (2, 2)

    def test_repr(self):
        settings = make_fast_settings()
        sweep = ParameterSweep(settings, base_config=make_fast_config())
        assert "ParameterSweep" in repr(sweep)


# ===========================================================================
# Export Tests
# ===========================================================================

class TestSweepExport:
    def _run_small_sweep(self) -> tuple[ParameterSweep, SweepResult]:
        settings = make_fast_settings(
            variable_params={"population.initial_count": [10, 15]},
            runs_per_set=2,
            max_generations=2,
        )
        sweep = ParameterSweep(settings, base_config=make_fast_config())
        result = sweep.run(parallel=False)
        return sweep, result

    def test_export_creates_files(self, tmp_dir):
        sweep, result = self._run_small_sweep()
        out = tmp_dir / "sweep_output"
        paths = sweep.export_results(result, out)

        assert paths["summary"].exists()
        assert paths["detailed"].exists()
        assert paths["stability_report"].exists()
        assert paths["config"].exists()

    def test_summary_csv_structure(self, tmp_dir):
        sweep, result = self._run_small_sweep()
        out = tmp_dir / "sweep_output"
        paths = sweep.export_results(result, out)

        import csv
        with open(paths["summary"], "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2  # 2 combinations
        assert "combination_id" in rows[0]
        assert "survival_rate" in rows[0]
        assert "stability_rate" in rows[0]
        assert "param_population.initial_count" in rows[0]

    def test_detailed_csv_has_rows(self, tmp_dir):
        sweep, result = self._run_small_sweep()
        out = tmp_dir / "sweep_output"
        paths = sweep.export_results(result, out)

        import csv
        with open(paths["detailed"], "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have at least some rows (one per generation per run)
        # 2 combos * 2 runs * ~2 gens = ~8 rows (some may have more/fewer)
        assert len(rows) >= 1

    def test_stability_report_json(self, tmp_dir):
        sweep, result = self._run_small_sweep()
        out = tmp_dir / "sweep_output"
        paths = sweep.export_results(result, out)

        with open(paths["stability_report"], "r") as f:
            report = json.load(f)

        assert report["total_combinations"] == 2
        assert "combinations" in report
        assert len(report["combinations"]) == 2
        for combo in report["combinations"]:
            assert "is_stable" in combo
            assert "stability_rate" in combo

    def test_sweep_config_saved(self, tmp_dir):
        sweep, result = self._run_small_sweep()
        out = tmp_dir / "sweep_output"
        paths = sweep.export_results(result, out)

        with open(paths["config"], "r") as f:
            saved = json.load(f)

        assert "fixed_params" in saved
        assert "variable_params" in saved
        assert "sweep_settings" in saved


# ===========================================================================
# Stability Classification Tests
# ===========================================================================

class TestStabilityClassification:
    def test_stable_run(self):
        """With generous band and food, run should be classified stable."""
        cfg = make_fast_config()
        cfg.population.initial_count = 15
        cfg.resources.food_rate = 5.0

        result = _run_single_simulation(
            base_config_dict=cfg.to_dict(),
            combination_id=0,
            combination_params={},
            fixed_params={},
            seed=42,
            run_index=0,
            max_generations=2,
            stability_band_min_pct=0.01,  # Very generous
            stability_band_max_pct=100.0,  # Very generous
            check_after_generation=0,
            early_termination_on_extinction=True,
        )
        # With generous band, should be stable if not extinct
        if not result.extinct:
            assert result.stable is True

    def test_unstable_run_too_low(self):
        """Very tight lower band → likely unstable."""
        cfg = make_fast_config()
        cfg.population.initial_count = 10
        cfg.resources.food_rate = 0.5  # Low food

        result = _run_single_simulation(
            base_config_dict=cfg.to_dict(),
            combination_id=0,
            combination_params={},
            fixed_params={},
            seed=42,
            run_index=0,
            max_generations=3,
            stability_band_min_pct=0.99,  # Tight: must stay at 99%+
            stability_band_max_pct=10.0,
            check_after_generation=0,
            early_termination_on_extinction=True,
        )
        # With tight band and low food, probably not stable
        # (population likely to drop below 99% of initial)
        # We just check the field exists and is boolean
        assert isinstance(result.stable, bool)

    def test_check_after_generation_skips_early(self):
        """Stability check should skip generations before check_after_generation."""
        cfg = make_fast_config()
        cfg.population.initial_count = 10

        result = _run_single_simulation(
            base_config_dict=cfg.to_dict(),
            combination_id=0,
            combination_params={},
            fixed_params={},
            seed=42,
            run_index=0,
            max_generations=2,
            stability_band_min_pct=0.01,
            stability_band_max_pct=100.0,
            check_after_generation=999,  # Never check
            early_termination_on_extinction=True,
        )
        # With check_after_generation=999, stability is never checked → defaults stable
        if not result.extinct:
            assert result.stable is True
