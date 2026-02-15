"""
Parameter Sweep for the Evolution Simulator.

Runs the simulation across a cartesian product of parameter combinations,
each repeated with multiple seeds. Evaluates stability of each combination
and aggregates KPIs for comparison.

Supports parallel execution via concurrent.futures.ProcessPoolExecutor.
"""

from __future__ import annotations

import csv
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.core.config import (
    SimConfig,
    apply_param_override,
    get_default_config,
    save_config,
)
from src.simulation.engine import SimulationEngine, RunResult
from src.simulation.generation import GenerationStats
from src.simulation.metrics import MetricsCollector


# ---------------------------------------------------------------------------
# Data classes for sweep results
# ---------------------------------------------------------------------------

@dataclass
class SingleRunResult:
    """Result of a single simulation run within a sweep."""
    combination_id: int
    combination_params: dict[str, Any]
    seed: int
    run_index: int
    total_ticks: int = 0
    total_generations: int = 0
    final_alive_count: int = 0
    initial_count: int = 0
    extinct: bool = False
    extinction_tick: Optional[int] = None
    stable: bool = False
    instability_generation: Optional[int] = None
    generation_kpis: list[dict] = field(default_factory=list)


@dataclass
class CombinationResult:
    """Aggregated results for a single parameter combination."""
    combination_id: int
    params: dict[str, Any]
    runs: list[SingleRunResult] = field(default_factory=list)

    # Aggregated stats (computed by aggregate())
    total_runs: int = 0
    extinction_count: int = 0
    survival_rate: float = 0.0
    stable_count: int = 0
    stability_rate: float = 0.0
    avg_final_alive: float = 0.0
    std_final_alive: float = 0.0
    avg_generations: float = 0.0
    kpi_aggregates: dict[str, dict[str, float]] = field(default_factory=dict)

    def aggregate(self) -> None:
        """Compute aggregated statistics from individual runs."""
        self.total_runs = len(self.runs)
        if self.total_runs == 0:
            return

        self.extinction_count = sum(1 for r in self.runs if r.extinct)
        self.survival_rate = 1.0 - (self.extinction_count / self.total_runs)
        self.stable_count = sum(1 for r in self.runs if r.stable)
        self.stability_rate = self.stable_count / self.total_runs

        alive_counts = [r.final_alive_count for r in self.runs]
        self.avg_final_alive = float(np.mean(alive_counts))
        self.std_final_alive = float(np.std(alive_counts))
        self.avg_generations = float(np.mean([r.total_generations for r in self.runs]))

        # Aggregate last-generation KPIs across runs
        self._aggregate_kpis()

    def _aggregate_kpis(self) -> None:
        """Compute mean/std/min/max of each KPI across runs (last generation)."""
        self.kpi_aggregates = {}

        # Collect last-gen KPIs from each run
        last_kpis: list[dict] = []
        for run in self.runs:
            if run.generation_kpis:
                last_kpis.append(run.generation_kpis[-1])

        if not last_kpis:
            return

        # Get numeric KPI keys
        numeric_keys = []
        for key, val in last_kpis[0].items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_keys.append(key)

        for key in numeric_keys:
            values = [kpi[key] for kpi in last_kpis if key in kpi]
            if values:
                arr = np.array(values, dtype=float)
                self.kpi_aggregates[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }


@dataclass
class SweepResult:
    """Complete results of a parameter sweep."""
    combinations: list[CombinationResult] = field(default_factory=list)
    total_combinations: int = 0
    total_runs: int = 0
    elapsed_seconds: float = 0.0

    def best_stable_combination(self) -> Optional[CombinationResult]:
        """
        Return the combination with the highest stability rate.
        Ties broken by survival rate, then average final alive count.
        """
        stable = [c for c in self.combinations if c.stability_rate > 0]
        if not stable:
            return None
        return max(
            stable,
            key=lambda c: (c.stability_rate, c.survival_rate, c.avg_final_alive),
        )


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

@dataclass
class SweepSettings:
    """Settings parsed from sweep config JSON."""
    fixed_params: dict[str, Any]
    variable_params: dict[str, list[Any]]
    runs_per_set: int = 9
    max_generations: int = 99
    base_seed: int = 42
    stability_band_min_pct: float = 0.20
    stability_band_max_pct: float = 5.00
    check_after_generation: int = 10
    early_termination_on_extinction: bool = True
    parallel_workers: int = 4
    stability_required_pct: float = 0.66

    @classmethod
    def from_dict(cls, data: dict) -> SweepSettings:
        """Parse sweep settings from a JSON-style dict."""
        fixed = data.get("fixed_params", {})
        variable = data.get("variable_params", {})
        ss = data.get("sweep_settings", {})
        sb = ss.get("stability_band", {})

        return cls(
            fixed_params=fixed,
            variable_params=variable,
            runs_per_set=ss.get("runs_per_set", 9),
            max_generations=ss.get("max_generations", 99),
            base_seed=ss.get("base_seed", 42),
            stability_band_min_pct=sb.get("min_population_pct", 0.20),
            stability_band_max_pct=sb.get("max_population_pct", 5.00),
            check_after_generation=sb.get("check_after_generation", 10),
            early_termination_on_extinction=ss.get("early_termination_on_extinction", True),
            parallel_workers=ss.get("parallel_workers", 4),
            stability_required_pct=ss.get("stability_required_pct", 0.66),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> SweepSettings:
        """Load sweep settings from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate(self) -> list[str]:
        """Validate sweep settings."""
        errors = []
        if self.runs_per_set < 1:
            errors.append(f"runs_per_set must be >= 1, got {self.runs_per_set}")
        if self.max_generations < 1:
            errors.append(f"max_generations must be >= 1, got {self.max_generations}")
        if self.parallel_workers < 1:
            errors.append(f"parallel_workers must be >= 1, got {self.parallel_workers}")
        if self.stability_band_min_pct < 0:
            errors.append(f"stability_band_min_pct must be >= 0, got {self.stability_band_min_pct}")
        if self.stability_band_max_pct <= self.stability_band_min_pct:
            errors.append("stability_band_max_pct must be > min_pct")
        if self.check_after_generation < 0:
            errors.append(f"check_after_generation must be >= 0, got {self.check_after_generation}")
        if not (0.0 <= self.stability_required_pct <= 1.0):
            errors.append(f"stability_required_pct must be in [0, 1], got {self.stability_required_pct}")
        if not self.variable_params:
            errors.append("variable_params must have at least one parameter")
        for key, values in self.variable_params.items():
            if not isinstance(values, list) or len(values) == 0:
                errors.append(f"variable_params['{key}'] must be a non-empty list")
        return errors


# ---------------------------------------------------------------------------
# Combination generation
# ---------------------------------------------------------------------------

def generate_combinations(
    variable_params: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """
    Generate cartesian product of variable parameters.

    Args:
        variable_params: Dict of param_name → list of values.

    Returns:
        List of dicts, each representing one parameter combination.
        Example for {"a": [1, 2], "b": [3, 4]}:
          [{"a": 1, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 3}, {"a": 2, "b": 4}]
    """
    if not variable_params:
        return [{}]

    keys = list(variable_params.keys())
    value_lists = [variable_params[k] for k in keys]
    combos = []

    for values in itertools.product(*value_lists):
        combo = dict(zip(keys, values))
        combos.append(combo)

    return combos


# ---------------------------------------------------------------------------
# Single run worker (must be top-level for pickling in multiprocessing)
# ---------------------------------------------------------------------------

def _run_single_simulation(
    base_config_dict: dict,
    combination_id: int,
    combination_params: dict[str, Any],
    fixed_params: dict[str, Any],
    seed: int,
    run_index: int,
    max_generations: int,
    stability_band_min_pct: float,
    stability_band_max_pct: float,
    check_after_generation: int,
    early_termination_on_extinction: bool,
) -> SingleRunResult:
    """
    Run a single simulation and return results.

    This function is designed to be called in a separate process.

    Args:
        base_config_dict: Base config as dict (for pickling).
        combination_id: ID of this parameter combination.
        combination_params: Variable parameter overrides.
        fixed_params: Fixed parameter overrides.
        seed: Random seed for this run.
        run_index: Index of this run within the combination.
        max_generations: Maximum generations to run.
        stability_band_min_pct: Min population as fraction of initial.
        stability_band_max_pct: Max population as fraction of initial.
        check_after_generation: Start stability check after this gen.
        early_termination_on_extinction: Stop on extinction.

    Returns:
        SingleRunResult with all metrics.
    """
    # Reconstruct config
    config = SimConfig.from_dict(base_config_dict)

    # Apply fixed params
    for key, value in fixed_params.items():
        apply_param_override(config, key, value)

    # Apply variable params (combination)
    for key, value in combination_params.items():
        apply_param_override(config, key, value)

    # Set seed
    config.world.seed = seed

    # Get initial count for stability calculations
    initial_count = config.population.initial_count
    pop_min = initial_count * stability_band_min_pct
    pop_max = initial_count * stability_band_max_pct

    # Create and initialize engine
    engine = SimulationEngine(config)
    engine.initialize()

    # Set up metrics collector
    metrics = MetricsCollector(config)

    result = SingleRunResult(
        combination_id=combination_id,
        combination_params=combination_params,
        seed=seed,
        run_index=run_index,
        initial_count=initial_count,
    )

    stable_so_far = True
    instability_gen = None

    # Collect KPIs at each generation boundary
    def on_generation(gen_number: int, eng: SimulationEngine) -> None:
        nonlocal stable_so_far, instability_gen

        gen_stats = eng.generation_manager.all_gen_stats[-1] if eng.generation_manager.all_gen_stats else GenerationStats()
        tick_totals = eng.get_accumulated_stats()
        kpis = metrics.collect(eng.world, gen_stats, tick_totals)
        eng.reset_accumulated_stats()

        result.generation_kpis.append(kpis)

        # Stability check
        if stable_so_far and gen_number >= check_after_generation:
            alive = kpis.get("alive_count", 0)
            if alive < pop_min or alive > pop_max:
                stable_so_far = False
                instability_gen = gen_number

    engine.on_generation = on_generation

    # Run simulation
    run_result = engine.run(max_generations=max_generations)

    result.total_ticks = run_result.total_ticks
    result.total_generations = run_result.total_generations
    result.final_alive_count = run_result.final_alive_count
    result.extinct = run_result.extinct
    result.extinction_tick = run_result.extinction_tick
    result.stable = stable_so_far and not run_result.extinct
    result.instability_generation = instability_gen

    return result


# ---------------------------------------------------------------------------
# ParameterSweep class
# ---------------------------------------------------------------------------

class ParameterSweep:
    """
    Runs multiple simulations across parameter combinations to find stable baselines.

    Usage:
        sweep = ParameterSweep(settings)
        result = sweep.run()
        sweep.export_results(result, output_dir)

    Attributes:
        settings: Sweep settings (fixed/variable params, run config).
        combinations: Generated parameter combinations.
    """

    def __init__(self, settings: SweepSettings, base_config: Optional[SimConfig] = None):
        """
        Initialize the parameter sweep.

        Args:
            settings: Sweep settings (from JSON or manual).
            base_config: Base simulation config. None = use default config.
        """
        self.settings = settings
        self.base_config = base_config or get_default_config()
        self.combinations = generate_combinations(settings.variable_params)

    @property
    def total_combinations(self) -> int:
        """Number of parameter combinations."""
        return len(self.combinations)

    @property
    def total_runs(self) -> int:
        """Total number of simulation runs."""
        return self.total_combinations * self.settings.runs_per_set

    def run(
        self,
        parallel: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> SweepResult:
        """
        Execute the full parameter sweep.

        Args:
            parallel: Whether to use parallel execution. False = sequential.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            SweepResult with all combination results.
        """
        start_time = time.time()

        # Build all jobs
        jobs = self._build_jobs()

        # Execute
        if parallel and self.settings.parallel_workers > 1 and len(jobs) > 1:
            run_results = self._run_parallel(jobs, progress_callback)
        else:
            run_results = self._run_sequential(jobs, progress_callback)

        # Group by combination
        sweep_result = self._aggregate_results(run_results)
        sweep_result.elapsed_seconds = time.time() - start_time

        return sweep_result

    def _build_jobs(self) -> list[dict]:
        """Build the list of all simulation jobs to execute."""
        base_config_dict = self.base_config.to_dict()
        jobs = []

        for combo_id, combo_params in enumerate(self.combinations):
            for run_idx in range(self.settings.runs_per_set):
                seed = self.settings.base_seed + combo_id * 1000 + run_idx
                jobs.append({
                    "base_config_dict": base_config_dict,
                    "combination_id": combo_id,
                    "combination_params": combo_params,
                    "fixed_params": self.settings.fixed_params,
                    "seed": seed,
                    "run_index": run_idx,
                    "max_generations": self.settings.max_generations,
                    "stability_band_min_pct": self.settings.stability_band_min_pct,
                    "stability_band_max_pct": self.settings.stability_band_max_pct,
                    "check_after_generation": self.settings.check_after_generation,
                    "early_termination_on_extinction": self.settings.early_termination_on_extinction,
                })

        return jobs

    def _run_sequential(
        self,
        jobs: list[dict],
        progress_callback: Optional[callable],
    ) -> list[SingleRunResult]:
        """Execute jobs sequentially."""
        results = []
        for i, job in enumerate(jobs):
            result = _run_single_simulation(**job)
            results.append(result)
            if progress_callback is not None:
                progress_callback(i + 1, len(jobs))
        return results

    def _run_parallel(
        self,
        jobs: list[dict],
        progress_callback: Optional[callable],
    ) -> list[SingleRunResult]:
        """Execute jobs in parallel using ProcessPoolExecutor."""
        results = []
        completed = 0

        with ProcessPoolExecutor(max_workers=self.settings.parallel_workers) as executor:
            futures = {
                executor.submit(_run_single_simulation, **job): job
                for job in jobs
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, len(jobs))

        return results

    def _aggregate_results(self, run_results: list[SingleRunResult]) -> SweepResult:
        """Group run results by combination and compute aggregates."""
        # Group by combination_id
        combo_map: dict[int, list[SingleRunResult]] = {}
        for r in run_results:
            combo_map.setdefault(r.combination_id, []).append(r)

        combinations = []
        for combo_id, combo_params in enumerate(self.combinations):
            runs = combo_map.get(combo_id, [])
            combo_result = CombinationResult(
                combination_id=combo_id,
                params=combo_params,
                runs=runs,
            )
            combo_result.aggregate()
            combinations.append(combo_result)

        sweep_result = SweepResult(
            combinations=combinations,
            total_combinations=len(combinations),
            total_runs=sum(len(c.runs) for c in combinations),
        )

        return sweep_result

    # ------------------------------------------------------------------
    # Export results
    # ------------------------------------------------------------------

    def export_results(
        self,
        result: SweepResult,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """
        Export sweep results to files.

        Creates:
          - summary.csv: one row per combination with aggregated stats
          - detailed.csv: one row per generation per run
          - stability_report.json: which combinations are stable
          - sweep_config.json: copy of the sweep settings

        Args:
            result: The sweep result to export.
            output_dir: Output directory.

        Returns:
            Dict of file type → path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Summary CSV
        paths["summary"] = self._export_summary_csv(result, out / "summary.csv")

        # Detailed CSV
        paths["detailed"] = self._export_detailed_csv(result, out / "detailed.csv")

        # Stability report
        paths["stability_report"] = self._export_stability_report(result, out / "stability_report.json")

        # Sweep config copy
        config_path = out / "sweep_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "fixed_params": self.settings.fixed_params,
                "variable_params": self.settings.variable_params,
                "sweep_settings": {
                    "runs_per_set": self.settings.runs_per_set,
                    "max_generations": self.settings.max_generations,
                    "base_seed": self.settings.base_seed,
                    "stability_band": {
                        "min_population_pct": self.settings.stability_band_min_pct,
                        "max_population_pct": self.settings.stability_band_max_pct,
                        "check_after_generation": self.settings.check_after_generation,
                    },
                    "early_termination_on_extinction": self.settings.early_termination_on_extinction,
                    "parallel_workers": self.settings.parallel_workers,
                    "stability_required_pct": self.settings.stability_required_pct,
                },
            }, f, indent=2)
        paths["config"] = config_path

        return paths

    def _export_summary_csv(self, result: SweepResult, path: Path) -> Path:
        """Write summary CSV: one row per combination."""
        fieldnames = [
            "combination_id",
            "total_runs",
            "extinction_count",
            "survival_rate",
            "stable_count",
            "stability_rate",
            "avg_final_alive",
            "std_final_alive",
            "avg_generations",
        ]
        # Add param columns
        param_keys = sorted(self.settings.variable_params.keys())
        fieldnames = [f"param_{k}" for k in param_keys] + fieldnames

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for combo in result.combinations:
                row = {
                    "combination_id": combo.combination_id,
                    "total_runs": combo.total_runs,
                    "extinction_count": combo.extinction_count,
                    "survival_rate": round(combo.survival_rate, 4),
                    "stable_count": combo.stable_count,
                    "stability_rate": round(combo.stability_rate, 4),
                    "avg_final_alive": round(combo.avg_final_alive, 2),
                    "std_final_alive": round(combo.std_final_alive, 2),
                    "avg_generations": round(combo.avg_generations, 2),
                }
                for k in param_keys:
                    row[f"param_{k}"] = combo.params.get(k, "")
                writer.writerow(row)

        return path

    def _export_detailed_csv(self, result: SweepResult, path: Path) -> Path:
        """Write detailed CSV: one row per generation per run."""
        # Determine all KPI columns from the first run that has data
        kpi_columns = []
        for combo in result.combinations:
            for run in combo.runs:
                if run.generation_kpis:
                    kpi_columns = list(run.generation_kpis[0].keys())
                    break
            if kpi_columns:
                break

        fieldnames = [
            "combination_id", "run_index", "seed",
        ]
        # Add param columns
        param_keys = sorted(self.settings.variable_params.keys())
        fieldnames += [f"param_{k}" for k in param_keys]
        fieldnames += kpi_columns

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for combo in result.combinations:
                for run in combo.runs:
                    for kpi in run.generation_kpis:
                        row = {
                            "combination_id": combo.combination_id,
                            "run_index": run.run_index,
                            "seed": run.seed,
                        }
                        for k in param_keys:
                            row[f"param_{k}"] = combo.params.get(k, "")
                        row.update(kpi)
                        writer.writerow(row)

        return path

    def _export_stability_report(self, result: SweepResult, path: Path) -> Path:
        """Write stability report as JSON."""
        report = {
            "total_combinations": result.total_combinations,
            "total_runs": result.total_runs,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "stability_required_pct": self.settings.stability_required_pct,
            "combinations": [],
        }

        for combo in result.combinations:
            is_stable = combo.stability_rate >= self.settings.stability_required_pct
            entry = {
                "combination_id": combo.combination_id,
                "params": combo.params,
                "total_runs": combo.total_runs,
                "extinction_count": combo.extinction_count,
                "survival_rate": round(combo.survival_rate, 4),
                "stable_count": combo.stable_count,
                "stability_rate": round(combo.stability_rate, 4),
                "is_stable": is_stable,
                "avg_final_alive": round(combo.avg_final_alive, 2),
            }
            report["combinations"].append(entry)

        # Overall summary
        stable_combos = [c for c in report["combinations"] if c["is_stable"]]
        report["stable_combinations_count"] = len(stable_combos)
        report["best_combination"] = None
        if stable_combos:
            best = max(stable_combos, key=lambda c: (c["stability_rate"], c["survival_rate"]))
            report["best_combination"] = best

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return path

    def __repr__(self) -> str:
        return (
            f"ParameterSweep(combinations={self.total_combinations}, "
            f"runs_per_set={self.settings.runs_per_set}, "
            f"total_runs={self.total_runs})"
        )
