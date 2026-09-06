"""
Parameter sweep: independent (config, seed) jobs, local process pool.

Each job is a pure function. No shared engine state.

Author: Cursor Grok 4.6 High Fast
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.config import SimConfig, apply_param_override, get_default_config
from src.engine import SimulationEngine
from src.metrics import EpochMetrics


def _ensure_v3_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SingleRunResult:
    combination_id: int
    combination_params: dict[str, Any]
    seed: int
    run_index: int
    total_ticks: int = 0
    total_epochs: int = 0
    final_alive_count: int = 0
    initial_count: int = 0
    extinct: bool = False
    extinction_tick: Optional[int] = None
    stable: bool = False
    instability_epoch: Optional[int] = None
    epoch_kpis: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "combination_id": self.combination_id,
            "combination_params": self.combination_params,
            "seed": self.seed,
            "run_index": self.run_index,
            "total_ticks": self.total_ticks,
            "total_epochs": self.total_epochs,
            "final_alive_count": self.final_alive_count,
            "initial_count": self.initial_count,
            "extinct": self.extinct,
            "extinction_tick": self.extinction_tick,
            "stable": self.stable,
            "instability_epoch": self.instability_epoch,
            "epoch_kpis": self.epoch_kpis,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SingleRunResult:
        return cls(
            combination_id=int(data["combination_id"]),
            combination_params=data.get("combination_params") or {},
            seed=int(data["seed"]),
            run_index=int(data["run_index"]),
            total_ticks=int(data.get("total_ticks", 0)),
            total_epochs=int(data.get("total_epochs", 0)),
            final_alive_count=int(data.get("final_alive_count", 0)),
            initial_count=int(data.get("initial_count", 0)),
            extinct=bool(data.get("extinct", False)),
            extinction_tick=data.get("extinction_tick"),
            stable=bool(data.get("stable", False)),
            instability_epoch=data.get("instability_epoch"),
            epoch_kpis=list(data.get("epoch_kpis") or []),
        )


@dataclass
class CombinationResult:
    combination_id: int
    params: dict[str, Any]
    runs: list[SingleRunResult] = field(default_factory=list)
    total_runs: int = 0
    extinction_count: int = 0
    survival_rate: float = 0.0
    stable_count: int = 0
    stability_rate: float = 0.0
    avg_final_alive: float = 0.0
    std_final_alive: float = 0.0
    avg_epochs: float = 0.0
    kpi_aggregates: dict[str, dict[str, float]] = field(default_factory=dict)

    def aggregate(self) -> None:
        self.total_runs = len(self.runs)
        if self.total_runs == 0:
            return
        self.extinction_count = sum(1 for r in self.runs if r.extinct)
        self.survival_rate = 1.0 - (self.extinction_count / self.total_runs)
        self.stable_count = sum(1 for r in self.runs if r.stable)
        self.stability_rate = self.stable_count / self.total_runs
        alive = [r.final_alive_count for r in self.runs]
        self.avg_final_alive = float(np.mean(alive))
        self.std_final_alive = float(np.std(alive))
        self.avg_epochs = float(np.mean([r.total_epochs for r in self.runs]))
        last: list[dict] = [r.epoch_kpis[-1] for r in self.runs if r.epoch_kpis]
        self.kpi_aggregates = {}
        if not last:
            return
        for key, val in last[0].items():
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                continue
            values = [kpi[key] for kpi in last if key in kpi]
            arr = np.array(values, dtype=float)
            self.kpi_aggregates[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }


@dataclass
class SweepResult:
    combinations: list[CombinationResult] = field(default_factory=list)
    total_combinations: int = 0
    total_runs: int = 0
    elapsed_seconds: float = 0.0

    def best_stable_combination(self) -> Optional[CombinationResult]:
        stable = [c for c in self.combinations if c.stability_rate > 0]
        if not stable:
            return None
        return max(
            stable,
            key=lambda c: (c.stability_rate, c.survival_rate, c.avg_final_alive),
        )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class SweepSettings:
    fixed_params: dict[str, Any]
    variable_params: dict[str, list[Any]]
    runs_per_set: int = 9
    max_epochs: int = 99
    base_seed: int = 42
    stability_band_min_pct: float = 0.20
    stability_band_max_pct: float = 5.00
    check_after_epoch: int = 10
    early_termination_on_extinction: bool = True
    parallel_workers: int = 4
    stability_required_pct: float = 0.66

    @classmethod
    def from_dict(cls, data: dict) -> SweepSettings:
        ss = data.get("sweep_settings", {})
        sb = ss.get("stability_band", {})
        check = sb.get("check_after_epoch", sb.get("check_after_generation", 10))
        max_epochs = ss.get("max_epochs", ss.get("max_generations", 99))
        return cls(
            fixed_params=data.get("fixed_params", {}),
            variable_params=data.get("variable_params", {}),
            runs_per_set=ss.get("runs_per_set", 9),
            max_epochs=max_epochs,
            base_seed=ss.get("base_seed", 42),
            stability_band_min_pct=sb.get("min_population_pct", 0.20),
            stability_band_max_pct=sb.get("max_population_pct", 5.00),
            check_after_epoch=check,
            early_termination_on_extinction=ss.get("early_termination_on_extinction", True),
            parallel_workers=ss.get("parallel_workers", 4),
            stability_required_pct=ss.get("stability_required_pct", 0.66),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> SweepSettings:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def validate(self) -> list[str]:
        errors = []
        if self.runs_per_set < 1:
            errors.append(f"runs_per_set must be >= 1, got {self.runs_per_set}")
        if self.max_epochs < 1:
            errors.append(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.parallel_workers < 1:
            errors.append(f"parallel_workers must be >= 1, got {self.parallel_workers}")
        if self.stability_band_max_pct <= self.stability_band_min_pct:
            errors.append("stability_band max must be > min")
        if not self.variable_params:
            errors.append("variable_params must have at least one parameter")
        for key, values in self.variable_params.items():
            if not isinstance(values, list) or len(values) == 0:
                errors.append(f"variable_params['{key}'] must be a non-empty list")
        return errors

    def to_export_dict(self) -> dict:
        return {
            "fixed_params": self.fixed_params,
            "variable_params": self.variable_params,
            "sweep_settings": {
                "runs_per_set": self.runs_per_set,
                "max_epochs": self.max_epochs,
                "base_seed": self.base_seed,
                "stability_band": {
                    "min_population_pct": self.stability_band_min_pct,
                    "max_population_pct": self.stability_band_max_pct,
                    "check_after_epoch": self.check_after_epoch,
                },
                "early_termination_on_extinction": self.early_termination_on_extinction,
                "parallel_workers": self.parallel_workers,
                "stability_required_pct": self.stability_required_pct,
            },
        }


def generate_combinations(variable_params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not variable_params:
        return [{}]
    keys = list(variable_params.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(variable_params[k] for k in keys))]


def classify_stability(
    alive_series: list[int],
    initial_count: int,
    min_pct: float,
    max_pct: float,
    check_after_epoch: int,
) -> tuple[bool, Optional[int]]:
    """Return (stable, first_bad_epoch). Epoch index is the KPI epoch number."""
    lo = initial_count * min_pct
    hi = initial_count * max_pct
    for epoch, alive in enumerate(alive_series):
        if epoch < check_after_epoch:
            continue
        if alive < lo or alive > hi:
            return False, epoch
    return True, None


# ---------------------------------------------------------------------------
# Worker (top-level for Windows spawn)
# ---------------------------------------------------------------------------

def run_single_job(job: dict) -> SingleRunResult:
    _ensure_v3_on_path()
    config = SimConfig.from_dict(job["base_config_dict"])
    for key, value in job["fixed_params"].items():
        apply_param_override(config, key, value)
    for key, value in job["combination_params"].items():
        apply_param_override(config, key, value)
    config.world.seed = job["seed"]
    config.viz.snapshot_every_epoch = False
    if config.population.initial_count > config.perf.max_animals:
        config.perf.max_animals = int(config.population.initial_count * 10)

    initial = config.population.initial_count
    engine = SimulationEngine(config)
    engine.initialize()

    result = SingleRunResult(
        combination_id=job["combination_id"],
        combination_params=job["combination_params"],
        seed=job["seed"],
        run_index=job["run_index"],
        initial_count=initial,
    )
    run = engine.run(max_epochs=job["max_epochs"])
    result.total_ticks = run.total_ticks
    result.total_epochs = run.total_epochs
    result.final_alive_count = run.final_alive
    result.extinct = run.extinct
    result.extinction_tick = run.extinction_tick
    result.epoch_kpis = [m.to_dict() if isinstance(m, EpochMetrics) else m for m in run.epoch_metrics]

    alive_series = [int(k.get("alive_count", 0)) for k in result.epoch_kpis]
    stable, bad = classify_stability(
        alive_series,
        initial,
        job["stability_band_min_pct"],
        job["stability_band_max_pct"],
        job["check_after_epoch"],
    )
    result.stable = bool(stable and not run.extinct)
    result.instability_epoch = bad
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ParameterSweep:
    def __init__(self, settings: SweepSettings, base_config: Optional[SimConfig] = None):
        errors = settings.validate()
        if errors:
            raise ValueError("Invalid sweep settings:\n" + "\n".join(f"  - {e}" for e in errors))
        self.settings = settings
        self.base_config = base_config or get_default_config()
        self.combinations = generate_combinations(settings.variable_params)

    @property
    def total_combinations(self) -> int:
        return len(self.combinations)

    @property
    def total_runs(self) -> int:
        return self.total_combinations * self.settings.runs_per_set

    def _build_jobs(self) -> list[dict]:
        base = self.base_config.to_dict()
        jobs = []
        for combo_id, combo in enumerate(self.combinations):
            for run_idx in range(self.settings.runs_per_set):
                seed = self.settings.base_seed + combo_id * 1000 + run_idx
                jobs.append({
                    "base_config_dict": base,
                    "combination_id": combo_id,
                    "combination_params": combo,
                    "fixed_params": self.settings.fixed_params,
                    "seed": seed,
                    "run_index": run_idx,
                    "max_epochs": self.settings.max_epochs,
                    "stability_band_min_pct": self.settings.stability_band_min_pct,
                    "stability_band_max_pct": self.settings.stability_band_max_pct,
                    "check_after_epoch": self.settings.check_after_epoch,
                    "early_termination_on_extinction": self.settings.early_termination_on_extinction,
                })
        return jobs

    def run(
        self,
        parallel: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SweepResult:
        start = time.time()
        jobs = self._build_jobs()
        workers = self.settings.parallel_workers
        use_pool = parallel and workers > 1 and len(jobs) > 1
        if use_pool:
            run_results = self._run_parallel(jobs, progress_callback)
        else:
            run_results = self._run_sequential(jobs, progress_callback)
        sweep = self._aggregate(run_results)
        sweep.elapsed_seconds = time.time() - start
        return sweep

    def _run_sequential(self, jobs: list[dict], progress_callback) -> list[SingleRunResult]:
        results = []
        for i, job in enumerate(jobs):
            results.append(run_single_job(job))
            if progress_callback is not None:
                progress_callback(i + 1, len(jobs))
        return results

    def _run_parallel(self, jobs: list[dict], progress_callback) -> list[SingleRunResult]:
        results: list[SingleRunResult] = []
        completed = 0
        with ProcessPoolExecutor(
            max_workers=self.settings.parallel_workers,
            initializer=_ensure_v3_on_path,
        ) as ex:
            futures = [ex.submit(run_single_job, job) for job in jobs]
            for fut in as_completed(futures):
                results.append(fut.result())
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, len(jobs))
        results.sort(key=lambda r: (r.combination_id, r.run_index))
        return results

    def _aggregate(self, run_results: list[SingleRunResult]) -> SweepResult:
        by_id: dict[int, list[SingleRunResult]] = {}
        for r in run_results:
            by_id.setdefault(r.combination_id, []).append(r)
        combinations = []
        for combo_id, params in enumerate(self.combinations):
            runs = sorted(by_id.get(combo_id, []), key=lambda r: r.run_index)
            combo = CombinationResult(combination_id=combo_id, params=params, runs=runs)
            combo.aggregate()
            combinations.append(combo)
        return SweepResult(
            combinations=combinations,
            total_combinations=len(combinations),
            total_runs=sum(len(c.runs) for c in combinations),
        )

    def export_results(self, result: SweepResult, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "summary": self._export_summary(result, out / "summary.csv"),
            "detailed": self._export_detailed(result, out / "detailed.csv"),
            "stability_report": self._export_stability(result, out / "stability_report.json"),
        }
        cfg_path = out / "sweep_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.settings.to_export_dict(), f, indent=2)
        paths["config"] = cfg_path
        return paths

    def _export_summary(self, result: SweepResult, path: Path) -> Path:
        param_keys = sorted(self.settings.variable_params.keys())
        fieldnames = [f"param_{k}" for k in param_keys] + [
            "combination_id", "total_runs", "extinction_count", "survival_rate",
            "stable_count", "stability_rate", "avg_final_alive", "std_final_alive",
            "avg_epochs",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
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
                    "avg_epochs": round(combo.avg_epochs, 2),
                }
                for k in param_keys:
                    row[f"param_{k}"] = combo.params.get(k, "")
                w.writerow(row)
        return path

    def _export_detailed(self, result: SweepResult, path: Path) -> Path:
        kpi_columns: list[str] = []
        for combo in result.combinations:
            for run in combo.runs:
                if run.epoch_kpis:
                    kpi_columns = list(run.epoch_kpis[0].keys())
                    break
            if kpi_columns:
                break
        param_keys = sorted(self.settings.variable_params.keys())
        fieldnames = ["combination_id", "run_index", "seed"] + [f"param_{k}" for k in param_keys] + kpi_columns
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for combo in result.combinations:
                for run in combo.runs:
                    for kpi in run.epoch_kpis:
                        row = {
                            "combination_id": combo.combination_id,
                            "run_index": run.run_index,
                            "seed": run.seed,
                        }
                        for k in param_keys:
                            row[f"param_{k}"] = combo.params.get(k, "")
                        row.update(kpi)
                        w.writerow(row)
        return path

    def _export_stability(self, result: SweepResult, path: Path) -> Path:
        combos = []
        for combo in result.combinations:
            combos.append({
                "combination_id": combo.combination_id,
                "params": combo.params,
                "total_runs": combo.total_runs,
                "extinction_count": combo.extinction_count,
                "survival_rate": round(combo.survival_rate, 4),
                "stable_count": combo.stable_count,
                "stability_rate": round(combo.stability_rate, 4),
                "is_stable": combo.stability_rate >= self.settings.stability_required_pct,
                "avg_final_alive": round(combo.avg_final_alive, 2),
            })
        stable = [c for c in combos if c["is_stable"]]
        report = {
            "total_combinations": result.total_combinations,
            "total_runs": result.total_runs,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
            "stability_required_pct": self.settings.stability_required_pct,
            "stable_combinations_count": len(stable),
            "best_combination": max(
                stable,
                key=lambda c: (c["stability_rate"], c["survival_rate"], c["avg_final_alive"]),
            ) if stable else None,
            "combinations": combos,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return path

    def export_job_bundle(self, output_dir: str | Path) -> dict[str, Path]:
        """Write a cluster-ready job bundle (shared config + jsonl jobs)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        from src.config import save_config
        cfg_path = out / "base_config.json"
        save_config(self.base_config, cfg_path)
        settings_path = out / "sweep_settings.json"
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(self.settings.to_export_dict(), f, indent=2)
        jobs_path = out / "jobs.jsonl"
        slim = []
        for job in self._build_jobs():
            slim.append({k: v for k, v in job.items() if k != "base_config_dict"})
        with open(jobs_path, "w", encoding="utf-8") as f:
            for row in slim:
                f.write(json.dumps(row) + "\n")
        meta = {
            "n_jobs": len(slim),
            "total_combinations": self.total_combinations,
            "runs_per_set": self.settings.runs_per_set,
        }
        meta_path = out / "manifest.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return {
            "base_config": cfg_path,
            "sweep_settings": settings_path,
            "jobs": jobs_path,
            "manifest": meta_path,
        }


def load_job_from_bundle(jobs_dir: str | Path, index: int) -> dict:
    """Reconstruct a full worker job dict from an exported bundle."""
    from src.config import load_config

    jobs_dir = Path(jobs_dir)
    lines = jobs_dir.joinpath("jobs.jsonl").read_text(encoding="utf-8").splitlines()
    if index < 0 or index >= len(lines):
        raise IndexError(f"job index {index} out of range 0..{len(lines)-1}")
    job = json.loads(lines[index])
    job["base_config_dict"] = load_config(jobs_dir / "base_config.json").to_dict()
    return job


def collect_job_result_files(results_dir: str | Path) -> list[Path]:
    """JSON files that look like SingleRunResult dumps (not bundle metadata)."""
    files = []
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "combination_id" in data and "seed" in data:
            files.append(path)
    return files


def merge_job_results(
    result_files: list[Path],
    settings: SweepSettings,
    output_dir: str | Path,
) -> SweepResult:
    """Merge run-job JSON files into a SweepResult and export CSVs."""
    runs = [
        SingleRunResult.from_dict(json.loads(Path(p).read_text(encoding="utf-8")))
        for p in result_files
    ]
    sweep = ParameterSweep(settings)
    result = sweep._aggregate(runs)
    sweep.export_results(result, output_dir)
    return result
