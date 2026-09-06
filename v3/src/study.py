"""
Scientific batch-run core: Monte Carlo replicates from a starting point.

A "study" runs N re-seeded replicates from a saved checkpoint (or a fresh
config), grouped into "arms" (each arm = optional dotted-key config overrides).
Results are aggregated into mean +/- CI trajectories, survival curves, and
pairwise arm comparisons, then written under the origin checkpoint folder.

This module is UI-agnostic (no tkinter). The GUI lives in src/study_gui.py.

Author: Cursor Claude Opus 4.8 High
Edited on 2026-09-06 by Cursor Claude Opus 4.8 High: per-epoch progress
streaming from worker processes via a Manager queue + drain thread.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, Callable, Optional

import numpy as np

from src.config import (
    SimConfig,
    apply_param_override,
    get_default_config,
    load_config,
    save_config,
)
from src.checkpoint import load_checkpoint, save_checkpoint
from src.engine import SimulationEngine


# ---------------------------------------------------------------------------
# Path / process helpers
# ---------------------------------------------------------------------------

def _ensure_v3_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _slug(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", (name or "").strip(), flags=re.UNICODE)
    s = s.strip("._") or "study"
    return s[:80]


def _hash_dir(path: Path) -> str:
    """Cheap content hash of a checkpoint folder (arrays + engine + config)."""
    h = hashlib.sha256()
    for fname in ("config.json", "engine.json", "arrays.npz", "meta.json"):
        fp = path / fname
        if fp.is_file():
            h.update(fname.encode("utf-8"))
            h.update(fp.read_bytes())
    return h.hexdigest()[:16]


# KPIs surfaced in the scientific report (others are still aggregated).
KEY_KPIS = (
    "adaptation_score",
    "adapted_frac",
    "alive_count",
    "genetic_diversity",
    "avg_energy",
)


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    label: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"label": self.label, "overrides": dict(self.overrides)}

    @classmethod
    def from_dict(cls, data: dict) -> "Arm":
        return cls(label=str(data["label"]), overrides=dict(data.get("overrides") or {}))


@dataclass
class StudySpec:
    name: str
    origin_path: str                      # checkpoint folder or config file
    origin_kind: str = "checkpoint"       # "checkpoint" | "config"
    arms: list[Arm] = field(default_factory=list)
    replicates_per_arm: int = 10
    max_epochs: int = 20
    base_seed: int = 1234
    random_base_seed: bool = False
    burn_in_epochs: int = 0
    compare_metric: str = "adaptation_score"
    compare_epoch: Optional[int] = None   # None => last aligned epoch index
    bootstrap: bool = True
    n_bootstrap: int = 2000
    quantiles: tuple[float, float] = (0.1, 0.9)
    workers: Optional[int] = None         # None => os.cpu_count()
    early_stop_all_extinct: bool = True
    save_end_checkpoints: bool = False

    def resolved_workers(self) -> int:
        if self.workers and self.workers > 0:
            return int(self.workers)
        return max(1, os.cpu_count() or 1)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.arms:
            errors.append("study needs at least one arm")
        labels = [a.label for a in self.arms]
        if len(labels) != len(set(labels)):
            errors.append("arm labels must be unique")
        if self.replicates_per_arm < 1:
            errors.append("replicates_per_arm must be >= 1")
        if self.max_epochs < 1:
            errors.append("max_epochs must be >= 1")
        if self.burn_in_epochs < 0 or self.burn_in_epochs >= self.max_epochs:
            errors.append("burn_in_epochs must be in [0, max_epochs)")
        if self.origin_kind not in ("checkpoint", "config"):
            errors.append("origin_kind must be 'checkpoint' or 'config'")
        if not Path(self.origin_path).exists():
            errors.append(f"origin_path does not exist: {self.origin_path}")
        return errors

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "origin_path": str(self.origin_path),
            "origin_kind": self.origin_kind,
            "arms": [a.to_dict() for a in self.arms],
            "replicates_per_arm": self.replicates_per_arm,
            "max_epochs": self.max_epochs,
            "base_seed": self.base_seed,
            "random_base_seed": self.random_base_seed,
            "burn_in_epochs": self.burn_in_epochs,
            "compare_metric": self.compare_metric,
            "compare_epoch": self.compare_epoch,
            "bootstrap": self.bootstrap,
            "n_bootstrap": self.n_bootstrap,
            "quantiles": list(self.quantiles),
            "workers": self.workers,
            "early_stop_all_extinct": self.early_stop_all_extinct,
            "save_end_checkpoints": self.save_end_checkpoints,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StudySpec":
        q = data.get("quantiles") or [0.1, 0.9]
        return cls(
            name=str(data.get("name", "study")),
            origin_path=str(data["origin_path"]),
            origin_kind=str(data.get("origin_kind", "checkpoint")),
            arms=[Arm.from_dict(a) for a in data.get("arms", [])],
            replicates_per_arm=int(data.get("replicates_per_arm", 10)),
            max_epochs=int(data.get("max_epochs", 20)),
            base_seed=int(data.get("base_seed", 1234)),
            random_base_seed=bool(data.get("random_base_seed", False)),
            burn_in_epochs=int(data.get("burn_in_epochs", 0)),
            compare_metric=str(data.get("compare_metric", "adaptation_score")),
            compare_epoch=data.get("compare_epoch"),
            bootstrap=bool(data.get("bootstrap", True)),
            n_bootstrap=int(data.get("n_bootstrap", 2000)),
            quantiles=(float(q[0]), float(q[1])),
            workers=data.get("workers"),
            early_stop_all_extinct=bool(data.get("early_stop_all_extinct", True)),
            save_end_checkpoints=bool(data.get("save_end_checkpoints", False)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "StudySpec":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ReplicateResult:
    arm_index: int
    arm_label: str
    replicate_index: int
    seed: int
    total_ticks: int = 0
    total_epochs: int = 0
    final_alive: int = 0
    extinct: bool = False
    extinction_tick: Optional[int] = None
    epoch_kpis: list[dict] = field(default_factory=list)
    end_checkpoint: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReplicateResult":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


# ---------------------------------------------------------------------------
# Seed derivation (deterministic, reproducible)
# ---------------------------------------------------------------------------

def replicate_seed(base_seed: int, arm_index: int, replicate_index: int) -> int:
    return int(base_seed) + arm_index * 10_000 + replicate_index


# ---------------------------------------------------------------------------
# Worker (top-level so ProcessPool can pickle it on Windows spawn)
# ---------------------------------------------------------------------------

def _terminate_pool(ex: ProcessPoolExecutor) -> None:
    """Forcibly kill all worker processes of a ProcessPoolExecutor.

    Used for a hard, immediate cancel: futures already executing cannot be
    cancelled cooperatively, so we terminate the OS processes directly.
    """
    procs = list(getattr(ex, "_processes", {}).values())
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            if hasattr(p, "kill"):
                p.kill()
        except Exception:
            pass


def run_replicate_job(job: dict, progress_queue: Any = None) -> dict:
    """Run one replicate. Returns a ReplicateResult dict. Never raises.

    If ``progress_queue`` is given (a queue with a ``put`` method), the worker
    emits ``("epoch", arm_index, replicate_index, arm_label, epochs_done, max)``
    after every completed epoch so the UI can show live intra-replicate progress.
    """
    _ensure_v3_on_path()
    res = ReplicateResult(
        arm_index=job["arm_index"],
        arm_label=job["arm_label"],
        replicate_index=job["replicate_index"],
        seed=job["seed"],
    )
    try:
        engine = _build_engine(
            origin_path=job["origin_path"],
            origin_kind=job["origin_kind"],
            overrides=job["overrides"],
            seed=job["seed"],
        )
        # Run max_epochs ADDITIONAL epochs beyond the checkpoint's start, and keep
        # only the new epochs so every replicate's series shares the common start.
        start_epochs = len(engine.epoch_history)
        target = engine.epochs_completed + int(job["max_epochs"])
        if progress_queue is not None:
            _jmax = int(job["max_epochs"])
            _ai = job["arm_index"]
            _ri = job["replicate_index"]
            _al = job["arm_label"]

            def _report(_metrics, _eng, _base=start_epochs) -> None:
                cur = _eng.epochs_completed - _base
                cur = 0 if cur < 0 else (_jmax if cur > _jmax else cur)
                try:
                    progress_queue.put(("epoch", _ai, _ri, _al, cur, _jmax))
                except Exception:
                    pass

            engine.on_epoch = _report
        run = engine.run(max_epochs=target)
        new_metrics = run.epoch_metrics[start_epochs:]
        res.total_ticks = run.total_ticks
        res.total_epochs = len(new_metrics)
        res.final_alive = run.final_alive
        res.extinct = run.extinct
        res.extinction_tick = run.extinction_tick
        res.epoch_kpis = [m.to_dict() for m in new_metrics]
        if job.get("save_end_checkpoint") and job.get("end_ckpt_dir"):
            folder = save_checkpoint(
                engine,
                job["end_ckpt_dir"],
                name=f"{job['arm_label']}_rep{job['replicate_index']:03d}",
                notes=f"study end state, seed={job['seed']}",
                parent=Path(job["origin_path"]).name if job["origin_kind"] == "checkpoint" else None,
            )
            res.end_checkpoint = str(folder)
    except Exception as exc:  # keep the pool alive; surface per-replicate errors
        res.error = f"{type(exc).__name__}: {exc}"
    return res.to_dict()


def _build_engine(origin_path: str, origin_kind: str, overrides: dict, seed: int) -> SimulationEngine:
    if origin_kind == "checkpoint":
        base_cfg = load_config(Path(origin_path) / "config.json")
        arm_cfg = base_cfg.copy()
        for key, value in (overrides or {}).items():
            apply_param_override(arm_cfg, key, value)
        arm_cfg.viz.snapshot_every_epoch = False
        engine = load_checkpoint(origin_path, config_override=arm_cfg)
        # CRUX: give this replicate a fresh, divergent future. load_checkpoint
        # restored the saved RNG state; rebind BOTH engine and world generators.
        engine.rng = np.random.default_rng(seed)
        engine.world.rng = engine.rng
        return engine

    # origin_kind == "config"
    cfg = load_config(origin_path)
    for key, value in (overrides or {}).items():
        apply_param_override(cfg, key, value)
    cfg.world.seed = int(seed)
    cfg.viz.snapshot_every_epoch = False
    if cfg.population.initial_count > cfg.perf.max_animals:
        cfg.perf.max_animals = int(cfg.population.initial_count * 10)
    engine = SimulationEngine(cfg)
    engine.initialize()
    return engine


# ---------------------------------------------------------------------------
# Statistics (numpy-only)
# ---------------------------------------------------------------------------

def _mean_ci(values: np.ndarray) -> dict:
    v = values[~np.isnan(values)]
    n = int(v.size)
    if n == 0:
        return {"mean": float("nan"), "std": 0.0, "sem": 0.0,
                "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else 0.0
    half = 1.96 * sem
    return {"mean": mean, "std": std, "sem": sem,
            "ci_low": mean - half, "ci_high": mean + half, "n": n}


def _bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    v = values[~np.isnan(values)]
    if v.size < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = math.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size < 1 or b.size < 1:
        return float("nan")
    observed = abs(np.mean(a) - np.mean(b))
    pool = np.concatenate([a, b])
    na = a.size
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(np.mean(pool[:na]) - np.mean(pool[na:])) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def welch_ttest(a: np.ndarray, b: np.ndarray) -> Optional[dict]:
    """Welch t-test if scipy is importable, else None."""
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return None
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return None
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p": float(p)}


def sample_size_hint(d: float, power: float = 0.8, alpha: float = 0.05) -> Optional[int]:
    """Approx replicates per arm to detect effect size d (two-sided z-approx)."""
    if not d or math.isnan(d) or d == 0:
        return None
    # z_{1-alpha/2} + z_{power}; for 0.05/0.8 -> 1.96 + 0.84
    z_a = 1.959963985
    z_b = 0.841621234 if abs(power - 0.8) < 1e-6 else _inv_norm(power)
    n = ((z_a + z_b) ** 2) * 2.0 / (d * d)
    return int(math.ceil(n))


def _inv_norm(p: float) -> float:
    # Acklam's inverse-normal approximation (good enough for a power hint)
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.357751867269, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _numeric_metric_keys(replicates: list[ReplicateResult]) -> list[str]:
    for r in replicates:
        if r.epoch_kpis:
            keys = []
            for k, v in r.epoch_kpis[0].items():
                if k in ("epoch", "tick"):
                    continue
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    keys.append(k)
            return keys
    return []


def _series_matrix(replicates: list[ReplicateResult], metric: str, max_len: int,
                   fill_extinct_zero: bool) -> np.ndarray:
    """[n_rep, max_len] with NaN where a replicate has no epoch at that index."""
    mat = np.full((len(replicates), max_len), np.nan, dtype=float)
    for i, r in enumerate(replicates):
        for j, kpi in enumerate(r.epoch_kpis):
            if j >= max_len:
                break
            val = kpi.get(metric)
            if isinstance(val, bool):
                val = float(val)
            if val is not None:
                mat[i, j] = float(val)
        if fill_extinct_zero and r.extinct:
            last = len(r.epoch_kpis)
            mat[i, last:max_len] = 0.0
    return mat


def _replicate_scalar(r: ReplicateResult, metric: str, burn_in: int) -> float:
    """Per-replicate scalar for comparison: steady-state mean if burn_in>0, else final value."""
    if metric == "alive_count" and r.extinct:
        # honest: an extinct replicate's final population is zero
        if burn_in <= 0:
            return 0.0
    if not r.epoch_kpis:
        return float("nan")
    if burn_in > 0:
        vals = [k.get(metric) for k in r.epoch_kpis[burn_in:] if k.get(metric) is not None]
        vals = [float(v) for v in vals if not isinstance(v, bool)]
        return float(np.mean(vals)) if vals else float("nan")
    val = r.epoch_kpis[-1].get(metric)
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else float("nan")


@dataclass
class ArmAggregate:
    arm_index: int
    label: str
    overrides: dict
    n_replicates: int
    n_extinct: int
    survival_prob: float
    extinction_rate: float
    mean_extinction_tick: Optional[float]
    epochs: list[int]
    ticks: list[float]
    survival_curve: list[float]
    metrics: dict[str, dict]     # key -> {mean:[...], ci_low:[...], ci_high:[...], q_low, q_high, boot_low, boot_high}
    final: dict[str, dict]       # key -> _mean_ci(+bootstrap)

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate_arm(arm_index: int, label: str, overrides: dict,
                  replicates: list[ReplicateResult], spec: StudySpec,
                  rng: np.random.Generator) -> ArmAggregate:
    n = len(replicates)
    n_ext = sum(1 for r in replicates if r.extinct)
    ext_ticks = [r.extinction_tick for r in replicates if r.extinct and r.extinction_tick is not None]
    max_len = max((len(r.epoch_kpis) for r in replicates), default=0)

    # epoch/tick axis from the longest replicate
    epochs: list[int] = []
    ticks: list[float] = []
    for j in range(max_len):
        epoch_val, tick_val = j, float("nan")
        for r in replicates:
            if j < len(r.epoch_kpis):
                epoch_val = int(r.epoch_kpis[j].get("epoch", j))
                tick_val = float(r.epoch_kpis[j].get("tick", float("nan")))
                break
        epochs.append(epoch_val)
        ticks.append(tick_val)

    survival_curve = [
        float(np.mean([1.0 if len(r.epoch_kpis) > j else 0.0 for r in replicates])) if n else 0.0
        for j in range(max_len)
    ]

    keys = _numeric_metric_keys(replicates)
    metrics: dict[str, dict] = {}
    for key in keys:
        fill0 = key == "alive_count"
        mat = _series_matrix(replicates, key, max_len, fill_extinct_zero=fill0)
        mean = np.full(max_len, np.nan)
        ci_low = np.full(max_len, np.nan)
        ci_high = np.full(max_len, np.nan)
        q_low = np.full(max_len, np.nan)
        q_high = np.full(max_len, np.nan)
        for j in range(max_len):
            col = mat[:, j]
            stats = _mean_ci(col)
            mean[j] = stats["mean"]
            ci_low[j] = stats["ci_low"]
            ci_high[j] = stats["ci_high"]
            valid = col[~np.isnan(col)]
            if valid.size:
                q_low[j] = float(np.quantile(valid, spec.quantiles[0]))
                q_high[j] = float(np.quantile(valid, spec.quantiles[1]))
        metrics[key] = {
            "mean": mean.tolist(),
            "ci_low": ci_low.tolist(),
            "ci_high": ci_high.tolist(),
            "q_low": q_low.tolist(),
            "q_high": q_high.tolist(),
        }

    # final / steady-state summary per key metric (+ bootstrap)
    final: dict[str, dict] = {}
    for key in keys:
        scalars = np.array([_replicate_scalar(r, key, spec.burn_in_epochs) for r in replicates], dtype=float)
        summary = _mean_ci(scalars)
        if spec.bootstrap:
            bl, bh = _bootstrap_ci(scalars, spec.n_bootstrap, rng)
            summary["boot_low"] = bl
            summary["boot_high"] = bh
        final[key] = summary

    return ArmAggregate(
        arm_index=arm_index,
        label=label,
        overrides=dict(overrides),
        n_replicates=n,
        n_extinct=n_ext,
        survival_prob=1.0 - (n_ext / n) if n else 0.0,
        extinction_rate=(n_ext / n) if n else 0.0,
        mean_extinction_tick=float(np.mean(ext_ticks)) if ext_ticks else None,
        epochs=epochs,
        ticks=ticks,
        survival_curve=survival_curve,
        metrics=metrics,
        final=final,
    )


@dataclass
class Comparison:
    metric: str
    arm_a: str
    arm_b: str
    mean_a: float
    mean_b: float
    diff: float
    cohens_d: float
    p_permutation: float
    welch: Optional[dict]
    significant: bool
    n_a: int
    n_b: int
    sample_size_hint: Optional[int]

    def to_dict(self) -> dict:
        return asdict(self)


def compare_arms(arm_a: ArmAggregate, reps_a: list[ReplicateResult],
                 arm_b: ArmAggregate, reps_b: list[ReplicateResult],
                 spec: StudySpec, rng: np.random.Generator) -> Comparison:
    metric = spec.compare_metric
    a = np.array([_replicate_scalar(r, metric, spec.burn_in_epochs) for r in reps_a], dtype=float)
    b = np.array([_replicate_scalar(r, metric, spec.burn_in_epochs) for r in reps_b], dtype=float)
    d = cohens_d(a, b)
    p = permutation_pvalue(a, b, min(spec.n_bootstrap, 5000), rng)
    av = a[~np.isnan(a)]
    bv = b[~np.isnan(b)]
    return Comparison(
        metric=metric,
        arm_a=arm_a.label,
        arm_b=arm_b.label,
        mean_a=float(np.mean(av)) if av.size else float("nan"),
        mean_b=float(np.mean(bv)) if bv.size else float("nan"),
        diff=(float(np.mean(av)) - float(np.mean(bv))) if av.size and bv.size else float("nan"),
        cohens_d=d,
        p_permutation=p,
        welch=welch_ttest(a, b),
        significant=bool(not math.isnan(p) and p < 0.05),
        n_a=int(av.size),
        n_b=int(bv.size),
        sample_size_hint=sample_size_hint(d),
    )


@dataclass
class StudyResult:
    spec: StudySpec
    arms: list[ArmAggregate]
    comparisons: list[Comparison]
    replicates: list[ReplicateResult]
    elapsed_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Study:
    def __init__(self, spec: StudySpec):
        errors = spec.validate()
        if errors:
            raise ValueError("Invalid study spec:\n" + "\n".join(f"  - {e}" for e in errors))
        self.spec = spec
        if spec.random_base_seed:
            import secrets
            spec.base_seed = secrets.randbelow(2**31 - 1) or 1

    @property
    def total_replicates(self) -> int:
        return len(self.spec.arms) * self.spec.replicates_per_arm

    def build_jobs(self, start_index: int = 0, end_ckpt_dir: Optional[Path] = None) -> list[dict]:
        """start_index lets extend-study append replicates beyond existing ones."""
        jobs: list[dict] = []
        for arm_index, arm in enumerate(self.spec.arms):
            for rep in range(start_index, start_index + self.spec.replicates_per_arm):
                jobs.append({
                    "origin_path": str(self.spec.origin_path),
                    "origin_kind": self.spec.origin_kind,
                    "arm_index": arm_index,
                    "arm_label": arm.label,
                    "overrides": arm.overrides,
                    "seed": replicate_seed(self.spec.base_seed, arm_index, rep),
                    "replicate_index": rep,
                    "max_epochs": self.spec.max_epochs,
                    "save_end_checkpoint": self.spec.save_end_checkpoints,
                    "end_ckpt_dir": str(end_ckpt_dir) if end_ckpt_dir else None,
                })
        return jobs

    def run(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[Event] = None,
        end_ckpt_dir: Optional[Path] = None,
        start_index: int = 0,
        epoch_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> StudyResult:
        start = time.time()
        started_at = datetime.now().isoformat(timespec="seconds")
        jobs = self.build_jobs(start_index=start_index, end_ckpt_dir=end_ckpt_dir)
        workers = self.spec.resolved_workers()
        use_pool = workers > 1 and len(jobs) > 1
        if use_pool:
            raw = self._run_parallel(jobs, workers, progress_callback, epoch_callback, cancel_event)
        else:
            raw = self._run_sequential(jobs, progress_callback, epoch_callback, cancel_event)
        replicates = [ReplicateResult.from_dict(d) for d in raw]
        replicates.sort(key=lambda r: (r.arm_index, r.replicate_index))
        result = self._aggregate(replicates)
        result.elapsed_seconds = time.time() - start
        result.started_at = started_at
        result.finished_at = datetime.now().isoformat(timespec="seconds")
        return result

    def _run_sequential(self, jobs, progress_callback, epoch_callback, cancel_event) -> list[dict]:
        out = []
        total_units = max(sum(int(j["max_epochs"]) for j in jobs), 1)
        per_job: dict[tuple, int] = {}

        class _Shim:
            def put(_self, msg) -> None:
                if not msg or msg[0] != "epoch":
                    return
                _, ai, ri, al, cur, jmax = msg
                per_job[(ai, ri)] = cur
                if epoch_callback:
                    epoch_callback(sum(per_job.values()), total_units,
                                   f"{al}: rep {ri} epoch {cur}/{jmax}")

        shim = _Shim()
        for i, job in enumerate(jobs):
            if cancel_event is not None and cancel_event.is_set():
                break
            out.append(run_replicate_job(job, shim))
            per_job[(job["arm_index"], job["replicate_index"])] = int(job["max_epochs"])
            if progress_callback:
                progress_callback(i + 1, len(jobs))
        return out

    def _run_parallel(self, jobs, workers, progress_callback, epoch_callback, cancel_event) -> list[dict]:
        out: list[dict] = []
        done = 0
        total_units = max(sum(int(j["max_epochs"]) for j in jobs), 1)
        per_job: dict[tuple, int] = {}
        manager = mp.Manager()
        pq = manager.Queue()
        stop = threading.Event()

        def _emit(label: str) -> None:
            if epoch_callback:
                epoch_callback(sum(per_job.values()), total_units, label)

        def _drain() -> None:
            while not stop.is_set():
                try:
                    msg = pq.get(timeout=0.2)
                except Exception:
                    continue
                if msg is None or msg[0] != "epoch":
                    continue
                _, ai, ri, al, cur, jmax = msg
                per_job[(ai, ri)] = cur
                _emit(f"{al}: rep {ri} epoch {cur}/{jmax}")

        drainer = threading.Thread(target=_drain, daemon=True)
        drainer.start()
        ex = ProcessPoolExecutor(max_workers=workers, initializer=_ensure_v3_on_path)
        cancelled = False
        try:
            futures = {ex.submit(run_replicate_job, job, pq): job for job in jobs}
            pending = set(futures)
            while pending:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                # Poll so a cancel is honored within ~0.2s instead of only
                # after an in-flight replicate finishes.
                finished, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                for fut in finished:
                    job = futures[fut]
                    try:
                        out.append(fut.result())
                    except Exception:
                        continue  # worker died/killed; ignore
                    done += 1
                    # Snap the finished replicate to full so the bar can't stall
                    # below 100% if its last epoch message is still in flight.
                    per_job[(job["arm_index"], job["replicate_index"])] = int(job["max_epochs"])
                    _emit(f"{job['arm_label']}: rep {job['replicate_index']} done")
                    if progress_callback:
                        progress_callback(done, len(jobs))
        finally:
            stop.set()
            try:
                pq.put(None)
            except Exception:
                pass
            drainer.join(timeout=1.0)
            if cancelled:
                _terminate_pool(ex)          # kill running workers immediately
                ex.shutdown(wait=False)
            else:
                ex.shutdown(wait=True)
            try:
                manager.shutdown()
            except Exception:
                pass
        return out

    def _aggregate(self, replicates: list[ReplicateResult]) -> StudyResult:
        rng = np.random.default_rng(self.spec.base_seed)
        by_arm: dict[int, list[ReplicateResult]] = {}
        for r in replicates:
            by_arm.setdefault(r.arm_index, []).append(r)

        arms: list[ArmAggregate] = []
        for arm_index, arm in enumerate(self.spec.arms):
            reps = sorted(by_arm.get(arm_index, []), key=lambda r: r.replicate_index)
            arms.append(aggregate_arm(arm_index, arm.label, arm.overrides, reps, self.spec, rng))

        comparisons: list[Comparison] = []
        if len(arms) >= 2:
            base = arms[0]
            base_reps = sorted(by_arm.get(0, []), key=lambda r: r.replicate_index)
            for j in range(1, len(arms)):
                other_reps = sorted(by_arm.get(j, []), key=lambda r: r.replicate_index)
                comparisons.append(
                    compare_arms(arms[j], other_reps, base, base_reps, self.spec, rng)
                )
        return StudyResult(
            spec=self.spec,
            arms=arms,
            comparisons=comparisons,
            replicates=replicates,
        )


# ---------------------------------------------------------------------------
# Output tree + export
# ---------------------------------------------------------------------------

def study_output_dir(spec: StudySpec, when: Optional[datetime] = None) -> Path:
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    leaf = f"{stamp}_{_slug(spec.name)}"
    if spec.origin_kind == "checkpoint":
        return Path(spec.origin_path) / "studies" / leaf
    return Path("runs") / "studies" / leaf


def export_results(result: StudyResult, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["replicates_jsonl"] = _write_replicates_jsonl(result, out / "replicates.jsonl")
    paths["replicates_csv"] = _write_replicates_csv(result, out / "replicates.csv")
    paths["arm_summary"] = _write_arm_summary(result, out / "arm_summary.csv")
    paths["comparison"] = _write_comparison(result, out / "comparison.json")
    paths["manifest"] = _write_manifest(result, out / "manifest.json")
    report_json, report_txt = build_report(result)
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)
    (out / "report.txt").write_text(report_txt, encoding="utf-8")
    paths["report_json"] = out / "report.json"
    paths["report_txt"] = out / "report.txt"
    # plots (matplotlib if available, else numpy PNG writer)
    try:
        from src import plots
        written = plots.save_study_plots(result, out / "plots")
        if written:
            paths["plots_dir"] = out / "plots"
    except Exception:
        pass
    # per-arm detail
    for arm in result.arms:
        arm_dir = out / "arms" / _slug(arm.label)
        arm_dir.mkdir(parents=True, exist_ok=True)
        reps = [r for r in result.replicates if r.arm_index == arm.arm_index]
        _write_replicate_rows(reps, arm_dir / "replicates.csv")
    return paths


def _write_replicates_jsonl(result: StudyResult, path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in result.replicates:
            f.write(json.dumps(r.to_dict()) + "\n")
    return path


def _flat_rows(result: StudyResult) -> tuple[list[str], list[dict]]:
    rows: list[dict] = []
    kpi_cols: list[str] = []
    for r in result.replicates:
        for kpi in r.epoch_kpis:
            if not kpi_cols:
                kpi_cols = list(kpi.keys())
            row = {"arm": r.arm_label, "replicate": r.replicate_index, "seed": r.seed}
            row.update(kpi)
            rows.append(row)
    fieldnames = ["arm", "replicate", "seed"] + kpi_cols
    return fieldnames, rows


def _write_replicates_csv(result: StudyResult, path: Path) -> Path:
    fieldnames, rows = _flat_rows(result)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def _write_replicate_rows(reps: list[ReplicateResult], path: Path) -> Path:
    kpi_cols: list[str] = []
    for r in reps:
        if r.epoch_kpis:
            kpi_cols = list(r.epoch_kpis[0].keys())
            break
    fieldnames = ["arm", "replicate", "seed"] + kpi_cols
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in reps:
            for kpi in r.epoch_kpis:
                row = {"arm": r.arm_label, "replicate": r.replicate_index, "seed": r.seed}
                row.update(kpi)
                w.writerow(row)
    return path


def _write_arm_summary(result: StudyResult, path: Path) -> Path:
    keys = list(result.arms[0].metrics.keys()) if result.arms else []
    fieldnames = ["arm", "epoch", "tick", "survival"]
    for k in keys:
        fieldnames += [f"{k}_mean", f"{k}_ci_low", f"{k}_ci_high", f"{k}_q_low", f"{k}_q_high"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for arm in result.arms:
            for j, epoch in enumerate(arm.epochs):
                row = {
                    "arm": arm.label,
                    "epoch": epoch,
                    "tick": arm.ticks[j],
                    "survival": arm.survival_curve[j],
                }
                for k in keys:
                    m = arm.metrics[k]
                    row[f"{k}_mean"] = m["mean"][j]
                    row[f"{k}_ci_low"] = m["ci_low"][j]
                    row[f"{k}_ci_high"] = m["ci_high"][j]
                    row[f"{k}_q_low"] = m["q_low"][j]
                    row[f"{k}_q_high"] = m["q_high"][j]
                w.writerow(row)
    return path


def _write_comparison(result: StudyResult, path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in result.comparisons], f, indent=2, ensure_ascii=False)
    return path


def _write_manifest(result: StudyResult, path: Path) -> Path:
    spec = result.spec
    origin = Path(spec.origin_path)
    manifest = {
        "study_name": spec.name,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "origin": {
            "kind": spec.origin_kind,
            "path": str(origin),
            "hash": _hash_dir(origin) if spec.origin_kind == "checkpoint" and origin.is_dir() else None,
        },
        "base_seed": spec.base_seed,
        "replicates_per_arm": spec.replicates_per_arm,
        "max_epochs": spec.max_epochs,
        "burn_in_epochs": spec.burn_in_epochs,
        "compare_metric": spec.compare_metric,
        "workers": spec.resolved_workers(),
        "spec": spec.to_dict(),
        "arms": [
            {
                "label": arm.label,
                "overrides": arm.overrides,
                "seeds": [
                    replicate_seed(spec.base_seed, arm.arm_index, rep)
                    for rep in range(spec.replicates_per_arm)
                ],
            }
            for arm in result.arms
        ],
        "errors": [
            {"arm": r.arm_label, "replicate": r.replicate_index, "seed": r.seed, "error": r.error}
            for r in result.replicates if r.error
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Scientific report
# ---------------------------------------------------------------------------

def build_report(result: StudyResult) -> tuple[dict, str]:
    spec = result.spec
    arms_out = []
    for arm in result.arms:
        kpis = {}
        for key in KEY_KPIS:
            if key in arm.final:
                s = arm.final[key]
                kpis[key] = {
                    "mean": round(s["mean"], 4) if not math.isnan(s["mean"]) else None,
                    "ci_low": round(s["ci_low"], 4) if not math.isnan(s["ci_low"]) else None,
                    "ci_high": round(s["ci_high"], 4) if not math.isnan(s["ci_high"]) else None,
                    "n": s["n"],
                }
        arms_out.append({
            "label": arm.label,
            "overrides": arm.overrides,
            "replicates": arm.n_replicates,
            "survival_prob": round(arm.survival_prob, 4),
            "extinction_rate": round(arm.extinction_rate, 4),
            "mean_extinction_tick": (round(arm.mean_extinction_tick, 1)
                                     if arm.mean_extinction_tick is not None else None),
            "final_kpis": kpis,
        })

    comparisons_out = []
    for c in result.comparisons:
        better = c.arm_a if c.diff > 0 else c.arm_b
        comparisons_out.append({
            "metric": c.metric,
            "arm_a": c.arm_a,
            "arm_b": c.arm_b,
            "mean_a": round(c.mean_a, 4) if not math.isnan(c.mean_a) else None,
            "mean_b": round(c.mean_b, 4) if not math.isnan(c.mean_b) else None,
            "difference": round(c.diff, 4) if not math.isnan(c.diff) else None,
            "cohens_d": round(c.cohens_d, 3) if not math.isnan(c.cohens_d) else None,
            "p_permutation": round(c.p_permutation, 4) if not math.isnan(c.p_permutation) else None,
            "welch": c.welch,
            "significant_at_0.05": c.significant,
            "higher_arm": better if not math.isnan(c.diff) else None,
            "replicates_needed_for_power_0.8": c.sample_size_hint,
        })

    verdict = _verdict_text(result)
    report = {
        "study": spec.name,
        "origin": {"kind": spec.origin_kind, "path": str(spec.origin_path)},
        "generated_at": result.finished_at,
        "compare_metric": spec.compare_metric,
        "burn_in_epochs": spec.burn_in_epochs,
        "arms": arms_out,
        "comparisons": comparisons_out,
        "verdict": verdict,
    }
    return report, _report_text(report)


def _verdict_text(result: StudyResult) -> str:
    if not result.comparisons:
        if result.arms:
            a = result.arms[0]
            return (f"Single arm '{a.label}': survival {a.survival_prob:.0%}, "
                    f"final {result.spec.compare_metric} "
                    f"{a.final.get(result.spec.compare_metric, {}).get('mean', float('nan')):.3f}.")
        return "No arms."
    c = result.comparisons[0]
    if math.isnan(c.p_permutation):
        return "Not enough data for a comparison."
    metric = c.metric
    if c.significant:
        better = c.arm_a if c.diff > 0 else c.arm_b
        return (f"'{better}' had significantly higher {metric} "
                f"(diff {c.diff:+.3f}, d={c.cohens_d:.2f}, p={c.p_permutation:.3f}).")
    hint = f" ~{c.sample_size_hint} replicates/arm would be needed for 80% power." if c.sample_size_hint else ""
    return (f"No significant difference in {metric} "
            f"(diff {c.diff:+.3f}, d={c.cohens_d:.2f}, p={c.p_permutation:.3f}).{hint}")


def _report_text(report: dict) -> str:
    lines = [
        f"Study: {report['study']}",
        f"Origin: {report['origin']['kind']} {report['origin']['path']}",
        f"Generated: {report['generated_at']}",
        f"Compare metric: {report['compare_metric']} (burn-in {report['burn_in_epochs']} epochs)",
        "",
        "Arms:",
    ]
    for a in report["arms"]:
        lines.append(f"  - {a['label']}  (n={a['replicates']}, "
                     f"survival {a['survival_prob']:.0%}, extinct {a['extinction_rate']:.0%})")
        for key, s in a["final_kpis"].items():
            if s["mean"] is not None:
                lines.append(f"      {key}: {s['mean']} [{s['ci_low']}, {s['ci_high']}] (n={s['n']})")
    if report["comparisons"]:
        lines.append("")
        lines.append("Comparisons (vs first arm):")
        for c in report["comparisons"]:
            lines.append(
                f"  - {c['arm_a']} vs {c['arm_b']} on {c['metric']}: "
                f"diff {c['difference']}, d={c['cohens_d']}, p={c['p_permutation']} "
                f"{'(significant)' if c['significant_at_0.05'] else '(n.s.)'}"
            )
    lines.append("")
    lines.append(f"Verdict: {report['verdict']}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Reload / merge (extend-study, cluster merge)
# ---------------------------------------------------------------------------

def load_replicate_results(path: str | Path) -> list[ReplicateResult]:
    """Load replicates from a study dir (replicates.jsonl) or a dir of *.json."""
    p = Path(path)
    reps: list[ReplicateResult] = []
    jsonl = p / "replicates.jsonl" if p.is_dir() else p
    if jsonl.is_file() and jsonl.suffix == ".jsonl":
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                reps.append(ReplicateResult.from_dict(json.loads(line)))
        return reps
    if p.is_dir():
        for f in sorted(p.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and "arm_index" in data and "seed" in data:
                reps.append(ReplicateResult.from_dict(data))
    return reps


def aggregate_existing(spec: StudySpec, replicates: list[ReplicateResult]) -> StudyResult:
    study = Study(spec)
    replicates = sorted(replicates, key=lambda r: (r.arm_index, r.replicate_index))
    result = study._aggregate(replicates)
    result.finished_at = datetime.now().isoformat(timespec="seconds")
    return result
