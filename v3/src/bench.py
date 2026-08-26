"""Tick-rate benchmarks for default / medium / large worlds."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src import kernels
from src.config import get_default_config
from src.engine import SimulationEngine


@dataclass(frozen=True)
class BenchCase:
    name: str
    width: int
    height: int
    n_animals: int
    max_animals: int
    timed_ticks: int
    warmup_ticks: int = 20


CASES = (
    BenchCase("default", 80, 80, 80, 800, 400),
    BenchCase("medium", 200, 200, 400, 4000, 150),
    BenchCase("large", 500, 500, 1000, 10000, 60),
)


def _config_for(case: BenchCase, backend: str, seed: int = 42):
    cfg = get_default_config()
    cfg.world.width = case.width
    cfg.world.height = case.height
    cfg.world.seed = seed
    cfg.population.initial_count = case.n_animals
    cfg.perf.max_animals = case.max_animals
    cfg.perf.backend = backend
    cfg.viz.snapshot_every_epoch = False
    cfg.metrics.interval = 10_000_000
    return cfg


def measure_case(case: BenchCase, backend: str) -> dict:
    resolved = kernels.resolve_backend(backend)
    kernels.warmup()
    engine = SimulationEngine(_config_for(case, backend))
    engine.initialize()
    for _ in range(case.warmup_ticks):
        engine.tick()
    t0 = time.perf_counter()
    for _ in range(case.timed_ticks):
        engine.tick()
    elapsed = time.perf_counter() - t0
    ticks_s = case.timed_ticks / elapsed if elapsed > 0 else 0.0
    return {
        "name": case.name,
        "width": case.width,
        "height": case.height,
        "n_animals": case.n_animals,
        "requested_backend": backend,
        "resolved_backend": resolved,
        "alive": engine.world.n,
        "timed_ticks": case.timed_ticks,
        "elapsed_s": elapsed,
        "ticks_per_s": ticks_s,
    }


def run_bench(backend: str = "numba", cases: tuple[BenchCase, ...] | None = None) -> list[dict]:
    return [measure_case(case, backend) for case in (cases or CASES)]


def format_results(rows: list[dict]) -> str:
    lines = [
        f"backend requested={rows[0]['requested_backend']}  "
        f"resolved={rows[0]['resolved_backend']}",
        f"{'world':<10} {'grid':<11} {'n':>5} {'alive':>6} {'ticks':>6} {'sec':>8} {'ticks/s':>10}",
    ]
    for r in rows:
        grid = f"{r['width']}x{r['height']}"
        lines.append(
            f"{r['name']:<10} {grid:<11} {r['n_animals']:5d} {r['alive']:6d} "
            f"{r['timed_ticks']:6d} {r['elapsed_s']:8.3f} {r['ticks_per_s']:10.1f}"
        )
    return "\n".join(lines)
