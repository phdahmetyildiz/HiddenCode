"""CLI for Evolution Simulator v3: budget, run, watch, sweep, bench, cluster jobs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Allow `python main.py` from the v3 folder
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import SimConfig, get_default_config, load_config
from src.engine import SimulationEngine
from src.livability import evaluate
from src.logging_io import RunManager
from src.sweep import (
    ParameterSweep,
    SweepSettings,
    collect_job_result_files,
    load_job_from_bundle,
    merge_job_results,
    run_single_job,
)


def _load(path: str | None) -> SimConfig:
    if path is None:
        default = Path(__file__).resolve().parent / "config" / "default_config.json"
        if default.exists():
            return load_config(default)
        return get_default_config()
    return load_config(path)


def cmd_budget(args: argparse.Namespace) -> None:
    config = _load(args.config)
    print(evaluate(config).as_text())


def cmd_run(args: argparse.Namespace) -> None:
    config = _load(args.config)
    if args.seed is not None:
        config.world.seed = args.seed
    report = evaluate(config)
    print(report.as_text())
    if report.warns and args.strict_livability:
        raise SystemExit("Livability warnings with --strict-livability; aborting.")

    engine = SimulationEngine(config)
    engine.initialize()
    out_base = args.output_dir or config.viz.output_dir
    run = RunManager(config, base_dir=out_base)

    def on_epoch(metrics, _eng):
        run.log_epoch(metrics)
        if config.viz.snapshot_every_epoch:
            run.save_snapshot(engine.world, metrics.epoch)
        print(
            f"  epoch {metrics.epoch:4d}  tick {metrics.tick:6d}  "
            f"alive {metrics.alive_count:5d}  avgE {metrics.avg_energy:.3f}  "
            f"births {metrics.births_count}  "
            f"d.em {metrics.deaths_emergency} d.st {metrics.deaths_starvation} "
            f"d.age {metrics.deaths_max_age}"
        )

    engine.on_epoch = on_epoch
    result = engine.run(max_ticks=args.max_ticks, max_epochs=args.max_epochs)
    run.finalize(
        {
            "total_ticks": result.total_ticks,
            "total_epochs": result.total_epochs,
            "final_alive": result.final_alive,
            "extinct": result.extinct,
            "extinction_tick": result.extinction_tick,
        }
    )
    print(f"Done. alive={result.final_alive} extinct={result.extinct}")
    print(f"Output: {run.run_dir}")


def cmd_sweep(args: argparse.Namespace) -> None:
    sweep_path = Path(args.sweep_config) if args.sweep_config else (
        Path(__file__).resolve().parent / "config" / "sweep_template.json"
    )
    settings = SweepSettings.from_file(sweep_path)
    if args.workers is not None:
        settings.parallel_workers = args.workers
    base = _load(args.config)
    sweep = ParameterSweep(settings, base_config=base)
    print(
        f"Sweep: {sweep.total_combinations} combinations × "
        f"{settings.runs_per_set} seeds = {sweep.total_runs} runs "
        f"(workers={settings.parallel_workers}, max_epochs={settings.max_epochs})"
    )

    def progress(done: int, total: int) -> None:
        print(f"  {done}/{total} jobs", flush=True)

    result = sweep.run(parallel=settings.parallel_workers > 1, progress_callback=progress)
    out = Path(args.output_dir) if args.output_dir else Path(base.viz.output_dir) / "sweeps"
    dest = out / datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = sweep.export_results(result, dest)
    best = result.best_stable_combination()
    print(f"Elapsed {result.elapsed_seconds:.1f}s")
    if best is None:
        print("No combination had a non-zero stability rate.")
    else:
        print(
            f"Best stable combo #{best.combination_id}: {best.params}  "
            f"stability={best.stability_rate:.2f} survival={best.survival_rate:.2f} "
            f"avg_alive={best.avg_final_alive:.1f}"
        )
    print(f"Output: {dest}")
    for kind, path in paths.items():
        print(f"  {kind}: {path.name}")


def cmd_bench(args: argparse.Namespace) -> None:
    from src.bench import CASES, BenchCase, format_results, run_bench

    cases = CASES
    if args.quick:
        cases = tuple(
            BenchCase(c.name, c.width, c.height, c.n_animals, c.max_animals, max(8, c.timed_ticks // 10), 5)
            for c in CASES
        )
    rows = run_bench(backend=args.backend, cases=cases)
    print(format_results(rows))


def cmd_export_jobs(args: argparse.Namespace) -> None:
    sweep_path = Path(args.sweep_config) if args.sweep_config else (
        Path(__file__).resolve().parent / "config" / "sweep_template.json"
    )
    settings = SweepSettings.from_file(sweep_path)
    base = _load(args.config)
    sweep = ParameterSweep(settings, base_config=base)
    dest = Path(args.output_dir)
    paths = sweep.export_job_bundle(dest)
    print(
        f"Exported {sweep.total_runs} jobs "
        f"({sweep.total_combinations} combinations × {settings.runs_per_set} seeds)"
    )
    print(f"Bundle: {dest.resolve()}")
    for kind, path in paths.items():
        print(f"  {kind}: {path.name}")
    print("Run one job with:")
    print(f"  python main.py run-job --jobs-dir {dest} --index 0 --out {dest / 'results' / 'job_000000.json'}")


def cmd_run_job(args: argparse.Namespace) -> None:
    import json

    job = load_job_from_bundle(args.jobs_dir, args.index)
    result = run_single_job(job)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    print(
        f"job {args.index}: combo={result.combination_id} seed={result.seed} "
        f"alive={result.final_alive_count} extinct={result.extinct} -> {out}"
    )


def cmd_merge_sweep(args: argparse.Namespace) -> None:
    jobs_dir = Path(args.jobs_dir)
    settings_path = Path(args.sweep_config) if args.sweep_config else jobs_dir / "sweep_settings.json"
    settings = SweepSettings.from_file(settings_path)
    files = collect_job_result_files(args.results_dir)
    if not files:
        raise SystemExit(f"No job result JSON files in {args.results_dir}")
    dest = Path(args.output_dir)
    result = merge_job_results(files, settings, dest)
    print(f"Merged {result.total_runs} runs / {result.total_combinations} combinations")
    best = result.best_stable_combination()
    if best is None:
        print("No combination had a non-zero stability rate.")
    else:
        print(
            f"Best stable combo #{best.combination_id}: {best.params}  "
            f"stability={best.stability_rate:.2f} survival={best.survival_rate:.2f} "
            f"avg_alive={best.avg_final_alive:.1f}"
        )
    print(f"Output: {dest}")


def cmd_watch(args: argparse.Namespace) -> None:
    config = _load(args.config)
    if args.seed is not None:
        config.world.seed = args.seed
    from src.watch import run_watch

    run_watch(config)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evolution Simulator v3")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("budget", help="Print energy/foraging livability budget")
    b.add_argument("--config", default=None)
    b.set_defaults(func=cmd_budget)

    r = sub.add_parser("run", help="Headless run")
    r.add_argument("--config", default=None)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--max-ticks", type=int, default=None)
    r.add_argument("--max-epochs", type=int, default=10)
    r.add_argument("--output-dir", default=None)
    r.add_argument("--strict-livability", action="store_true")
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("watch", help="Live grid window")
    w.add_argument("--config", default=None)
    w.add_argument("--seed", type=int, default=None)
    w.set_defaults(func=cmd_watch)

    s = sub.add_parser("sweep", help="Parameter sweep (local process pool)")
    s.add_argument("--sweep-config", default=None, help="Sweep JSON (default: config/sweep_template.json)")
    s.add_argument("--config", default=None, help="Base sim config JSON")
    s.add_argument("--output-dir", default=None)
    s.add_argument("--workers", type=int, default=None)
    s.set_defaults(func=cmd_sweep)

    n = sub.add_parser("bench", help="Measure ticks/second on default, medium, and large worlds")
    n.add_argument("--backend", default="numba", help="numpy | numba | cuda")
    n.add_argument("--quick", action="store_true", help="Fewer ticks (smoke timing)")
    n.set_defaults(func=cmd_bench)

    e = sub.add_parser("export-jobs", help="Write a cluster job bundle (config + jobs.jsonl)")
    e.add_argument("--sweep-config", default=None)
    e.add_argument("--config", default=None)
    e.add_argument("--output-dir", required=True)
    e.set_defaults(func=cmd_export_jobs)

    j = sub.add_parser("run-job", help="Run one exported sweep job by index")
    j.add_argument("--jobs-dir", required=True)
    j.add_argument("--index", type=int, required=True)
    j.add_argument("--out", required=True, help="JSON result path")
    j.set_defaults(func=cmd_run_job)

    m = sub.add_parser("merge-sweep", help="Merge run-job JSON files into sweep CSVs")
    m.add_argument("--jobs-dir", required=True, help="Bundle from export-jobs")
    m.add_argument("--results-dir", required=True, help="Directory of job_*.json results")
    m.add_argument("--output-dir", required=True)
    m.add_argument("--sweep-config", default=None, help="Override jobs-dir/sweep_settings.json")
    m.set_defaults(func=cmd_merge_sweep)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
