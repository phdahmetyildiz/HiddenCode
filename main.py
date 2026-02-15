"""
Evolution Simulator — CLI Entry Point

Usage:
    python main.py --mode single --config config/default_config.json
    python main.py --mode sweep --config config/sweep_template.json
    python main.py --ui
"""

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolution Simulator — Test stress-induced mutagenesis in evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ui                                         Launch Streamlit UI
  python main.py --mode single --config config/default_config.json   Run single simulation
  python main.py --mode sweep --config config/sweep_template.json    Run parameter sweep
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["single", "sweep"],
        default=None,
        help="Run mode: 'single' for one simulation, 'sweep' for parameter sweep",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to JSON config file (default: config/default_config.json)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Streamlit web UI (ignores --mode and --config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed (overrides config value)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless mode (no visualization)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override max generations (for single mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory",
    )

    return parser.parse_args()


def launch_ui() -> None:
    """Launch the Streamlit web UI."""
    import subprocess
    ui_path = Path(__file__).parent / "src" / "ui" / "app.py"
    if not ui_path.exists():
        print(f"Error: UI app not found at {ui_path}")
        sys.exit(1)
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run", str(ui_path),
            "--server.port=8501",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
        check=True,
    )


def run_single(config_path: str, seed_override: int | None = None,
               headless: bool = False, max_generations: int | None = None,
               output_dir: str | None = None) -> None:
    """Run a single simulation."""
    from src.core.config import load_config
    from src.simulation.engine import SimulationEngine
    from src.simulation.generation import GenerationStats
    from src.simulation.metrics import MetricsCollector
    from src.logging.run_manager import RunManager

    config = load_config(config_path)

    if seed_override is not None:
        config.world.seed = seed_override
    if headless:
        config.viz.mode = "headless"
    if max_generations is not None:
        config.sweep.max_generations = max_generations

    out_dir = output_dir or config.viz.output_dir
    gen_limit = config.sweep.max_generations

    print(f"[Evolution Simulator] Single run")
    print(f"  Config: {config_path}")
    print(f"  Grid: {config.world.width}x{config.world.height}")
    print(f"  Population: {config.population.initial_count}")
    print(f"  Seed: {config.world.seed}")
    print(f"  Max Generations: {gen_limit}")
    print(f"  Output: {out_dir}")
    print()

    # Initialize
    engine = SimulationEngine(config, seed=config.world.seed)
    engine.initialize()
    metrics = MetricsCollector(config)
    run_manager = RunManager(config, base_dir=out_dir)

    last_gen = -1
    start_time = time.time()

    def on_generation(gen_number: int, eng: SimulationEngine) -> None:
        nonlocal last_gen
        last_gen = gen_number
        gen_stats = (eng.generation_manager.all_gen_stats[-1]
                     if eng.generation_manager.all_gen_stats else GenerationStats())
        tick_totals = eng.get_accumulated_stats()
        kpis = metrics.collect(eng.world, gen_stats, tick_totals)
        eng.reset_accumulated_stats()
        run_manager.log_generation(kpis)

        pop = kpis.get("alive_count", 0)
        energy = kpis.get("avg_energy", 0)
        print(f"  Gen {gen_number:4d} | Pop: {pop:6d} | Avg Energy: {energy:.4f}")

    engine.on_generation = on_generation

    # Run
    result = engine.run(max_generations=gen_limit)
    elapsed = time.time() - start_time

    print()
    print(f"[Result]")
    print(f"  Ticks: {result.total_ticks}")
    print(f"  Generations: {result.total_generations}")
    print(f"  Final population: {result.final_alive_count}")
    print(f"  Extinct: {result.extinct}")
    print(f"  Elapsed: {elapsed:.1f}s")

    summary = {
        "total_ticks": result.total_ticks,
        "total_generations": result.total_generations,
        "final_alive": result.final_alive_count,
        "extinct": result.extinct,
        "elapsed_seconds": round(elapsed, 2),
        "seed": config.world.seed,
    }
    run_manager.finalize(summary)
    print(f"  Output saved to: {run_manager.run_dir}")


def run_sweep(config_path: str, output_dir: str | None = None) -> None:
    """Run a parameter sweep."""
    from src.core.config import get_default_config
    from src.simulation.sweep import ParameterSweep, SweepSettings

    print(f"[Evolution Simulator] Parameter sweep")
    print(f"  Config: {config_path}")
    print()

    settings = SweepSettings.from_file(config_path)
    base_config = get_default_config()
    sweep = ParameterSweep(settings, base_config=base_config)

    total = len(sweep.combinations) * settings.runs_per_set
    print(f"  Combinations: {len(sweep.combinations)}")
    print(f"  Runs per set: {settings.runs_per_set}")
    print(f"  Total simulations: {total}")
    print(f"  Max generations: {settings.max_generations}")
    print()

    start_time = time.time()

    def progress_cb(done: int, total_sims: int) -> None:
        pct = done / total_sims * 100 if total_sims > 0 else 100
        print(f"\r  Progress: {done}/{total_sims} ({pct:.0f}%)", end="", flush=True)

    result = sweep.run(parallel=True, progress_callback=progress_cb)
    elapsed = time.time() - start_time
    print()

    # Export results
    out_dir = Path(output_dir or "runs") / f"sweep_{int(time.time())}"
    exported = sweep.export_results(result, out_dir)

    print()
    print(f"[Sweep Results]")
    print(f"  Combinations tested: {result.total_combinations}")
    print(f"  Total runs: {result.total_runs}")
    print(f"  Elapsed: {elapsed:.1f}s")

    best = result.best_stable_combination()
    if best:
        print(f"  Best stable combination: {best.params}")
        print(f"    Stability rate: {best.stability_rate:.0%}")
        print(f"    Survival rate: {best.survival_rate:.0%}")
        print(f"    Avg final alive: {best.avg_final_alive:.0f}")
    else:
        print(f"  No stable combination found.")

    print(f"  Results exported to: {out_dir}")


def main() -> None:
    args = parse_args()

    if args.ui:
        launch_ui()
        return

    if args.mode is None:
        print("Error: Specify --mode (single|sweep) or --ui to launch the web interface.")
        print("Run with --help for usage information.")
        sys.exit(1)

    if args.mode == "single":
        run_single(
            args.config,
            seed_override=args.seed,
            headless=args.headless,
            max_generations=args.generations,
            output_dir=args.output,
        )
    elif args.mode == "sweep":
        run_sweep(args.config, output_dir=args.output)


if __name__ == "__main__":
    main()
