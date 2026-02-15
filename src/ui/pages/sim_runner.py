"""
Single Run page for the Evolution Simulator UI.

Allows users to:
  - Start/stop a single simulation run
  - See live generation progress and KPIs
  - Manually trigger/deactivate stress events
  - View KPI evolution charts
"""

import time
from pathlib import Path
from copy import deepcopy

import streamlit as st
import pandas as pd

from src.core.config import SimConfig, get_default_config
from src.simulation.engine import SimulationEngine, TickStats
from src.simulation.generation import GenerationStats
from src.simulation.metrics import MetricsCollector
from src.logging.run_manager import RunManager


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialize session state for the sim runner."""
    defaults = {
        "sim_running": False,
        "sim_engine": None,
        "sim_metrics": None,
        "sim_run_manager": None,
        "sim_generation_data": [],  # list of KPI dicts
        "sim_log": [],              # text log entries
        "sim_finished": False,
        "sim_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_sim_runner() -> None:
    """Render the single simulation runner page."""
    _init_session_state()
    st.title("▶️ Single Simulation Run")

    # --- Configuration source ---
    config = st.session_state.get("config", get_default_config())
    config = deepcopy(config)  # work on a copy

    st.markdown("Using configuration from **Config Editor** (or defaults).")

    # --- Run parameters ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        max_generations = st.number_input(
            "Max Generations", min_value=1, max_value=100000,
            value=config.sweep.max_generations, step=10, key="sr_max_gen",
        )
    with col2:
        seed = st.number_input(
            "Seed", min_value=0, max_value=999999999,
            value=config.world.seed, step=1, key="sr_seed",
        )
    with col3:
        save_snapshots = st.checkbox("Save snapshots", value=True, key="sr_snap")
    with col4:
        output_dir = st.text_input("Output dir", value=config.viz.output_dir, key="sr_outdir")

    st.markdown("---")

    # --- Control buttons ---
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        start_btn = st.button(
            "🚀 Start Simulation",
            disabled=st.session_state.sim_running,
            key="sr_start",
        )
    with btn_col2:
        stress_btn = st.button(
            "⚡ Toggle Stress",
            disabled=not st.session_state.sim_running,
            key="sr_stress",
        )
    with btn_col3:
        reset_btn = st.button("🔄 Reset", key="sr_reset")

    if reset_btn:
        for k in ["sim_running", "sim_engine", "sim_metrics", "sim_run_manager",
                   "sim_generation_data", "sim_log", "sim_finished", "sim_result"]:
            st.session_state[k] = None if k in ("sim_engine", "sim_metrics", "sim_run_manager", "sim_result") else ([] if "data" in k or "log" in k else False)
        st.rerun()

    # --- Start simulation ---
    if start_btn:
        _run_simulation(config, max_generations, seed, save_snapshots, output_dir)

    # --- Display results ---
    if st.session_state.sim_generation_data:
        _display_results()

    if st.session_state.sim_finished and st.session_state.sim_result:
        _display_summary(st.session_state.sim_result)


# ---------------------------------------------------------------------------
# Simulation execution
# ---------------------------------------------------------------------------

def _run_simulation(
    config: SimConfig,
    max_generations: int,
    seed: int,
    save_snapshots: bool,
    output_dir: str,
) -> None:
    """Run the full simulation with live progress display."""

    st.session_state.sim_running = True
    st.session_state.sim_finished = False
    st.session_state.sim_generation_data = []
    st.session_state.sim_log = []

    config.world.seed = seed

    # Create engine and metrics collector
    engine = SimulationEngine(config, seed=seed)
    engine.initialize()
    metrics = MetricsCollector(config)
    run_manager = RunManager(config, base_dir=output_dir) if output_dir else None

    st.session_state.sim_engine = engine
    st.session_state.sim_metrics = metrics
    st.session_state.sim_run_manager = run_manager

    # Progress containers
    progress_bar = st.progress(0.0, text="Starting simulation...")
    status_container = st.empty()
    metrics_container = st.container()

    # KPI placeholders for live updating
    kpi_cols = st.columns(5)
    kpi_pop = kpi_cols[0].empty()
    kpi_gen = kpi_cols[1].empty()
    kpi_energy = kpi_cols[2].empty()
    kpi_food = kpi_cols[3].empty()
    kpi_deaths = kpi_cols[4].empty()

    chart_placeholder = st.empty()

    gen_data_list = []
    start_time = time.time()
    current_gen = 0

    # Set up generation callback to collect KPIs
    def on_generation(gen_number: int, eng: SimulationEngine) -> None:
        nonlocal current_gen
        current_gen = gen_number

    engine.on_generation = on_generation

    # Run tick by tick, checking generation boundaries
    last_gen_collected = -1
    total_ticks = 0

    try:
        while True:
            # Tick
            tick_stats = engine.tick()
            total_ticks += 1

            # Check if a generation boundary was crossed
            completed_gen = engine.generation_manager.total_generations_completed
            if completed_gen > last_gen_collected:
                last_gen_collected = completed_gen

                gen_stats = (engine.generation_manager.all_gen_stats[-1]
                             if engine.generation_manager.all_gen_stats else GenerationStats())
                tick_totals = engine.get_accumulated_stats()
                kpis = metrics.collect(engine.world, gen_stats, tick_totals)
                engine.reset_accumulated_stats()

                gen_data_list.append(kpis)
                st.session_state.sim_generation_data = gen_data_list

                # Log to run manager
                if run_manager:
                    run_manager.log_generation(kpis)
                    if save_snapshots:
                        run_manager.save_snapshot(engine.world, completed_gen)

                # Update progress
                pct = min(completed_gen / max_generations, 1.0) if max_generations > 0 else 0
                progress_bar.progress(pct, text=f"Generation {completed_gen}/{max_generations}")

                # Update live KPIs
                kpi_pop.metric("🐾 Population", kpis.get("alive_count", 0))
                kpi_gen.metric("🔄 Generation", completed_gen)
                kpi_energy.metric("⚡ Avg Energy", f"{kpis.get('avg_energy', 0):.3f}")
                kpi_food.metric("🍎 Food Eaten", kpis.get("food_eaten", 0))
                kpi_deaths.metric("💀 Deaths", kpis.get("deaths_total", 0))

                # Update chart every generation
                if gen_data_list:
                    df = pd.DataFrame(gen_data_list)
                    if "alive_count" in df.columns:
                        chart_placeholder.line_chart(
                            df[["alive_count"]].rename(columns={"alive_count": "Population"}),
                            use_container_width=True,
                        )

            # Stop conditions
            if completed_gen >= max_generations:
                break
            if engine.is_extinct:
                status_container.warning(f"⚠️ Population extinct at tick {engine.current_tick}!")
                break

    except Exception as e:
        status_container.error(f"Simulation error: {e}")
        st.session_state.sim_running = False
        return

    elapsed = time.time() - start_time
    progress_bar.progress(1.0, text="✅ Simulation complete!")

    result = {
        "total_ticks": total_ticks,
        "total_generations": completed_gen,
        "final_alive": engine.alive_count,
        "extinct": engine.is_extinct,
        "elapsed_seconds": round(elapsed, 2),
        "seed": seed,
    }

    st.session_state.sim_result = result
    st.session_state.sim_running = False
    st.session_state.sim_finished = True

    if run_manager:
        run_manager.finalize(result)

    status_container.success(
        f"✅ Done: {completed_gen} generations, {total_ticks} ticks in {elapsed:.1f}s. "
        f"Final population: {engine.alive_count}."
    )


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _display_results() -> None:
    """Display generation KPI data as charts and table."""
    gen_data = st.session_state.sim_generation_data
    if not gen_data:
        return

    st.markdown("---")
    st.subheader("📈 Generation KPIs")

    df = pd.DataFrame(gen_data)
    if df.empty:
        return

    # Chart selection
    available_kpis = [c for c in df.columns if c != "generation"]

    tab_pop, tab_energy, tab_food, tab_deaths, tab_genetic, tab_all = st.tabs([
        "Population", "Energy", "Food & Pitfalls", "Deaths", "Genetics", "All KPIs"
    ])

    with tab_pop:
        pop_cols = [c for c in ["alive_count", "births_total", "deaths_total",
                                "pop_at_primary_repro", "pop_at_survival_check"] if c in df.columns]
        if pop_cols:
            st.line_chart(df[pop_cols], use_container_width=True)

    with tab_energy:
        energy_cols = [c for c in ["avg_energy", "std_energy", "min_energy", "max_energy",
                                    "median_energy"] if c in df.columns]
        if energy_cols:
            st.line_chart(df[energy_cols], use_container_width=True)

    with tab_food:
        food_cols = [c for c in ["food_eaten", "food_spawned", "food_expired",
                                  "food_available", "pitfall_encounters",
                                  "pitfall_zero_damage"] if c in df.columns]
        if food_cols:
            st.line_chart(df[food_cols], use_container_width=True)

    with tab_deaths:
        death_cols = [c for c in ["deaths_starvation", "deaths_emergency",
                                   "deaths_pitfall", "deaths_age", "deaths_total"] if c in df.columns]
        if death_cols:
            st.line_chart(df[death_cols], use_container_width=True)

    with tab_genetic:
        gen_cols = [c for c in ["avg_weight", "avg_speed", "avg_defense_ones",
                                "genetic_diversity", "unique_defense_seqs"] if c in df.columns]
        if gen_cols:
            st.line_chart(df[gen_cols], use_container_width=True)

    with tab_all:
        selected_kpis = st.multiselect(
            "Select KPIs to plot", options=available_kpis,
            default=["alive_count"] if "alive_count" in available_kpis else [],
            key="sr_kpi_sel",
        )
        if selected_kpis:
            st.line_chart(df[selected_kpis], use_container_width=True)

    # Raw data table
    with st.expander("📋 Raw Data Table"):
        st.dataframe(df, use_container_width=True)

    # CSV download
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV",
        data=csv,
        file_name="simulation_kpis.csv",
        mime="text/csv",
        key="sr_dl_csv",
    )


def _display_summary(result: dict) -> None:
    """Display the final run summary."""
    st.markdown("---")
    st.subheader("📋 Run Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ticks", result["total_ticks"])
    col2.metric("Generations", result["total_generations"])
    col3.metric("Final Population", result["final_alive"])
    col4.metric("Elapsed Time", f"{result['elapsed_seconds']}s")

    if result["extinct"]:
        st.error("💀 Population went extinct!")
    else:
        st.success("🎉 Population survived!")
