"""
Sweep Mode page for the Evolution Simulator UI.

Allows users to:
  - Define fixed vs variable parameters
  - Add multiple values per variable parameter
  - Set sweep settings (runs, generations, stability band)
  - Launch sweep with progress tracking
  - Display results summary and comparison charts
"""

import json
import time
from pathlib import Path
from copy import deepcopy

import streamlit as st
import pandas as pd

from src.core.config import SimConfig, get_default_config
from src.simulation.sweep import ParameterSweep, SweepSettings, SweepResult


# ---------------------------------------------------------------------------
# Common parameter options for the UI
# ---------------------------------------------------------------------------

SWEEPABLE_PARAMS = {
    "world.width": {"label": "Grid Width", "type": "int", "default": 500},
    "world.height": {"label": "Grid Height", "type": "int", "default": 500},
    "population.initial_count": {"label": "Initial Population", "type": "int", "default": 200},
    "genetics.dna_length": {"label": "DNA Length", "type": "int", "default": 2048},
    "genetics.base_mutation_rate": {"label": "Base Mutation Rate", "type": "float", "default": 0.01},
    "genetics.stress_mutation_rate": {"label": "Stress Mutation Rate", "type": "float", "default": 0.20},
    "energy.base_metabolism": {"label": "Base Metabolism", "type": "float", "default": 0.001},
    "energy.k_weight_speed": {"label": "k_weight_speed", "type": "float", "default": 0.01},
    "energy.food_gain": {"label": "Food Energy Gain", "type": "float", "default": 0.2},
    "energy.max_pitfall_loss_pct": {"label": "Max Pitfall Loss %", "type": "float", "default": 0.5},
    "resources.food_rate": {"label": "Food Spawn Rate", "type": "float", "default": 5.0},
    "resources.food_lifespan": {"label": "Food Lifespan", "type": "int", "default": 50},
    "resources.pitfall_rate": {"label": "Pitfall Spawn Rate", "type": "float", "default": 2.0},
    "resources.pitfall_lifespan": {"label": "Pitfall Lifespan", "type": "int", "default": 100},
    "generation.gen_length": {"label": "Generation Length", "type": "int", "default": 1000},
    "generation.repro_checkpoint_pct": {"label": "Repro Checkpoint %", "type": "float", "default": 0.70},
    "generation.survival_threshold": {"label": "Survival Threshold", "type": "float", "default": 0.50},
    "properties.eyesight_radius": {"label": "Eyesight Radius", "type": "int", "default": 10},
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults = {
        "sweep_fixed_params": {},
        "sweep_variable_params": {},
        "sweep_running": False,
        "sweep_result": None,
        "sweep_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_sweep_mode() -> None:
    """Render the parameter sweep page."""
    _init_session_state()
    st.title("🔄 Parameter Sweep Mode")

    st.markdown("""
    Define **fixed** parameters (single value) and **variable** parameters (multiple values)
    to systematically test combinations and find stable baseline configurations.
    """)

    # --- Sweep settings ---
    st.subheader("⚙️ Sweep Settings")
    set_col1, set_col2, set_col3, set_col4 = st.columns(4)
    with set_col1:
        runs_per_set = st.number_input("Runs per Set", min_value=1, max_value=100,
                                        value=9, step=1, key="sw_runs")
    with set_col2:
        max_generations = st.number_input("Max Generations", min_value=1, max_value=10000,
                                           value=99, step=10, key="sw_maxgen")
    with set_col3:
        base_seed = st.number_input("Base Seed", min_value=0, max_value=999999999,
                                     value=42, step=1, key="sw_seed")
    with set_col4:
        parallel_workers = st.number_input("Parallel Workers", min_value=1, max_value=32,
                                            value=4, step=1, key="sw_workers")

    # Stability settings
    st.markdown("**Stability Band**")
    stab_col1, stab_col2, stab_col3 = st.columns(3)
    with stab_col1:
        min_pop_pct = st.number_input("Min Population % of initial", min_value=0.0, max_value=1.0,
                                       value=0.20, step=0.05, format="%.2f", key="sw_minpop")
    with stab_col2:
        max_pop_pct = st.number_input("Max Population % of initial", min_value=1.0, max_value=100.0,
                                       value=5.00, step=0.5, format="%.2f", key="sw_maxpop")
    with stab_col3:
        check_after = st.number_input("Check stability after gen", min_value=1, max_value=1000,
                                       value=10, step=1, key="sw_checkafter")

    st.markdown("---")

    # --- Fixed Parameters ---
    st.subheader("📌 Fixed Parameters")
    st.markdown("These will have the **same** value across all combinations.")

    fixed_params = st.session_state.sweep_fixed_params

    fix_col1, fix_col2, fix_col3 = st.columns([2, 2, 1])
    with fix_col1:
        fix_param = st.selectbox("Parameter", options=list(SWEEPABLE_PARAMS.keys()),
                                  key="fix_param_sel")
    with fix_col2:
        pinfo = SWEEPABLE_PARAMS[fix_param]
        if pinfo["type"] == "int":
            fix_value = st.number_input("Value", value=pinfo["default"], step=1, key="fix_val")
        else:
            fix_value = st.number_input("Value", value=pinfo["default"], step=0.01,
                                         format="%.4f", key="fix_val")
    with fix_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add Fixed", key="fix_add"):
            fixed_params[fix_param] = fix_value
            st.session_state.sweep_fixed_params = fixed_params
            st.rerun()

    if fixed_params:
        for k, v in list(fixed_params.items()):
            fc1, fc2, fc3 = st.columns([2, 2, 1])
            fc1.write(f"`{k}`")
            fc2.write(f"{v}")
            if fc3.button("🗑️", key=f"fix_del_{k}"):
                del fixed_params[k]
                st.session_state.sweep_fixed_params = fixed_params
                st.rerun()
    else:
        st.info("No fixed parameters set. Defaults from config will be used.")

    st.markdown("---")

    # --- Variable Parameters ---
    st.subheader("🔀 Variable Parameters")
    st.markdown("These will be **varied** — each combination of values will be tested.")

    variable_params = st.session_state.sweep_variable_params

    var_col1, var_col2, var_col3 = st.columns([2, 3, 1])
    with var_col1:
        var_param = st.selectbox("Parameter", options=list(SWEEPABLE_PARAMS.keys()),
                                  key="var_param_sel")
    with var_col2:
        var_values_str = st.text_input(
            "Values (comma-separated)",
            value="",
            placeholder="e.g. 500, 1000, 2500",
            key="var_vals",
        )
    with var_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add Variable", key="var_add"):
            if var_values_str.strip():
                pinfo = SWEEPABLE_PARAMS[var_param]
                try:
                    if pinfo["type"] == "int":
                        values = [int(v.strip()) for v in var_values_str.split(",")]
                    else:
                        values = [float(v.strip()) for v in var_values_str.split(",")]
                    variable_params[var_param] = values
                    st.session_state.sweep_variable_params = variable_params
                    st.rerun()
                except ValueError:
                    st.error("Invalid values. Use comma-separated numbers.")

    if variable_params:
        for k, vals in list(variable_params.items()):
            vc1, vc2, vc3 = st.columns([2, 3, 1])
            vc1.write(f"`{k}`")
            vc2.write(f"{vals}")
            if vc3.button("🗑️", key=f"var_del_{k}"):
                del variable_params[k]
                st.session_state.sweep_variable_params = variable_params
                st.rerun()
    else:
        st.info("No variable parameters set. Add at least one to run a sweep.")

    # --- Combination preview ---
    if variable_params:
        from src.simulation.sweep import generate_combinations
        combos = generate_combinations(variable_params)
        st.markdown(f"**{len(combos)} combinations** × {runs_per_set} runs = **{len(combos) * runs_per_set} total simulations**")

        with st.expander("Preview combinations"):
            for i, combo in enumerate(combos[:20]):
                st.write(f"#{i+1}: {combo}")
            if len(combos) > 20:
                st.write(f"... and {len(combos) - 20} more")

    st.markdown("---")

    # --- Load / Save sweep config ---
    lcol, scol = st.columns(2)
    with lcol:
        uploaded = st.file_uploader("Load sweep config JSON", type=["json"], key="sw_upload")
        if uploaded is not None:
            try:
                data = json.loads(uploaded.read().decode("utf-8"))
                st.session_state.sweep_fixed_params = data.get("fixed_params", {})
                vp = data.get("variable_params", {})
                st.session_state.sweep_variable_params = vp
                ss = data.get("sweep_settings", {})
                st.success("Sweep config loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load: {e}")

    with scol:
        if st.button("💾 Save Sweep Config", key="sw_save"):
            sweep_data = {
                "fixed_params": dict(fixed_params),
                "variable_params": dict(variable_params),
                "sweep_settings": {
                    "runs_per_set": runs_per_set,
                    "max_generations": max_generations,
                    "base_seed": base_seed,
                    "stability_band": {
                        "min_population_pct": min_pop_pct,
                        "max_population_pct": max_pop_pct,
                        "check_after_generation": check_after,
                    },
                    "early_termination_on_extinction": True,
                    "parallel_workers": parallel_workers,
                },
            }
            save_path = Path("config/sweep_ui_config.json")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(sweep_data, f, indent=2)
            st.success(f"Saved to {save_path}")

    st.markdown("---")

    # --- Run sweep ---
    run_col1, run_col2 = st.columns([1, 3])
    with run_col1:
        can_run = bool(variable_params) and not st.session_state.sweep_running
        run_btn = st.button(
            "🚀 Run Sweep",
            disabled=not can_run,
            key="sw_run",
        )

    if run_btn:
        _run_sweep(
            fixed_params=dict(fixed_params),
            variable_params=dict(variable_params),
            runs_per_set=runs_per_set,
            max_generations=max_generations,
            base_seed=base_seed,
            min_pop_pct=min_pop_pct,
            max_pop_pct=max_pop_pct,
            check_after=check_after,
            parallel_workers=parallel_workers,
        )

    # --- Display results ---
    if st.session_state.sweep_result is not None:
        _display_sweep_results(st.session_state.sweep_result)


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------

def _run_sweep(
    fixed_params: dict,
    variable_params: dict,
    runs_per_set: int,
    max_generations: int,
    base_seed: int,
    min_pop_pct: float,
    max_pop_pct: float,
    check_after: int,
    parallel_workers: int,
) -> None:
    """Execute the parameter sweep."""

    st.session_state.sweep_running = True
    st.session_state.sweep_result = None

    settings = SweepSettings(
        fixed_params=fixed_params,
        variable_params=variable_params,
        runs_per_set=runs_per_set,
        max_generations=max_generations,
        base_seed=base_seed,
        stability_band_min_pct=min_pop_pct,
        stability_band_max_pct=max_pop_pct,
        check_after_generation=check_after,
        early_termination_on_extinction=True,
        parallel_workers=parallel_workers,
    )

    base_config = st.session_state.get("config", get_default_config())
    sweep = ParameterSweep(settings, base_config=deepcopy(base_config))

    total_sims = len(sweep.combinations) * runs_per_set
    progress_bar = st.progress(0.0, text=f"Running sweep: 0/{total_sims} simulations...")
    status = st.empty()
    start_time = time.time()

    completed = [0]

    def progress_cb(done: int, total: int) -> None:
        completed[0] = done
        pct = done / total if total > 0 else 1.0
        progress_bar.progress(pct, text=f"Running sweep: {done}/{total} simulations...")

    try:
        # Run sequentially in the Streamlit process to avoid pickling issues
        result = sweep.run(parallel=False, progress_callback=progress_cb)
    except Exception as e:
        status.error(f"Sweep failed: {e}")
        st.session_state.sweep_running = False
        return

    elapsed = time.time() - start_time
    progress_bar.progress(1.0, text="✅ Sweep complete!")
    status.success(f"Sweep finished in {elapsed:.1f}s — {total_sims} simulations across {len(sweep.combinations)} combinations.")

    # Export results
    out_dir = Path("runs") / f"sweep_{int(time.time())}"
    try:
        sweep.export_results(result, out_dir)
        st.info(f"Results exported to `{out_dir}`")
    except Exception as e:
        st.warning(f"Export failed: {e}")

    st.session_state.sweep_result = result
    st.session_state.sweep_running = False


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _display_sweep_results(result: SweepResult) -> None:
    """Display the sweep results."""
    st.markdown("---")
    st.subheader("📊 Sweep Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Combinations", result.total_combinations)
    col2.metric("Total Runs", result.total_runs)
    col3.metric("Time", f"{result.elapsed_seconds:.1f}s")

    best = result.best_stable_combination()
    if best:
        col4.metric("Best Stability", f"{best.stability_rate:.0%}")
    else:
        col4.metric("Best Stability", "N/A")

    # Best combination
    if best:
        st.success(f"🏆 **Best stable combination:** {best.params}")
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Stability Rate", f"{best.stability_rate:.0%}")
        bc2.metric("Survival Rate", f"{best.survival_rate:.0%}")
        bc3.metric("Avg Final Population", f"{best.avg_final_alive:.0f}")
        bc4.metric("Avg Generations", f"{best.avg_generations:.1f}")
    else:
        st.warning("No stable combination found. Try adjusting parameters or increasing generations.")

    # Comparison table
    st.markdown("### Combination Comparison")
    rows = []
    for combo in result.combinations:
        row = {
            "ID": combo.combination_id,
            **combo.params,
            "Total Runs": combo.total_runs,
            "Extinction Count": combo.extinction_count,
            "Survival Rate": f"{combo.survival_rate:.0%}",
            "Stable Count": combo.stable_count,
            "Stability Rate": f"{combo.stability_rate:.0%}",
            "Avg Final Alive": f"{combo.avg_final_alive:.0f}",
            "Avg Generations": f"{combo.avg_generations:.1f}",
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Summary CSV",
            data=csv,
            file_name="sweep_summary.csv",
            mime="text/csv",
            key="sw_dl_csv",
        )

    # Bar charts
    if result.combinations:
        st.markdown("### Stability & Survival Rates")
        chart_data = pd.DataFrame({
            "Combination": [str(c.params) for c in result.combinations],
            "Stability Rate": [c.stability_rate for c in result.combinations],
            "Survival Rate": [c.survival_rate for c in result.combinations],
        })
        chart_data = chart_data.set_index("Combination")
        st.bar_chart(chart_data, use_container_width=True)

        st.markdown("### Average Final Population")
        pop_data = pd.DataFrame({
            "Combination": [str(c.params) for c in result.combinations],
            "Avg Final Alive": [c.avg_final_alive for c in result.combinations],
        })
        pop_data = pop_data.set_index("Combination")
        st.bar_chart(pop_data, use_container_width=True)
