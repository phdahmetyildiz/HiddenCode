"""
Config Editor page for the Evolution Simulator UI.

Allows users to:
  - Load a config from JSON file
  - Edit all parameters grouped by section
  - Validate in real-time
  - Save to file
  - Load presets (small, medium, large)
"""

import json
from pathlib import Path

import streamlit as st

from src.core.config import (
    SimConfig,
    load_config,
    save_config,
    get_default_config,
)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "Default": {},
    "Small Test (10×10, 20 animals)": {
        "world.width": 10, "world.height": 10,
        "population.initial_count": 20,
        "generation.gen_length": 100,
        "resources.food_rate": 3.0,
    },
    "Medium (100×100, 200 animals)": {
        "world.width": 100, "world.height": 100,
        "population.initial_count": 200,
        "generation.gen_length": 500,
        "resources.food_rate": 5.0,
    },
    "Large (500×500, 2000 animals)": {
        "world.width": 500, "world.height": 500,
        "population.initial_count": 2000,
        "generation.gen_length": 1000,
        "resources.food_rate": 10.0,
    },
}


def render_config_editor() -> None:
    """Render the configuration editor page."""
    st.title("⚙️ Configuration Editor")

    # Initialize session config
    if "config" not in st.session_state:
        st.session_state.config = get_default_config()

    config: SimConfig = st.session_state.config

    # --- Top bar: Load / Save / Presets ---
    col_load, col_save, col_preset = st.columns([1, 1, 1])

    with col_load:
        uploaded = st.file_uploader("Load config JSON", type=["json"], key="config_upload")
        if uploaded is not None:
            try:
                data = json.loads(uploaded.read().decode("utf-8"))
                st.session_state.config = SimConfig.from_dict(data)
                config = st.session_state.config
                st.success("Config loaded!")
            except Exception as e:
                st.error(f"Failed to load config: {e}")

    with col_save:
        save_path = st.text_input("Save path", value="config/my_config.json", key="save_path")
        if st.button("💾 Save Config", key="save_btn"):
            try:
                save_config(config, save_path)
                st.success(f"Saved to {save_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    with col_preset:
        preset_name = st.selectbox("Load preset", options=list(PRESETS.keys()), key="preset_sel")
        if st.button("📋 Apply Preset", key="preset_btn"):
            cfg = get_default_config()
            overrides = PRESETS[preset_name]
            from src.core.config import apply_param_override
            for k, v in overrides.items():
                apply_param_override(cfg, k, v)
            st.session_state.config = cfg
            config = cfg
            st.success(f"Applied preset: {preset_name}")
            st.rerun()

    st.markdown("---")

    # --- Validation status ---
    errors = config.validate()
    if errors:
        st.error("⚠️ Configuration has validation errors:")
        for err in errors:
            st.markdown(f"- `{err}`")
    else:
        st.success("✅ Configuration is valid")

    # --- Tabbed sections ---
    tabs = st.tabs([
        "🌍 World",
        "🧬 Genetics",
        "🏋️ Properties",
        "⚡ Energy",
        "🍎 Resources",
        "👥 Population",
        "🔄 Generation",
        "⚡ Stress",
        "📊 Viz / Output",
    ])

    with tabs[0]:
        _render_world_section(config)
    with tabs[1]:
        _render_genetics_section(config)
    with tabs[2]:
        _render_properties_section(config)
    with tabs[3]:
        _render_energy_section(config)
    with tabs[4]:
        _render_resources_section(config)
    with tabs[5]:
        _render_population_section(config)
    with tabs[6]:
        _render_generation_section(config)
    with tabs[7]:
        _render_stress_section(config)
    with tabs[8]:
        _render_viz_section(config)

    # --- Raw JSON view ---
    with st.expander("📄 Raw JSON"):
        st.json(config.to_dict())


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_world_section(config: SimConfig) -> None:
    st.subheader("World Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        config.world.width = st.number_input(
            "Grid Width", min_value=10, max_value=10000,
            value=config.world.width, step=10, key="world_width",
        )
    with col2:
        config.world.height = st.number_input(
            "Grid Height", min_value=10, max_value=10000,
            value=config.world.height, step=10, key="world_height",
        )
    with col3:
        config.world.seed = st.number_input(
            "Random Seed", min_value=0, max_value=999999999,
            value=config.world.seed, step=1, key="world_seed",
        )


def _render_genetics_section(config: SimConfig) -> None:
    st.subheader("Genetics / DNA")
    col1, col2 = st.columns(2)
    with col1:
        config.genetics.dna_length = st.number_input(
            "DNA Length (bits)", min_value=64, max_value=65536,
            value=config.genetics.dna_length, step=64, key="gen_dna_len",
        )
        config.genetics.encoding = st.selectbox(
            "Encoding", options=["binary", "gray"],
            index=0 if config.genetics.encoding == "binary" else 1,
            key="gen_encoding",
        )
    with col2:
        config.genetics.base_mutation_rate = st.number_input(
            "Base Mutation Rate", min_value=0.0, max_value=1.0,
            value=config.genetics.base_mutation_rate, step=0.001,
            format="%.4f", key="gen_base_mut",
        )
        config.genetics.stress_mutation_rate = st.number_input(
            "Stress Mutation Rate", min_value=0.0, max_value=1.0,
            value=config.genetics.stress_mutation_rate, step=0.01,
            format="%.4f", key="gen_stress_mut",
        )
        config.genetics.stress_mode_coding_only = st.checkbox(
            "Stress mode: coding regions only",
            value=config.genetics.stress_mode_coding_only,
            key="gen_stress_coding",
        )

    st.markdown("**Bit Ranges** (start, end — exclusive)")
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        w_start = st.number_input("Weight start", value=config.genetics.weight_bits[0], key="wb_s")
        w_end = st.number_input("Weight end", value=config.genetics.weight_bits[1], key="wb_e")
        config.genetics.weight_bits = [int(w_start), int(w_end)]
    with bc2:
        s_start = st.number_input("Speed start", value=config.genetics.speed_bits[0], key="sb_s")
        s_end = st.number_input("Speed end", value=config.genetics.speed_bits[1], key="sb_e")
        config.genetics.speed_bits = [int(s_start), int(s_end)]
    with bc3:
        d_start = st.number_input("Defense start", value=config.genetics.defense_bits[0], key="db_s")
        d_end = st.number_input("Defense end", value=config.genetics.defense_bits[1], key="db_e")
        config.genetics.defense_bits = [int(d_start), int(d_end)]


def _render_properties_section(config: SimConfig) -> None:
    st.subheader("Animal Properties")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Weight Limits**")
        wl = config.properties.weight_limits
        wl_low = st.number_input("Min Weight", value=wl[0], step=0.1, format="%.2f", key="wl_low")
        wl_high = st.number_input("Max Weight", value=wl[1], step=0.1, format="%.2f", key="wl_high")
        config.properties.weight_limits = [wl_low, wl_high]

    with col2:
        st.markdown("**Speed Limits**")
        sl = config.properties.speed_limits
        sl_low = st.number_input("Min Speed", value=sl[0], step=0.1, format="%.2f", key="sl_low")
        sl_high = st.number_input("Max Speed", value=sl[1], step=0.1, format="%.2f", key="sl_high")
        config.properties.speed_limits = [sl_low, sl_high]

    config.properties.eyesight_radius = st.number_input(
        "Eyesight Radius", min_value=1, max_value=100,
        value=config.properties.eyesight_radius, step=1, key="eye_r",
    )


def _render_energy_section(config: SimConfig) -> None:
    st.subheader("Energy Mechanics")
    col1, col2 = st.columns(2)
    with col1:
        config.energy.base_metabolism = st.number_input(
            "Base Metabolism (per tick)", min_value=0.0, max_value=1.0,
            value=config.energy.base_metabolism, step=0.0001,
            format="%.5f", key="e_metab",
        )
        config.energy.k_weight_speed = st.number_input(
            "k_weight_speed", min_value=0.0, max_value=1.0,
            value=config.energy.k_weight_speed, step=0.001,
            format="%.4f", key="e_kws",
        )
        config.energy.food_gain = st.number_input(
            "Food Energy Gain", min_value=0.01, max_value=1.0,
            value=config.energy.food_gain, step=0.01,
            format="%.3f", key="e_foodgain",
        )
    with col2:
        config.energy.max_pitfall_loss_pct = st.number_input(
            "Max Pitfall Loss %", min_value=0.0, max_value=1.0,
            value=config.energy.max_pitfall_loss_pct, step=0.05,
            format="%.2f", key="e_maxpit",
        )
        config.energy.low_energy_death_threshold = st.number_input(
            "Low Energy Death Threshold", min_value=0.0, max_value=1.0,
            value=config.energy.low_energy_death_threshold, step=0.01,
            format="%.3f", key="e_lowdeath",
        )
        config.energy.defense_cost_enabled = st.checkbox(
            "Defense cost enabled",
            value=config.energy.defense_cost_enabled, key="e_defcost",
        )
        if config.energy.defense_cost_enabled:
            config.energy.k_defense_cost = st.number_input(
                "k_defense_cost", min_value=0.0, max_value=0.1,
                value=config.energy.k_defense_cost, step=0.0001,
                format="%.5f", key="e_kdc",
            )


def _render_resources_section(config: SimConfig) -> None:
    st.subheader("Resources (Food & Pitfalls)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Food**")
        config.resources.food_rate = st.number_input(
            "Food Spawn Rate (per tick)", min_value=0.0, max_value=100.0,
            value=config.resources.food_rate, step=0.5, key="r_foodrate",
        )
        config.resources.food_lifespan = st.number_input(
            "Food Lifespan (ticks)", min_value=1, max_value=10000,
            value=config.resources.food_lifespan, step=10, key="r_foodlife",
        )
    with col2:
        st.markdown("**Pitfalls**")
        config.resources.pitfall_rate = st.number_input(
            "Pitfall Spawn Rate (per tick)", min_value=0.0, max_value=100.0,
            value=config.resources.pitfall_rate, step=0.5, key="r_pitrate",
        )
        config.resources.pitfall_lifespan = st.number_input(
            "Pitfall Lifespan (ticks)", min_value=1, max_value=10000,
            value=config.resources.pitfall_lifespan, step=10, key="r_pitlife",
        )

    st.markdown("**Initial Pitfall Types**")
    for i, pt in enumerate(config.resources.initial_pitfall_types):
        c1, c2 = st.columns([1, 3])
        with c1:
            pt["name"] = st.text_input(f"Type {i+1} Name", value=pt.get("name", "A"), key=f"pt_name_{i}")
        with c2:
            pt["sequence"] = st.text_input(
                f"Type {i+1} Sequence (32 bits)",
                value=pt.get("sequence", "1" * 32), key=f"pt_seq_{i}",
            )


def _render_population_section(config: SimConfig) -> None:
    st.subheader("Population")
    config.population.initial_count = st.number_input(
        "Initial Animal Count", min_value=2, max_value=1000000,
        value=config.population.initial_count, step=10, key="pop_init",
    )


def _render_generation_section(config: SimConfig) -> None:
    st.subheader("Generation Lifecycle")
    col1, col2 = st.columns(2)
    with col1:
        config.generation.gen_length = st.number_input(
            "Generation Length (ticks)", min_value=10, max_value=100000,
            value=config.generation.gen_length, step=100, key="g_len",
        )
        config.generation.repro_checkpoint_pct = st.number_input(
            "Primary Repro Checkpoint %", min_value=0.01, max_value=1.0,
            value=config.generation.repro_checkpoint_pct, step=0.05,
            format="%.2f", key="g_repro",
        )
        config.generation.survival_check_pct = st.number_input(
            "Survival Check %", min_value=0.01, max_value=2.0,
            value=config.generation.survival_check_pct, step=0.05,
            format="%.2f", key="g_surv",
        )
        config.generation.bonus_repro_pct = st.number_input(
            "Bonus Repro %", min_value=0.01, max_value=3.0,
            value=config.generation.bonus_repro_pct, step=0.05,
            format="%.2f", key="g_bonus",
        )
    with col2:
        config.generation.survival_threshold = st.number_input(
            "Survival Energy Threshold", min_value=0.0, max_value=1.0,
            value=config.generation.survival_threshold, step=0.05,
            format="%.2f", key="g_surv_thresh",
        )
        config.generation.repro_energy_low = st.number_input(
            "Repro Energy Low (0 offspring below)", min_value=0.0, max_value=1.0,
            value=config.generation.repro_energy_low, step=0.05,
            format="%.2f", key="g_repro_low",
        )
        config.generation.repro_energy_high = st.number_input(
            "Repro Energy High (2 offspring above)", min_value=0.0, max_value=1.0,
            value=config.generation.repro_energy_high, step=0.05,
            format="%.2f", key="g_repro_high",
        )


def _render_stress_section(config: SimConfig) -> None:
    st.subheader("Stress Events")
    col1, col2 = st.columns(2)
    with col1:
        use_auto = st.checkbox(
            "Auto-trigger stress",
            value=config.stress.trigger_tick is not None, key="s_auto",
        )
        if use_auto:
            config.stress.trigger_tick = st.number_input(
                "Trigger at tick", min_value=0, max_value=1000000,
                value=config.stress.trigger_tick or 0, step=100, key="s_tick",
            )
        else:
            config.stress.trigger_tick = None

        use_duration = st.checkbox(
            "Auto-deactivate after duration",
            value=config.stress.duration_ticks is not None, key="s_dur_en",
        )
        if use_duration:
            config.stress.duration_ticks = st.number_input(
                "Duration (ticks)", min_value=1, max_value=1000000,
                value=config.stress.duration_ticks or 100, step=100, key="s_dur",
            )
        else:
            config.stress.duration_ticks = None

    with col2:
        config.stress.pitfall_burst_count = st.number_input(
            "Pitfall Burst Count", min_value=0, max_value=10000,
            value=config.stress.pitfall_burst_count, step=10, key="s_burst",
        )

        use_food_override = st.checkbox(
            "Override food rate during stress",
            value=config.stress.food_rate_during_stress is not None, key="s_food_en",
        )
        if use_food_override:
            config.stress.food_rate_during_stress = st.number_input(
                "Food rate during stress", min_value=0.0, max_value=100.0,
                value=config.stress.food_rate_during_stress or 1.0, step=0.5, key="s_food",
            )
        else:
            config.stress.food_rate_during_stress = None

    st.markdown("**Stress Pitfall Types**")
    for i, pt in enumerate(config.stress.post_event_pitfall_types):
        c1, c2 = st.columns([1, 3])
        with c1:
            pt["name"] = st.text_input(f"Stress Type {i+1} Name", value=pt.get("name", "B"), key=f"spt_name_{i}")
        with c2:
            pt["sequence"] = st.text_input(
                f"Stress Type {i+1} Sequence (32 bits)",
                value=pt.get("sequence", "0" * 32), key=f"spt_seq_{i}",
            )


def _render_viz_section(config: SimConfig) -> None:
    st.subheader("Visualization & Output")
    col1, col2 = st.columns(2)
    with col1:
        config.viz.mode = st.selectbox(
            "Mode", options=["headless", "realtime"],
            index=0 if config.viz.mode == "headless" else 1,
            key="v_mode",
        )
        config.viz.snapshot_every_gen = st.checkbox(
            "Save snapshot every generation",
            value=config.viz.snapshot_every_gen, key="v_snap",
        )
    with col2:
        config.viz.realtime_every_n_ticks = st.number_input(
            "Realtime update interval (ticks)", min_value=1, max_value=10000,
            value=config.viz.realtime_every_n_ticks, step=1, key="v_rt_int",
        )
        config.viz.output_dir = st.text_input(
            "Output directory", value=config.viz.output_dir, key="v_outdir",
        )
