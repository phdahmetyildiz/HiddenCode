"""
Evolution Simulator — Streamlit Web UI

Multi-page application with sidebar navigation:
  1. Config Editor — Load/edit/save simulation configuration
  2. Single Run   — Run one simulation with live KPI tracking
  3. Sweep Mode   — Parameter sweep across multiple combinations
  4. Results       — Browse and compare past runs
"""

import streamlit as st
from pathlib import Path

# Must be the very first Streamlit command
st.set_page_config(
    page_title="Evolution Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main entry point for the Streamlit app."""

    # --- Sidebar navigation ---
    st.sidebar.title("🧬 Evolution Simulator")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "🏠 Home",
            "⚙️ Config Editor",
            "▶️ Single Run",
            "🔄 Sweep Mode",
            "📊 Results Viewer",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("v2.0 — Phase 10")

    # --- Page routing ---
    if page == "🏠 Home":
        _render_home()
    elif page == "⚙️ Config Editor":
        from src.ui.pages.config_editor import render_config_editor
        render_config_editor()
    elif page == "▶️ Single Run":
        from src.ui.pages.sim_runner import render_sim_runner
        render_sim_runner()
    elif page == "🔄 Sweep Mode":
        from src.ui.pages.sweep_mode import render_sweep_mode
        render_sweep_mode()
    elif page == "📊 Results Viewer":
        from src.ui.pages.results_viewer import render_results_viewer
        render_results_viewer()


def _render_home() -> None:
    """Render the home page."""
    st.title("🧬 Evolution Simulator")
    st.markdown("""
    Welcome to the **Evolution Simulator** — a grid-based artificial life simulator
    that explores how stress-induced mutagenesis drives evolutionary adaptation.

    ### Quick Start

    1. **⚙️ Config Editor** — Set up your simulation parameters (grid size, population, 
       energy mechanics, genetics, etc.)
    2. **▶️ Single Run** — Launch a single simulation and watch KPIs evolve in real-time
    3. **🔄 Sweep Mode** — Run many simulations across parameter combinations to find
       stable baseline values
    4. **📊 Results Viewer** — Browse past runs, compare metrics, and export data

    ### Key Concepts

    | Concept | Description |
    |---------|-------------|
    | **DNA** | Each animal has a binary genome encoding weight, speed, and defense |
    | **Toroidal Grid** | 2D wrap-around world where animals, food, and pitfalls coexist |
    | **Generations** | Reproduction at 70%, survival check at 100%, bonus at 120% of gen length |
    | **Stress Events** | Triggered environmental changes that increase mutation and add new hazards |
    | **Parameter Sweep** | Systematic testing of parameter combinations to find stable worlds |
    """)

    # Show project stats
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    runs_dir = Path("runs")
    run_count = len(list(runs_dir.iterdir())) if runs_dir.exists() else 0

    col1.metric("📁 Past Runs", run_count)
    col2.metric("⚙️ Config Sections", 9)
    col3.metric("📈 KPIs Tracked", 41)
    col4.metric("🧪 Tests", "598+")


if __name__ == "__main__":
    main()
