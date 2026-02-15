"""
Results Viewer page for the Evolution Simulator UI.

Allows users to:
  - Browse past runs from the runs/ directory
  - View metrics over generations (line charts)
  - Compare multiple runs side-by-side
  - Export data
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_runs(base_dir: str = "runs") -> list[dict]:
    """Discover all past runs in the output directory."""
    runs_path = Path(base_dir)
    if not runs_path.exists():
        return []

    runs = []
    for entry in sorted(runs_path.iterdir(), reverse=True):
        if not entry.is_dir():
            continue

        run_info = {
            "name": entry.name,
            "path": entry,
            "has_metrics": (entry / "metrics.csv").exists(),
            "has_config": (entry / "config.json").exists(),
            "has_summary": (entry / "summary.json").exists(),
            "has_snapshots": (entry / "snapshots").exists(),
        }

        # Try to load summary
        summary_path = entry / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    run_info["summary"] = json.load(f)
            except Exception:
                run_info["summary"] = {}

        # Check for sweep results
        run_info["is_sweep"] = (entry / "summary.csv").exists()

        runs.append(run_info)

    return runs


def _load_metrics_csv(path: Path) -> pd.DataFrame:
    """Load a metrics CSV file into a DataFrame."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_results_viewer() -> None:
    """Render the results viewer page."""
    st.title("📊 Results Viewer")

    # Discover runs
    base_dir = st.text_input("Output directory", value="runs", key="rv_basedir")
    runs = _discover_runs(base_dir)

    if not runs:
        st.info("No runs found. Run a simulation or sweep first!")
        return

    st.markdown(f"Found **{len(runs)}** runs in `{base_dir}/`")

    # --- Run selection ---
    tab_browse, tab_compare, tab_sweep = st.tabs([
        "📁 Browse Runs",
        "📊 Compare Runs",
        "🔄 Sweep Results",
    ])

    with tab_browse:
        _render_browse(runs)

    with tab_compare:
        _render_compare(runs)

    with tab_sweep:
        _render_sweep_results(runs)


# ---------------------------------------------------------------------------
# Browse tab
# ---------------------------------------------------------------------------

def _render_browse(runs: list[dict]) -> None:
    """Browse individual runs."""
    run_names = [r["name"] for r in runs]
    selected_name = st.selectbox("Select run", options=run_names, key="rv_run_sel")

    selected_run = next((r for r in runs if r["name"] == selected_name), None)
    if selected_run is None:
        return

    run_path = selected_run["path"]

    # Run info
    st.markdown(f"### Run: `{selected_name}`")
    info_cols = st.columns(4)
    info_cols[0].metric("📄 Metrics", "✅" if selected_run["has_metrics"] else "❌")
    info_cols[1].metric("⚙️ Config", "✅" if selected_run["has_config"] else "❌")
    info_cols[2].metric("📋 Summary", "✅" if selected_run["has_summary"] else "❌")
    info_cols[3].metric("📸 Snapshots", "✅" if selected_run["has_snapshots"] else "❌")

    # Summary
    if selected_run.get("summary"):
        summary = selected_run["summary"]
        st.markdown("**Run Summary:**")
        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.metric("Ticks", summary.get("total_ticks", "N/A"))
        scol2.metric("Generations", summary.get("total_generations", "N/A"))
        scol3.metric("Final Pop.", summary.get("final_alive", "N/A"))
        scol4.metric("Elapsed", f"{summary.get('elapsed_seconds', 'N/A')}s")

    # Config
    if selected_run["has_config"]:
        with st.expander("⚙️ Configuration"):
            try:
                with open(run_path / "config.json", "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                st.json(config_data)
            except Exception as e:
                st.error(f"Failed to load config: {e}")

    # Metrics
    if selected_run["has_metrics"]:
        st.markdown("---")
        st.subheader("📈 Metrics Over Generations")

        df = _load_metrics_csv(run_path / "metrics.csv")
        if not df.empty:
            available = [c for c in df.columns if c != "generation"]

            # Preset chart tabs
            tab_pop, tab_energy, tab_food, tab_deaths, tab_custom = st.tabs([
                "Population", "Energy", "Food & Pitfalls", "Deaths", "Custom"
            ])

            with tab_pop:
                pop_cols = [c for c in ["alive_count", "births_total", "deaths_total",
                                        "pop_at_primary_repro", "pop_at_survival_check"] if c in df.columns]
                if pop_cols:
                    st.line_chart(df[pop_cols], use_container_width=True)

            with tab_energy:
                energy_cols = [c for c in ["avg_energy", "std_energy", "min_energy", "max_energy"] if c in df.columns]
                if energy_cols:
                    st.line_chart(df[energy_cols], use_container_width=True)

            with tab_food:
                food_cols = [c for c in ["food_eaten", "food_spawned",
                                          "pitfall_encounters", "food_available"] if c in df.columns]
                if food_cols:
                    st.line_chart(df[food_cols], use_container_width=True)

            with tab_deaths:
                death_cols = [c for c in ["deaths_starvation", "deaths_emergency",
                                           "deaths_pitfall", "deaths_age", "deaths_total"] if c in df.columns]
                if death_cols:
                    st.line_chart(df[death_cols], use_container_width=True)

            with tab_custom:
                selected_kpis = st.multiselect(
                    "Select KPIs to plot", options=available,
                    default=[], key="rv_custom_kpi",
                )
                if selected_kpis:
                    st.line_chart(df[selected_kpis], use_container_width=True)

            # Raw data
            with st.expander("📋 Raw Data Table"):
                st.dataframe(df, use_container_width=True)

            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name=f"{selected_name}_metrics.csv",
                mime="text/csv",
                key="rv_dl_csv",
            )

    # Snapshots
    if selected_run["has_snapshots"]:
        snap_dir = run_path / "snapshots"
        snapshots = sorted(snap_dir.glob("*.json"))
        if snapshots:
            with st.expander(f"📸 Snapshots ({len(snapshots)} available)"):
                snap_names = [s.name for s in snapshots]
                sel_snap = st.selectbox("Select snapshot", options=snap_names, key="rv_snap_sel")
                if sel_snap:
                    snap_path = snap_dir / sel_snap
                    try:
                        with open(snap_path, "r", encoding="utf-8") as f:
                            snap_data = json.load(f)
                        st.json(snap_data)
                    except Exception as e:
                        st.error(f"Failed to load snapshot: {e}")


# ---------------------------------------------------------------------------
# Compare tab
# ---------------------------------------------------------------------------

def _render_compare(runs: list[dict]) -> None:
    """Compare multiple runs side by side."""
    runs_with_metrics = [r for r in runs if r["has_metrics"]]
    if len(runs_with_metrics) < 2:
        st.info("Need at least 2 runs with metrics to compare. Run more simulations!")
        return

    run_names = [r["name"] for r in runs_with_metrics]
    selected = st.multiselect(
        "Select runs to compare", options=run_names,
        default=run_names[:2] if len(run_names) >= 2 else run_names,
        key="rv_cmp_sel",
    )

    if len(selected) < 2:
        st.info("Select at least 2 runs to compare.")
        return

    # Load metrics for selected runs
    dfs = {}
    for name in selected:
        run = next(r for r in runs_with_metrics if r["name"] == name)
        df = _load_metrics_csv(run["path"] / "metrics.csv")
        if not df.empty:
            dfs[name] = df

    if len(dfs) < 2:
        st.warning("Could not load metrics for enough runs.")
        return

    # Find common KPI columns
    all_columns = set()
    for df in dfs.values():
        all_columns.update(df.columns)
    common_cols = sorted([c for c in all_columns if c != "generation" and all(c in df.columns for df in dfs.values())])

    kpi_to_compare = st.selectbox("KPI to compare", options=common_cols,
                                   index=common_cols.index("alive_count") if "alive_count" in common_cols else 0,
                                   key="rv_cmp_kpi")

    # Build comparison chart
    chart_data = pd.DataFrame()
    for name, df in dfs.items():
        if kpi_to_compare in df.columns:
            series = df[kpi_to_compare].reset_index(drop=True)
            chart_data[name] = series

    if not chart_data.empty:
        st.line_chart(chart_data, use_container_width=True)

    # Summary comparison table
    st.markdown("### Summary Comparison")
    summary_rows = []
    for name in selected:
        run = next(r for r in runs_with_metrics if r["name"] == name)
        row = {"Run": name}
        if run.get("summary"):
            s = run["summary"]
            row["Generations"] = s.get("total_generations", "N/A")
            row["Final Pop."] = s.get("final_alive", "N/A")
            row["Extinct"] = s.get("extinct", "N/A")
            row["Elapsed"] = s.get("elapsed_seconds", "N/A")
        summary_rows.append(row)

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Sweep results tab
# ---------------------------------------------------------------------------

def _render_sweep_results(runs: list[dict]) -> None:
    """Display sweep results from past sweeps."""
    sweep_runs = [r for r in runs if r["is_sweep"]]
    if not sweep_runs:
        st.info("No sweep results found. Run a parameter sweep first!")
        return

    sweep_names = [r["name"] for r in sweep_runs]
    selected_name = st.selectbox("Select sweep", options=sweep_names, key="rv_sw_sel")

    selected_run = next((r for r in sweep_runs if r["name"] == selected_name), None)
    if selected_run is None:
        return

    run_path = selected_run["path"]

    # Load summary CSV
    summary_csv = run_path / "summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        st.subheader("Sweep Summary")
        st.dataframe(df, use_container_width=True)

        # Bar chart for stability
        if "stability_rate" in df.columns:
            st.markdown("### Stability Rates")
            chart_df = df.set_index(df.columns[0]) if len(df.columns) > 0 else df
            if "stability_rate" in chart_df.columns and "survival_rate" in chart_df.columns:
                st.bar_chart(chart_df[["stability_rate", "survival_rate"]], use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Summary CSV",
            data=csv,
            file_name=f"{selected_name}_summary.csv",
            mime="text/csv",
            key="rv_sw_dl_csv",
        )

    # Load stability report
    stability_report = run_path / "stability_report.json"
    if stability_report.exists():
        with st.expander("📋 Stability Report"):
            try:
                with open(stability_report, "r", encoding="utf-8") as f:
                    report = json.load(f)
                st.json(report)
            except Exception as e:
                st.error(f"Failed to load report: {e}")

    # Detailed CSV
    detailed_csv = run_path / "detailed.csv"
    if detailed_csv.exists():
        with st.expander("📋 Detailed Results (per generation per run)"):
            df_detail = pd.read_csv(detailed_csv)
            st.dataframe(df_detail, use_container_width=True, height=400)
