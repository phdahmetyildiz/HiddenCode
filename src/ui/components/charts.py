"""
Reusable chart components for the Evolution Simulator UI.

Provides helper functions that return Plotly figures for:
  - Population over time
  - Energy distribution histogram
  - Trait evolution lines (weight, speed, defense)
  - Sweep comparison bar charts
"""

from typing import Optional

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Population charts
# ---------------------------------------------------------------------------

def population_over_time(
    df: pd.DataFrame,
    title: str = "Population Over Time",
) -> go.Figure:
    """
    Line chart of population metrics over generations.

    Args:
        df: DataFrame with generation KPIs (must have 'population_alive').
        title: Chart title.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    pop_cols = {
        "alive_count": ("Alive", "#2ecc71"),
        "births_total": ("Born", "#3498db"),
        "deaths_total": ("Died", "#e74c3c"),
    }

    for col, (label, color) in pop_cols.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index if "generation" not in df.columns else df["generation"],
                y=df[col],
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Generation",
        yaxis_title="Count",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Energy charts
# ---------------------------------------------------------------------------

def energy_distribution(
    energies: list[float] | np.ndarray,
    title: str = "Energy Distribution",
    bins: int = 30,
) -> go.Figure:
    """
    Histogram of animal energy levels.

    Args:
        energies: List of energy values.
        title: Chart title.
        bins: Number of histogram bins.

    Returns:
        Plotly figure.
    """
    fig = go.Figure(data=[
        go.Histogram(
            x=energies,
            nbinsx=bins,
            marker_color="#f39c12",
            opacity=0.75,
        )
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Energy",
        yaxis_title="Count",
        template="plotly_white",
    )
    return fig


def energy_over_time(
    df: pd.DataFrame,
    title: str = "Energy Metrics Over Time",
) -> go.Figure:
    """
    Line chart of energy statistics over generations.

    Args:
        df: DataFrame with energy KPIs.
        title: Chart title.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    energy_cols = {
        "avg_energy": ("Mean", "#f39c12"),
        "std_energy": ("Std Dev", "#e67e22"),
        "min_energy": ("Min", "#e74c3c"),
        "max_energy": ("Max", "#27ae60"),
    }

    x = df.index if "generation" not in df.columns else df["generation"]

    for col, (label, color) in energy_cols.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df[col],
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            ))

    # Add shaded band between min and max
    if "min_energy" in df.columns and "max_energy" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([x, x[::-1]]) if isinstance(x, pd.Series) else list(range(len(df))) + list(range(len(df) - 1, -1, -1)),
            y=pd.concat([df["max_energy"], df["min_energy"][::-1]]) if isinstance(df["max_energy"], pd.Series) else [],
            fill="toself",
            fillcolor="rgba(46, 204, 113, 0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            name="Range",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Generation",
        yaxis_title="Energy",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Trait evolution charts
# ---------------------------------------------------------------------------

def trait_evolution(
    df: pd.DataFrame,
    title: str = "Trait Evolution",
) -> go.Figure:
    """
    Line chart of trait means (weight, speed, defense) over generations.

    Args:
        df: DataFrame with trait KPIs.
        title: Chart title.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    trait_cols = {
        "avg_weight": ("Weight", "#3498db"),
        "avg_speed": ("Speed", "#e74c3c"),
        "avg_defense_ones": ("Defense (1-bits)", "#2ecc71"),
    }

    x = df.index if "generation" not in df.columns else df["generation"]

    for col, (label, color) in trait_cols.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df[col],
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Generation",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def genetic_diversity(
    df: pd.DataFrame,
    title: str = "Genetic Diversity Over Time",
) -> go.Figure:
    """
    Line chart of genetic diversity metrics.

    Args:
        df: DataFrame with genetic diversity KPIs.
        title: Chart title.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    diversity_cols = {
        "genetic_diversity": ("Pairwise Hamming", "#9b59b6"),
        "unique_defense_seqs": ("Unique Defense Sequences", "#1abc9c"),
    }

    x = df.index if "generation" not in df.columns else df["generation"]

    for col, (label, color) in diversity_cols.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=x, y=df[col],
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Generation",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Sweep comparison charts
# ---------------------------------------------------------------------------

def sweep_comparison_bars(
    combinations: list[dict],
    metric: str = "stability_rate",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Bar chart comparing a metric across sweep combinations.

    Args:
        combinations: List of dicts with 'params' and metric value.
        metric: Name of the metric to compare.
        title: Chart title (auto-generated if None).

    Returns:
        Plotly figure.
    """
    if title is None:
        title = f"Sweep Comparison: {metric}"

    labels = [str(c.get("params", c.get("combination_id", ""))) for c in combinations]
    values = [c.get(metric, 0) for c in combinations]

    # Color by value (higher = greener)
    max_val = max(values) if values else 1
    colors = [f"rgba({int(255 * (1 - v / max_val))}, {int(255 * (v / max_val))}, 50, 0.8)"
              for v in values]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition="auto",
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Parameter Combination",
        yaxis_title=metric.replace("_", " ").title(),
        template="plotly_white",
        xaxis_tickangle=-45,
    )
    return fig


def sweep_population_comparison(
    combinations: list[dict],
    title: str = "Average Final Population by Combination",
) -> go.Figure:
    """
    Bar chart comparing average final populations across combinations.

    Args:
        combinations: List of dicts with 'params', 'avg_final_alive', 'std_final_alive'.
        title: Chart title.

    Returns:
        Plotly figure.
    """
    labels = [str(c.get("params", "")) for c in combinations]
    means = [c.get("avg_final_alive", 0) for c in combinations]
    stds = [c.get("std_final_alive", 0) for c in combinations]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=means,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color="#3498db",
            text=[f"{m:.0f}" for m in means],
            textposition="auto",
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Parameter Combination",
        yaxis_title="Average Final Population",
        template="plotly_white",
        xaxis_tickangle=-45,
    )
    return fig
