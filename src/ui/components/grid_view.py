"""
2D Grid View component for the Evolution Simulator UI.

Renders a snapshot of the world grid using Plotly:
  - Animals as dots (size proportional to weight, color encodes energy)
  - Food as green dots
  - Pitfalls as red squares
  - Supports both World objects and snapshot dicts

Preparation for future real-time visualization (Phase 12).
"""

from typing import Optional

import plotly.graph_objects as go
import numpy as np

from src.core.world import World


# ---------------------------------------------------------------------------
# Grid rendering from live World object
# ---------------------------------------------------------------------------

def render_world_grid(
    world: World,
    title: Optional[str] = None,
    width: int = 700,
    height: int = 700,
    max_entities: int = 5000,
) -> go.Figure:
    """
    Render a 2D grid snapshot of the world.

    Args:
        world: World object with animals, food, and pitfalls.
        title: Optional chart title.
        width: Plot width in pixels.
        height: Plot height in pixels.
        max_entities: Maximum entities to render (performance cap).

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    grid_w = world.config.world.width
    grid_h = world.config.world.height

    if title is None:
        title = f"World Grid ({grid_w}×{grid_h}) | Tick {world.tick_count}"

    # --- Pitfalls (red squares, background layer) ---
    pitfall_positions = list(world.pitfalls.keys())[:max_entities]
    if pitfall_positions:
        px_list = [p[0] for p in pitfall_positions]
        py_list = [p[1] for p in pitfall_positions]
        fig.add_trace(go.Scatter(
            x=px_list, y=py_list,
            mode="markers",
            marker=dict(
                symbol="square",
                size=6,
                color="rgba(231, 76, 60, 0.6)",
                line=dict(width=0),
            ),
            name=f"Pitfalls ({len(pitfall_positions)})",
            hovertemplate="Pitfall (%{x}, %{y})<extra></extra>",
        ))

    # --- Food (green dots) ---
    food_positions = list(world.food.keys())[:max_entities]
    if food_positions:
        fx_list = [f[0] for f in food_positions]
        fy_list = [f[1] for f in food_positions]
        fig.add_trace(go.Scatter(
            x=fx_list, y=fy_list,
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=5,
                color="rgba(46, 204, 113, 0.7)",
                line=dict(width=0),
            ),
            name=f"Food ({len(food_positions)})",
            hovertemplate="Food (%{x}, %{y})<extra></extra>",
        ))

    # --- Animals (colored dots) ---
    animals = list(world.animals.values())[:max_entities]
    if animals:
        ax = [a.x for a in animals]
        ay = [a.y for a in animals]
        energies = [a.energy for a in animals]
        weights = [a.weight for a in animals]

        # Size: map weight [0.1..1.0] → marker size [4..14]
        sizes = [4 + (w - 0.1) / 0.9 * 10 for w in weights]

        # Color: map energy [0..1] → yellow(low) to blue(high)
        fig.add_trace(go.Scatter(
            x=ax, y=ay,
            mode="markers",
            marker=dict(
                size=sizes,
                color=energies,
                colorscale="Viridis",
                cmin=0, cmax=1,
                colorbar=dict(title="Energy", thickness=15, len=0.5),
                line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
            ),
            name=f"Animals ({len(animals)})",
            hovertemplate=(
                "Animal (%{x}, %{y})<br>"
                "Energy: %{marker.color:.3f}<extra></extra>"
            ),
        ))

    # --- Layout ---
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis=dict(
            range=[-0.5, grid_w - 0.5],
            title="X",
            scaleanchor="y",
            scaleratio=1,
            constrain="domain",
        ),
        yaxis=dict(
            range=[-0.5, grid_h - 0.5],
            title="Y",
        ),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


# ---------------------------------------------------------------------------
# Grid rendering from snapshot dict
# ---------------------------------------------------------------------------

def render_snapshot_grid(
    snapshot: dict,
    title: Optional[str] = None,
    width: int = 700,
    height: int = 700,
    max_entities: int = 5000,
) -> go.Figure:
    """
    Render a 2D grid from a saved snapshot dict.

    Args:
        snapshot: Dict with 'animals', 'food', 'pitfalls', and optionally
                  'width', 'height', 'tick_count'.
        title: Optional chart title.
        width: Plot width.
        height: Plot height.
        max_entities: Max entities to render.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    grid_w = snapshot.get("width", 500)
    grid_h = snapshot.get("height", 500)
    tick = snapshot.get("tick_count", "?")

    if title is None:
        title = f"Snapshot Grid ({grid_w}×{grid_h}) | Tick {tick}"

    # Pitfalls
    pitfalls = snapshot.get("pitfalls", [])[:max_entities]
    if pitfalls:
        px_list = [p.get("x", 0) for p in pitfalls]
        py_list = [p.get("y", 0) for p in pitfalls]
        fig.add_trace(go.Scatter(
            x=px_list, y=py_list,
            mode="markers",
            marker=dict(symbol="square", size=6, color="rgba(231, 76, 60, 0.6)"),
            name=f"Pitfalls ({len(pitfalls)})",
        ))

    # Food
    food = snapshot.get("food", [])[:max_entities]
    if food:
        fx_list = [f.get("x", 0) for f in food]
        fy_list = [f.get("y", 0) for f in food]
        fig.add_trace(go.Scatter(
            x=fx_list, y=fy_list,
            mode="markers",
            marker=dict(symbol="diamond", size=5, color="rgba(46, 204, 113, 0.7)"),
            name=f"Food ({len(food)})",
        ))

    # Animals
    animals = snapshot.get("animals", [])[:max_entities]
    if animals:
        ax = [a.get("x", 0) for a in animals]
        ay = [a.get("y", 0) for a in animals]
        energies = [a.get("energy", 0.5) for a in animals]
        weights = [a.get("weight", 0.5) for a in animals]
        sizes = [4 + (w - 0.1) / 0.9 * 10 for w in weights]

        fig.add_trace(go.Scatter(
            x=ax, y=ay,
            mode="markers",
            marker=dict(
                size=sizes,
                color=energies,
                colorscale="Viridis", cmin=0, cmax=1,
                colorbar=dict(title="Energy", thickness=15, len=0.5),
            ),
            name=f"Animals ({len(animals)})",
        ))

    fig.update_layout(
        title=title,
        width=width, height=height,
        xaxis=dict(range=[-0.5, grid_w - 0.5], title="X",
                    scaleanchor="y", scaleratio=1, constrain="domain"),
        yaxis=dict(range=[-0.5, grid_h - 0.5], title="Y"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
