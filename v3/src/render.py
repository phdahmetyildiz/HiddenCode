"""
RGB grid for optional live view (watch / studio).

Author: Cursor Grok 4.6 High Fast
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.world import World

BG = (18, 18, 24)
FOOD = (40, 160, 70)
PIT = (180, 40, 40)


def world_rgb(world: World) -> NDArray[np.uint8]:
    """Return (height, width, 3) uint8 image. Food, then pitfalls, then animals."""
    img = np.empty((world.height, world.width, 3), dtype=np.uint8)
    img[:] = BG
    fx, fy = world.food_positions()
    if fx.size:
        img[fy, fx] = FOOD
    px, py = world.pitfall_positions()
    if px.size:
        img[py, px] = PIT
    n = world.n
    if n:
        xs = world.x[:n]
        ys = world.y[:n]
        energy = np.clip(world.energy[:n], 0.0, 1.0)
        r = (220 * (1.0 - energy)).astype(np.uint8)
        g = (40 + 200 * energy).astype(np.uint8)
        img[ys, xs, 0] = r
        img[ys, xs, 1] = g
        img[ys, xs, 2] = 40
    return img
