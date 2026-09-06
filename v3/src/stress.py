"""
Stress mode: mutation rate, pitfall burst, optional food rate.

Author: Cursor Grok 4.6 High Fast
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import SimConfig
from src.world import World


@dataclass
class StressState:
    active: bool = False
    started_tick: int | None = None


class StressManager:
    def __init__(self, config: SimConfig):
        self.config = config
        self.state = StressState()

    def trigger(self, world: World) -> int:
        self.state.active = True
        self.state.started_tick = world.tick
        world.stress_mode = True
        types = list(world.active_pitfall_types)
        extra = self.config.stress.get_post_event_types()
        # Merge by name
        names = {t.name for t in types}
        for t in extra:
            if t.name not in names:
                types.append(t)
        world.active_pitfall_types = types
        burst = self.config.stress.pitfall_burst_count
        spawned = 0
        if burst > 0 and extra:
            spawned = world.spawn_pitfalls_batch(burst, extra)
        return spawned

    def deactivate(self, world: World) -> None:
        self.state.active = False
        self.state.started_tick = None
        world.stress_mode = False

    def check_tick(self, world: World) -> str:
        cfg = self.config.stress
        event = "none"
        if not self.state.active and cfg.trigger_tick is not None:
            if world.tick == cfg.trigger_tick:
                self.trigger(world)
                event = "triggered"
        if self.state.active and cfg.duration_ticks is not None and self.state.started_tick is not None:
            if world.tick >= self.state.started_tick + cfg.duration_ticks:
                self.deactivate(world)
                event = "deactivated"
        return event

    def food_rate(self, world: World) -> float:
        if world.stress_mode and self.config.stress.food_rate_during_stress is not None:
            return float(self.config.stress.food_rate_during_stress)
        return float(self.config.resources.food_rate)
