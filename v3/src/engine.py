"""Synchronous tick engine for Evolution Simulator v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from src.aging import age_curves
from src.config import SimConfig
from src import kernels
from src.metrics import EpochMetrics, collect_epoch
from src.reproduction import reproduce
from src.spatial import move_toward, nearest_food, random_steps, wrap
from src.stress import StressManager
from src.world import World


def popcount32(x: NDArray[np.uint32]) -> NDArray[np.int32]:
    return kernels.popcount32(x)


@dataclass
class TickStats:
    food_spawned: int = 0
    food_expired: int = 0
    food_eaten: int = 0
    pitfalls_spawned: int = 0
    pitfalls_expired: int = 0
    pitfall_encounters: int = 0
    pitfall_total_damage: int = 0
    pitfall_zero_damage: int = 0
    deaths_starvation: int = 0
    deaths_emergency: int = 0
    deaths_pitfall: int = 0
    deaths_max_age: int = 0
    deaths_cull: int = 0
    births: int = 0
    births_skipped: int = 0


@dataclass
class RunResult:
    config: SimConfig
    seed: int
    total_ticks: int = 0
    total_epochs: int = 0
    final_alive: int = 0
    extinct: bool = False
    extinction_tick: Optional[int] = None
    epoch_metrics: list[EpochMetrics] = field(default_factory=list)


class SimulationEngine:
    def __init__(self, config: SimConfig, seed: Optional[int] = None):
        self.config = config
        if seed is not None:
            self.config.world.seed = seed
        self.rng = np.random.default_rng(self.config.world.seed)
        self.backend = kernels.resolve_and_set(self.config.perf.backend)
        self.world = World(self.config, rng=self.rng)
        self.stress = StressManager(self.config)
        self.tick_stats = TickStats()
        self.epoch_counters = TickStats()
        self.epochs_completed = 0
        self.epoch_history: list[EpochMetrics] = []
        self.on_epoch: Optional[Callable[[EpochMetrics, "SimulationEngine"], None]] = None
        self.on_tick: Optional[Callable[[int, "SimulationEngine"], None]] = None

    def initialize(self) -> None:
        self.world.initialize_population()
        # Standing crop so founders are not in an empty desert
        initial = max(
            1,
            int(self.config.resources.food_rate * self.config.resources.food_lifespan * 0.5),
        )
        for _ in range(initial):
            self.world.spawn_food(rate=1.0)

    def _slice(self):
        n = self.world.n
        return n, self.world.x[:n], self.world.y[:n], self.world.energy[:n]

    def _compact(self, keep: NDArray[np.bool_]) -> int:
        return self.world.compact(keep)

    def _kill_mask(self, mask: NDArray[np.bool_], cause: str) -> int:
        n = self.world.n
        if n == 0 or not np.any(mask):
            return 0
        keep = ~mask
        removed = self._compact(keep)
        if cause == "starvation":
            self.tick_stats.deaths_starvation += removed
            self.epoch_counters.deaths_starvation += removed
        elif cause == "emergency":
            self.tick_stats.deaths_emergency += removed
            self.epoch_counters.deaths_emergency += removed
        elif cause == "pitfall":
            self.tick_stats.deaths_pitfall += removed
            self.epoch_counters.deaths_pitfall += removed
        elif cause == "max_age":
            self.tick_stats.deaths_max_age += removed
            self.epoch_counters.deaths_max_age += removed
        elif cause == "cull":
            self.tick_stats.deaths_cull += removed
            self.epoch_counters.deaths_cull += removed
        return removed

    def tick(self) -> TickStats:
        self.tick_stats = TickStats()
        w = self.world
        cfg = self.config

        food_rate = self.stress.food_rate(w)
        self.tick_stats.food_spawned = w.spawn_food(rate=food_rate)
        self.tick_stats.pitfalls_spawned = w.spawn_pitfalls()
        fe, pe = w.decay_resources()
        self.tick_stats.food_expired = fe
        self.tick_stats.pitfalls_expired = pe
        self.epoch_counters.food_spawned += self.tick_stats.food_spawned
        self.epoch_counters.food_expired += fe
        self.epoch_counters.pitfalls_spawned += self.tick_stats.pitfalls_spawned
        self.epoch_counters.pitfalls_expired += pe

        # Max age
        if w.n > 0:
            self._kill_mask(w.age() >= cfg.aging.max_age, "max_age")

        # Drain
        n = w.n
        if n > 0:
            kernels.apply_drain(
                w.energy[:n],
                w.weight[:n],
                w.speed[:n],
                cfg.energy.base_metabolism,
                cfg.energy.k_weight_speed,
                defense=w.defense[:n],
                k_def=cfg.energy.k_defense_cost,
                defense_cost=cfg.energy.defense_cost_enabled,
            )
            self._kill_mask(w.energy[: w.n] <= 0.0, "starvation")

        # Emergency (pre-move food map)
        n = w.n
        if n > 0:
            fx, fy = w.food_positions()
            in_range, _, _ = nearest_food(
                w.x[:n], w.y[:n], fx, fy,
                cfg.properties.eyesight_radius, w.width, w.height,
            )
            low = w.energy[:n] < cfg.energy.low_energy_death_threshold
            self._kill_mask(low & ~in_range, "emergency")

        # Sense + move
        n = w.n
        if n > 0:
            age = w.age()
            mobility, _ = age_curves(age, cfg.aging)
            move_p = np.clip(w.speed[:n] * mobility, 0.0, 1.0)
            do_move = self.rng.random(n) < move_p
            fx, fy = w.food_positions()
            in_range, tx, ty = nearest_food(
                w.x[:n], w.y[:n], fx, fy,
                cfg.properties.eyesight_radius, w.width, w.height,
            )
            toward = do_move & in_range
            if np.any(toward):
                t_idx = np.nonzero(toward)[0]
                nx, ny = move_toward(
                    w.x[t_idx], w.y[t_idx],
                    tx[t_idx], ty[t_idx],
                    w.width, w.height,
                )
                w.x[t_idx] = nx
                w.y[t_idx] = ny
            rnd = do_move & ~in_range
            r_idx = np.nonzero(rnd)[0]
            if r_idx.size:
                dx, dy = random_steps(int(r_idx.size), self.rng)
                w.x[r_idx] = wrap(w.x[r_idx] + dx, w.width).astype(np.int32)
                w.y[r_idx] = wrap(w.y[r_idx] + dy, w.height).astype(np.int32)

            eaten = self._resolve_feeding()
            self.tick_stats.food_eaten = eaten
            self.epoch_counters.food_eaten += eaten
            self._resolve_pitfalls()
            self._kill_mask(w.energy[: w.n] <= 0.0, "starvation")

        # Reproduction
        skipped_before = w.births_skipped
        births = reproduce(w, stress_mode=w.stress_mode)
        self.tick_stats.births = births
        self.tick_stats.births_skipped = w.births_skipped - skipped_before
        self.epoch_counters.births += births
        self.epoch_counters.births_skipped += self.tick_stats.births_skipped

        # Advance time
        w.tick += 1

        event = self.stress.check_tick(w)

        if w.tick > 0 and w.tick % cfg.metrics.interval == 0:
            if cfg.metrics.cull_enabled and w.n > 0:
                self._kill_mask(w.energy[: w.n] <= cfg.metrics.survival_threshold, "cull")
            metrics = collect_epoch(w, self.epoch_counters, self.epochs_completed)
            self.epoch_history.append(metrics)
            self.epochs_completed += 1
            w.cohort = self.epochs_completed
            self.epoch_counters = TickStats()
            if self.on_epoch is not None:
                self.on_epoch(metrics, self)

        if self.on_tick is not None:
            self.on_tick(w.tick, self)

        _ = event
        return self.tick_stats

    def _resolve_feeding(self) -> int:
        w = self.world
        n = w.n
        if n == 0:
            return 0
        on_food = w.food_life[w.x[:n], w.y[:n]] > 0
        if not np.any(on_food):
            return 0
        idx = np.nonzero(on_food)[0]
        key = w.x[:n][idx].astype(np.int64) * w.height + w.y[:n][idx]
        weights = w.weight[:n][idx]
        tie = self.rng.random(idx.size)
        order = np.lexsort((-tie, -weights, key))
        sorted_key = key[order]
        sorted_idx = idx[order]
        first = np.ones(sorted_key.size, dtype=bool)
        if sorted_key.size > 1:
            first[1:] = sorted_key[1:] != sorted_key[:-1]
        winners = sorted_idx[first]
        _, absorption = age_curves(w.age()[winners], self.config.aging)
        gain = (self.config.energy.food_gain * absorption).astype(np.float32)
        w.energy[winners] = np.clip(w.energy[winners] + gain, 0.0, 1.0)
        w.food_life[w.x[winners], w.y[winners]] = 0
        return int(winners.size)

    def _resolve_pitfalls(self) -> None:
        w = self.world
        n = w.n
        if n == 0:
            return
        here = w.pitfall_life[w.x[:n], w.y[:n]] > 0
        if not np.any(here):
            return
        idx = np.nonzero(here)[0]
        seq = w.pitfall_seq[w.x[:n][idx], w.y[:n][idx]]
        defense = w.defense[:n][idx]
        hit = seq & (~defense)
        damage = popcount32(hit)
        self.tick_stats.pitfall_encounters += int(idx.size)
        self.tick_stats.pitfall_total_damage += int(damage.sum())
        self.tick_stats.pitfall_zero_damage += int(np.count_nonzero(damage == 0))
        self.epoch_counters.pitfall_encounters += int(idx.size)
        self.epoch_counters.pitfall_total_damage += int(damage.sum())
        self.epoch_counters.pitfall_zero_damage += int(np.count_nonzero(damage == 0))
        loss = (damage.astype(np.float32) / 32.0) * self.config.energy.max_pitfall_loss_pct
        # Deaths from this blow are tagged pitfall if they hit 0
        energy_before = w.energy[idx].copy()
        w.energy[idx] = np.clip(w.energy[idx] - loss, 0.0, 1.0)
        killed_now = (energy_before > 0) & (w.energy[idx] <= 0) & (damage > 0)
        if np.any(killed_now):
            # Mark these for pitfall death: set a sentinel then kill
            pit_mask = np.zeros(n, dtype=bool)
            pit_mask[idx[killed_now]] = True
            self._kill_mask(pit_mask, "pitfall")

    def run(
        self,
        max_ticks: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> RunResult:
        if max_ticks is None and max_epochs is None:
            max_epochs = 1
        result = RunResult(config=self.config, seed=self.config.world.seed)
        ticks = 0
        while True:
            if max_ticks is not None and ticks >= max_ticks:
                break
            if max_epochs is not None and self.epochs_completed >= max_epochs:
                break
            self.tick()
            ticks += 1
            if self.world.is_extinct:
                result.extinct = True
                result.extinction_tick = self.world.tick
                break
        result.total_ticks = ticks
        result.total_epochs = self.epochs_completed
        result.final_alive = self.world.n
        result.epoch_metrics = list(self.epoch_history)
        return result
