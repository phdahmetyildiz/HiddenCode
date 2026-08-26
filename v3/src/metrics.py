"""Per-metrics-epoch KPIs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.aging import age_curves
from src import dna as dnalib
from src.world import World


@dataclass
class EpochMetrics:
    epoch: int = 0
    tick: int = 0
    alive_count: int = 0
    births_count: int = 0
    births_skipped: int = 0
    deaths_starvation: int = 0
    deaths_emergency: int = 0
    deaths_pitfall: int = 0
    deaths_max_age: int = 0
    deaths_cull: int = 0
    extinction_flag: bool = False
    avg_energy: float = 0.0
    median_energy: float = 0.0
    min_energy: float = 0.0
    max_energy: float = 0.0
    std_energy: float = 0.0
    avg_weight: float = 0.0
    avg_speed: float = 0.0
    avg_defense_ones: float = 0.0
    avg_repro_age: float = 0.0
    avg_age: float = 0.0
    median_age: float = 0.0
    max_age_alive: int = 0
    avg_mobility: float = 0.0
    avg_food_absorption: float = 0.0
    avg_move_probability: float = 0.0
    genetic_diversity: float = 0.0
    unique_defense_seqs: int = 0
    food_spawned: int = 0
    food_eaten: int = 0
    food_expired: int = 0
    food_available: int = 0
    pitfall_encounters: int = 0
    pitfall_avg_damage: float = 0.0
    pitfall_zero_damage: int = 0
    stress_mode_active: bool = False
    mutation_rate_effective: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_epoch(world: World, counters, epoch: int) -> EpochMetrics:
    n = world.n
    m = EpochMetrics(epoch=epoch, tick=world.tick, alive_count=n)
    m.births_count = counters.births
    m.births_skipped = counters.births_skipped
    m.deaths_starvation = counters.deaths_starvation
    m.deaths_emergency = counters.deaths_emergency
    m.deaths_pitfall = counters.deaths_pitfall
    m.deaths_max_age = counters.deaths_max_age
    m.deaths_cull = counters.deaths_cull
    m.extinction_flag = n == 0
    m.food_spawned = counters.food_spawned
    m.food_eaten = counters.food_eaten
    m.food_expired = counters.food_expired
    m.food_available = int(np.count_nonzero(world.food_life > 0))
    m.pitfall_encounters = counters.pitfall_encounters
    m.pitfall_zero_damage = counters.pitfall_zero_damage
    if counters.pitfall_encounters > 0:
        m.pitfall_avg_damage = counters.pitfall_total_damage / counters.pitfall_encounters
    m.stress_mode_active = world.stress_mode
    g = world.config.genetics
    m.mutation_rate_effective = g.stress_mutation_rate if world.stress_mode else g.base_mutation_rate

    if n == 0:
        return m

    energy = world.energy[:n]
    m.avg_energy = float(energy.mean())
    m.median_energy = float(np.median(energy))
    m.min_energy = float(energy.min())
    m.max_energy = float(energy.max())
    m.std_energy = float(energy.std())
    m.avg_weight = float(world.weight[:n].mean())
    m.avg_speed = float(world.speed[:n].mean())
    ones = np.zeros(n, dtype=np.int32)
    d = world.defense[:n].astype(np.uint32)
    for _ in range(32):
        ones += (d & np.uint32(1)).astype(np.int32)
        d >>= np.uint32(1)
    m.avg_defense_ones = float(ones.mean())
    m.avg_repro_age = float(world.repro_age[:n].mean())
    age = world.age()
    m.avg_age = float(age.mean())
    m.median_age = float(np.median(age))
    m.max_age_alive = int(age.max())
    mobility, absorption = age_curves(age, world.config.aging)
    m.avg_mobility = float(mobility.mean())
    m.avg_food_absorption = float(absorption.mean())
    move_p = np.clip(world.speed[:n] * mobility, 0.0, 1.0)
    m.avg_move_probability = float(move_p.mean())
    m.unique_defense_seqs = int(np.unique(world.defense[:n]).size)
    sample_n = min(n, 100)
    m.genetic_diversity = dnalib.hamming_sample(world.dna[:n], world.rng, max_animals=sample_n)
    return m
