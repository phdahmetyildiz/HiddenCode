"""Pre-run energy / foraging budget. No simulation required."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import SimConfig


@dataclass
class LivabilityReport:
    mean_weight: float
    mean_speed: float
    drain_per_tick: float
    ticks_to_emergency: float
    repro_age_min: int
    food_standing_if_uneaten: float
    eyesight_cells: float
    expected_food_in_sight: float
    food_per_animal_per_tick: float
    warns: list[str]

    def as_text(self) -> str:
        lines = [
            "Livability budget",
            f"  mean founder weight/speed: {self.mean_weight:.3f} / {self.mean_speed:.3f}",
            f"  drain per tick:            {self.drain_per_tick:.5f}",
            f"  ticks to emergency (no food): {self.ticks_to_emergency:.1f}",
            f"  first possible clutch age: {self.repro_age_min}",
            f"  uneaten food standing crop: {self.food_standing_if_uneaten:.1f}",
            f"  expected food in eyesight: {self.expected_food_in_sight:.3f}",
            f"  food items / animal / tick (if shared): {self.food_per_animal_per_tick:.4f}",
        ]
        if self.warns:
            lines.append("  WARNINGS:")
            for w in self.warns:
                lines.append(f"    - {w}")
        else:
            lines.append("  warnings: none")
        return "\n".join(lines)


def evaluate(config: SimConfig) -> LivabilityReport:
    p = config.properties
    mean_w = 0.5 * (p.weight_init_range[0] + p.weight_init_range[1])
    mean_s = 0.5 * (p.speed_init_range[0] + p.speed_init_range[1])
    e = config.energy
    drain = e.base_metabolism + e.k_weight_speed * mean_w * mean_s
    if e.defense_cost_enabled:
        drain += e.k_defense_cost * 16.0  # expected ones in 32 bits
    thr = e.low_energy_death_threshold
    ticks_to_em = (1.0 - thr) / drain if drain > 0 else math.inf

    r = float(p.eyesight_radius)
    eyesight_cells = math.pi * r * r
    cells = config.world.width * config.world.height
    standing = config.resources.food_rate * config.resources.food_lifespan
    density = standing / cells if cells else 0.0
    expected_in_sight = eyesight_cells * density
    n = max(config.population.initial_count, 1)
    food_per = config.resources.food_rate / n

    warns: list[str] = []
    repro_min = config.reproduction.repro_age_min
    # Zero-food emergency time is *expected* to be < repro_age — they must forage.
    # Warn only when they cannot find food, or when even perfectly shared food
    # cannot cover metabolism.
    shared_energy = food_per * config.energy.food_gain
    if expected_in_sight < 0.5:
        warns.append(
            f"expected food in eyesight is {expected_in_sight:.4f}; "
            "animals may not find food (grid too large / food too sparse)"
        )
    if shared_energy + 1e-12 < 0.5 * drain:
        warns.append(
            f"shared food energy/tick ({shared_energy:.5f}) is far below drain "
            f"({drain:.5f}); population likely starves even with perfect foraging"
        )

    return LivabilityReport(
        mean_weight=mean_w,
        mean_speed=mean_s,
        drain_per_tick=drain,
        ticks_to_emergency=ticks_to_em,
        repro_age_min=repro_min,
        food_standing_if_uneaten=standing,
        eyesight_cells=eyesight_cells,
        expected_food_in_sight=expected_in_sight,
        food_per_animal_per_tick=food_per,
        warns=warns,
    )
