"""
Per-animal one-clutch reproduction.

Author: Cursor Grok 4.6 High Fast
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src import dna as dnalib
from src.config import SimConfig
from src.world import World


def offspring_counts(energy: NDArray[np.float32], config: SimConfig) -> NDArray[np.int32]:
    low = config.reproduction.repro_energy_low
    high = config.reproduction.repro_energy_high
    counts = np.zeros(energy.shape[0], dtype=np.int32)
    counts[energy >= low] = 1
    counts[energy >= high] = 2
    return counts


def reproduce(world: World, stress_mode: bool) -> int:
    """
    Animals with age == repro_age and not yet reproduced get one clutch.
    Returns number of births actually added.
    """
    n = world.n
    if n == 0:
        return 0
    age = world.age()
    due = (age == world.repro_age[:n]) & (~world.has_reproduced[:n])
    world.has_reproduced[:n] = world.has_reproduced[:n] | due
    if not np.any(due):
        return 0

    parent_idx = np.nonzero(due)[0]
    counts = offspring_counts(world.energy[:n][due], world.config)
    # Expand parent indices by offspring count
    if int(counts.sum()) == 0:
        return 0
    expanded = np.repeat(parent_idx, counts)
    parent_dna = world.dna[:n][expanded].copy()
    cfg = world.config.genetics
    rate = cfg.stress_mutation_rate if stress_mode else cfg.base_mutation_rate
    coding_only = cfg.stress_mode_coding_only if stress_mode else True
    dnalib.mutate_coding(
        parent_dna,
        rate=rate,
        coding_regions=cfg.coding_regions,
        rng=world.rng,
        dna_length=cfg.dna_length,
        coding_only=coding_only,
    )
    return world.add_offspring(expanded, parent_dna)
