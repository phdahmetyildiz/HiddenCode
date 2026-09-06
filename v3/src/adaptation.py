"""
Pitfall adaptation: coverage of dangerous bits (the inverse of damage).

Damage counts bits where the pitfall is 1 and defense is 0:

    damage = popcount(seq & ~defense)

Adaptation is the fraction of those threatening bits the animal actually
covers. Pitfall 0-bits are ignored (they never hurt, so they are not a
target to 'adapt' to):

    need    = popcount(seq)                 # dangerous bits
    covered = popcount(seq & defense)
    score   = covered / need                # 1 if need == 0

An encounter is fully adapted when score == 1 (damage == 0 on all
dangerous bits). The reported score is the mean over encounters.

Author: Cursor Grok 4.6 High Fast
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src import kernels


def encounter_adaptation(
    seq: NDArray[np.uint32],
    defense: NDArray[np.uint32],
) -> NDArray[np.float32]:
    seq = np.asarray(seq, dtype=np.uint32)
    defense = np.asarray(defense, dtype=np.uint32)
    need = kernels.popcount32(seq).astype(np.float32)
    covered = kernels.popcount32(seq & defense).astype(np.float32)
    out = np.ones(seq.shape[0], dtype=np.float32)
    nz = need > 0
    out[nz] = covered[nz] / need[nz]
    return out
