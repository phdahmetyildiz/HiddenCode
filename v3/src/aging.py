"""Age mobility and food-absorption curves: plateau until onset, then decline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.config import AgingConfig


def age_curves(
    age: NDArray[np.int32],
    cfg: AgingConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Returns (age_mobility, food_absorption), both float32, shape = age.shape.

    age <= onset → 1.0
    onset < age < max_age → interpolate toward *_end
    age >= max_age → *_end (caller should have already killed these)
    """
    age_f = age.astype(np.float32)
    onset = float(cfg.onset)
    max_age = float(cfg.max_age)
    span = max(max_age - onset, 1.0)
    t = np.clip((age_f - onset) / span, 0.0, 1.0)
    if cfg.curve == "quadratic":
        t = t * t
    mobility = np.where(
        age_f <= onset,
        1.0,
        1.0 + t * (cfg.mobility_end - 1.0),
    ).astype(np.float32)
    absorption = np.where(
        age_f <= onset,
        1.0,
        1.0 + t * (cfg.absorption_end - 1.0),
    ).astype(np.float32)
    return mobility, absorption
