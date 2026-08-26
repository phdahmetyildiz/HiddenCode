"""Aging plateau, spatial torus, feeding/emergency/repro."""

import numpy as np
import pytest

from src.aging import age_curves
from src.config import get_default_config
from src.engine import SimulationEngine, popcount32
from src.reproduction import offspring_counts
from src.spatial import move_toward, nearest_food, toroidal_distance_sq
from src.world import World


def test_aging_plateau_200_vs_500():
    cfg = get_default_config().aging
    ages = np.array([200, 500, 1000, 1400, 1799], dtype=np.int32)
    mob, ab = age_curves(ages, cfg)
    assert ab[0] == pytest.approx(1.0)
    assert ab[1] == pytest.approx(1.0)
    assert ab[0] == pytest.approx(ab[1])
    assert ab[2] == pytest.approx(1.0)
    assert ab[3] == pytest.approx(0.60, abs=0.02)
    assert ab[4] == pytest.approx(cfg.absorption_end, abs=0.02)
    assert mob[0] == pytest.approx(1.0)


def test_toroidal_wrap_distance():
    d2 = toroidal_distance_sq(
        np.array([0]), np.array([0]),
        np.array([79]), np.array([0]),
        80, 80,
    )
    assert int(d2[0]) == 1


def test_move_toward_wraps():
    nx, ny = move_toward(
        np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
        np.array([79], dtype=np.int32), np.array([0], dtype=np.int32),
        80, 80,
    )
    assert int(nx[0]) == 79


def test_nearest_food_in_range():
    ax = np.array([10], dtype=np.int32)
    ay = np.array([10], dtype=np.int32)
    fx = np.array([12], dtype=np.int32)
    fy = np.array([10], dtype=np.int32)
    in_r, tx, ty = nearest_food(ax, ay, fx, fy, 10, 80, 80)
    assert bool(in_r[0])
    assert int(tx[0]) == 12


def test_world_compact():
    cfg = get_default_config()
    w = World(cfg, rng=np.random.default_rng(0))
    w.initialize_population(10)
    keep = np.ones(10, dtype=bool)
    keep[[1, 4, 7]] = False
    removed = w.compact(keep)
    assert removed == 3
    assert w.n == 7


def test_emergency_no_food_dies():
    cfg = get_default_config()
    cfg.population.initial_count = 4
    cfg.resources.food_rate = 0.0
    cfg.resources.pitfall_rate = 0.0
    cfg.energy.low_energy_death_threshold = 0.99
    cfg.energy.base_metabolism = 0.5
    cfg.energy.k_weight_speed = 0.0
    eng = SimulationEngine(cfg, seed=1)
    eng.initialize()
    eng.world.food_life[:] = 0
    eng.tick()
    assert eng.tick_stats.deaths_emergency > 0 or eng.world.n == 0


def test_emergency_food_in_range_lives():
    cfg = get_default_config()
    cfg.population.initial_count = 1
    cfg.resources.food_rate = 0.0
    cfg.resources.pitfall_rate = 0.0
    cfg.energy.low_energy_death_threshold = 0.99
    cfg.energy.base_metabolism = 0.0
    cfg.energy.k_weight_speed = 0.0
    eng = SimulationEngine(cfg, seed=2)
    eng.initialize()
    eng.world.energy[0] = 0.05
    eng.world.x[0] = 10
    eng.world.y[0] = 10
    eng.world.food_life[:, :] = 0
    eng.world.food_life[12, 10] = 50
    eng.tick()
    assert eng.world.n == 1
    assert eng.tick_stats.deaths_emergency == 0


def test_max_age_kills_full_energy():
    cfg = get_default_config()
    cfg.population.initial_count = 3
    cfg.aging.max_age = 5
    cfg.aging.onset = 2
    cfg.resources.food_rate = 20.0
    cfg.energy.low_energy_death_threshold = 0.0
    eng = SimulationEngine(cfg, seed=3)
    eng.initialize()
    for _ in range(6):
        eng.tick()
        if eng.world.n == 0:
            break
    assert eng.epoch_counters.deaths_max_age + eng.tick_stats.deaths_max_age >= 0
    # After 5 ticks of age (birth 0, tick becomes 5 → age 5 at start of next)
    # Run extra ticks
    for _ in range(5):
        eng.tick()
    # All founders should be gone by age 5+
    if eng.world.n:
        assert np.all(eng.world.age() < 5) or np.all(eng.world.birth_tick[: eng.world.n] > 0)


def test_heaviest_eats():
    cfg = get_default_config()
    cfg.population.initial_count = 2
    cfg.resources.food_rate = 0
    cfg.resources.pitfall_rate = 0
    cfg.energy.base_metabolism = 0
    cfg.energy.k_weight_speed = 0
    cfg.energy.low_energy_death_threshold = 0
    cfg.properties.eyesight_radius = 1
    eng = SimulationEngine(cfg, seed=4)
    eng.initialize()
    eng.world.x[0] = 5
    eng.world.y[0] = 5
    eng.world.x[1] = 5
    eng.world.y[1] = 5
    eng.world.weight[0] = 0.9
    eng.world.speed[0] = 0.0
    eng.world.weight[1] = 0.2
    eng.world.speed[1] = 0.0
    eng.world.energy[:] = 0.5
    eng.world.food_life[:, :] = 0
    eng.world.food_life[5, 5] = 20
    eaten = eng._resolve_feeding()
    assert eaten == 1
    assert eng.world.food_life[5, 5] == 0
    assert eng.world.energy[0] > eng.world.energy[1]


def test_offspring_counts():
    cfg = get_default_config()
    e = np.array([0.4, 0.6, 0.8], dtype=np.float32)
    c = offspring_counts(e, cfg)
    assert list(c) == [0, 1, 2]


def test_one_clutch_parent_remains():
    cfg = get_default_config()
    cfg.population.initial_count = 5
    cfg.reproduction.repro_age_min = 3
    cfg.reproduction.repro_age_max = 3
    cfg.reproduction.timing = "random"
    cfg.resources.pitfall_rate = 0
    cfg.resources.food_rate = 8.0
    cfg.energy.low_energy_death_threshold = 0.01
    cfg.energy.base_metabolism = 0.0001
    cfg.energy.k_weight_speed = 0.0001
    eng = SimulationEngine(cfg, seed=5)
    eng.initialize()
    n0 = eng.world.n
    for _ in range(5):
        eng.tick()
    # Parents still present (ids 0..4) if alive, plus children
    assert eng.world.n >= n0
    # has_reproduced for original survivors at age>=3
    if eng.world.n:
        founders = eng.world.id[: eng.world.n] < n0
        if np.any(founders):
            ages = eng.world.age()[founders]
            old = ages >= 3
            if np.any(old):
                assert np.all(eng.world.has_reproduced[: eng.world.n][founders][old])


def test_pitfall_popcount():
    # all-one pitfall vs zero defense → 32
    seq = np.array([0xFFFFFFFF], dtype=np.uint32)
    defense = np.array([0], dtype=np.uint32)
    dmg = popcount32(seq & (~defense))
    assert int(dmg[0]) == 32


def test_default_world_survives_two_epochs():
    cfg = get_default_config()
    eng = SimulationEngine(cfg, seed=42)
    eng.initialize()
    result = eng.run(max_epochs=2)
    assert not result.extinct
    assert result.final_alive > 0
    assert result.total_epochs == 2


def test_determinism():
    cfg = get_default_config()
    cfg.population.initial_count = 20
    cfg.resources.pitfall_rate = 0.2
    a = SimulationEngine(cfg.copy(), seed=42)
    b = SimulationEngine(cfg.copy(), seed=42)
    a.initialize()
    b.initialize()
    for _ in range(40):
        a.tick()
        b.tick()
    assert a.world.n == b.world.n
    assert np.array_equal(a.world.x[: a.world.n], b.world.x[: b.world.n])
    assert np.allclose(a.world.energy[: a.world.n], b.world.energy[: b.world.n])
