"""
Pitfall adaptation score (coverage of dangerous bits).

Author: Cursor Grok 4.6 High Fast
"""

import numpy as np
import pytest

from src.adaptation import encounter_adaptation
from src.config import PitfallType, get_default_config
from src.engine import SimulationEngine, TickStats
from src.watch import format_pitfall_line, format_totals_line, write_adaptation_csv
from src.world import World


def test_adaptation_full_cover():
    seq = np.array([0b1111], dtype=np.uint32)
    defense = np.array([0b1111], dtype=np.uint32)
    assert encounter_adaptation(seq, defense)[0] == pytest.approx(1.0)


def test_adaptation_none_cover():
    seq = np.array([0b1111], dtype=np.uint32)
    defense = np.array([0], dtype=np.uint32)
    assert encounter_adaptation(seq, defense)[0] == pytest.approx(0.0)


def test_adaptation_ignores_safe_zero_bits():
    seq = np.array([np.uint32(1)], dtype=np.uint32)
    unarmed = np.array([np.uint32(0xFFFFFFFE)], dtype=np.uint32)
    armed = np.array([np.uint32(1)], dtype=np.uint32)
    assert encounter_adaptation(seq, unarmed)[0] == pytest.approx(0.0)
    assert encounter_adaptation(seq, armed)[0] == pytest.approx(1.0)


def test_adaptation_half_of_dangerous_bits():
    seq = np.array([0b1111], dtype=np.uint32)
    defense = np.array([0b0011], dtype=np.uint32)
    assert encounter_adaptation(seq, defense)[0] == pytest.approx(0.5)


def test_empty_pitfall_is_vacuously_adapted():
    seq = np.array([0], dtype=np.uint32)
    defense = np.array([0], dtype=np.uint32)
    assert encounter_adaptation(seq, defense)[0] == pytest.approx(1.0)


def test_pitfall_counts_by_name():
    cfg = get_default_config()
    cfg.world.width = 8
    cfg.world.height = 8
    cfg.population.initial_count = 1
    cfg.resources.pitfall_rate = 0
    w = World(cfg, rng=np.random.default_rng(0))
    w.initialize_population(1)
    a = PitfallType(name="A", sequence="11110000111100001111000011110000")
    b = PitfallType(name="B", sequence="00001111000011110000111100001111")
    w.pitfall_life[0, 0] = 10
    w.pitfall_seq[0, 0] = np.uint32(a.as_uint32())
    w.pitfall_type_id[0, 0] = w.register_pitfall_type(a)
    w.pitfall_life[1, 0] = 10
    w.pitfall_seq[1, 0] = np.uint32(b.as_uint32())
    w.pitfall_type_id[1, 0] = w.register_pitfall_type(b)
    w.pitfall_life[2, 0] = 10
    w.pitfall_seq[2, 0] = np.uint32(a.as_uint32())
    w.pitfall_type_id[2, 0] = w.register_pitfall_type(a)
    counts = w.pitfall_counts_by_name()
    assert counts["A"] == 2
    assert counts["B"] == 1


def test_lifetime_counts_accumulate():
    cfg = get_default_config()
    cfg.world.width = 24
    cfg.world.height = 24
    cfg.population.initial_count = 12
    cfg.resources.food_rate = 8.0
    eng = SimulationEngine(cfg, seed=1)
    eng.initialize()
    for _ in range(30):
        eng.tick()
    assert eng.lifetime.food_spawned >= eng.tick_stats.food_spawned
    assert len(eng.adaptation_series) >= 2


def test_hud_lines_use_totals():
    life = TickStats(births=12, deaths_emergency=3, deaths_starvation=1, deaths_max_age=4)
    line = format_totals_line(life)
    assert "births 12" in line
    assert "d.em 3" in line
    life.pitfall_encounters = 10
    life.pitfall_adapt_sum = 4.0
    life.pitfall_zero_damage = 2
    life.deaths_pitfall = 1
    pit = format_pitfall_line({"A": 5, "B": 2}, life, TickStats())
    assert "A:5" in pit
    assert "B:2" in pit
    assert "enc 10" in pit
    assert "d.pit 1" in pit
    assert "adapt 0.40" in pit
    assert "full 20%" in pit


def test_write_adaptation_csv(tmp_path):
    path = tmp_path / "adaptation.csv"
    write_adaptation_csv(path, [0.1, 0.2, float("nan")])
    text = path.read_text(encoding="utf-8")
    assert "adaptation_cum" in text
    assert "0.100000" in text
