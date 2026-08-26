"""Config + livability tests."""

import json
import warnings
from pathlib import Path

import pytest

from src.config import (
    SimConfig,
    apply_param_override,
    get_default_config,
    load_config,
    save_config,
)
from src.livability import evaluate


def test_defaults_validate():
    cfg = get_default_config()
    assert cfg.world.width == 80
    assert cfg.population.initial_count == 80
    assert cfg.aging.onset == 1000
    assert cfg.reproduction.repro_age_min == 700


def test_partial_json_fills_defaults(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"world": {"width": 40}}', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.world.width == 40
    assert cfg.world.height == 80


def test_invalid_grid_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"world": {"width": -1}}', encoding="utf-8")
    with pytest.raises(ValueError, match="world.width"):
        load_config(p)


def test_onset_not_less_than_max_age(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"aging": {"onset": 2000, "max_age": 1000}}', encoding="utf-8")
    with pytest.raises(ValueError, match="max_age"):
        load_config(p)


def test_mutation_rate_bounds(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"genetics": {"base_mutation_rate": 1.5}}', encoding="utf-8")
    with pytest.raises(ValueError, match="base_mutation_rate"):
        load_config(p)


def test_roundtrip(tmp_path):
    cfg = get_default_config()
    p = tmp_path / "out.json"
    save_config(cfg, p)
    cfg2 = load_config(p)
    assert cfg2.to_dict() == cfg.to_dict()


def test_unknown_key_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SimConfig.from_dict({"not_a_section": 1})
        assert any("Unknown config key" in str(x.message) for x in w)


def test_load_default_file():
    path = Path(__file__).resolve().parents[1] / "config" / "default_config.json"
    cfg = load_config(path)
    assert cfg.aging.onset == 1000


def test_livability_v3_default_no_sparse_warning():
    report = evaluate(get_default_config())
    assert report.expected_food_in_sight >= 0.5
    sparse = [w for w in report.warns if "eyesight" in w or "shared food" in w]
    assert sparse == []


def test_livability_v2_size_warns():
    cfg = get_default_config()
    cfg.world.width = 500
    cfg.world.height = 500
    cfg.population.initial_count = 200
    cfg.resources.food_rate = 5.0
    cfg.perf.max_animals = 2000
    report = evaluate(cfg)
    assert report.warns
    assert any("eyesight" in w or "shared food" in w for w in report.warns)


def test_override():
    cfg = get_default_config()
    apply_param_override(cfg, "population.initial_count", 40)
    assert cfg.population.initial_count == 40
