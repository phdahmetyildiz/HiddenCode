"""
Unit tests for the configuration system.

Tests cover:
- Default config creation and validation
- JSON load/save roundtrip
- Partial config loading (missing fields use defaults)
- Invalid value detection
- Unknown key warnings
- Dot-notation parameter overrides
- Edge cases
"""

import json
import warnings
from pathlib import Path

import pytest

from src.core.config import (
    SimConfig,
    WorldConfig,
    GeneticsConfig,
    PropertyConfig,
    EnergyConfig,
    ResourceConfig,
    GenerationConfig,
    PopulationConfig,
    StressConfig,
    VizConfig,
    SweepConfig,
    SweepStabilityConfig,
    GPUConfig,
    PitfallType,
    load_config,
    save_config,
    get_default_config,
    apply_param_override,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> SimConfig:
    """Fresh default config."""
    return get_default_config()


@pytest.fixture
def tmp_config_path(tmp_path) -> Path:
    """Temporary directory for config files."""
    return tmp_path / "test_config.json"


@pytest.fixture
def minimal_config_path(tmp_path) -> Path:
    """Config file with only a few overrides."""
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps({
        "world": {"width": 100, "height": 100},
        "population": {"initial_count": 50}
    }))
    return path


@pytest.fixture
def invalid_config_path(tmp_path) -> Path:
    """Config file with invalid values."""
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({
        "world": {"width": -5, "height": 0},
        "genetics": {"base_mutation_rate": 2.5}
    }))
    return path


# ---------------------------------------------------------------------------
# Default Config Tests
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    """Tests for default configuration creation."""

    def test_default_config_valid(self, default_config: SimConfig):
        """Default config must pass validation with zero errors."""
        errors = default_config.validate()
        assert errors == [], f"Default config has errors: {errors}"

    def test_default_world_values(self, default_config: SimConfig):
        """Verify default world parameters."""
        assert default_config.world.width == 500
        assert default_config.world.height == 500
        assert default_config.world.seed == 42

    def test_default_genetics_values(self, default_config: SimConfig):
        """Verify default genetics parameters."""
        assert default_config.genetics.dna_length == 2048
        assert default_config.genetics.encoding == "binary"
        assert default_config.genetics.base_mutation_rate == 0.01
        assert default_config.genetics.stress_mutation_rate == 0.20
        assert default_config.genetics.weight_bits == [0, 32]
        assert default_config.genetics.speed_bits == [32, 64]
        assert default_config.genetics.defense_bits == [128, 160]

    def test_default_energy_values(self, default_config: SimConfig):
        """Verify default energy parameters."""
        assert default_config.energy.base_metabolism == 0.001
        assert default_config.energy.k_weight_speed == 0.01
        assert default_config.energy.food_gain == 0.2
        assert default_config.energy.max_pitfall_loss_pct == 0.5
        assert default_config.energy.defense_cost_enabled is False

    def test_default_population_values(self, default_config: SimConfig):
        """Verify default population parameters."""
        assert default_config.population.initial_count == 200

    def test_default_generation_values(self, default_config: SimConfig):
        """Verify default generation parameters."""
        assert default_config.generation.gen_length == 1000
        assert default_config.generation.repro_checkpoint_pct == 0.70
        assert default_config.generation.survival_threshold == 0.50

    def test_default_sweep_values(self, default_config: SimConfig):
        """Verify default sweep parameters."""
        assert default_config.sweep.runs_per_set == 9
        assert default_config.sweep.max_generations == 99
        assert default_config.sweep.parallel_workers == 4
        assert default_config.sweep.early_termination_on_extinction is True

    def test_default_stability_values(self, default_config: SimConfig):
        """Verify default stability band parameters."""
        assert default_config.sweep.stability.min_population_pct == 0.20
        assert default_config.sweep.stability.max_population_pct == 5.00
        assert default_config.sweep.stability.check_after_generation == 10

    def test_default_resource_values(self, default_config: SimConfig):
        """Verify default resource parameters."""
        assert default_config.resources.food_rate == 5.0
        assert default_config.resources.food_lifespan == 50
        assert default_config.resources.pitfall_rate == 2.0
        assert default_config.resources.pitfall_lifespan == 100
        assert len(default_config.resources.initial_pitfall_types) == 1

    def test_default_pitfall_type(self, default_config: SimConfig):
        """Verify default pitfall type A."""
        pt = default_config.resources.get_pitfall_types()[0]
        assert pt.name == "A"
        assert len(pt.sequence) == 32
        assert all(c in "01" for c in pt.sequence)


# ---------------------------------------------------------------------------
# JSON Load / Save Tests
# ---------------------------------------------------------------------------

class TestConfigIO:
    """Tests for config file I/O."""

    def test_save_and_load_roundtrip(self, default_config: SimConfig, tmp_config_path: Path):
        """Save config → load it back → values must match."""
        save_config(default_config, tmp_config_path)
        loaded = load_config(tmp_config_path)

        assert loaded.world.width == default_config.world.width
        assert loaded.world.height == default_config.world.height
        assert loaded.genetics.dna_length == default_config.genetics.dna_length
        assert loaded.population.initial_count == default_config.population.initial_count
        assert loaded.energy.food_gain == default_config.energy.food_gain
        assert loaded.sweep.runs_per_set == default_config.sweep.runs_per_set

    def test_save_creates_parent_dirs(self, default_config: SimConfig, tmp_path: Path):
        """save_config should create parent directories if needed."""
        deep_path = tmp_path / "a" / "b" / "c" / "config.json"
        save_config(default_config, deep_path)
        assert deep_path.exists()

    def test_load_partial_config_uses_defaults(self, minimal_config_path: Path):
        """Loading a partial config should fill missing fields with defaults."""
        config = load_config(minimal_config_path)

        # Overridden values
        assert config.world.width == 100
        assert config.world.height == 100
        assert config.population.initial_count == 50

        # Default values (not in file)
        assert config.genetics.dna_length == 2048
        assert config.energy.base_metabolism == 0.001
        assert config.generation.gen_length == 1000

    def test_load_nonexistent_file_raises(self):
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_path/config.json")

    def test_load_malformed_json_raises(self, tmp_path: Path):
        """Loading invalid JSON should raise json.JSONDecodeError."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("{invalid json content!!}")
        with pytest.raises(json.JSONDecodeError):
            load_config(bad_path)

    def test_load_invalid_values_raises(self, invalid_config_path: Path):
        """Loading config with invalid values should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid configuration"):
            load_config(invalid_config_path)

    def test_save_load_preserves_nested_structures(self, default_config: SimConfig, tmp_config_path: Path):
        """Nested structures like pitfall_types and coding_regions survive roundtrip."""
        default_config.resources.initial_pitfall_types = [
            {"name": "X", "sequence": "10101010101010101010101010101010"},
            {"name": "Y", "sequence": "01010101010101010101010101010101"},
        ]
        save_config(default_config, tmp_config_path)
        loaded = load_config(tmp_config_path)

        assert len(loaded.resources.initial_pitfall_types) == 2
        assert loaded.resources.initial_pitfall_types[0]["name"] == "X"
        assert loaded.resources.initial_pitfall_types[1]["name"] == "Y"


# ---------------------------------------------------------------------------
# Unknown Keys Warning Test
# ---------------------------------------------------------------------------

class TestUnknownKeys:
    """Tests for handling unknown config keys."""

    def test_unknown_top_level_key_warns(self, tmp_path: Path):
        """Unknown keys in config should produce a warning but not crash."""
        path = tmp_path / "extra.json"
        path.write_text(json.dumps({
            "world": {"width": 100, "height": 100},
            "unknown_section": {"foo": "bar"}
        }))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = load_config(path)
            assert any("Unknown config key" in str(warning.message) for warning in w)

        # Config should still be valid
        assert config.world.width == 100

    def test_unknown_nested_key_warns(self, tmp_path: Path):
        """Unknown keys within a known section should warn."""
        path = tmp_path / "extra_nested.json"
        path.write_text(json.dumps({
            "world": {"width": 100, "height": 100, "nonexistent_param": 999}
        }))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = load_config(path)
            assert any("Unknown config key" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for config validation rules."""

    def test_world_width_too_small(self):
        config = SimConfig()
        config.world.width = 5
        errors = config.validate()
        assert any("world.width" in e for e in errors)

    def test_world_width_too_large(self):
        config = SimConfig()
        config.world.width = 20_000
        errors = config.validate()
        assert any("world.width" in e for e in errors)

    def test_world_height_too_small(self):
        config = SimConfig()
        config.world.height = 3
        errors = config.validate()
        assert any("world.height" in e for e in errors)

    def test_mutation_rate_negative(self):
        config = SimConfig()
        config.genetics.base_mutation_rate = -0.5
        errors = config.validate()
        assert any("base_mutation_rate" in e for e in errors)

    def test_mutation_rate_above_one(self):
        config = SimConfig()
        config.genetics.base_mutation_rate = 1.5
        errors = config.validate()
        assert any("base_mutation_rate" in e for e in errors)

    def test_invalid_encoding(self):
        config = SimConfig()
        config.genetics.encoding = "quaternary"
        errors = config.validate()
        assert any("encoding" in e for e in errors)

    def test_food_gain_zero(self):
        config = SimConfig()
        config.energy.food_gain = 0.0
        errors = config.validate()
        assert any("food_gain" in e for e in errors)

    def test_food_gain_negative(self):
        config = SimConfig()
        config.energy.food_gain = -0.5
        errors = config.validate()
        assert any("food_gain" in e for e in errors)

    def test_population_count_zero(self):
        config = SimConfig()
        config.population.initial_count = 0
        errors = config.validate()
        assert any("initial_count" in e for e in errors)

    def test_population_count_one(self):
        """At least 2 animals needed for meaningful simulation."""
        config = SimConfig()
        config.population.initial_count = 1
        errors = config.validate()
        assert any("initial_count" in e for e in errors)

    def test_gen_length_too_short(self):
        config = SimConfig()
        config.generation.gen_length = 5
        errors = config.validate()
        assert any("gen_length" in e for e in errors)

    def test_repro_pct_greater_than_survival(self):
        """repro_checkpoint_pct must be less than survival_check_pct."""
        config = SimConfig()
        config.generation.repro_checkpoint_pct = 1.5
        config.generation.survival_check_pct = 1.0
        errors = config.validate()
        assert any("repro_checkpoint_pct" in e for e in errors)

    def test_survival_pct_greater_than_bonus(self):
        config = SimConfig()
        config.generation.survival_check_pct = 1.5
        config.generation.bonus_repro_pct = 1.2
        errors = config.validate()
        assert any("survival_check_pct" in e for e in errors)

    def test_repro_energy_low_greater_than_high(self):
        config = SimConfig()
        config.generation.repro_energy_low = 0.9
        config.generation.repro_energy_high = 0.5
        errors = config.validate()
        assert any("repro_energy" in e for e in errors)

    def test_eyesight_radius_zero(self):
        config = SimConfig()
        config.properties.eyesight_radius = 0
        errors = config.validate()
        assert any("eyesight_radius" in e for e in errors)

    def test_eyesight_radius_too_large(self):
        config = SimConfig()
        config.properties.eyesight_radius = 200
        errors = config.validate()
        assert any("eyesight_radius" in e for e in errors)

    def test_invalid_pitfall_sequence_length(self):
        config = SimConfig()
        config.resources.initial_pitfall_types = [
            {"name": "bad", "sequence": "1111"}  # only 4 chars, need 32
        ]
        errors = config.validate()
        assert any("sequence must be 32 chars" in e for e in errors)

    def test_invalid_pitfall_sequence_chars(self):
        config = SimConfig()
        config.resources.initial_pitfall_types = [
            {"name": "bad", "sequence": "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"}
        ]
        errors = config.validate()
        assert any("only '0' and '1'" in e for e in errors)

    def test_sweep_runs_per_set_zero(self):
        config = SimConfig()
        config.sweep.runs_per_set = 0
        errors = config.validate()
        assert any("runs_per_set" in e for e in errors)

    def test_stability_band_inverted(self):
        config = SimConfig()
        config.sweep.stability.min_population_pct = 5.0
        config.sweep.stability.max_population_pct = 0.2
        errors = config.validate()
        assert any("max_population_pct" in e for e in errors)

    def test_valid_config_no_errors(self, default_config: SimConfig):
        """A valid config should produce zero errors."""
        errors = default_config.validate()
        assert errors == []

    def test_multiple_errors_reported(self):
        """Multiple invalid fields should all be reported."""
        config = SimConfig()
        config.world.width = -1
        config.world.height = -1
        config.genetics.base_mutation_rate = 5.0
        config.population.initial_count = 0
        errors = config.validate()
        assert len(errors) >= 4

    def test_invalid_weight_bits_range(self):
        config = SimConfig()
        config.genetics.weight_bits = [100, 50]  # start >= end
        errors = config.validate()
        assert any("weight_bits" in e for e in errors)

    def test_invalid_coding_region(self):
        config = SimConfig()
        config.genetics.coding_regions = [[0, 3000]]  # exceeds dna_length
        errors = config.validate()
        assert any("coding_regions" in e for e in errors)

    def test_defense_cost_negative(self):
        config = SimConfig()
        config.energy.k_defense_cost = -0.01
        errors = config.validate()
        assert any("k_defense_cost" in e for e in errors)

    def test_viz_mode_invalid(self):
        config = SimConfig()
        config.viz.mode = "turbo"
        errors = config.validate()
        assert any("viz.mode" in e for e in errors)

    def test_gpu_device_negative(self):
        config = SimConfig()
        config.gpu.device_id = -1
        errors = config.validate()
        assert any("device_id" in e for e in errors)


# ---------------------------------------------------------------------------
# Parameter Override Tests
# ---------------------------------------------------------------------------

class TestParamOverride:
    """Tests for dot-notation parameter overrides."""

    def test_override_top_level(self, default_config: SimConfig):
        apply_param_override(default_config, "world.width", 200)
        assert default_config.world.width == 200

    def test_override_nested(self, default_config: SimConfig):
        apply_param_override(default_config, "sweep.stability.min_population_pct", 0.10)
        assert default_config.sweep.stability.min_population_pct == 0.10

    def test_override_population(self, default_config: SimConfig):
        apply_param_override(default_config, "population.initial_count", 1000)
        assert default_config.population.initial_count == 1000

    def test_override_invalid_path_raises(self, default_config: SimConfig):
        with pytest.raises(KeyError):
            apply_param_override(default_config, "world.nonexistent", 42)

    def test_override_invalid_section_raises(self, default_config: SimConfig):
        with pytest.raises(KeyError):
            apply_param_override(default_config, "nonexistent.width", 42)

    def test_override_preserves_other_values(self, default_config: SimConfig):
        original_height = default_config.world.height
        apply_param_override(default_config, "world.width", 200)
        assert default_config.world.height == original_height


# ---------------------------------------------------------------------------
# Copy / Deep Copy Tests
# ---------------------------------------------------------------------------

class TestConfigCopy:
    """Tests for config deep copying."""

    def test_copy_is_independent(self, default_config: SimConfig):
        """Modifying a copy should not affect the original."""
        copy = default_config.copy()
        copy.world.width = 999
        assert default_config.world.width == 500

    def test_copy_preserves_values(self, default_config: SimConfig):
        copy = default_config.copy()
        assert copy.world.width == default_config.world.width
        assert copy.genetics.dna_length == default_config.genetics.dna_length
        assert copy.population.initial_count == default_config.population.initial_count

    def test_copy_nested_independence(self, default_config: SimConfig):
        """Nested structures should also be independent."""
        copy = default_config.copy()
        copy.resources.initial_pitfall_types.append(
            {"name": "Z", "sequence": "00000000000000000000000000000000"}
        )
        assert len(default_config.resources.initial_pitfall_types) == 1
        assert len(copy.resources.initial_pitfall_types) == 2


# ---------------------------------------------------------------------------
# PitfallType Tests
# ---------------------------------------------------------------------------

class TestPitfallType:
    """Tests for PitfallType validation."""

    def test_valid_pitfall_type(self):
        pt = PitfallType(name="A", sequence="11110000111100001111000011110000")
        assert pt.validate() == []

    def test_empty_name(self):
        pt = PitfallType(name="", sequence="11110000111100001111000011110000")
        errors = pt.validate()
        assert any("name" in e for e in errors)

    def test_wrong_length_sequence(self):
        pt = PitfallType(name="A", sequence="1111")
        errors = pt.validate()
        assert any("32 chars" in e for e in errors)

    def test_invalid_chars_in_sequence(self):
        pt = PitfallType(name="A", sequence="1111000011110000111100001111ABCD")
        errors = pt.validate()
        assert any("only '0' and '1'" in e for e in errors)

    def test_all_zeros_sequence(self):
        pt = PitfallType(name="Zero", sequence="0" * 32)
        assert pt.validate() == []

    def test_all_ones_sequence(self):
        pt = PitfallType(name="Max", sequence="1" * 32)
        assert pt.validate() == []


# ---------------------------------------------------------------------------
# to_dict / from_dict Tests
# ---------------------------------------------------------------------------

class TestSerialization:
    """Tests for dict serialization."""

    def test_to_dict_returns_dict(self, default_config: SimConfig):
        d = default_config.to_dict()
        assert isinstance(d, dict)
        assert "world" in d
        assert "genetics" in d
        assert "population" in d

    def test_from_dict_roundtrip(self, default_config: SimConfig):
        d = default_config.to_dict()
        restored = SimConfig.from_dict(d)
        assert restored.world.width == default_config.world.width
        assert restored.genetics.dna_length == default_config.genetics.dna_length
        assert restored.sweep.stability.min_population_pct == default_config.sweep.stability.min_population_pct

    def test_from_empty_dict_uses_defaults(self):
        config = SimConfig.from_dict({})
        assert config.world.width == 500
        assert config.genetics.dna_length == 2048

    def test_from_dict_with_overrides(self):
        config = SimConfig.from_dict({
            "world": {"width": 100},
            "population": {"initial_count": 500}
        })
        assert config.world.width == 100
        assert config.world.height == 500  # default
        assert config.population.initial_count == 500
