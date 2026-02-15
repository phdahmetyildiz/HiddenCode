"""
Configuration system for the Evolution Simulator.

Provides a hierarchical dataclass-based config with JSON serialization,
validation, and sensible defaults for all simulation parameters.
"""

from __future__ import annotations

import json
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Sub-config dataclasses (grouped by domain)
# ---------------------------------------------------------------------------

@dataclass
class WorldConfig:
    """Grid and time settings."""
    width: int = 500
    height: int = 500
    seed: int = 42

    def validate(self) -> list[str]:
        errors = []
        if self.width < 10:
            errors.append(f"world.width must be >= 10, got {self.width}")
        if self.height < 10:
            errors.append(f"world.height must be >= 10, got {self.height}")
        if self.width > 10_000:
            errors.append(f"world.width must be <= 10000, got {self.width}")
        if self.height > 10_000:
            errors.append(f"world.height must be <= 10000, got {self.height}")
        return errors


@dataclass
class GeneticsConfig:
    """DNA / Genome settings."""
    dna_length: int = 2048
    encoding: str = "binary"  # "binary" or "gray"

    # Coding regions: list of [start, end] pairs (inclusive start, exclusive end)
    # Default: bits 0-63 = weight/speed, bits 64-127 = reserved, bits 128-159 = defense
    coding_regions: list[list[int]] = field(
        default_factory=lambda: [[0, 64], [64, 128], [128, 160]]
    )

    # Property mapping: which bit ranges map to which properties
    weight_bits: list[int] = field(default_factory=lambda: [0, 32])     # [start, end)
    speed_bits: list[int] = field(default_factory=lambda: [32, 64])     # [start, end)
    defense_bits: list[int] = field(default_factory=lambda: [128, 160]) # [start, end) = 32 bits

    # Mutation rates
    base_mutation_rate: float = 0.01
    stress_mutation_rate: float = 0.20
    stress_mode_coding_only: bool = True

    def validate(self) -> list[str]:
        errors = []
        if self.dna_length < 64:
            errors.append(f"genetics.dna_length must be >= 64, got {self.dna_length}")
        if self.encoding not in ("binary", "gray"):
            errors.append(f"genetics.encoding must be 'binary' or 'gray', got '{self.encoding}'")
        if not (0.0 <= self.base_mutation_rate <= 1.0):
            errors.append(f"genetics.base_mutation_rate must be in [0, 1], got {self.base_mutation_rate}")
        if not (0.0 <= self.stress_mutation_rate <= 1.0):
            errors.append(f"genetics.stress_mutation_rate must be in [0, 1], got {self.stress_mutation_rate}")

        # Validate bit ranges
        for name, bits in [("weight_bits", self.weight_bits),
                           ("speed_bits", self.speed_bits),
                           ("defense_bits", self.defense_bits)]:
            if len(bits) != 2:
                errors.append(f"genetics.{name} must have exactly 2 elements [start, end)")
            elif bits[0] < 0 or bits[1] > self.dna_length or bits[0] >= bits[1]:
                errors.append(f"genetics.{name} invalid range {bits} for dna_length={self.dna_length}")

        # Validate coding regions
        for i, region in enumerate(self.coding_regions):
            if len(region) != 2:
                errors.append(f"genetics.coding_regions[{i}] must have 2 elements [start, end)")
            elif region[0] < 0 or region[1] > self.dna_length or region[0] >= region[1]:
                errors.append(f"genetics.coding_regions[{i}] invalid range {region}")

        return errors


@dataclass
class PropertyConfig:
    """Agent property ranges and limits."""
    weight_init_range: list[float] = field(default_factory=lambda: [0.2, 0.8])
    weight_limits: list[float] = field(default_factory=lambda: [0.1, 1.0])
    speed_init_range: list[float] = field(default_factory=lambda: [0.2, 0.8])
    speed_limits: list[float] = field(default_factory=lambda: [0.1, 1.0])
    eyesight_radius: int = 10

    def validate(self) -> list[str]:
        errors = []
        if self.eyesight_radius < 1:
            errors.append(f"properties.eyesight_radius must be >= 1, got {self.eyesight_radius}")
        if self.eyesight_radius > 100:
            errors.append(f"properties.eyesight_radius must be <= 100, got {self.eyesight_radius}")
        for name, rng in [("weight_init_range", self.weight_init_range),
                          ("speed_init_range", self.speed_init_range)]:
            if len(rng) != 2 or rng[0] >= rng[1]:
                errors.append(f"properties.{name} must be [low, high] with low < high")
        for name, lim in [("weight_limits", self.weight_limits),
                          ("speed_limits", self.speed_limits)]:
            if len(lim) != 2 or lim[0] >= lim[1]:
                errors.append(f"properties.{name} must be [min, max] with min < max")
            elif lim[0] < 0.0 or lim[1] > 1.0:
                errors.append(f"properties.{name} values must be in [0, 1]")
        return errors


@dataclass
class EnergyConfig:
    """Energy mechanics parameters."""
    base_metabolism: float = 0.001
    k_weight_speed: float = 0.01
    food_gain: float = 0.2
    max_pitfall_loss_pct: float = 0.5
    k_defense_cost: float = 0.0001
    defense_cost_enabled: bool = False
    low_energy_death_threshold: float = 0.10

    def validate(self) -> list[str]:
        errors = []
        if self.base_metabolism < 0:
            errors.append(f"energy.base_metabolism must be >= 0, got {self.base_metabolism}")
        if self.k_weight_speed < 0:
            errors.append(f"energy.k_weight_speed must be >= 0, got {self.k_weight_speed}")
        if not (0.0 < self.food_gain <= 1.0):
            errors.append(f"energy.food_gain must be in (0, 1], got {self.food_gain}")
        if not (0.0 <= self.max_pitfall_loss_pct <= 1.0):
            errors.append(f"energy.max_pitfall_loss_pct must be in [0, 1], got {self.max_pitfall_loss_pct}")
        if self.k_defense_cost < 0:
            errors.append(f"energy.k_defense_cost must be >= 0, got {self.k_defense_cost}")
        if not (0.0 <= self.low_energy_death_threshold <= 1.0):
            errors.append(f"energy.low_energy_death_threshold must be in [0, 1], got {self.low_energy_death_threshold}")
        return errors


@dataclass
class PitfallType:
    """Definition of a single pitfall type."""
    name: str = "A"
    sequence: str = "11110000111100001111000011110000"  # 32-char binary string

    def validate(self) -> list[str]:
        errors = []
        if not self.name:
            errors.append("pitfall_type.name must not be empty")
        if len(self.sequence) != 32:
            errors.append(f"pitfall_type '{self.name}' sequence must be 32 chars, got {len(self.sequence)}")
        if not all(c in "01" for c in self.sequence):
            errors.append(f"pitfall_type '{self.name}' sequence must contain only '0' and '1'")
        return errors


@dataclass
class ResourceConfig:
    """Food and pitfall spawning parameters."""
    food_rate: float = 5.0          # expected items per tick
    food_lifespan: int = 50         # ticks before decay
    pitfall_rate: float = 2.0       # expected items per tick
    pitfall_lifespan: int = 100     # ticks before decay
    initial_pitfall_types: list[dict] = field(
        default_factory=lambda: [
            {"name": "A", "sequence": "11110000111100001111000011110000"}
        ]
    )

    def validate(self) -> list[str]:
        errors = []
        if self.food_rate < 0:
            errors.append(f"resources.food_rate must be >= 0, got {self.food_rate}")
        if self.food_lifespan < 1:
            errors.append(f"resources.food_lifespan must be >= 1, got {self.food_lifespan}")
        if self.pitfall_rate < 0:
            errors.append(f"resources.pitfall_rate must be >= 0, got {self.pitfall_rate}")
        if self.pitfall_lifespan < 1:
            errors.append(f"resources.pitfall_lifespan must be >= 1, got {self.pitfall_lifespan}")
        # Validate pitfall types
        for i, pt_dict in enumerate(self.initial_pitfall_types):
            pt = PitfallType(**pt_dict)
            for err in pt.validate():
                errors.append(f"resources.initial_pitfall_types[{i}]: {err}")
        return errors

    def get_pitfall_types(self) -> list[PitfallType]:
        """Convert raw dicts to PitfallType objects."""
        return [PitfallType(**pt) for pt in self.initial_pitfall_types]


@dataclass
class GenerationConfig:
    """Generation lifecycle and reproduction parameters."""
    gen_length: int = 1000                # ticks per generation
    repro_checkpoint_pct: float = 0.70    # primary reproduction at 70%
    survival_check_pct: float = 1.00      # survival check at 100%
    bonus_repro_pct: float = 1.20         # secondary reproduction at 120%
    survival_threshold: float = 0.50      # energy needed to survive past 100%
    repro_energy_low: float = 0.50        # below this: 0 offspring
    repro_energy_high: float = 0.75       # above this: 2 offspring; between low-high: 1

    def validate(self) -> list[str]:
        errors = []
        if self.gen_length < 10:
            errors.append(f"generation.gen_length must be >= 10, got {self.gen_length}")
        if not (0.0 < self.repro_checkpoint_pct < self.survival_check_pct):
            errors.append("generation.repro_checkpoint_pct must be < survival_check_pct")
        if not (self.survival_check_pct < self.bonus_repro_pct):
            errors.append("generation.survival_check_pct must be < bonus_repro_pct")
        if not (0.0 <= self.survival_threshold <= 1.0):
            errors.append(f"generation.survival_threshold must be in [0, 1], got {self.survival_threshold}")
        if not (0.0 <= self.repro_energy_low <= self.repro_energy_high <= 1.0):
            errors.append("generation: need 0 <= repro_energy_low <= repro_energy_high <= 1")
        return errors


@dataclass
class PopulationConfig:
    """Initial population settings."""
    initial_count: int = 200

    def validate(self) -> list[str]:
        errors = []
        if self.initial_count < 2:
            errors.append(f"population.initial_count must be >= 2, got {self.initial_count}")
        if self.initial_count > 1_000_000:
            errors.append(f"population.initial_count must be <= 1000000, got {self.initial_count}")
        return errors


@dataclass
class StressConfig:
    """Stress event settings."""
    trigger_tick: Optional[int] = None  # None = manual only; int = auto-trigger at tick N
    duration_ticks: Optional[int] = None  # None = permanent until manual deactivation; int = auto-off after N ticks
    pitfall_burst_count: int = 50  # number of new pitfalls to spawn immediately on trigger
    post_event_pitfall_types: list[dict] = field(
        default_factory=lambda: [
            {"name": "B", "sequence": "00001111000011110000111100001111"}
        ]
    )
    food_rate_during_stress: Optional[float] = None  # None = unchanged

    def validate(self) -> list[str]:
        errors = []
        if self.trigger_tick is not None and self.trigger_tick < 0:
            errors.append(f"stress.trigger_tick must be >= 0 or null, got {self.trigger_tick}")
        if self.duration_ticks is not None and self.duration_ticks < 1:
            errors.append(f"stress.duration_ticks must be >= 1 or null, got {self.duration_ticks}")
        if self.pitfall_burst_count < 0:
            errors.append(f"stress.pitfall_burst_count must be >= 0, got {self.pitfall_burst_count}")
        for i, pt_dict in enumerate(self.post_event_pitfall_types):
            pt = PitfallType(**pt_dict)
            for err in pt.validate():
                errors.append(f"stress.post_event_pitfall_types[{i}]: {err}")
        if self.food_rate_during_stress is not None and self.food_rate_during_stress < 0:
            errors.append(f"stress.food_rate_during_stress must be >= 0, got {self.food_rate_during_stress}")
        return errors


@dataclass
class VizConfig:
    """Visualization and output settings."""
    mode: str = "headless"              # "headless" or "realtime"
    snapshot_every_gen: bool = True
    realtime_every_n_ticks: int = 10
    output_dir: str = "runs"

    def validate(self) -> list[str]:
        errors = []
        if self.mode not in ("headless", "realtime"):
            errors.append(f"viz.mode must be 'headless' or 'realtime', got '{self.mode}'")
        if self.realtime_every_n_ticks < 1:
            errors.append(f"viz.realtime_every_n_ticks must be >= 1, got {self.realtime_every_n_ticks}")
        return errors


@dataclass
class SweepStabilityConfig:
    """Stability criteria for parameter sweep mode."""
    min_population_pct: float = 0.20    # population must stay above initial * this
    max_population_pct: float = 5.00    # population must stay below initial * this
    check_after_generation: int = 10    # start checking after this generation

    def validate(self) -> list[str]:
        errors = []
        if self.min_population_pct < 0:
            errors.append(f"sweep.stability.min_population_pct must be >= 0, got {self.min_population_pct}")
        if self.max_population_pct <= self.min_population_pct:
            errors.append("sweep.stability.max_population_pct must be > min_population_pct")
        if self.check_after_generation < 1:
            errors.append(f"sweep.stability.check_after_generation must be >= 1, got {self.check_after_generation}")
        return errors


@dataclass
class SweepConfig:
    """Parameter sweep mode settings."""
    runs_per_set: int = 9
    max_generations: int = 99
    base_seed: int = 42
    parallel_workers: int = 4
    early_termination_on_extinction: bool = True
    stability: SweepStabilityConfig = field(default_factory=SweepStabilityConfig)

    def validate(self) -> list[str]:
        errors = []
        if self.runs_per_set < 1:
            errors.append(f"sweep.runs_per_set must be >= 1, got {self.runs_per_set}")
        if self.max_generations < 1:
            errors.append(f"sweep.max_generations must be >= 1, got {self.max_generations}")
        if self.parallel_workers < 1:
            errors.append(f"sweep.parallel_workers must be >= 1, got {self.parallel_workers}")
        errors.extend(self.stability.validate())
        return errors


@dataclass
class GPUConfig:
    """Optional GPU acceleration settings."""
    enabled: bool = False
    device_id: int = 0

    def validate(self) -> list[str]:
        errors = []
        if self.device_id < 0:
            errors.append(f"gpu.device_id must be >= 0, got {self.device_id}")
        return errors


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    Top-level simulation configuration.

    All parameters are adjustable. Nested dataclasses group related settings.
    Load from JSON with `load_config()`, validate with `validate()`.
    """
    world: WorldConfig = field(default_factory=WorldConfig)
    genetics: GeneticsConfig = field(default_factory=GeneticsConfig)
    properties: PropertyConfig = field(default_factory=PropertyConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)

    def validate(self) -> list[str]:
        """Validate all config sections. Returns list of error messages (empty = valid)."""
        errors = []
        for f in fields(self):
            sub = getattr(self, f.name)
            if hasattr(sub, "validate"):
                errors.extend(sub.validate())
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimConfig:
        """Create SimConfig from nested dict, merging with defaults."""
        config = cls()
        _merge_into_dataclass(config, data)
        return config

    def copy(self) -> SimConfig:
        """Deep copy of this config."""
        return deepcopy(self)


# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------

def _merge_into_dataclass(target: Any, source: dict[str, Any]) -> None:
    """
    Recursively merge a dict into a dataclass instance.
    Unknown keys emit a warning but don't raise.
    """
    if not isinstance(source, dict):
        return

    known_fields = {f.name for f in fields(target)}
    for key, value in source.items():
        if key not in known_fields:
            warnings.warn(
                f"Unknown config key '{key}' in section {type(target).__name__} — ignored.",
                UserWarning,
                stacklevel=3,
            )
            continue

        current = getattr(target, key)

        # If the current field is a dataclass, recurse
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_into_dataclass(current, value)
        else:
            setattr(target, key, value)


def load_config(path: str | Path) -> SimConfig:
    """
    Load config from a JSON file. Missing fields use defaults.

    Args:
        path: Path to JSON config file.

    Returns:
        Validated SimConfig instance.

    Raises:
        FileNotFoundError: If path doesn't exist.
        json.JSONDecodeError: If JSON is malformed.
        ValueError: If config values are invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = SimConfig.from_dict(data)

    errors = config.validate()
    if errors:
        msg = "Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)

    return config


def save_config(config: SimConfig, path: str | Path) -> None:
    """Save config to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def get_default_config() -> SimConfig:
    """Return a fresh default config (all defaults, validated)."""
    config = SimConfig()
    errors = config.validate()
    assert not errors, f"Default config is invalid: {errors}"
    return config


def apply_param_override(config: SimConfig, dotted_key: str, value: Any) -> None:
    """
    Apply a single parameter override using dot notation.

    Example:
        apply_param_override(config, "population.initial_count", 1000)
        apply_param_override(config, "resources.food_gain", 50)

    Args:
        config: SimConfig to modify in-place.
        dotted_key: Dot-separated path like "world.width" or "sweep.stability.min_population_pct"
        value: New value to set.

    Raises:
        KeyError: If the path doesn't exist.
    """
    parts = dotted_key.split(".")
    obj = config
    for part in parts[:-1]:
        if not hasattr(obj, part):
            raise KeyError(f"Config path '{dotted_key}' invalid: '{part}' not found in {type(obj).__name__}")
        obj = getattr(obj, part)

    final_key = parts[-1]
    if not hasattr(obj, final_key):
        raise KeyError(f"Config path '{dotted_key}' invalid: '{final_key}' not found in {type(obj).__name__}")

    setattr(obj, final_key, value)
