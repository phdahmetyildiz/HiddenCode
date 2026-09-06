"""
Simulation configuration for Evolution Simulator v3.

Nested dataclasses, JSON load/save, validation, dotted-key overrides.
Does not import v2 `src.*`.

Author: Cursor Grok 4.6 High Fast
"""

from __future__ import annotations

import json
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


@dataclass
class WorldConfig:
    width: int = 80
    height: int = 80
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
    dna_length: int = 2048
    encoding: str = "binary"
    coding_regions: list[list[int]] = field(
        default_factory=lambda: [[0, 64], [64, 128], [128, 160]]
    )
    weight_bits: list[int] = field(default_factory=lambda: [0, 32])
    speed_bits: list[int] = field(default_factory=lambda: [32, 64])
    repro_bits: list[int] = field(default_factory=lambda: [64, 96])
    defense_bits: list[int] = field(default_factory=lambda: [128, 160])
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
        for name, bits in [
            ("weight_bits", self.weight_bits),
            ("speed_bits", self.speed_bits),
            ("repro_bits", self.repro_bits),
            ("defense_bits", self.defense_bits),
        ]:
            if len(bits) != 2:
                errors.append(f"genetics.{name} must have exactly 2 elements [start, end)")
            elif bits[0] < 0 or bits[1] > self.dna_length or bits[0] >= bits[1]:
                errors.append(f"genetics.{name} invalid range {bits} for dna_length={self.dna_length}")
        for i, region in enumerate(self.coding_regions):
            if len(region) != 2:
                errors.append(f"genetics.coding_regions[{i}] must have 2 elements [start, end)")
            elif region[0] < 0 or region[1] > self.dna_length or region[0] >= region[1]:
                errors.append(f"genetics.coding_regions[{i}] invalid range {region}")
        return errors


@dataclass
class PropertyConfig:
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
        for name, rng in [
            ("weight_init_range", self.weight_init_range),
            ("speed_init_range", self.speed_init_range),
        ]:
            if len(rng) != 2 or rng[0] >= rng[1]:
                errors.append(f"properties.{name} must be [low, high] with low < high")
        for name, lim in [
            ("weight_limits", self.weight_limits),
            ("speed_limits", self.speed_limits),
        ]:
            if len(lim) != 2 or lim[0] >= lim[1]:
                errors.append(f"properties.{name} must be [min, max] with min < max")
            elif lim[0] < 0.0 or lim[1] > 1.0:
                errors.append(f"properties.{name} values must be in [0, 1]")
        return errors


@dataclass
class EnergyConfig:
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
            errors.append(
                f"energy.low_energy_death_threshold must be in [0, 1], got {self.low_energy_death_threshold}"
            )
        return errors


@dataclass
class PitfallType:
    name: str = "A"
    sequence: str = "11110000111100001111000011110000"

    def validate(self) -> list[str]:
        errors = []
        if not self.name:
            errors.append("pitfall_type.name must not be empty")
        if len(self.sequence) != 32:
            errors.append(f"pitfall_type '{self.name}' sequence must be 32 chars, got {len(self.sequence)}")
        if not all(c in "01" for c in self.sequence):
            errors.append(f"pitfall_type '{self.name}' sequence must contain only '0' and '1'")
        return errors

    def as_uint32(self) -> int:
        """Pack sequence[0] as bit 0 (LSB)."""
        value = 0
        for i, ch in enumerate(self.sequence):
            if ch == "1":
                value |= 1 << i
        return value


@dataclass
class ResourceConfig:
    food_rate: float = 4.0
    food_lifespan: int = 50
    pitfall_rate: float = 0.5
    pitfall_lifespan: int = 100
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
        for i, pt_dict in enumerate(self.initial_pitfall_types):
            pt = PitfallType(**pt_dict)
            errors.extend(f"resources.initial_pitfall_types[{i}]: {e}" for e in pt.validate())
        return errors

    def get_pitfall_types(self) -> list[PitfallType]:
        return [PitfallType(**pt) for pt in self.initial_pitfall_types]


@dataclass
class AgingConfig:
    onset: int = 1000
    max_age: int = 1800
    mobility_end: float = 0.05
    absorption_end: float = 0.20
    curve: str = "linear"

    def validate(self) -> list[str]:
        errors = []
        if self.onset < 0:
            errors.append(f"aging.onset must be >= 0, got {self.onset}")
        if self.max_age <= self.onset:
            errors.append(f"aging.max_age must be > onset, got max_age={self.max_age} onset={self.onset}")
        if not (0.0 <= self.mobility_end <= 1.0):
            errors.append(f"aging.mobility_end must be in [0, 1], got {self.mobility_end}")
        if not (0.0 <= self.absorption_end <= 1.0):
            errors.append(f"aging.absorption_end must be in [0, 1], got {self.absorption_end}")
        if self.curve not in ("linear", "quadratic"):
            errors.append(f"aging.curve must be 'linear' or 'quadratic', got '{self.curve}'")
        return errors


@dataclass
class ReproductionConfig:
    timing: str = "genetic"
    repro_age_min: int = 700
    repro_age_max: int = 1100
    repro_energy_low: float = 0.50
    repro_energy_high: float = 0.75

    def validate(self) -> list[str]:
        errors = []
        if self.timing not in ("genetic", "random"):
            errors.append(f"reproduction.timing must be 'genetic' or 'random', got '{self.timing}'")
        if self.repro_age_min < 1:
            errors.append(f"reproduction.repro_age_min must be >= 1, got {self.repro_age_min}")
        if self.repro_age_max < self.repro_age_min:
            errors.append("reproduction.repro_age_max must be >= repro_age_min")
        if not (0.0 <= self.repro_energy_low <= self.repro_energy_high <= 1.0):
            errors.append("reproduction: need 0 <= repro_energy_low <= repro_energy_high <= 1")
        return errors


@dataclass
class MetricsConfig:
    interval: int = 1000
    cull_enabled: bool = False
    survival_threshold: float = 0.50

    def validate(self) -> list[str]:
        errors = []
        if self.interval < 1:
            errors.append(f"metrics.interval must be >= 1, got {self.interval}")
        if not (0.0 <= self.survival_threshold <= 1.0):
            errors.append(f"metrics.survival_threshold must be in [0, 1], got {self.survival_threshold}")
        return errors


@dataclass
class PopulationConfig:
    initial_count: int = 80

    def validate(self) -> list[str]:
        errors = []
        if self.initial_count < 2:
            errors.append(f"population.initial_count must be >= 2, got {self.initial_count}")
        if self.initial_count > 1_000_000:
            errors.append(f"population.initial_count must be <= 1000000, got {self.initial_count}")
        return errors


@dataclass
class StressConfig:
    trigger_tick: Optional[int] = None
    duration_ticks: Optional[int] = None
    pitfall_burst_count: int = 50
    post_event_pitfall_types: list[dict] = field(
        default_factory=lambda: [
            {"name": "B", "sequence": "00001111000011110000111100001111"}
        ]
    )
    food_rate_during_stress: Optional[float] = None

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
            errors.extend(f"stress.post_event_pitfall_types[{i}]: {e}" for e in pt.validate())
        if self.food_rate_during_stress is not None and self.food_rate_during_stress < 0:
            errors.append(f"stress.food_rate_during_stress must be >= 0, got {self.food_rate_during_stress}")
        return errors

    def get_post_event_types(self) -> list[PitfallType]:
        return [PitfallType(**pt) for pt in self.post_event_pitfall_types]


@dataclass
class VizConfig:
    mode: str = "headless"
    snapshot_every_epoch: bool = True
    render_every_n_ticks: int = 1
    cell_size: int = 8
    output_dir: str = "runs"

    def validate(self) -> list[str]:
        errors = []
        if self.mode not in ("headless", "watch"):
            errors.append(f"viz.mode must be 'headless' or 'watch', got '{self.mode}'")
        if self.render_every_n_ticks < 1:
            errors.append(f"viz.render_every_n_ticks must be >= 1, got {self.render_every_n_ticks}")
        if self.cell_size < 1:
            errors.append(f"viz.cell_size must be >= 1, got {self.cell_size}")
        return errors


@dataclass
class SweepStabilityConfig:
    min_population_pct: float = 0.20
    max_population_pct: float = 5.00
    check_after_epoch: int = 10

    def validate(self) -> list[str]:
        errors = []
        if self.min_population_pct < 0:
            errors.append(f"sweep.stability.min_population_pct must be >= 0, got {self.min_population_pct}")
        if self.max_population_pct <= self.min_population_pct:
            errors.append("sweep.stability.max_population_pct must be > min_population_pct")
        if self.check_after_epoch < 1:
            errors.append(f"sweep.stability.check_after_epoch must be >= 1, got {self.check_after_epoch}")
        return errors


@dataclass
class SweepConfig:
    runs_per_set: int = 9
    max_epochs: int = 99
    base_seed: int = 42
    parallel_workers: int = 4
    early_termination_on_extinction: bool = True
    stability: SweepStabilityConfig = field(default_factory=SweepStabilityConfig)

    def validate(self) -> list[str]:
        errors = []
        if self.runs_per_set < 1:
            errors.append(f"sweep.runs_per_set must be >= 1, got {self.runs_per_set}")
        if self.max_epochs < 1:
            errors.append(f"sweep.max_epochs must be >= 1, got {self.max_epochs}")
        if self.parallel_workers < 1:
            errors.append(f"sweep.parallel_workers must be >= 1, got {self.parallel_workers}")
        errors.extend(self.stability.validate())
        return errors


@dataclass
class PerfConfig:
    max_animals: int = 800
    backend: str = "numpy"

    def validate(self) -> list[str]:
        errors = []
        if self.max_animals < 2:
            errors.append(f"perf.max_animals must be >= 2, got {self.max_animals}")
        if self.backend not in ("numpy", "numba", "cuda", "numba_cuda"):
            errors.append(
                f"perf.backend must be 'numpy', 'numba', 'cuda', or 'numba_cuda', got '{self.backend}'"
            )
        return errors


@dataclass
class SimConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    genetics: GeneticsConfig = field(default_factory=GeneticsConfig)
    properties: PropertyConfig = field(default_factory=PropertyConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    aging: AgingConfig = field(default_factory=AgingConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)

    def validate(self) -> list[str]:
        errors: list[str] = []
        for f in fields(self):
            sub = getattr(self, f.name)
            if hasattr(sub, "validate"):
                errors.extend(sub.validate())
        if self.reproduction.repro_age_max >= self.aging.max_age:
            errors.append("reproduction.repro_age_max must be < aging.max_age")
        if self.population.initial_count > self.perf.max_animals:
            errors.append("population.initial_count must be <= perf.max_animals")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimConfig:
        config = cls()
        _merge_into_dataclass(config, data)
        return config

    def copy(self) -> SimConfig:
        return deepcopy(self)


def _merge_into_dataclass(target: Any, source: dict[str, Any]) -> None:
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
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_into_dataclass(current, value)
        else:
            setattr(target, key, value)


def load_config(path: str | Path) -> SimConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    config = SimConfig.from_dict(data)
    errors = config.validate()
    if errors:
        raise ValueError("Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))
    return config


def save_config(config: SimConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def get_default_config() -> SimConfig:
    config = SimConfig()
    errors = config.validate()
    assert not errors, f"Default config is invalid: {errors}"
    return config


def apply_param_override(config: SimConfig, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    obj = config
    for part in parts[:-1]:
        if not hasattr(obj, part):
            raise KeyError(f"Config path '{dotted_key}' invalid: '{part}' not found")
        obj = getattr(obj, part)
    final_key = parts[-1]
    if not hasattr(obj, final_key):
        raise KeyError(f"Config path '{dotted_key}' invalid: '{final_key}' not found")
    setattr(obj, final_key, value)
