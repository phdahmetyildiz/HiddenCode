"""
Unit tests for Phase 7: KPI Logging & Metrics.

Tests cover:
- MetricsCollector:
  - KPI computation from known world states
  - Energy statistics (avg, median, min, max, std)
  - Death and birth counting
  - Genetic diversity (identical vs. diverse populations)
  - Defense match rate
  - Unique defense sequences
  - Food/pitfall stats
  - Stress mode flag
  - History tracking
  - Empty world edge cases
- CSVLogger:
  - Header written on first row
  - Incremental appending
  - Write-all mode
  - Read-back verification
- SnapshotManager:
  - Save/load roundtrip
  - List snapshots
  - Missing snapshot error
- RunManager:
  - Directory creation
  - Config copy
  - Integration with CSV and snapshot
  - Run listing
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.config import SimConfig
from src.core.animal import Animal, reset_animal_id_counter
from src.core.dna import DNA
from src.core.food import Food
from src.core.pitfall import Pitfall
from src.core.world import World
from src.simulation.generation import GenerationStats
from src.simulation.metrics import MetricsCollector
from src.logging.csv_logger import CSVLogger
from src.logging.snapshot import SnapshotManager
from src.logging.run_manager import RunManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ids():
    reset_animal_id_counter()
    yield
    reset_animal_id_counter()


@pytest.fixture
def config() -> SimConfig:
    cfg = SimConfig()
    cfg.world.width = 20
    cfg.world.height = 20
    cfg.world.seed = 42
    cfg.genetics.base_mutation_rate = 0.01
    cfg.genetics.stress_mutation_rate = 0.20
    return cfg


@pytest.fixture
def world(config) -> World:
    return World(config)


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory, cleaned up after test."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def place_animal(world: World, config: SimConfig, x: int, y: int, energy: float) -> Animal:
    dna = DNA.create_random(length=config.genetics.dna_length, rng=world.rng)
    animal = Animal(dna=dna, x=x, y=y, config=config, energy=energy)
    world.add_animal(animal)
    return animal


def make_gen_stats(
    generation: int = 0,
    primary_births: int = 0,
    bonus_births: int = 0,
    survival_deaths: int = 0,
    pop_primary: int = 0,
    pop_survival: int = 0,
    pop_bonus: int = 0,
    pop_end: int = 0,
) -> GenerationStats:
    gs = GenerationStats(generation=generation)
    gs.primary_repro_births = primary_births
    gs.bonus_repro_births = bonus_births
    gs.total_births = primary_births + bonus_births
    gs.survival_check_deaths = survival_deaths
    gs.pop_at_primary_repro = pop_primary
    gs.pop_at_survival_check = pop_survival
    gs.pop_at_bonus_repro = pop_bonus
    gs.pop_at_generation_end = pop_end
    return gs


def make_tick_totals(**kwargs) -> dict[str, int]:
    defaults = {
        "food_spawned": 0,
        "food_expired": 0,
        "food_eaten": 0,
        "pitfalls_spawned": 0,
        "pitfalls_expired": 0,
        "pitfall_encounters": 0,
        "pitfall_total_damage": 0,
        "pitfall_zero_damage_encounters": 0,
        "deaths_starvation": 0,
        "deaths_emergency": 0,
        "deaths_pitfall": 0,
        "moves_toward_food": 0,
        "moves_random": 0,
    }
    defaults.update(kwargs)
    return defaults


# ===========================================================================
# MetricsCollector Tests
# ===========================================================================

class TestMetricsCollectorBasic:
    def test_collect_returns_dict(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.7)
        gs = make_gen_stats(generation=0, pop_end=1)
        tick_totals = make_tick_totals()

        kpis = mc.collect(world, gs, tick_totals)
        assert isinstance(kpis, dict)
        assert kpis["generation"] == 0

    def test_all_kpi_names_present(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.7)
        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())

        for name in MetricsCollector.kpi_names():
            assert name in kpis, f"Missing KPI: {name}"

    def test_alive_count(self, config, world):
        mc = MetricsCollector(config)
        for i in range(5):
            place_animal(world, config, i, 0, energy=0.8)
        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())
        assert kpis["alive_count"] == 5

    def test_extinction_flag(self, config, world):
        mc = MetricsCollector(config)
        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())
        assert kpis["extinction_flag"] is True
        assert kpis["alive_count"] == 0


class TestEnergyStatistics:
    def test_known_energies(self, config, world):
        """5 animals with known energies → verify avg, median, min, max, std."""
        mc = MetricsCollector(config)
        energies = [0.2, 0.4, 0.6, 0.8, 1.0]
        for i, e in enumerate(energies):
            place_animal(world, config, i, 0, energy=e)

        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())

        assert kpis["avg_energy"] == pytest.approx(0.6, abs=0.01)
        assert kpis["median_energy"] == pytest.approx(0.6, abs=0.01)
        assert kpis["min_energy"] == pytest.approx(0.2, abs=0.01)
        assert kpis["max_energy"] == pytest.approx(1.0, abs=0.01)
        assert kpis["std_energy"] == pytest.approx(np.std(energies), abs=0.01)

    def test_single_animal_energy(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.75)
        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())

        assert kpis["avg_energy"] == pytest.approx(0.75, abs=0.01)
        assert kpis["std_energy"] == pytest.approx(0.0, abs=0.001)

    def test_empty_world_energy(self, config, world):
        mc = MetricsCollector(config)
        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())
        assert kpis["avg_energy"] == 0.0
        assert kpis["std_energy"] == 0.0


class TestDeathAndBirthCounting:
    def test_birth_counts(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.8)
        gs = make_gen_stats(primary_births=5, bonus_births=3)
        kpis = mc.collect(world, gs, make_tick_totals())

        assert kpis["births_primary"] == 5
        assert kpis["births_bonus"] == 3
        assert kpis["births_total"] == 8

    def test_death_counts(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.8)
        gs = make_gen_stats(survival_deaths=2)
        tick_totals = make_tick_totals(
            deaths_starvation=3,
            deaths_emergency=1,
            deaths_pitfall=2,
        )
        kpis = mc.collect(world, gs, tick_totals)

        assert kpis["deaths_starvation"] == 3
        assert kpis["deaths_emergency"] == 1
        assert kpis["deaths_pitfall"] == 2
        assert kpis["deaths_age"] == 2
        assert kpis["deaths_total"] == 8  # 3+1+2+2


class TestGeneticDiversity:
    def test_identical_animals_zero_diversity(self, config, world):
        """5 animals with identical DNA → diversity = 0."""
        mc = MetricsCollector(config)
        bits = np.ones(config.genetics.dna_length, dtype=np.uint8)
        for i in range(5):
            dna = DNA(length=config.genetics.dna_length, bits=bits.copy())
            animal = Animal(dna=dna, x=i, y=0, config=config, energy=0.8)
            world.add_animal(animal)

        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())
        assert kpis["genetic_diversity"] == 0.0

    def test_diverse_animals_positive_diversity(self, config, world):
        """Animals with different DNA → diversity > 0."""
        mc = MetricsCollector(config)
        rng = np.random.default_rng(42)
        for i in range(10):
            dna = DNA.create_random(length=config.genetics.dna_length, rng=rng)
            animal = Animal(dna=dna, x=i, y=0, config=config, energy=0.8)
            world.add_animal(animal)

        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())
        assert kpis["genetic_diversity"] > 0

    def test_single_animal_zero_diversity(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.8)
        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())
        assert kpis["genetic_diversity"] == 0.0


class TestUniqueDefenseSequences:
    def test_identical_defense(self, config, world):
        mc = MetricsCollector(config)
        bits = np.ones(config.genetics.dna_length, dtype=np.uint8)
        for i in range(5):
            dna = DNA(length=config.genetics.dna_length, bits=bits.copy())
            animal = Animal(dna=dna, x=i, y=0, config=config, energy=0.8)
            world.add_animal(animal)

        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())
        assert kpis["unique_defense_seqs"] == 1

    def test_diverse_defense(self, config, world):
        mc = MetricsCollector(config)
        rng = np.random.default_rng(42)
        for i in range(10):
            dna = DNA.create_random(length=config.genetics.dna_length, rng=rng)
            animal = Animal(dna=dna, x=i, y=0, config=config, energy=0.8)
            world.add_animal(animal)

        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())
        assert kpis["unique_defense_seqs"] >= 1


class TestDefenseMatchRate:
    def test_no_pitfalls_zero_rate(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.8)
        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())
        assert kpis["defense_match_rate"] == 0.0

    def test_perfect_defense_match(self, config, world):
        """Animal with all-1 defense vs all-1 pitfall → 100% match."""
        mc = MetricsCollector(config)
        bits = np.ones(config.genetics.dna_length, dtype=np.uint8)
        dna = DNA(length=config.genetics.dna_length, bits=bits)
        animal = Animal(dna=dna, x=5, y=5, config=config, energy=0.8)
        world.add_animal(animal)

        pitfall = Pitfall.from_string(x=0, y=0, name="A", sequence_str="1" * 32, lifespan=100)
        world.add_pitfall(pitfall)

        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())
        assert kpis["defense_match_rate"] == pytest.approx(1.0, abs=0.01)

    def test_no_defense_match(self, config, world):
        """Animal with all-0 defense vs all-1 pitfall → 0% match."""
        mc = MetricsCollector(config)
        bits = np.zeros(config.genetics.dna_length, dtype=np.uint8)
        dna = DNA(length=config.genetics.dna_length, bits=bits)
        animal = Animal(dna=dna, x=5, y=5, config=config, energy=0.8)
        world.add_animal(animal)

        pitfall = Pitfall.from_string(x=0, y=0, name="A", sequence_str="1" * 32, lifespan=100)
        world.add_pitfall(pitfall)

        gs = make_gen_stats()
        kpis = mc.collect(world, gs, make_tick_totals())
        assert kpis["defense_match_rate"] == pytest.approx(0.0, abs=0.01)


class TestFoodPitfallStats:
    def test_food_stats(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.8)
        world.add_food(Food(x=1, y=1, remaining_lifespan=10, energy_value=0.2))

        tick_totals = make_tick_totals(food_spawned=100, food_eaten=30, food_expired=20)
        gs = make_gen_stats()
        kpis = mc.collect(world, gs, tick_totals)

        assert kpis["food_spawned"] == 100
        assert kpis["food_eaten"] == 30
        assert kpis["food_expired"] == 20
        assert kpis["food_available"] == 1

    def test_pitfall_stats(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.8)

        tick_totals = make_tick_totals(
            pitfall_encounters=50,
            pitfall_total_damage=200,
            pitfall_zero_damage_encounters=10,
            deaths_pitfall=5,
        )
        gs = make_gen_stats()
        kpis = mc.collect(world, gs, tick_totals)

        assert kpis["pitfall_encounters"] == 50
        assert kpis["pitfall_avg_damage"] == pytest.approx(4.0, abs=0.01)
        assert kpis["pitfall_zero_damage"] == 10
        assert kpis["pitfall_deaths_caused"] == 5

    def test_no_encounters_avg_damage_zero(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.8)
        kpis = mc.collect(world, make_gen_stats(), make_tick_totals())
        assert kpis["pitfall_avg_damage"] == 0.0


class TestStressAndMutation:
    def test_stress_mode_flag(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.8)

        world.stress_mode = True
        kpis = mc.collect(world, make_gen_stats(), make_tick_totals())
        assert kpis["stress_mode_active"] is True
        assert kpis["mutation_rate_effective"] == config.genetics.stress_mutation_rate

    def test_normal_mode_flag(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.8)

        world.stress_mode = False
        kpis = mc.collect(world, make_gen_stats(), make_tick_totals())
        assert kpis["stress_mode_active"] is False
        assert kpis["mutation_rate_effective"] == config.genetics.base_mutation_rate


class TestMetricsHistory:
    def test_history_appended(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.8)

        for gen in range(3):
            gs = make_gen_stats(generation=gen)
            mc.collect(world, gs, make_tick_totals())

        assert len(mc.history) == 3
        assert mc.history[0]["generation"] == 0
        assert mc.history[2]["generation"] == 2

    def test_get_last(self, config, world):
        mc = MetricsCollector(config)
        place_animal(world, config, 5, 5, energy=0.8)
        mc.collect(world, make_gen_stats(generation=5), make_tick_totals())
        assert mc.get_last()["generation"] == 5

    def test_get_last_empty(self, config):
        mc = MetricsCollector(config)
        assert mc.get_last() is None

    def test_get_kpi_series(self, config, world):
        mc = MetricsCollector(config)
        for i in range(5):
            place_animal(world, config, i, 0, energy=0.5 + i * 0.1)

        for gen in range(3):
            gs = make_gen_stats(generation=gen, primary_births=gen * 2)
            mc.collect(world, gs, make_tick_totals())

        series = mc.get_kpi_series("births_primary")
        assert series == [0, 2, 4]


# ===========================================================================
# CSVLogger Tests
# ===========================================================================

class TestCSVLogger:
    def test_log_row_creates_file(self, tmp_dir):
        path = tmp_dir / "test.csv"
        logger = CSVLogger(path, columns=["a", "b", "c"])
        logger.log_row({"a": 1, "b": 2, "c": 3})
        assert path.exists()

    def test_header_written(self, tmp_dir):
        path = tmp_dir / "test.csv"
        logger = CSVLogger(path, columns=["a", "b", "c"])
        logger.log_row({"a": 1, "b": 2, "c": 3})

        content = path.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "a,b,c"
        assert lines[1] == "1,2,3"

    def test_append_multiple_rows(self, tmp_dir):
        path = tmp_dir / "test.csv"
        logger = CSVLogger(path, columns=["gen", "pop"])
        logger.log_row({"gen": 0, "pop": 100})
        logger.log_row({"gen": 1, "pop": 120})
        logger.log_row({"gen": 2, "pop": 95})

        rows = logger.read_back()
        assert len(rows) == 3
        assert rows[0]["gen"] == "0"
        assert rows[2]["pop"] == "95"

    def test_log_all_overwrites(self, tmp_dir):
        path = tmp_dir / "test.csv"
        logger = CSVLogger(path, columns=["x", "y"])
        logger.log_row({"x": 1, "y": 2})
        logger.log_all([{"x": 10, "y": 20}, {"x": 30, "y": 40}])

        rows = logger.read_back()
        assert len(rows) == 2
        assert rows[0]["x"] == "10"

    def test_read_back_empty(self, tmp_dir):
        path = tmp_dir / "nonexistent.csv"
        logger = CSVLogger(path)
        assert logger.read_back() == []

    def test_extra_keys_ignored(self, tmp_dir):
        path = tmp_dir / "test.csv"
        logger = CSVLogger(path, columns=["a", "b"])
        logger.log_row({"a": 1, "b": 2, "c": 3, "d": 4})

        rows = logger.read_back()
        assert len(rows) == 1
        assert "c" not in rows[0]

    def test_full_kpi_columns(self, tmp_dir):
        """Log a full KPI dict with all columns from MetricsCollector."""
        path = tmp_dir / "metrics.csv"
        logger = CSVLogger(path)

        # Build a fake KPI dict with all fields
        kpi = {name: 0 for name in MetricsCollector.kpi_names()}
        kpi["generation"] = 0
        kpi["alive_count"] = 42
        logger.log_row(kpi)

        rows = logger.read_back()
        assert len(rows) == 1
        assert rows[0]["alive_count"] == "42"


# ===========================================================================
# SnapshotManager Tests
# ===========================================================================

class TestSnapshotManager:
    def test_save_creates_file(self, config, world, tmp_dir):
        sm = SnapshotManager(tmp_dir)
        place_animal(world, config, 5, 5, energy=0.8)
        world.add_food(Food(x=1, y=1, remaining_lifespan=10, energy_value=0.2))

        path = sm.save(world, generation=0)
        assert path.exists()
        assert "gen_0000.json" in path.name

    def test_save_load_roundtrip(self, config, world, tmp_dir):
        sm = SnapshotManager(tmp_dir)
        place_animal(world, config, 5, 5, energy=0.8)
        world.add_food(Food(x=1, y=1, remaining_lifespan=10, energy_value=0.2))
        world.add_pitfall(
            Pitfall.from_string(x=3, y=3, name="A", sequence_str="1" * 32, lifespan=50)
        )

        sm.save(world, generation=3)
        loaded = sm.load(3)

        assert loaded["generation"] == 3
        assert loaded["alive_count"] == 1
        assert loaded["food_count"] == 1
        assert loaded["pitfall_count"] == 1
        assert len(loaded["animals"]) == 1
        assert loaded["animals"][0]["energy"] == pytest.approx(0.8, abs=0.01)
        assert len(loaded["food"]) == 1
        assert len(loaded["pitfalls"]) == 1

    def test_list_snapshots(self, config, world, tmp_dir):
        sm = SnapshotManager(tmp_dir)
        place_animal(world, config, 5, 5, energy=0.8)

        sm.save(world, generation=0)
        sm.save(world, generation=5)
        sm.save(world, generation=10)

        available = sm.list_snapshots()
        assert available == [0, 5, 10]

    def test_load_missing_raises(self, tmp_dir):
        sm = SnapshotManager(tmp_dir)
        with pytest.raises(FileNotFoundError):
            sm.load(99)

    def test_empty_world_snapshot(self, config, world, tmp_dir):
        sm = SnapshotManager(tmp_dir)
        sm.save(world, generation=0)
        loaded = sm.load(0)
        assert loaded["alive_count"] == 0
        assert loaded["animals"] == []

    def test_pitfall_sequence_preserved(self, config, world, tmp_dir):
        sm = SnapshotManager(tmp_dir)
        seq = "10101010" * 4
        world.add_pitfall(Pitfall.from_string(x=0, y=0, name="X", sequence_str=seq, lifespan=50))
        sm.save(world, generation=0)

        loaded = sm.load(0)
        assert loaded["pitfalls"][0]["sequence"] == seq


# ===========================================================================
# RunManager Tests
# ===========================================================================

class TestRunManager:
    def test_creates_directory(self, config, tmp_dir):
        rm = RunManager(config, base_dir=tmp_dir, run_name="test_run")
        assert rm.run_dir.exists()
        assert (rm.run_dir / "config.json").exists()

    def test_config_saved(self, config, tmp_dir):
        rm = RunManager(config, base_dir=tmp_dir, run_name="test_run")
        with open(rm.config_path) as f:
            saved = json.load(f)
        assert saved["world"]["width"] == config.world.width

    def test_log_generation(self, config, tmp_dir):
        rm = RunManager(config, base_dir=tmp_dir, run_name="test_run")
        rm.log_generation({"generation": 0, "alive_count": 42})
        rm.log_generation({"generation": 1, "alive_count": 50})

        rows = rm.csv_logger.read_back()
        assert len(rows) == 2

    def test_save_snapshot(self, config, world, tmp_dir):
        rm = RunManager(config, base_dir=tmp_dir, run_name="test_run")
        place_animal(world, config, 5, 5, energy=0.8)
        path = rm.save_snapshot(world, generation=0)
        assert path.exists()

    def test_finalize_with_summary(self, config, tmp_dir):
        rm = RunManager(config, base_dir=tmp_dir, run_name="test_run")
        rm.finalize(summary={"total_generations": 5, "extinct": False})

        summary_path = rm.run_dir / "summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            data = json.load(f)
        assert data["total_generations"] == 5

    def test_list_runs(self, config, tmp_dir):
        RunManager(config, base_dir=tmp_dir, run_name="run_001")
        RunManager(config, base_dir=tmp_dir, run_name="run_002")

        runs = RunManager.list_runs(tmp_dir)
        assert "run_001" in runs
        assert "run_002" in runs

    def test_list_runs_empty(self, tmp_dir):
        assert RunManager.list_runs(tmp_dir / "nonexistent") == []

    def test_repr(self, config, tmp_dir):
        rm = RunManager(config, base_dir=tmp_dir, run_name="test_run")
        assert "RunManager" in repr(rm)

    def test_auto_timestamp_name(self, config, tmp_dir):
        """With no run_name, should create a timestamped directory."""
        rm = RunManager(config, base_dir=tmp_dir)
        assert rm.run_dir.exists()
        # Should be a directory with a timestamp-like name
        assert len(rm.run_dir.name) > 8


# ===========================================================================
# Integration: Metrics + CSV + Snapshot
# ===========================================================================

class TestMetricsIntegration:
    def test_full_pipeline(self, config, world, tmp_dir):
        """Collect metrics → log to CSV → save snapshot → verify all."""
        mc = MetricsCollector(config)
        rm = RunManager(config, base_dir=tmp_dir, run_name="integration")

        # Set up world
        for i in range(5):
            place_animal(world, config, i, 0, energy=0.5 + i * 0.1)
        world.add_food(Food(x=10, y=10, remaining_lifespan=20, energy_value=0.3))

        # Collect metrics
        gs = make_gen_stats(generation=0, primary_births=3, pop_end=5)
        tick_totals = make_tick_totals(food_spawned=50, food_eaten=20)
        kpis = mc.collect(world, gs, tick_totals)

        # Log to CSV
        rm.log_generation(kpis)

        # Save snapshot
        rm.save_snapshot(world, generation=0)

        # Verify CSV
        rows = rm.csv_logger.read_back()
        assert len(rows) == 1
        assert rows[0]["alive_count"] == "5"

        # Verify snapshot
        snap = rm.snapshot_manager.load(0)
        assert snap["alive_count"] == 5
        assert len(snap["animals"]) == 5

    def test_multi_generation_pipeline(self, config, world, tmp_dir):
        """Multiple generations → CSV has multiple rows."""
        mc = MetricsCollector(config)
        rm = RunManager(config, base_dir=tmp_dir, run_name="multi_gen")

        for gen in range(3):
            place_animal(world, config, gen, 0, energy=0.8)
            gs = make_gen_stats(generation=gen, primary_births=gen, pop_end=gen + 1)
            kpis = mc.collect(world, gs, make_tick_totals())
            rm.log_generation(kpis)
            rm.save_snapshot(world, generation=gen)

        assert len(mc.history) == 3
        rows = rm.csv_logger.read_back()
        assert len(rows) == 3
        assert rm.snapshot_manager.list_snapshots() == [0, 1, 2]
