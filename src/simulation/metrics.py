"""
KPI Metrics collection for the Evolution Simulator.

MetricsCollector gathers per-generation Key Performance Indicators (KPIs)
from the world state and accumulated tick statistics. It produces a flat
dictionary per generation suitable for CSV export and analysis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.core.config import SimConfig
from src.core.world import World
from src.core.animal import Animal
from src.simulation.generation import GenerationStats


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Collects and computes KPIs per generation.

    Usage:
      1. At generation end, call `collect(world, gen_stats, tick_stats_totals)`
      2. Resulting dict is appended to `history`
      3. Call `get_history()` to retrieve all collected snapshots

    Attributes:
        config: Simulation configuration.
        history: List of KPI dicts, one per generation.
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.history: list[dict] = []

    def collect(
        self,
        world: World,
        gen_stats: GenerationStats,
        tick_stats_totals: dict[str, int],
    ) -> dict:
        """
        Compute all KPIs for the current generation and append to history.

        Args:
            world: Current world state (for live population stats).
            gen_stats: Generation lifecycle stats (births, survival deaths).
            tick_stats_totals: Accumulated tick counters for this generation
                               (from engine.get_accumulated_stats()).

        Returns:
            Dict of KPI_name → value.
        """
        kpis: dict = {}

        # --- Population ---
        alive = world.get_alive_animals()
        kpis["generation"] = gen_stats.generation
        kpis["alive_count"] = len(alive)
        kpis["extinction_flag"] = len(alive) == 0

        # --- Births ---
        kpis["births_primary"] = gen_stats.primary_repro_births
        kpis["births_bonus"] = gen_stats.bonus_repro_births
        kpis["births_total"] = gen_stats.total_births
        kpis["parents_primary"] = gen_stats.parents_at_primary
        kpis["parents_bonus"] = gen_stats.parents_at_bonus

        # --- Deaths (from tick stats) ---
        kpis["deaths_starvation"] = tick_stats_totals.get("deaths_starvation", 0)
        kpis["deaths_emergency"] = tick_stats_totals.get("deaths_emergency", 0)
        kpis["deaths_pitfall"] = tick_stats_totals.get("deaths_pitfall", 0)
        kpis["deaths_age"] = gen_stats.survival_check_deaths
        kpis["deaths_total"] = (
            kpis["deaths_starvation"]
            + kpis["deaths_emergency"]
            + kpis["deaths_pitfall"]
            + kpis["deaths_age"]
        )

        # --- Energy statistics ---
        if alive:
            energies = np.array([a.energy for a in alive])
            kpis["avg_energy"] = float(np.mean(energies))
            kpis["median_energy"] = float(np.median(energies))
            kpis["min_energy"] = float(np.min(energies))
            kpis["max_energy"] = float(np.max(energies))
            kpis["std_energy"] = float(np.std(energies))
        else:
            kpis["avg_energy"] = 0.0
            kpis["median_energy"] = 0.0
            kpis["min_energy"] = 0.0
            kpis["max_energy"] = 0.0
            kpis["std_energy"] = 0.0

        # --- Trait statistics ---
        if alive:
            kpis["avg_weight"] = float(np.mean([a.weight for a in alive]))
            kpis["avg_speed"] = float(np.mean([a.speed for a in alive]))
            kpis["avg_defense_ones"] = float(np.mean([a.defense_ones_count for a in alive]))
        else:
            kpis["avg_weight"] = 0.0
            kpis["avg_speed"] = 0.0
            kpis["avg_defense_ones"] = 0.0

        # --- Genetic diversity ---
        kpis["genetic_diversity"] = self._compute_genetic_diversity(alive)
        kpis["unique_defense_seqs"] = self._count_unique_defense_sequences(alive)

        # --- Defense match rate ---
        kpis["defense_match_rate"] = self._compute_defense_match_rate(alive, world)

        # --- Food stats ---
        kpis["food_spawned"] = tick_stats_totals.get("food_spawned", 0)
        kpis["food_eaten"] = tick_stats_totals.get("food_eaten", 0)
        kpis["food_expired"] = tick_stats_totals.get("food_expired", 0)
        kpis["food_available"] = world.food_count

        # --- Pitfall stats ---
        encounters = tick_stats_totals.get("pitfall_encounters", 0)
        total_damage = tick_stats_totals.get("pitfall_total_damage", 0)
        kpis["pitfall_encounters"] = encounters
        kpis["pitfall_avg_damage"] = (
            total_damage / encounters if encounters > 0 else 0.0
        )
        kpis["pitfall_zero_damage"] = tick_stats_totals.get(
            "pitfall_zero_damage_encounters", 0
        )
        kpis["pitfall_deaths_caused"] = kpis["deaths_pitfall"]
        kpis["pitfalls_available"] = world.pitfall_count

        # --- Simulation state ---
        kpis["stress_mode_active"] = world.stress_mode
        kpis["mutation_rate_effective"] = (
            self.config.genetics.stress_mutation_rate
            if world.stress_mode
            else self.config.genetics.base_mutation_rate
        )

        # --- Population snapshots from generation lifecycle ---
        kpis["pop_at_primary_repro"] = gen_stats.pop_at_primary_repro
        kpis["pop_at_survival_check"] = gen_stats.pop_at_survival_check
        kpis["pop_at_bonus_repro"] = gen_stats.pop_at_bonus_repro
        kpis["pop_at_generation_end"] = gen_stats.pop_at_generation_end

        # --- Movement stats ---
        kpis["moves_toward_food"] = tick_stats_totals.get("moves_toward_food", 0)
        kpis["moves_random"] = tick_stats_totals.get("moves_random", 0)

        self.history.append(kpis)
        return kpis

    # ------------------------------------------------------------------
    # Genetic diversity
    # ------------------------------------------------------------------

    def _compute_genetic_diversity(
        self,
        animals: list[Animal],
        max_sample: int = 100,
    ) -> float:
        """
        Compute mean pairwise Hamming distance (sampled).

        For large populations, sample up to max_sample animals to keep
        computation tractable (O(N²) pairwise comparisons).

        Args:
            animals: List of alive animals.
            max_sample: Maximum animals to sample for pairwise comparison.

        Returns:
            Mean pairwise Hamming distance (float). 0.0 if < 2 animals.
        """
        if len(animals) < 2:
            return 0.0

        # Sample if too many
        if len(animals) > max_sample:
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            indices = rng.choice(len(animals), size=max_sample, replace=False)
            sampled = [animals[i] for i in indices]
        else:
            sampled = animals

        # Compute pairwise Hamming distances
        n = len(sampled)
        total_dist = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += sampled[i].dna.hamming_distance(sampled[j].dna)
                count += 1

        return total_dist / count if count > 0 else 0.0

    def _count_unique_defense_sequences(self, animals: list[Animal]) -> int:
        """Count distinct defense bit sequences among alive animals."""
        if not animals:
            return 0
        sequences = set()
        for a in animals:
            sequences.add(tuple(a.defense_bits.tolist()))
        return len(sequences)

    def _compute_defense_match_rate(
        self,
        animals: list[Animal],
        world: World,
    ) -> float:
        """
        Compute average defense match rate against active pitfall types.

        For each animal and each active pitfall type, compute what fraction
        of the pitfall's danger bits the animal is immune to. Average across
        all animals and pitfall types.

        Returns 0.0 if no animals or no pitfalls.
        """
        if not animals or not world.pitfalls:
            return 0.0

        # Collect unique pitfall sequences
        unique_sequences = {}
        for p in world.pitfalls.values():
            if p.active:
                key = tuple(p.sequence.tolist())
                if key not in unique_sequences:
                    unique_sequences[key] = p.sequence

        if not unique_sequences:
            return 0.0

        total_match = 0.0
        count = 0

        for animal in animals:
            defense = animal.defense_bits
            for seq in unique_sequences.values():
                # How many danger bits does the animal defend against?
                danger_bits = int(np.sum(seq))
                if danger_bits == 0:
                    total_match += 1.0  # No danger = 100% match
                else:
                    matched = int(np.sum(seq & defense))
                    total_match += matched / danger_bits
                count += 1

        return total_match / count if count > 0 else 0.0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_history(self) -> list[dict]:
        """Return all collected KPI snapshots."""
        return list(self.history)

    def get_last(self) -> Optional[dict]:
        """Return the last collected KPI snapshot, or None."""
        return self.history[-1] if self.history else None

    def get_kpi_series(self, kpi_name: str) -> list:
        """Extract a single KPI as a list across all generations."""
        return [snap[kpi_name] for snap in self.history if kpi_name in snap]

    @staticmethod
    def kpi_names() -> list[str]:
        """Return the ordered list of all KPI names."""
        return [
            "generation",
            "alive_count",
            "extinction_flag",
            "births_primary",
            "births_bonus",
            "births_total",
            "parents_primary",
            "parents_bonus",
            "deaths_starvation",
            "deaths_emergency",
            "deaths_pitfall",
            "deaths_age",
            "deaths_total",
            "avg_energy",
            "median_energy",
            "min_energy",
            "max_energy",
            "std_energy",
            "avg_weight",
            "avg_speed",
            "avg_defense_ones",
            "genetic_diversity",
            "unique_defense_seqs",
            "defense_match_rate",
            "food_spawned",
            "food_eaten",
            "food_expired",
            "food_available",
            "pitfall_encounters",
            "pitfall_avg_damage",
            "pitfall_zero_damage",
            "pitfall_deaths_caused",
            "pitfalls_available",
            "stress_mode_active",
            "mutation_rate_effective",
            "pop_at_primary_repro",
            "pop_at_survival_check",
            "pop_at_bonus_repro",
            "pop_at_generation_end",
            "moves_toward_food",
            "moves_random",
        ]
