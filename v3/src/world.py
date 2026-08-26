"""World: SoA animals + dense food/pitfall grids."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src import dna as dnalib
from src.config import PitfallType, SimConfig


class World:
    def __init__(self, config: SimConfig, rng: np.random.Generator | None = None):
        self.config = config
        self.width = config.world.width
        self.height = config.world.height
        self.tick = 0
        self.cohort = 0
        self.stress_mode = False
        self.rng = rng if rng is not None else np.random.default_rng(config.world.seed)

        self.capacity = int(config.perf.max_animals)
        self.n = 0
        self.next_id = 0
        self.dna_length = config.genetics.dna_length
        nw = dnalib.n_words(self.dna_length)

        self.id = np.zeros(self.capacity, dtype=np.int64)
        self.x = np.zeros(self.capacity, dtype=np.int32)
        self.y = np.zeros(self.capacity, dtype=np.int32)
        self.energy = np.zeros(self.capacity, dtype=np.float32)
        self.weight = np.zeros(self.capacity, dtype=np.float32)
        self.speed = np.zeros(self.capacity, dtype=np.float32)
        self.birth_tick = np.zeros(self.capacity, dtype=np.int32)
        self.repro_age = np.zeros(self.capacity, dtype=np.int32)
        self.has_reproduced = np.zeros(self.capacity, dtype=bool)
        self.cohort_of = np.zeros(self.capacity, dtype=np.int32)
        self.defense = np.zeros(self.capacity, dtype=np.uint32)
        self.dna = np.zeros((self.capacity, nw), dtype=np.uint64)

        self.food_life = np.zeros((self.width, self.height), dtype=np.int32)
        self.pitfall_life = np.zeros((self.width, self.height), dtype=np.int32)
        self.pitfall_seq = np.zeros((self.width, self.height), dtype=np.uint32)

        self.active_pitfall_types: list[PitfallType] = config.resources.get_pitfall_types()
        self.births_skipped = 0

    @property
    def n_alive(self) -> int:
        return self.n

    @property
    def is_extinct(self) -> bool:
        return self.n == 0

    def age(self) -> NDArray[np.int32]:
        return (self.tick - self.birth_tick[: self.n]).astype(np.int32)

    def food_positions(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        xs, ys = np.nonzero(self.food_life > 0)
        return xs.astype(np.int32), ys.astype(np.int32)

    def pitfall_positions(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        xs, ys = np.nonzero(self.pitfall_life > 0)
        return xs.astype(np.int32), ys.astype(np.int32)

    def _phenotype_from_dna(
        self,
        dna_rows: NDArray[np.uint64],
        founder: bool,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32], NDArray[np.int32]]:
        g = self.config.genetics
        p = self.config.properties
        raw_w = dnalib.extract_normalized(dna_rows, g.weight_bits[0], g.weight_bits[1], g.encoding)
        raw_s = dnalib.extract_normalized(dna_rows, g.speed_bits[0], g.speed_bits[1], g.encoding)
        raw_r = dnalib.extract_normalized(dna_rows, g.repro_bits[0], g.repro_bits[1], g.encoding)
        defense = dnalib.extract_uint32_slice(dna_rows, g.defense_bits[0], g.defense_bits[1])

        if founder:
            wlo, whi = p.weight_init_range
            slo, shi = p.speed_init_range
        else:
            wlo, whi = p.weight_limits
            slo, shi = p.speed_limits
        weight = np.clip(normalized_to_range_vec(raw_w, wlo, whi), wlo, whi).astype(np.float32)
        speed = np.clip(normalized_to_range_vec(raw_s, slo, shi), slo, shi).astype(np.float32)

        rcfg = self.config.reproduction
        if rcfg.timing == "genetic":
            span = rcfg.repro_age_max - rcfg.repro_age_min
            repro_age = (rcfg.repro_age_min + np.rint(raw_r * span)).astype(np.int32)
        else:
            repro_age = self.rng.integers(
                rcfg.repro_age_min, rcfg.repro_age_max + 1, size=dna_rows.shape[0], dtype=np.int32
            )
        return weight, speed, defense, repro_age

    def initialize_population(self, count: int | None = None) -> None:
        if count is None:
            count = self.config.population.initial_count
        if count > self.capacity:
            raise ValueError(f"initial_count {count} > max_animals {self.capacity}")
        genomes = dnalib.random_genomes(count, self.dna_length, self.rng)
        weight, speed, defense, repro_age = self._phenotype_from_dna(genomes, founder=True)
        self.n = count
        self.id[:count] = np.arange(count, dtype=np.int64)
        self.next_id = count
        self.x[:count] = self.rng.integers(0, self.width, size=count, dtype=np.int32)
        self.y[:count] = self.rng.integers(0, self.height, size=count, dtype=np.int32)
        self.energy[:count] = 1.0
        self.weight[:count] = weight
        self.speed[:count] = speed
        self.birth_tick[:count] = 0
        self.repro_age[:count] = repro_age
        self.has_reproduced[:count] = False
        self.cohort_of[:count] = 0
        self.defense[:count] = defense
        self.dna[:count] = genomes

    def spawn_food(self, rate: float | None = None) -> int:
        if rate is None:
            rate = self.config.resources.food_rate
        if rate <= 0:
            return 0
        count = int(self.rng.poisson(rate))
        spawned = 0
        life = self.config.resources.food_lifespan
        for _ in range(count):
            x = int(self.rng.integers(0, self.width))
            y = int(self.rng.integers(0, self.height))
            if self.food_life[x, y] == 0:
                self.food_life[x, y] = life
                spawned += 1
        return spawned

    def spawn_pitfalls(self, rate: float | None = None) -> int:
        if rate is None:
            rate = self.config.resources.pitfall_rate
        types = self.active_pitfall_types
        if rate <= 0 or not types:
            return 0
        count = int(self.rng.poisson(rate))
        return self._place_pitfalls(count, types)

    def spawn_pitfalls_batch(self, count: int, types: list[PitfallType]) -> int:
        return self._place_pitfalls(count, types)

    def _place_pitfalls(self, count: int, types: list[PitfallType]) -> int:
        spawned = 0
        life = self.config.resources.pitfall_lifespan
        n_types = len(types)
        for _ in range(count):
            x = int(self.rng.integers(0, self.width))
            y = int(self.rng.integers(0, self.height))
            if self.pitfall_life[x, y] == 0:
                pt = types[int(self.rng.integers(0, n_types))]
                self.pitfall_life[x, y] = life
                self.pitfall_seq[x, y] = np.uint32(pt.as_uint32())
                spawned += 1
        return spawned

    def decay_resources(self) -> tuple[int, int]:
        food_was = self.food_life > 0
        pit_was = self.pitfall_life > 0
        self.food_life = np.maximum(self.food_life - 1, 0)
        self.pitfall_life = np.maximum(self.pitfall_life - 1, 0)
        food_expired = int(np.count_nonzero(food_was & (self.food_life == 0)))
        pit_expired = int(np.count_nonzero(pit_was & (self.pitfall_life == 0)))
        self.pitfall_seq[self.pitfall_life == 0] = 0
        return food_expired, pit_expired

    def compact(self, keep: NDArray[np.bool_]) -> int:
        """Keep mask length n. Returns number removed."""
        n = self.n
        if keep.size != n:
            raise ValueError("keep mask length mismatch")
        n_new = int(keep.sum())
        removed = n - n_new
        if removed == 0:
            return 0
        self.id[:n_new] = self.id[:n][keep]
        self.x[:n_new] = self.x[:n][keep]
        self.y[:n_new] = self.y[:n][keep]
        self.energy[:n_new] = self.energy[:n][keep]
        self.weight[:n_new] = self.weight[:n][keep]
        self.speed[:n_new] = self.speed[:n][keep]
        self.birth_tick[:n_new] = self.birth_tick[:n][keep]
        self.repro_age[:n_new] = self.repro_age[:n][keep]
        self.has_reproduced[:n_new] = self.has_reproduced[:n][keep]
        self.cohort_of[:n_new] = self.cohort_of[:n][keep]
        self.defense[:n_new] = self.defense[:n][keep]
        self.dna[:n_new] = self.dna[:n][keep]
        self.n = n_new
        return removed

    def add_offspring(
        self,
        parent_idx: NDArray[np.intp],
        genomes: NDArray[np.uint64],
    ) -> int:
        """Append children of given parent indices. genomes shape (k, n_words)."""
        k = genomes.shape[0]
        if k == 0:
            return 0
        room = self.capacity - self.n
        if k > room:
            self.births_skipped += k - room
            k = room
            parent_idx = parent_idx[:k]
            genomes = genomes[:k]
        if k == 0:
            return 0
        sl = slice(self.n, self.n + k)
        weight, speed, defense, repro_age = self._phenotype_from_dna(genomes, founder=False)
        px = self.x[parent_idx]
        py = self.y[parent_idx]
        dx = self.rng.integers(-1, 2, size=k, dtype=np.int32)
        dy = self.rng.integers(-1, 2, size=k, dtype=np.int32)
        self.id[sl] = np.arange(self.next_id, self.next_id + k, dtype=np.int64)
        self.next_id += k
        self.x[sl] = np.mod(px + dx, self.width).astype(np.int32)
        self.y[sl] = np.mod(py + dy, self.height).astype(np.int32)
        self.energy[sl] = 1.0
        self.weight[sl] = weight
        self.speed[sl] = speed
        self.birth_tick[sl] = self.tick
        self.repro_age[sl] = repro_age
        self.has_reproduced[sl] = False
        self.cohort_of[sl] = self.cohort
        self.defense[sl] = defense
        self.dna[sl] = genomes
        self.n += k
        return k


def normalized_to_range_vec(raw: NDArray, low: float, high: float) -> NDArray:
    return low + raw * (high - low)
