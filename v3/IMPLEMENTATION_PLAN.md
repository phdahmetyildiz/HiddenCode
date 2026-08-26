# Evolution Simulator v3 — Implementation Plan

Implement **after** review of [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

Do not import v2 `src.*`. New package under `v3/`.

---

## 0. Review gates (before coding)

Confirm with the project owner:

- [x] Overlapping generations
- [x] Emergency death kept
- [x] Aging: **plateau until onset 1000**, then linear decline, `max_age` 1800 (ages 200 and 500 get the same full food energy)
- [x] Reproduction: **one clutch**, age in **[700, 1100]**, default **genetic** timing (optional `random`)
- [x] Offspring count 0/1/2 from energy at the fertility tick (unchanged)
- [x] Late breeders overlapping onset are OK
- [x] Genetic speed × age_mobility = move probability
- [x] Epoch energy cull default **off**
- [x] Default world 80×80, 80 animals
- [x] Watch UI in the first build (Pygame; engine stays UI-agnostic)
- [x] CPU NumPy/Numba; GPU optional with fallback; distributed = independent sweep jobs only

If any box is wrong, edit the spec first.

---

## Phase 1 — Skeleton, config, livability

**Goal:** You can load JSON, fail loud on bad values, and print an energy budget without simulating.

### Work

- `v3/src/config.py` — nested dataclasses: World, Genetics, Properties, Energy, Resources, Reproduction, Metrics, Population, Stress, Aging, Viz, Sweep, Perf (`max_animals`, `backend`)
- `v3/config/default_config.json` — spec defaults (small world + aging)
- `v3/config/sweep_template.json` — copy v2 idea, add aging keys as fixed or variable
- `v3/src/livability.py` — budget table from config (no world needed)
- `v3/main.py` — `python main.py budget --config ...` and `python main.py --help`

### Tests

- Load defaults; partial JSON fills defaults
- Invalid (negative grid, `onset >= max_age`, mutation rate > 1) → `ValueError`
- Roundtrip save/load
- Livability on v2-sized 500×500 / 200 / food_rate 5 **warns** (ticks-to-emergency ≪ `repro_age_min`)
- Livability on v3 defaults **does not** warn (or warns only mildly — tune thresholds in the test)

### Done when

`python main.py budget` prints drain, food, eyesight, ticks-to-emergency.

---

## Phase 2 — Packed DNA and encoding

**Goal:** 2048 bits as `uint64[32]` per genome; mutate and extract without Python bit lists.

### Work

- `encoding.py` — binary/Gray, bits↔int, normalize `[0,1]`
- `dna.py` — allocate batch `uint64[n, 32]`; `mutate_coding(batch, rate, regions, rng)`; `extract_weight/speed/defense/repro_age`; `hamming_sample`

### Tests (port intent from v2 `test_dna` / `test_encoding`)

- All-zero / all-one normalize to 0 and 1
- Gray adjacent integers differ by 1 bit
- Mutation rate 0 → unchanged
- Mutation coding_only → junk bits unchanged
- Defense 32 bits extracted from known pattern
- Packed vs slow reference implementation on 20 random genomes
- Fertility bits all-0 / all-1 map to `repro_age_min` / `repro_age_max`

### Done when

Batch of 1000 genomes mutates in well under 10 ms on CPU (order-of-magnitude check, not a hard SLA).

---

## Phase 3 — World SoA, resources, spatial kernels

**Goal:** A world that can spawn food/pitfalls, hold animals as arrays, answer nearest-food.

### Work

- `world.py` — grids, SoA, `add_animals`, `compact_dead`, spawn/decay
- Numba kernels: `nearest_food_and_in_range(xs, ys, food_grid, radius, width, height)`
- Pitfall sequences as `uint32` per cell or parallel array
- Toroidal wrap helpers

### Tests (from v2 spatial/world, plus arrays)

- Wrap `(width, y)` → `(0, y)`
- Distance across seam
- Nearest food on torus
- Food in range true/false
- Spawn Poisson mean over many ticks ≈ rate
- Compact: killing 3 of 10 leaves 7 dense rows, ids preserved

### Done when

World of 80×80, 80 animals, 200 ticks of spawn/decay/move-stub runs without Python objects for animals.

---

## Phase 4 — Tick phases: energy, emergency, move, eat, pitfall

**Goal:** One `engine.tick()` matching spec §10 steps 1–11 (no reproduction yet).

### Work

- `aging.py` — vectorized `age_mobility`, `food_absorption` from `age = tick - birth_tick`
- `engine.py` — drain, deaths, move probability `speed * mobility`, sync eat, pitfall damage
- Single RNG on the engine

### Tests

- Drain formula vs hand calculation
- Emergency: isolated low-energy animal dies; same with food in radius lives
- Max age: energy 1.0, age == max_age → dead, cause `max_age`
- Plateau: age 200 and age 500 on the same food gain **identical** energy (`food_gain`)
- Post-onset: animal near `max_age` gains ≈ `food_gain * absorption_end`
- Mobility 0 (force) → position unchanged
- Heaviest of two on one food cell eats; food gone
- Pitfall: all-zero defense vs all-one pitfall → 32 damage, loss `max_pitfall_loss_pct`
- Same seed, two engines, identical positions/energy after 50 ticks

### Done when

A tiny scripted world (no repro) shows energy going down and food being eaten; no extinction in 200 ticks on default food if we disable emergency for this test **or** use the default dense map.

---

## Phase 5 — Age-window reproduction and stress

**Goal:** Each animal gets one clutch at `repro_age`; parents remain; stress switches mutation rate.

### Work

- `reproduction.py` — mask `age == repro_age & ~has_reproduced`; energy → 0/1/2 children; set flag
- Encode `repro_age` from DNA (`genetic`) or draw at birth (`random`)
- Offspring rows appended to SoA (mutate DNA, energy 1, 3×3 position, own `repro_age`)
- `stress.py` — trigger, burst pitfalls, duration
- Cap `max_animals`: if full, skip extra births and count `births_skipped`
- Metrics epoch every `metrics.interval` ticks (CSV hook can be a stub until Phase 6)

### Tests

- Forced `repro_age=50`: children appear at age 50; parent still in the SoA
- After that clutch, same parent does not reproduce again
- Energy 0.4 → 0 offspring but `has_reproduced` true
- Energy 0.6 → 1; 0.8 → 2
- Child DNA ≠ parent (high mutation rate in test)
- Child energy 1.0, Chebyshev distance ≤ 1 (toroidal)
- Genetic: all-zero fertility bits → `repro_age_min`
- `timing: random`: two animals with identical DNA can have different `repro_age` (seeded)
- Stress: next births use stress rate; deactivate restores base

### Done when

A short run with min/max window [20, 30] produces staggered births, not one simultaneous wave.

---

## Phase 6 — Metrics, logging, CLI single run

**Goal:** Headless experiment you can plot in a spreadsheet.

### Work

- `metrics.py` — spec §11 KPIs including age fields and `avg_repro_age`
- `logging/` — run dir, CSV one row per **metrics epoch**, optional snapshot of SoA
- `main.py run --config --max-ticks --seed`
- Print livability, then run, then path to CSV

### Tests

- CSV headers include `deaths_max_age`, `avg_age`, `avg_repro_age`
- Snapshot roundtrip restores `n_alive` and first animal DNA
- Smoke: **default config, seed 42, 10 metrics epochs, not extinct**

If the smoke test fails, **stop and fix economy or bugs** before a sweep. Watch UI (Phase 7) is still useful to *see* why they die.

### Done when

`python main.py run --max-ticks 10000` writes `runs/.../metrics.csv` and population is alive.

---

## Phase 7 — Watch UI

**Goal:** You can watch a run and understand what is happening.

### Work

- `watch.py` — Pygame window driven by `engine.tick()`
- Grid: cell pixels or small sprites; animals colored by energy, sized by weight; food green; pitfalls red
- HUD: tick, alive, mean energy, recent births/deaths, stress
- Controls: pause, step, speed, render-every-N
- `main.py watch --config ...`
- Engine has no pygame import
- Optional: `watch --replay runs/...` slider over snapshots if Phase 6 snapshots exist

### Tests

- Manual: start watch, pause, step, see animals move toward food
- Automated: `pytest -m watch` skipped if pygame missing; otherwise construct a tiny world and call `draw_frame` once without crashing

### Done when

`python main.py watch` shows a living default world you can pause.

---

## Phase 8 — Parameter sweep (local parallel)

**Goal:** Many independent `(config, seed)` jobs on one machine.

### Work

- `sweep.py` — Cartesian product, `ProcessPoolExecutor`, each worker is a **fresh** engine (pure function)
- Summary CSV + detailed CSV + stability report (v2 semantics)
- `main.py sweep --sweep-config ...`

### Tests

- 2×2 variables × 2 seeds → 8 jobs
- Worker crash isolated (other jobs finish) — if cheap to test
- Stability band classification on synthetic metrics
- `workers=2` vs `workers=1` same summary given same seeds (determinism)

### Done when

A mini sweep (few minutes) finishes and names a “stable” combo if one exists.

---

## Phase 9 — Performance pass (CPU) ✅

**Goal:** Measure, then Numba the hot kernels if not already.

### Work

- `src/kernels.py` — Numba `@njit` for `popcount32`, `nearest_food`, `pack_bits`; drain stays vectorized NumPy (in-place clip)
- `python main.py bench --backend numba|numpy` — warmup then ticks/s
- Engine has no Python `Animal` objects

### Targets (not gates for correctness)

| World | Hope | Measured (`numba`, 2026-08-26, Windows / Ryzen) |
|---|---|---|
| 80×80, 80 animals | Hundreds of ticks/s | **1638 ticks/s** |
| 200×200, 400 animals | Comfortable watch / sweep | **803 ticks/s** |
| 500×500, 1000 animals | Beat v2’s 60 ticks/s | **178 ticks/s** |

NumPy-only on the same machine: 1314 / 473 / 136 ticks/s. Default config stays `"backend": "numpy"` so tests do not JIT on every engine.

### Done when

Numbers recorded; no Python `Animal` in `engine.tick`.

---

## Phase 10 — GPU backend with fallback ✅

`perf.backend`: `"numpy"` | `"numba"` | `"cuda"` (`"numba_cuda"` is an alias).

- CUDA kernels (cached) for popcount, nearest-food, drain when `n >= 256`
- `resolve_backend("cuda")` → cuda if `numba.cuda.is_available()`, else numba, else numpy
- Move RNG stays on the host so CPU seeds still match
- Tests: numba vs numpy parity; CUDA tests skipped when no GPU

A machine without NVIDIA is a supported path.

---

## Phase 11 — Cluster sweep ✅

Independent jobs, no engine changes.

```
python main.py export-jobs --sweep-config config/sweep_mini.json --output-dir jobs/mini
python main.py run-job --jobs-dir jobs/mini --index N --out jobs/mini/results/job_N.json
python main.py merge-sweep --jobs-dir jobs/mini --results-dir jobs/mini/results --output-dir jobs/mini/merged
```

Bundle: `base_config.json` + `sweep_settings.json` + `jobs.jsonl` (job lines omit the full config) + `manifest.json`. SLURM / Ray / extra VMs call `run-job`; `merge-sweep` rebuilds summary/detailed/stability CSVs.

---

## Implementation order (dependencies)

```
Phase 1 config + livability
    → Phase 2 DNA (incl. fertility bits)
        → Phase 3 world + spatial
            → Phase 4 tick (energy, aging plateau, eat)
                → Phase 5 per-animal reproduction + stress
                    → Phase 6 CLI + smoke survival
                        → Phase 7 watch UI
                            → Phase 8 local sweep
                                → Phase 9 profile/Numba
                                    → 10 GPU fallback / 11 cluster jobs
```

Stop after Phase 6 if the population still dies — unless you want Phase 7 (watch) to diagnose it. Do not start a sweep until the default world holds.

---

## Suggested first coding session (after review)

1. Create `v3/src/`, `v3/tests/`, `v3/config/`.
2. Phase 1 only.
3. Paste livability output for default vs old 500×500 into the review discussion.

No GPU required for that session. Watch UI comes after a runnable engine (Phase 7).

---

## Risk list

| Risk | Mitigation |
|---|---|
| Default still dies (emergency + sparse food) | Livability gate; denser default; do not scale grid until smoke passes |
| Speed×mobility too low → nobody reaches food | Clamp move_probability; check mean speed in init range ~0.5 |
| Population explodes (overlap + one clutch still many births) | `max_animals`; food is the real cap; watch `avg_repro_age` |
| Numba + Windows + multiprocessing | `'spawn'` start method; compile kernels before fork/spawn |
| Float32 drift vs tests | Use float32 everywhere in sim; tests with `pytest.approx` |
| Packed DNA bit-endian mistakes | Reference implementation in tests |

---

## Explicitly out of first build

- Porting the v2 Streamlit app
- Matching v2 CSVs
- Splitting **one** world across machines
- Video export
- Changing the scientific hypothesis

## Benchmarks

Recorded 2026-08-26 on Windows 11, Python 3.12, NumPy 2.4.2, Numba 0.67.0, AMD Ryzen (no CUDA GPU). Timed ticks after a short warmup. `python main.py bench --backend …`

| World | n | numpy ticks/s | numba ticks/s |
|---|---|---|---|
| 80×80 | 80 | 1314 | 1638 |
| 200×200 | 400 | 473 | 803 |
| 500×500 | 1000 | 136 | 178 |
