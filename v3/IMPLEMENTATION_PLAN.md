# Evolution Simulator v3 — Implementation Plan

Implement **after** review of [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

Do not import v2 `src.*`. New package under `v3/`.

---

## 0. Review gates (before coding)

Confirm with the project owner:

- [ ] Overlapping generations (already yes)
- [ ] Emergency death kept (already yes)
- [ ] Aging defaults: onset 800, max_age 1800, linear mobility → 0.05, absorption → 0.20
- [ ] Genetic speed × age_mobility = move probability
- [ ] 100% energy cull default **off**
- [ ] Default world 80×80, 80 animals (start small)
- [ ] No UI in first build
- [ ] CPU NumPy/Numba; GPU later; distributed = independent sweep jobs only

If any box is wrong, edit the spec first.

---

## Phase 1 — Skeleton, config, livability

**Goal:** You can load JSON, fail loud on bad values, and print an energy budget without simulating.

### Work

- `v3/src/config.py` — nested dataclasses: World, Genetics, Properties, Energy, Resources, Generation, Population, Stress, Aging, Viz, Sweep, Perf (`max_animals`, `backend`)
- `v3/config/default_config.json` — spec defaults (small world + aging)
- `v3/config/sweep_template.json` — copy v2 idea, add aging keys as fixed or variable
- `v3/src/livability.py` — budget table from config (no world needed)
- `v3/main.py` — `python main.py budget --config ...` and `python main.py --help`

### Tests

- Load defaults; partial JSON fills defaults
- Invalid (negative grid, `onset >= max_age`, mutation rate > 1) → `ValueError`
- Roundtrip save/load
- Livability on v2-sized 500×500 / 200 / food_rate 5 **warns** (ticks-to-emergency ≪ 700)
- Livability on v3 defaults **does not** warn (or warns only mildly — tune thresholds in the test)

### Done when

`python main.py budget` prints drain, food, eyesight, ticks-to-emergency.

---

## Phase 2 — Packed DNA and encoding

**Goal:** 2048 bits as `uint64[32]` per genome; mutate and extract without Python bit lists.

### Work

- `encoding.py` — binary/Gray, bits↔int, normalize `[0,1]`
- `dna.py` — allocate batch `uint64[n, 32]`; `mutate_coding(batch, rate, regions, rng)`; `extract_weight/speed/defense`; `hamming_sample`

### Tests (port intent from v2 `test_dna` / `test_encoding`)

- All-zero / all-one normalize to 0 and 1
- Gray adjacent integers differ by 1 bit
- Mutation rate 0 → unchanged
- Mutation coding_only → junk bits unchanged
- Defense 32 bits extracted from known pattern
- Packed vs slow reference implementation on 20 random genomes

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

**Goal:** One `engine.tick()` matching spec §10 steps 1–11 (no generation yet).

### Work

- `aging.py` — vectorized `age_mobility`, `food_absorption` from `age = tick - birth_tick`
- `engine.py` — drain, deaths, move probability `speed * mobility`, sync eat, pitfall damage
- Single RNG on the engine

### Tests

- Drain formula vs hand calculation
- Emergency: isolated low-energy animal dies; same with food in radius lives
- Max age: energy 1.0, age == max_age → dead, cause `max_age`
- Old vs young on same food: old gains `food_gain * absorption_end` (set age near max)
- Mobility 0 (force) → position unchanged
- Heaviest of two on one food cell eats; food gone
- Pitfall: all-zero defense vs all-one pitfall → 32 damage, loss `max_pitfall_loss_pct`
- Same seed, two engines, identical positions/energy after 50 ticks

### Done when

A tiny scripted world (no repro) shows energy going down and food being eaten; no extinction in 200 ticks on default food if we disable emergency for this test **or** use the default dense map.

---

## Phase 5 — Overlapping reproduction and stress

**Goal:** Checkpoints add children; parents remain; stress switches mutation rate.

### Work

- `generation.py` — offsets, flags, primary / optional cull / bonus / advance
- Offspring rows appended to SoA (mutate DNA, energy 1, 3×3 position)
- `stress.py` — trigger, burst pitfalls, duration
- Cap `max_animals`: if full, skip extra births and count `births_skipped`

### Tests

- Primary at tick 70 for `gen_length=100`
- After primary, `n_alive == n_parents + n_births`
- Energy 0.4 → 0 offspring; 0.6 → 1; 0.8 → 2
- Child DNA ≠ parent (with high mutation rate in test)
- Child energy 1.0, position Chebyshev distance ≤ 1 (toroidal)
- Cull off: animal with energy 0.2 still alive after 100% tick
- Cull on: that animal dies, cause `cull`
- Stress: next births use stress rate; deactivate restores base
- Generation index increments after bonus

### Done when

One full cycle (120 ticks at gen_length=100) produces children and a metrics row.

---

## Phase 6 — Metrics, logging, CLI single run

**Goal:** Headless experiment you can plot in a spreadsheet.

### Work

- `metrics.py` — spec §11 KPIs including new age fields
- `logging/` — run dir, CSV, optional snapshot of SoA
- `main.py run --config --max-generations --seed`
- Print livability, then run, then path to CSV

### Tests

- CSV headers include `deaths_max_age`, `avg_age`
- Snapshot roundtrip restores `n_alive` and first animal DNA
- Smoke: **default config, seed 42, 10 generation cycles, not extinct**

If the smoke test fails, **stop and fix economy or bugs** before Phase 7. Do not “tune in the sweep.”

### Done when

`python main.py run --max-generations 10` writes `runs/.../metrics.csv` and population is alive.

---

## Phase 7 — Parameter sweep (local parallel)

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

## Phase 8 — Performance pass (CPU)

**Goal:** Measure, then Numba the hot kernels if not already.

### Work

- Benchmark script: ticks/second for default world and for a “medium” world (e.g. 200×200, 400 animals)
- Profile one tick (eyesight vs mutate vs eat)
- Numba-compile remaining Python loops in the tick
- Document numbers in this file’s “Benchmarks” section when we have them

### Targets (not gates for correctness)

| World | Hope |
|---|---|
| 80×80, 80 animals | Hundreds of ticks/s (far above v2 large-world target) |
| 200×200, 400 animals | Comfortable interactive / sweep |
| 500×500, 1000 animals | Revisit v2’s 60 ticks/s target; should be reachable on CPU |

If 500×500 is still slow, it is a kernel problem, not “we need a GPU immediately.”

### Done when

Numbers recorded; no Python `Animal` in `engine.tick`.

---

## Phase 9 — Optional GPU backend (later)

Only if Phase 8 shows large-N array work dominating **and** you have an NVIDIA GPU.

### Work

- `backend: "numba_cuda"` or CuPy drain/mutate/pitfall
- Parity tests vs CPU on small N (floats ≈, or accept different RNG streams)
- Config flag; automatic fallback if no GPU

### Not done in the first delivery.

---

## Phase 10 — Optional cluster sweep (later)

### Work

- Job file: list of `(combo_id, seed, config_path)`
- Worker CLI: `python main.py run-job --job ... --out ...`
- Merge script for CSVs

SLURM / Ray / multiple VMs can call the same worker. **No engine changes** if Phase 7 kept runs pure.

### Not done in the first delivery.

---

## Phase 11 — UI (later, low priority)

Read-only: pick a `runs/` folder, plot CSV (population, energy, defense match). Desktop or Streamlit, whichever is less work. Must not own the engine.

---

## Implementation order (dependencies)

```
Phase 1 config + livability
    → Phase 2 DNA
        → Phase 3 world + spatial
            → Phase 4 tick (energy, aging, eat)
                → Phase 5 repro + stress
                    → Phase 6 CLI + smoke survival
                        → Phase 7 local sweep
                            → Phase 8 profile/Numba
                                → 9 GPU / 10 cluster / 11 UI   (optional)
```

Stop after Phase 6 if the population still dies. That is a spec/economy problem, not a sweep problem.

---

## Suggested first coding session (after review)

1. Create `v3/src/`, `v3/tests/`, `v3/config/`.
2. Phase 1 only.
3. Paste livability output for default vs old 500×500 into the review discussion.

No Streamlit, no GPU, no Docker change required for that session.

---

## Risk list

| Risk | Mitigation |
|---|---|
| Default still dies (emergency + sparse food) | Livability gate; denser default; do not scale grid until smoke passes |
| Speed×mobility too low → nobody reaches food | Clamp move_probability; check mean speed in init range ~0.5 |
| Population explodes (overlap + no cull) | `max_animals`; food is the real cap; watch stability band |
| Numba + Windows + multiprocessing | `'spawn'` start method; compile kernels before fork/spawn |
| Float32 drift vs tests | Use float32 everywhere in sim; tests with `pytest.approx` |
| Packed DNA bit-endian mistakes | Reference implementation in tests |

---

## Explicitly out of first build

- Porting Streamlit
- Matching v2 CSVs
- Distributed single-world
- Video / realtime grid
- Changing the scientific hypothesis
