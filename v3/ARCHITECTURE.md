# Evolution Simulator v3 — Architecture

How v3 is built. Behavior lives in [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md).

---

## 1. Design principles

1. **Arrays, not objects** in the hot path. One structure-of-arrays (SoA) for animals; dense or sparse grids for food and pitfalls.
2. **Tick phases are vectorized** (or Numba loops over arrays), never `for animal in animals: animal.step()`.
3. **Simulation is a library.** CLI and the watch window only start runs and draw; they do not own the tick loop internals.
4. **CPU first.** GPU is an optional backend for the same arrays (`perf.backend`), with automatic fallback.
5. **Sweeps scale out.** One world does not.

---

## 2. Repository layout (target)

v2 code stays at repo root (`src/`, `tests/`, `main.py`). v3 lives under `v3/`:

```
v3/
  README.md
  TECHNICAL_SPEC.md
  ARCHITECTURE.md
  IMPLEMENTATION_PLAN.md
  config/
    default_config.json
    sweep_template.json
  src/
    __init__.py
    config.py          # load / validate / save
    encoding.py        # binary / Gray
    dna.py             # packed bits, mutate, extract (array-native)
    world.py           # grids + SoA animal store
    engine.py          # tick phases
    reproduction.py    # per-animal fertility window, one clutch
    aging.py           # plateau-then-decline curves
    stress.py
    metrics.py
    livability.py      # pre-run energy/food budget
    kernels.py         # numpy / numba / cuda backends
    sweep.py           # process pool + cluster job bundle
    bench.py           # ticks/s cases
    watch.py           # Pygame grid viewer; imports engine, not vice versa
    logging_io.py      # run dir, CSV, snapshots
  tests/
  main.py              # CLI: budget | run | watch | sweep | bench | export-jobs | run-job | merge-sweep
```

Watch mode is part of the first build. The viewer must not sit in the hot path. Do not import v2 `src.*` from v3.

---

## 3. Data layout

### 3.1 Animals (SoA)

Alive animals are a **dense prefix** of arrays, length `n_alive`. Dead slots are compacted periodically (or each tick) so kernels stay tight.

```
id:        int64[n]
x, y:      int32[n]
energy:    float32[n]
weight:    float32[n]
speed:     float32[n]
birth_tick:int32[n]
repro_age: int32[n]
has_reproduced: bool[n]
cohort:    int32[n]
alive:     bool[n]          # or implicit via n_alive
dna:       uint64[n, 32]    # 2048 bits packed, 32×64
```

Defense bits = slice of `dna` (bits 128–159 by default), extracted as `uint32` per animal when needed.

Capacity: preallocate `max_animals` (config, e.g. 10× initial or hard cap). Births fail or extra deaths occur if cap is hit — must be explicit in logs.

### 3.2 Food and pitfalls

Two sparse maps plus optional occupancy rasters:

- `food_xy → remaining_lifespan, energy_value` (dict or parallel arrays + hash)
- `pitfall_xy → lifespan, sequence uint32, type_id`

For sensing, a dense `food_present[width, height]` bool (or uint8) is worth it on 80×80. On 500×500 it is still only 250k cells — cheap. **Prefer dense occupancy grids** for food/pitfalls; they make “food in eyesight” a stencil, not a Python loop over food items × animals.

### 3.3 Spatial queries

Phase 7 of the spec (nearest food) is the hotspot.

**v3 default (CPU):** for each animal, scan the eyesight window on the food occupancy grid with toroidal wrap. Radius 10 → ~300 cells. At 80 animals that is trivial. At 5k animals × 500×500 it is still OK in Numba.

**Later GPU:** batch distances from all animals to all food (if food count is small), or a coarse grid.

Do not use per-cell Python `set` of animal ids as the source of truth. Occupancy for **competition** is: after moves, `np` sort/unique on `(x,y)` among animals that sit on a food cell.

---

## 4. CPU vs GPU

### 4.1 What Python + NumPy + Numba actually uses

| Layer | Device | Role |
|---|---|---|
| NumPy | **CPU** | Arrays, RNG (`Generator`), indexing |
| Numba `@njit` | **CPU** (LLVM) | Tight loops: drain, aging curves, eyesight scan, mutate |
| Numba `@cuda` | **GPU** | Optional `perf.backend: cuda`; falls back if no device |

**The v3 speedup vs v2 is a CPU story:** no Python object per animal, packed DNA, one pass per phase. Expect **roughly 10–50×** on the same machine for the tick loop, depending on population and grid. Not a few percent.

A GPU is **not** required. Default backend is NumPy; set `perf.backend` to `"numba"` or `"cuda"` to opt in.

### 4.2 When a GPU helps

GPUs like large, regular, data-parallel work (thousands to millions of similar ops, little branching).

**Good fit later:**

- Energy drain, aging multipliers, death masks
- Packed-DNA mutation and Hamming distance
- Pitfall damage (`popcount` of `pitfall_bits & ~defense`)
- Pairwise diversity sampling

**Poor fit / extra work:**

- Sparse “who sits on this food cell” reductions (doable, but you write custom kernels)
- Branchy eyesight (food / no food / torus edges)
- Small N (80 animals on 80×80): GPU **loser** — PCIe overhead dominates

`perf.backend` is `numpy | numba | cuda` (`numba_cuda` alias). CUDA is used only when Numba reports a device **and** the batch is large (`n >= 256`); otherwise the request falls back to numba, then numpy. Move RNG stays on the host so a given seed on CPU backends still matches.

### 4.3 GPU kernels (optional)

- Same SoA layout.
- Cached Numba CUDA kernels for popcount, nearest-food, and drain.
- Integers (positions, deaths, DNA bits) match CPU when the same host RNG is used.
- Small N (80 animals on 80×80): GPU is usually slower than CPU.

---

## 5. Distributed servers (hundreds of CPUs / GPUs)

Two different problems:

### 5.1 Many simulations (what we want)

Each run is **independent**: config + seed → metrics.

This is embarrassingly parallel.

| Scale | How |
|---|---|
| One PC | `ProcessPoolExecutor` / `concurrent.futures` (v2 already did this) |
| One server, 64 cores | Same, `workers=64`, one process per run |
| Cluster / many servers | Split the sweep list: each machine gets a chunk of `(combo, seed)` jobs; write CSVs to shared storage or object store; merge summary |
| Many GPUs | Only useful if **each** GPU runs a **large** single sim. For default-sized worlds, **CPU processes beat GPUs**. Hundreds of GPUs would mean hundreds of huge populations, not one tiny world copied 100 times |

v3 will:

- Make one run a **pure function** of `(config_dict, seed) → RunResult + files`.
- Keep the sweep orchestrator **stateless** (job list in, paths out).
- Not require MPI.

Cluster runner: `export-jobs` writes a shared config + `jobs.jsonl`; each worker runs `run-job --index N`; `merge-sweep` rebuilds CSVs. Ray, Dask, SLURM array, or a GitHub Actions matrix can call the same CLI because jobs do not talk to each other.

**This is the path to “hundreds of CPUs.”** It is straightforward if we keep the engine free of global state (no process-wide animal id counter without passing it in; v2 had a global `_next_animal_id`).

### 5.2 One simulation split across machines (not v3)

Partitioning the torus, migrating animals that wrap, syncing food at borders every tick: high engineering cost, hard to keep deterministic, little gain until the grid is huge (far beyond 500×500).

**Out of scope.** If it ever happens, it is a new engine, not a flag.

### 5.3 Practical recommendation

```
Research loop:  laptop / one workstation, Numba CPU, small world
Calibration:    local sweep, N processes = CPU count
Big sweep:      many machines × many CPU workers, one sim per worker
GPU:            only after a single run is huge and profiled
```

---

## 6. Tick implementation sketch

`engine.tick()`:

```
spawn_resources()          # numpy poisson + random cells
decay_resources()
kill_max_age()             # age = tick - birth_tick
apply_drain()              # vectorized
kill_starved()
emergency_mask = (energy < thr) & ~food_in_eyesight(x, y)
kill(emergency_mask)
targets = nearest_food(x, y, radius)
move_mask = rng.random(n) < (speed * age_mobility(age))
apply_moves(move_mask, targets)
resolve_feeding()          # group by (x,y) ∩ food
resolve_pitfalls()
reproduce_due()            # age == repro_age and not has_reproduced
maybe_metrics_epoch()
stress.check()
tick += 1
if watch_callback: watch_callback()
```

`food_in_eyesight` and `nearest_food` share one Numba kernel.

RNG: a single `np.random.Generator(seed)` owned by the engine, passed in. No `np.random.seed` globals. No module-level id counter; `next_id` lives on the world.

---

## 7. Config and livability

`SimConfig` dataclass tree (as v2), plus `AgingConfig`, `ReproductionConfig`, `MetricsConfig`. JSON load with defaults and validation.

`livability.py` runs **before** initialize:

- Mean drain from expected weight/speed (~mid init range)
- Ticks to emergency with zero food
- Steady-state food if uneaten ≈ `food_rate * food_lifespan`
- Expected food in eyesight disk
- Food per animal per tick if perfectly shared

Print a short table. Exit with warning (or `--strict-livability` error) if the budget predicts death before `repro_age_min`.

This is how we avoid v2’s “all configs collapse” trap.

---

## 8. Logging

Same idea as v2: `RunManager` creates `runs/{timestamp}/`, copies config, appends one CSV row per **metrics epoch** (default every 1000 ticks), optional pickle/JSON snapshot of SoA + resource maps.

Do not snapshot every tick by default. Watch mode renders in memory; it may write snapshots on the same interval as headless, or never, via config.

---

## 9. Watch UI

- Module `watch.py` (Pygame). Window loop: while running, if not paused, call `engine.tick()` `speed` times, then draw occupancy grids.
- Draw from arrays: scatter animals by `(x,y)`, color by energy, radius by weight; blit food/pitfall grids.
- Keep drawing **O(n + cells)** using numpy → surface; do not create a sprite per animal if it stays slow — a pixel/cell grid is enough on 80×80.
- Headless tests never import pygame (optional extra).
- Replay (same window or a flag): load snapshots, slider over epoch index.

Toolkit is a choice, not a scientific one. Swap Pygame for pyglet/Qt later without touching `engine.py`.

---

## 10. Testing strategy

- Unit tests on **array functions** (aging curve, packed mutate, toroidal nearest, drain formula).
- Property tests: packed DNA extract equals a slow bit-list reference on random samples.
- Determinism: two engines, same config+seed, equal `metrics.csv`.
- Livability smoke: default config, 10 metrics epochs, seed 42, `alive > 0`.
- Watch UI: manual smoke; optional `pytest -m watch` skipped if pygame missing.

v2 tests are a **behavior checklist**, not copy-paste. Re-implement against this spec.

---

## 11. What we deliberately do not port from v2

- Streamlit pages and dual navigation
- Global animal id counter
- Shuffled per-agent tick
- Python `Animal` / `Food` / `Pitfall` in the hot path
- `gpu.py` stub that never ran
- Dual spatial index (`dict` of animals + `_animal_grid` sets) as source of truth
