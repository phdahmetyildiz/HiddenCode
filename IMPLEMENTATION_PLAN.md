# Evolution Simulator — Detailed Implementation Plan

## Architecture Overview

```
Simulator_v2-Cursor/
├── config/
│   ├── default_config.json       # Default simulation parameters
│   └── sweep_template.json       # Template for parameter sweep
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py             # Config loader, validation, defaults
│   │   ├── dna.py                # Binary genome: mutation, extraction, Gray code
│   │   ├── animal.py             # Agent: energy, movement, interaction
│   │   ├── world.py              # 2D toroidal grid, entity management
│   │   ├── food.py               # Food resource
│   │   └── pitfall.py            # Pitfall resource + damage calc
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── engine.py             # Main simulation loop (tick-by-tick)
│   │   ├── generation.py         # Generation lifecycle, reproduction, death
│   │   ├── stress.py             # Stress event triggers and effects
│   │   ├── metrics.py            # KPI collection per generation
│   │   └── sweep.py              # Parameter sweep orchestrator
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── csv_logger.py         # CSV export for metrics
│   │   ├── snapshot.py           # Grid/agent state snapshots (JSON/Pickle)
│   │   └── run_manager.py        # Output directory management (runs/{timestamp}/)
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py                # Streamlit main app entry point
│   │   ├── pages/
│   │   │   ├── __init__.py
│   │   │   ├── config_editor.py  # Edit simulation parameters
│   │   │   ├── sweep_mode.py     # Configure & launch parameter sweeps
│   │   │   ├── sim_runner.py     # Run single simulation with live status
│   │   │   └── results_viewer.py # Browse results, compare runs, plots
│   │   └── components/
│   │       ├── __init__.py
│   │       ├── grid_view.py      # Grid rendering component
│   │       └── charts.py         # Reusable chart components
│   └── utils/
│       ├── __init__.py
│       ├── spatial.py            # Toroidal distance, neighbor queries
│       ├── encoding.py           # Binary ↔ Gray code conversion
│       └── gpu.py                # Optional GPU acceleration (CuPy/Numba)
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_dna.py
│   ├── test_animal.py
│   ├── test_world.py
│   ├── test_food.py
│   ├── test_pitfall.py
│   ├── test_energy.py
│   ├── test_movement.py
│   ├── test_reproduction.py
│   ├── test_generation.py
│   ├── test_stress.py
│   ├── test_metrics.py
│   ├── test_sweep.py
│   ├── test_spatial.py
│   ├── test_encoding.py
│   └── test_integration.py
├── runs/                         # Output directory (git-ignored)
├── requirements.txt
├── main.py                       # CLI entry point
└── .gitignore
```

---

## Phase 1: Project Setup & Configuration System
**Goal**: Establish project structure, dependency management, and a robust config system.

### Sub-tasks:
1.1. Create project directory structure (all folders, `__init__.py` files)
1.2. Create `requirements.txt` with pinned versions:
     - numpy, streamlit, matplotlib, pytest, cupy-cuda (optional)
1.3. Create `src/core/config.py`:
     - `SimConfig` dataclass with ALL parameters and defaults
     - `load_config(path) -> SimConfig` — loads JSON, merges with defaults
     - `validate_config(config)` — range checks, type checks, raises on invalid
     - `save_config(config, path)` — export current config to JSON
1.4. Create `config/default_config.json` with all defaults from spec
1.5. Create `main.py` CLI entry point (argparse: --config, --mode [single|sweep], --headless)

### Tests (test_config.py):
- Load valid config → all fields populated correctly
- Load partial config → defaults fill missing fields
- Load config with invalid values (negative grid size, mutation rate > 1) → raises ValueError
- Save and reload config → roundtrip equality
- Config with unknown keys → warning but no crash

---

## Phase 2: DNA / Genome System
**Goal**: Implement the binary genome with coding regions, mutation, property extraction, and encoding options.

### Sub-tasks:
2.1. Create `src/utils/encoding.py`:
     - `binary_to_int(bits) -> int`
     - `int_to_binary(value, length) -> list[int]`
     - `binary_to_gray(bits) -> list[int]`
     - `gray_to_binary(bits) -> list[int]`
     - `bits_to_normalized(bits, encoding="binary") -> float` (0.0 to 1.0)
2.2. Create `src/core/dna.py`:
     - `DNA` class:
       - `__init__(length, bits=None)` — random bits if None
       - `bits: np.ndarray` (dtype=np.uint8 for memory efficiency)
       - `copy() -> DNA` — deep copy
       - `mutate(rate, coding_regions, coding_only=True)` — select N=round(coding_len*rate) random coding bits, set each to random 0/1
       - `get_slice(start, end) -> np.ndarray` — extract bit region
       - `get_property(start, end, encoding="binary") -> float` — extract + normalize
       - `get_defense_bits(start, length) -> np.ndarray` — defense bit sequence
       - `count_ones(start, length) -> int` — Hamming weight of a region
       - `hamming_distance(other: DNA) -> int` — for diversity metric
     - Coding region definitions stored in config (list of [start, end] pairs)

### Tests (test_dna.py + test_encoding.py):
- Create DNA with known bits → extract property matches expected value
- Mutation at rate=0.0 → no bits change
- Mutation at rate=1.0 → approximately 50% of coding bits change (since set to random)
- Mutation only affects coding regions (junk bits unchanged)
- Gray code: adjacent integers differ by exactly 1 bit
- Binary↔Gray roundtrip: encode then decode = original
- `bits_to_normalized` returns value in [0.0, 1.0]
- `bits_to_normalized` of all-zeros = 0.0, all-ones = 1.0
- `hamming_distance` of identical DNA = 0
- `hamming_distance` of opposite DNA = length
- `copy()` produces independent copy (mutating copy doesn't affect original)
- Defense bits extraction matches known bit pattern
- `count_ones` matches manual count

---

## Phase 3: Core Entities — Animal, Food, Pitfall
**Goal**: Implement the agent and resource data structures.

### Sub-tasks:
3.1. Create `src/core/food.py`:
     - `Food` class: position (x,y), remaining_lifespan, energy_value
     - `tick() -> bool` — decrement lifespan, return True if expired
3.2. Create `src/core/pitfall.py`:
     - `Pitfall` class: position (x,y), name/type, sequence (32-bit np.array), remaining_lifespan
     - `tick() -> bool` — decrement lifespan, return True if expired
     - `calculate_damage(defense_bits: np.ndarray) -> int` — bitwise comparison:
       pitfall=0 → no effect; both=1 → immune; pitfall=1 & animal=0 → +1 damage
     - `calculate_energy_loss(damage, max_loss_pct) -> float`
3.3. Create `src/core/animal.py`:
     - `Animal` class:
       - `__init__(dna, position, config, birth_tick=0)`
       - Properties (derived from DNA on creation, cached):
         `weight`, `speed`, `energy` (starts 1.0), `eyesight` (from config, fixed),
         `defense_bits` (from DNA)
       - State: `position`, `alive`, `birth_tick`, `age_in_ticks`
       - `energy_drain(config) -> float` — `base_metabolism + k_weight_speed * weight * speed`
       - `is_emergency(energy, food_in_range) -> bool`
       - `can_reproduce(energy, thresholds) -> int` — returns 0, 1, or 2 offspring
       - `create_offspring(world, config, current_tick, stress_mode) -> list[Animal]`

### Tests (test_food.py + test_pitfall.py + test_animal.py + test_energy.py):
- Food lifespan decrements correctly, expires at 0
- Pitfall damage: known defense vs known pitfall sequence → expected damage count
- Pitfall damage: perfect match (all immune) → 0 damage
- Pitfall damage: no defense (all zeros) vs all-ones pitfall → 32 damage
- Pitfall energy loss formula: damage=16, max_loss=0.5 → loss = 0.25
- Animal properties extracted from known DNA match expected values
- Energy drain formula: known weight/speed → expected drain value
- Emergency death: energy=0.05, no food → True; energy=0.05, food nearby → False
- Reproduction: energy=0.4 → 0 offspring; energy=0.6 → 1; energy=0.8 → 2
- Offspring inherits mutated copy of parent DNA (not identical)
- Offspring starts with energy=1.0
- Offspring position within 3x3 radius of parent

---

## Phase 4: World / Grid Engine
**Goal**: 2D toroidal grid with spatial queries and entity management.

### Sub-tasks:
4.1. Create `src/utils/spatial.py`:
     - `toroidal_distance(pos1, pos2, width, height) -> float` — min distance on torus
     - `toroidal_wrap(x, y, width, height) -> (x, y)`
     - `neighbors_in_radius(pos, radius, width, height) -> list[(x,y)]`
     - `move_toward(current, target, width, height) -> (x, y)` — one step closer (8-dir)
     - `random_direction() -> (dx, dy)` — uniform from 8 cardinal directions
4.2. Create `src/core/world.py`:
     - `World` class:
       - `__init__(config: SimConfig)`
       - `width, height` — from config (adjustable grid size)
       - `animals: list[Animal]` — all alive agents
       - `food: dict[(x,y), Food]` — active food items
       - `pitfalls: dict[(x,y), Pitfall]` — active pitfalls
       - `tick_count: int`, `generation: int`
       - `stress_mode: bool`
       - `spawn_food(rate)` — Poisson-distributed random positions
       - `spawn_pitfalls(rate, types)` — similar
       - `decay_resources()` — tick all resources, remove expired
       - `nearest_food_in_range(pos, radius) -> (x,y) | None`
       - `pitfall_at(pos) -> Pitfall | None`
       - `animals_at(pos) -> list[Animal]` — for food competition
       - `add_animal(animal)`, `remove_animal(animal)`
       - `initialize_population(count)` — random positions, random DNA
     - Spatial indexing: grid-based bucketing for O(1) lookups within radius

### Tests (test_world.py + test_spatial.py):
- Toroidal wrap: (501, 300) on 500x500 → (1, 300)
- Toroidal distance: (0,0) to (499,0) on 500-wide → 1 (wraps)
- `move_toward` on torus: prefers shorter wrapped path
- `neighbors_in_radius`: correct count, handles edge wrapping
- Food spawning: after N steps at rate R, total food ≈ N*R (within statistical bounds)
- `nearest_food_in_range`: finds closest food; returns None if none in range
- `pitfall_at`: returns correct pitfall or None
- `animals_at`: returns all animals at given position
- Food competition: heaviest animal wins; ties resolved randomly
- Resource decay: food at lifespan=1 removed after one tick
- Population initialization: correct count, all alive, valid positions

---

## Phase 5: Simulation Mechanics — Main Tick Loop
**Goal**: Implement the core simulation step (one tick).

### Sub-tasks:
5.1. Create `src/simulation/engine.py`:
     - `SimulationEngine` class:
       - `__init__(config: SimConfig, seed: int)`
       - `world: World`
       - `metrics: MetricsCollector`
       - `initialize()` — set up world, initial population, resources
       - `tick()` — one simulation step:
         1. Spawn food/pitfalls (rates from config)
         2. Decay resources (remove expired)
         3. For each alive agent (shuffled order each tick for fairness):
            a. Energy drain: `loss = base_metabolism + k * weight * speed` (+ optional defense cost)
            b. If energy <= 0: die (record starvation death)
            c. Emergency check: energy < threshold AND no food in eyesight → die
            d. Move: if food in eyesight → toward nearest; else → random direction
            e. Food interaction: if at food cell → compete (heaviest wins, ties random)
            f. Pitfall interaction: if at pitfall cell → calculate damage, subtract energy
         4. Generation checkpoint checks (delegated to generation.py)
         5. Increment tick counter
         6. If generation boundary → collect metrics, log
       - `run(max_generations) -> RunResult` — loop tick() until done or extinction
       - `is_extinct() -> bool`
     - Agent processing order: shuffled each tick to prevent positional bias

### Tests (test_simulation.py):
- Single tick with known state → verify energy drained correctly
- Animal moves toward food when food in range
- Animal moves randomly when no food in range
- Food consumed by heaviest animal at cell
- Pitfall damage applied correctly on encounter
- Dead animals removed from world
- Emergency death triggered correctly
- Tick counter increments
- Extinction detected when all animals die
- Deterministic with same seed (reproducibility test)
- Performance: 100x100 grid, 200 agents, 100 ticks completes in < 5 seconds

---

## Phase 6: Generations & Reproduction
**Goal**: Implement generation lifecycle with reproduction checkpoints.

### Sub-tasks:
6.1. Create `src/simulation/generation.py`:
     - `GenerationManager` class:
       - `__init__(config: SimConfig)`
       - `gen_length: int` (ticks per generation)
       - `current_generation: int`
       - `check_reproduction(tick, world, stress_mode)`:
         - At 70% of gen_length: primary reproduction checkpoint
           → For each alive animal: check energy → produce 0/1/2 offspring
         - At 100% of gen_length: survival check
           → Energy > survival_threshold → extend to 120%; else → die
         - At 120% of gen_length: secondary reproduction checkpoint
           → Same offspring logic for survivors
       - Offspring creation:
         - DNA: `parent.dna.copy()` → `mutate(base_rate or stress_rate)`
         - Energy: 1.0
         - Position: random cell in 3x3 around parent (toroidal)
         - birth_tick: current tick
       - After 120% checkpoint: reset generation counter, increment gen number
       - Track births/deaths per generation for metrics

### Tests (test_reproduction.py + test_generation.py):
- At tick 700 (70% of 1000): reproduction fires
- At tick 1000 (100%): survival check fires
- At tick 1200 (120%): secondary reproduction fires
- Energy=0.4 → 0 offspring; energy=0.6 → 1; energy=0.8 → 2
- Offspring DNA is mutated copy (not identical to parent)
- Offspring energy = 1.0
- Offspring position within 3x3 of parent, toroidally wrapped
- Low-energy animals die at 100% checkpoint
- Generation counter increments after 120% checkpoint
- All gen_length values adjustable (test with gen_length=100)
- Stress mode uses stress_rate for mutation during reproduction

---

## Phase 7: KPI Logging & Metrics
**Goal**: Collect, store, and export all KPIs per generation.

### Sub-tasks:
7.1. Create `src/simulation/metrics.py`:
     - `MetricsCollector` class:
       - Tracks per-generation counters (incremented during simulation):
         - births, deaths (by cause: starvation, emergency, age, pitfall)
         - food_spawned, food_eaten, food_expired
         - pitfall_encounters, pitfall_zero_damage, pitfall_deaths_caused
       - `collect_generation_snapshot(world, generation) -> dict`:
         Computes all KPIs from current world state + accumulated counters:
         | KPI                    | Computation                                        |
         |------------------------|----------------------------------------------------|
         | alive_count            | len(alive animals)                                 |
         | births_count           | accumulated this generation                        |
         | deaths_count           | total deaths this generation                       |
         | deaths_starvation      | energy <= 0 deaths                                 |
         | deaths_emergency       | low energy + no food deaths                        |
         | deaths_age             | 100% checkpoint deaths                             |
         | extinction_flag        | alive_count == 0                                   |
         | avg_energy             | mean energy of alive                               |
         | median_energy          | median energy of alive                             |
         | min_energy             | min energy of alive                                |
         | max_energy             | max energy of alive                                |
         | std_energy             | std dev energy                                     |
         | avg_weight             | mean weight of alive                               |
         | avg_speed              | mean speed of alive                                |
         | avg_defense_ones       | mean count of 1-bits in defense                    |
         | defense_match_rate     | avg % defense bits matching active pitfall types   |
         | genetic_diversity      | mean pairwise Hamming distance (sampled, max 100)  |
         | unique_defense_seqs    | count distinct defense sequences                   |
         | food_spawned           | total food spawned this generation                 |
         | food_eaten             | total food consumed this generation                |
         | food_expired           | total food expired uneaten                         |
         | food_available         | food items on grid at snapshot time                |
         | pitfall_encounters     | total encounters this generation                   |
         | pitfall_avg_damage     | mean damage per encounter                          |
         | pitfall_zero_damage    | encounters with 0 damage                           |
         | pitfall_deaths_caused  | deaths where pitfall was final blow                |
         | stress_mode_active     | boolean                                            |
         | mutation_rate_effective| current mutation rate                              |
       - `reset_generation_counters()` — call at start of each generation
7.2. Create `src/logging/csv_logger.py`:
     - `CSVLogger` class: append one row per generation to `metrics.csv`
7.3. Create `src/logging/snapshot.py`:
     - `SnapshotManager`: save full world state (agents, resources, grid) as JSON or Pickle per generation
7.4. Create `src/logging/run_manager.py`:
     - `RunManager`: create `runs/{timestamp}/` directory, copy config, manage output files

### Tests (test_metrics.py):
- Known world state (5 animals with known energy) → avg_energy, std_energy correct
- After 3 births and 2 deaths → counts correct
- Food eaten/spawned/expired tracking accurate over a short run
- Pitfall encounter counting works
- Genetic diversity: 5 identical animals → diversity = 0
- Genetic diversity: 5 fully different animals → diversity > 0
- CSV export produces valid file with correct headers and values
- Snapshot save/load roundtrip preserves state

---

## Phase 8: Stress Events
**Goal**: Implement user-triggered stress mode with all its effects.

### Sub-tasks:
8.1. Create `src/simulation/stress.py`:
     - `StressManager` class:
       - `__init__(config: SimConfig)`
       - `trigger_stress(world, new_pitfall_types)`:
         - Set `world.stress_mode = True`
         - Spawn batch of new pitfall types
         - Update effective mutation rate to stress_rate
       - `deactivate_stress(world)`:
         - Set `world.stress_mode = False`
         - Restore base mutation rate
       - Auto-trigger: if `config.stress.trigger_tick` is set → trigger at that tick
       - Manual trigger: callable from UI or script
     - Effects on simulation:
       - Mutation rate switches to `stress_rate` (coding only) during reproduction
       - New pitfall types appear on grid
       - Optionally: food rate changes (configurable)

### Tests (test_stress.py):
- Trigger stress → world.stress_mode = True
- Trigger stress → new pitfalls appear on grid
- Mutation rate during stress = stress_rate (not base_rate)
- Deactivate stress → mutation rate returns to base
- Auto-trigger at configured tick works
- New pitfall types have correct sequences
- Stress does not affect junk DNA mutation (coding only)

---

## Phase 9: Parameter Sweep / Simulation Mode
**Goal**: Run multiple parameter combinations × multiple seeds to find stable baseline values.

### Sub-tasks:
9.1. Design sweep configuration format (`config/sweep_template.json`):
     ```json
     {
       "fixed_params": {
         "world.width": 200,
         "world.height": 200,
         "genetics.dna_length": 2048
       },
       "variable_params": {
         "population.initial_count": [500, 1000, 2500],
         "resources.food_gain": [25, 50]
       },
       "sweep_settings": {
         "runs_per_set": 9,
         "max_generations": 99,
         "base_seed": 42,
         "stability_band": {
           "min_population_pct": 0.20,
           "max_population_pct": 5.00,
           "check_after_generation": 10
         },
         "early_termination_on_extinction": true,
         "parallel_workers": 4
       }
     }
     ```
9.2. Create `src/simulation/sweep.py`:
     - `ParameterSweep` class:
       - `__init__(sweep_config_path)`
       - `generate_combinations()` — cartesian product of variable params
         e.g., 3 × 2 = 6 combinations
       - `run_sweep()`:
         - For each combination × runs_per_set seeds:
           - Create SimConfig with fixed + this combination's variable params
           - Run simulation (via SimulationEngine)
           - Collect metrics per generation
           - Check stability criteria
         - Use `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`
       - `aggregate_results()`:
         - Per combination: mean, std, min, max of each KPI across runs
         - Survival rate: % of runs that didn't go extinct
         - Stability score: % of runs that stayed within population band
       - `export_results(output_dir)`:
         - Summary CSV: one row per combination with aggregated KPIs
         - Detailed CSV: one row per generation per run
         - Stability report: which combinations are stable
9.3. Stability criteria:
     - After `check_after_generation` generations:
       - Population stays between `initial_count * min_pct` and `initial_count * max_pct`
       - Measured at each generation end
     - A run is "stable" if it passes all generation checks
     - A combination is "stable" if >= X% of its runs are stable (configurable)

### Tests (test_sweep.py):
- 2 variable params with [2, 3] values → 6 combinations generated
- Each combination runs correct number of times (runs_per_set)
- Different seeds produce different results
- Extinction triggers early termination
- Stability band correctly classifies stable/unstable runs
- Aggregation math correct (mean, std across runs)
- Parallel execution produces same results as sequential (deterministic seeds)
- Output CSV files have correct structure

---

## Phase 10: Streamlit UI
**Goal**: Web-based interface for configuration, running, sweeping, and viewing results.

### Sub-tasks:
10.1. Create `src/ui/app.py`:
      - Streamlit multi-page app with sidebar navigation
      - Pages: Config Editor, Single Run, Sweep Mode, Results Viewer
10.2. Create `src/ui/pages/config_editor.py`:
      - Load/edit/save config JSON
      - Grouped parameter sections (World, Genetics, Energy, Resources, etc.)
      - Input validation with real-time feedback
      - Preset configs (small test, medium, large)
10.3. Create `src/ui/pages/sim_runner.py`:
      - Start/stop single simulation
      - Live progress bar (generation count)
      - Live KPI display (updating table/charts per generation)
      - Stress event trigger button (manual)
10.4. Create `src/ui/pages/sweep_mode.py`:
      - Define fixed vs variable params
      - Add multiple values per variable param
      - Set sweep settings (runs, generations, stability band)
      - Launch sweep with progress tracking
      - Display results summary table
      - Compare combinations with charts
10.5. Create `src/ui/pages/results_viewer.py`:
      - Browse past runs (from `runs/` directory)
      - View metrics over generations (line charts)
      - Compare multiple runs side-by-side
      - Export data
10.6. Create `src/ui/components/charts.py`:
      - Population over time chart
      - Energy distribution histogram
      - Trait evolution lines (weight, speed, defense)
      - Sweep comparison bar charts
10.7. Create `src/ui/components/grid_view.py`:
      - 2D grid snapshot renderer (Matplotlib/Plotly)
      - Animals as dots (size=weight, color=energy)
      - Food as green dots, pitfalls as red squares
      - (Preparation for future real-time view)

### Tests:
- Manual UI testing (Streamlit doesn't support easy automated testing)
- Verify config round-trip: edit in UI → save → load → values match
- Verify sweep launches and results display
- Verify charts render without errors for edge cases (0 animals, 1 animal)

---

## Phase 11: GPU Acceleration (Optional)
**Goal**: Accelerate compute-heavy operations using NVIDIA GPU.

### Sub-tasks:
11.1. Create `src/utils/gpu.py`:
      - `GPU_AVAILABLE: bool` — detect CuPy/Numba availability
      - Accelerated operations:
        - Spatial distance calculations (batch toroidal distances)
        - DNA mutation (batch random bit operations)
        - Pitfall damage calculation (batch bitwise operations)
        - Food nearest-neighbor queries
      - Fallback: If no GPU → use NumPy (seamless)
11.2. Config flag: `gpu.enabled: true/false`
11.3. Use Numba `@cuda.jit` for:
      - Batch energy drain calculation across all agents
      - Batch pitfall damage for all agents on pitfall cells
      - Genetic diversity sampling (pairwise Hamming distances)

### Tests:
- GPU results match CPU results (numerical equivalence)
- GPU path activated only when gpu.enabled=True AND hardware available
- Graceful fallback to CPU if GPU fails

---

## Phase 12: Real-time Visualization & Video Export (Future)
**Goal**: Watch simulation live and export recordings.

### Sub-tasks:
12.1. Streamlit real-time grid view:
      - Use `st.empty()` + Plotly/Matplotlib for updating grid
      - Throttle: render every N ticks (configurable)
      - Controls: play/pause/step/speed slider
12.2. Video export:
      - Save frames as images → compile with ffmpeg/OpenCV
      - Formats: MP4, GIF, WebM
      - Resolution and FPS configurable
12.3. Replay mode:
      - Load generation snapshots → step through with slider
      - Overlay KPI charts synchronized with grid view

---

## Implementation Order & Dependencies

```
Phase 1  ─── Project Setup & Config
  │
Phase 2  ─── DNA / Genome System
  │
Phase 3  ─── Core Entities (Animal, Food, Pitfall)
  │
Phase 4  ─── World / Grid Engine
  │
Phase 5  ─── Simulation Mechanics (Tick Loop)
  │
Phase 6  ─── Generations & Reproduction
  │
Phase 7  ─── KPI Logging & Metrics
  │
Phase 8  ─── Stress Events
  │
Phase 9  ─── Parameter Sweep Mode
  │
Phase 10 ─── Streamlit UI
  │
Phase 11 ─── GPU Acceleration (Optional)
  │
Phase 12 ─── Real-time Viz & Video (Future)
```

Each phase builds on the previous. Tests are written alongside each phase.

---

## Testing Strategy

### Per-Phase Testing:
- Every phase includes unit tests for all new classes/functions
- Tests written BEFORE or alongside implementation (TDD-lite)
- Run `pytest tests/` after each phase — must pass before proceeding

### Integration Testing:
- After Phase 6: Run a full 10-generation simulation on a small grid (50x50, 50 animals)
  → Verify no crashes, population doesn't instantly die or explode
- After Phase 8: Run with stress trigger → verify metrics change
- After Phase 9: Run a mini sweep (2 params × 2 values × 3 runs × 5 generations)
  → Verify results structure and stability detection

### Performance Testing:
- After Phase 5: Benchmark 200x200 grid, 500 agents, 1000 ticks → measure ticks/second
- Target: 500x500, 1000+ agents at 60+ ticks/second
- If below target: profile and optimize before Phase 9 (sweeps multiply runtime)

### Reproducibility Testing:
- Same config + same seed → identical results (bit-exact)
- Test at Phase 5 and after each subsequent phase

---

## Key Design Decisions

1. **Encoding**: Support both standard binary and Gray code (config option, default: binary)
2. **UI**: Streamlit (web-based, future-proof for hosting, supports video/real-time)
3. **Parallelism**: multiprocessing for sweep mode; optional GPU via CuPy/Numba
4. **Grid size**: Fully adjustable (width × height) via config
5. **All thresholds/counts**: Adjustable via config with sensible defaults
6. **Stability band**: Configurable min/max population percentages for sweep mode
7. **Output**: `runs/{timestamp}/` with config copy, metrics CSV, snapshots
