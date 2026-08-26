# Evolution Simulator v3 — Technical Specification

This document is the **rules of the simulation**. If architecture and this spec disagree, this spec wins for behavior; architecture wins for how data is stored.

v2 `technical_spec.txt` remains the historical source. This file is the v3 contract.

---

## 1. Purpose and scope

### 1.1 Purpose

Grid-based artificial life, single species, to compare:

- **Baseline:** low constant mutation in coding DNA
- **Stress:** user/config-triggered hypermutation plus new pitfall types

Question: does stress-activated mutation speed adaptation (defense vs new pitfalls) without destroying the population?

### 1.2 In scope (v3 first build)

- Tick loop, energy, movement, food, pitfalls
- Binary DNA, coding vs junk, mutation (base and stress)
- **Overlapping generations** with **per-animal** fertility (age window, not a global clock)
- **Biological aging**: full adult performance until `onset`, then decline, then hard max age
- Emergency death, starvation, pitfall death, max-age death
- Headless CLI, JSON config, periodic CSV, optional snapshots
- **Watch UI** (local window: grid + live stats, pause/step/speed)
- Parameter sweep (local process pool and cluster job files)
- Optional CUDA kernels with automatic CPU fallback
- Livability budget printed before a run

### 1.3 Out of scope (v3 first build)

- Splitting one world across many machines
- Multi-species, pathfinding, 3D, video export
- Porting the v2 Streamlit app

### 1.4 Compatibility with v2

Same **scientific rules** where they still make sense. **Not** bit-identical to v2 runs:

- Tick order is synchronous (not shuffled per-agent)
- Aging is new (plateau, then decline)
- Reproduction is per-animal age, not 70%/120% world checkpoints
- Default world is smaller and denser
- Speed gene affects movement, not only metabolism

Old `runs/` CSVs are not expected to match.

---

## 2. World

- 2D **toroidal** grid, integer cells, size `width × height` (config).
- Time is discrete **ticks**.
- Multiple animals may occupy one cell. No crowding penalty.
- At most **one food** and **one pitfall** per cell.
- Food and pitfalls spawn at uniform random empty-of-same-type cells (Poisson count per tick), decay by lifespan, then vanish.
- Food is consumed on eat. Pitfalls stay until they expire.

### 2.1 Default world (v3 start small)

Chosen so animals can actually see food. Scale up only after population holds.

| Parameter | v3 default | v2 default |
|---|---|---|
| width × height | **80 × 80** | 500 × 500 |
| initial population | **80** | 200 |
| eyesight radius | 10 | 10 |
| food_rate | **4.0** / tick | 5.0 |
| food_lifespan | 50 | 50 |
| food_gain (base) | 0.20 | 0.20 |
| pitfall_rate | **0.5** / tick | 2.0 |
| seed | 42 | 42 |

**Livability check (required):** before `run()`, compute and print (or log) expected drain, expected food per animal if evenly shared, ticks-to-emergency with zero food, food density vs eyesight area. Warn if ticks-to-emergency < 0.5 × `repro_age_min`.

---

## 3. Animal state

Per animal (conceptually; stored as arrays in code):

| Field | Meaning |
|---|---|
| id | Unique int |
| x, y | Cell |
| energy | `[0, 1]`, birth = 1.0 |
| dna | Fixed-length bit genome |
| weight, speed | Phenotype from DNA, cached until mutation (offspring only) |
| birth_tick | Tick of birth |
| repro_age | Age (ticks) at which this animal will attempt its one clutch |
| has_reproduced | True after the fertility tick has fired (even if 0 offspring) |
| cohort | Metrics-epoch index at birth (informational; not a biological generation) |
| alive | Bool |

**Age** (ticks) = `current_tick - birth_tick`. Computed, not stored twice.

Initial DNA is random bits. Weight/speed: map raw `[0,1]` DNA decode into **init range** `[0.2, 0.8]` at birth of gen-0. Offspring use full **limits** `[0.1, 1.0]` via the same decode (evolution may leave the init band). v2 mapped everything to limits and ignored init range; v3 honors init range for generation 0.

---

## 4. DNA

- Length default **2048** bits.
- Coding regions (default):
  - `[0, 32)` weight
  - `[32, 64)` speed
  - `[64, 96)` **fertility timing** (maps to `repro_age` in `[repro_age_min, repro_age_max]`)
  - `[96, 128)` reserved
  - `[128, 160)` defense (32 bits)
- Rest is junk.
- Encoding: `binary` (default) or `gray`.
- Mutation: pick `N = round(region_len * rate)` random bits in the allowed region(s), set each to random 0/1 (silent mutations allowed).
- Base rate default `0.01`, coding only.
- Stress rate default `0.20`, coding only unless config says otherwise.
- Inheritance: copy parent bits, then mutate.

Hamming distance used for diversity KPI (sample pairs, cap 100 animals).

---

## 5. Energy

### 5.1 Drain (every tick, all alive animals)

```
drain = base_metabolism + k_weight_speed * weight * speed
if defense_cost_enabled:
    drain += k_defense_cost * count_ones(defense_bits)
energy = clamp(energy - drain, 0, 1)
```

Defaults: `base_metabolism=0.001`, `k_weight_speed=0.01`, defense cost off.

### 5.2 Food gain (on successful eat)

```
gain = food_gain * food_absorption(age)
energy = clamp(energy + gain, 0, 1)
```

`food_absorption(age)` is in §7. Until `aging.onset`, this multiplier is **exactly 1.0** — a 200-tick animal and a 500-tick animal get the **same** energy from a meal.

### 5.3 Starvation

If `energy <= 0` after drain or pitfall: die, cause `starvation` (or `pitfall` if the pitfall was the blow that crossed zero — see §9).

### 5.4 Emergency death (kept)

Checked **after drain, before movement**, using the **pre-move** world (food already spawned/decayed this tick, no one has moved yet):

```
if energy < low_energy_death_threshold and no food within eyesight:
    die, cause = emergency
```

Default threshold `0.10`. This is harsh on sparse maps; the default map must be dense enough that this is a last resort, not the main cause of death.

---

## 6. Movement and sensing

- Eyesight: Euclidean radius (toroidal), default 10.
- Directions: 8-neighbor, one cell per successful move.
- If nearest food is in range → step toward it (shortest toroidal path).
- Else → one random 8-neighbor step.
- Wrap with modulo.

### 6.1 Move chance (new)

v2 always moved every tick. Speed only increased drain, so evolution favored minimum speed.

v3:

```
move_probability = clamp(speed * age_mobility(age), 0.0, 1.0)
```

Each tick, after emergency check: draw `U ~ Uniform(0,1)`. Move only if `U < move_probability`. Otherwise stay.

So:

- High genetic speed → more often reach food, higher metabolism.
- Old age → `age_mobility` drops → they move less → lose the race to food.

Ties at a food cell are still resolved by **weight** (heaviest eats; random among equals).

---

## 7. Aging

v2 did **not** implement aging. `Animal.age` returned 0. Death cause `"age"` meant “failed the 100% generation energy check.”

v3 aging is a **plateau then decline**, not a slope from birth.

### 7.1 Plateau (this is the important part)

Until `aging.onset` (default **1000** ticks of *this animal’s* age):

```
age_mobility = 1.0
food_absorption = 1.0
```

So **a 200-tick animal and a 500-tick animal get the same energy from food** and the same mobility. Nothing declines in between.

Only **after** onset does food energy (and mobility) start falling.

### 7.2 Parameters

| Name | Default | Meaning |
|---|---|---|
| `aging.onset` | **1000** ticks | Full performance until this age (inclusive) |
| `aging.max_age` | **1800** ticks | Hard death, any energy |
| `aging.mobility_end` | **0.05** | Mobility just before `max_age` |
| `aging.absorption_end` | **0.20** | Food absorption just before `max_age` |
| `aging.curve` | `"linear"` | Shape **only on the interval (onset, max_age)** |

Constraints: `0 <= onset < max_age`. End values in `[0, 1]`.

### 7.3 Curves

Let `age = current_tick - birth_tick`.

If `age >= max_age`: die, cause `max_age` (start of animal phase, before drain).

If `age <= onset`: full performance (§7.1).

If `onset < age < max_age` and `curve == "linear"`:

```
t = (age - onset) / (max_age - onset)   # 0 → 1 after onset only
age_mobility     = 1.0 + t * (mobility_end - 1.0)
food_absorption  = 1.0 + t * (absorption_end - 1.0)
```

`curve: "quadratic"` (`t^2`) may be added later; it still does **not** apply before onset.

Worked example (defaults):

| Animal age | food_absorption | Notes |
|---|---|---|
| 200 | **1.00** | Adult plateau |
| 500 | **1.00** | Same meal energy as age 200 |
| 1000 | **1.00** | Last tick of full strength |
| 1400 | **0.60** | Halfway from 1000→1800 toward 0.20 |
| 1799 | **~0.20** | Frail |
| 1800 | dead | `max_age` |

### 7.4 Why this removes old animals

After onset they skip more moves, get less per meal, still pay metabolism, may hit emergency death, then `max_age`.

Default life: adult 0–1000, decline 1000–1800, gone at 1800.

The fertility window is **[700, 1100]** (§8). That **overlaps** the first 100 ticks of senescence: an animal that breeds at 750 is at full strength; one that breeds at 1050 is slightly weaker. To keep every clutch at full absorption, set `onset >= repro_age_max` (e.g. 1100).

### 7.5 Optional energy cull

v2 killed everyone with `energy <= survival_threshold` on a global 100% checkpoint.

v3: **`metrics.cull_enabled: false` by default.** If enabled, at the end of each metrics epoch, `energy <= survival_threshold` → die, cause `cull`.

---

## 8. Overlapping generations and reproduction

Parents are **not** replaced. Children are added. Parents keep living until starvation, emergency, pitfall, `max_age`, or optional cull.

There is **no** global “everyone reproduces at world tick 700.” Fertility is a property of **the animal’s own age**.

### 8.1 One clutch per lifetime

Each animal attempts reproduction **once**, when:

```
age == repro_age
and has_reproduced == false
and alive
```

After that tick, `has_reproduced = true` even if energy was too low for offspring.

If they die before `repro_age`, they leave no descendants.

### 8.2 When: a range, not a fixed age

Config window (defaults match your example):

| Name | Default |
|---|---|
| `reproduction.repro_age_min` | **700** |
| `reproduction.repro_age_max` | **1100** |

`repro_age` is an integer in that closed interval.

**`timing: "genetic"` (default).** Bits `[64, 96)` decode to `[0, 1]`, then:

```
repro_age = repro_age_min + round(raw * (repro_age_max - repro_age_min))
```

Random genomes in generation 0 therefore **spread** across 700–1100. Timing can evolve.

**`timing: "random"`.** At birth, draw `repro_age` uniformly in the window (independent of DNA). Stored on the animal; not heritable.

### 8.3 Offspring count (energy at the fertility tick)

Evaluated **after** that tick’s drain / eat / pitfall:

```
energy < repro_energy_low   → 0     # default 0.50
energy < repro_energy_high  → 1     # default 0.75
else                        → 2
```

### 8.4 Offspring state

- DNA: copy + mutate (stress rate if stress mode, else base).
- Energy: 1.0.
- Position: uniform in 3×3 around parent, toroidal (including parent cell).
- `birth_tick`: current world tick.
- Own `repro_age`: from their DNA (`genetic`) or a new draw (`random`).
- `has_reproduced`: false.
- `cohort`: current metrics-epoch index.

Parents do **not** lose energy from reproducing.

### 8.5 Metrics epochs (logging only)

v2 mixed “generation” with reproduction checkpoints. v3 splits them:

- **Biology:** per-animal `repro_age`.
- **Logging:** every `metrics.interval` ticks (default **1000**) write one CSV row and optional snapshot. Increment `cohort` for animals born in the next interval.

Optional cull, if enabled, runs at epoch end — not as a substitute for aging.

### 8.6 Overlap in time

Animals born at different world ticks hit 700–1100 at different times. Several overlapping age classes live and breed on the same map. That is the intended overlapping-generations model.

---

## 9. Food and pitfalls

### 9.1 Food

- Spawn: Poisson(`food_rate`) positions per tick.
- Lifespan default 50.
- Eat: after movement, all animals on a food cell compete; **heaviest** wins (random among ties). Winner gets `food_gain * food_absorption(age)`. Food removed.
- Competition is **synchronous**: occupancy after all moves, then one winner per food cell. Not “whoever was processed first.”

### 9.2 Pitfalls

- Spawn: Poisson(`pitfall_rate`), type drawn from active type list.
- 32-bit sequence. Damage vs animal defense:

```
pitfall bit 0     → no effect
both 1            → immune
pitfall 1, def 0  → +1 damage
```

```
energy_loss = (damage / 32) * max_pitfall_loss_pct   # default 0.5
```

- Pitfall not consumed.
- If energy hits 0 from this hit: death cause `pitfall`.
- Every animal on a pitfall cell takes damage (no competition).

### 9.3 Stress

On trigger (config tick and/or API):

- `stress_mode = true`
- Mutation rate → `stress_mutation_rate` for subsequent births
- Optional pitfall burst + new types on the active list
- Optional `food_rate_during_stress`

Deactivate on duration expiry or API: restore base mutation rate. Existing animals unchanged; only new births use the current rate.

---

## 10. Main tick (order is part of the spec)

Synchronous. No per-agent shuffle.

1. **Spawn** food and pitfalls.
2. **Decay** resources; remove expired.
3. **Max-age deaths** (`age >= max_age`).
4. **Drain** energy (vectorized).
5. **Starvation deaths** (`energy <= 0`).
6. **Emergency deaths** (low energy AND no food in eyesight on the current food map).
7. **Sense** nearest food in eyesight (from current positions).
8. **Move** (probabilistic; see §6.1).
9. **Eat** (heaviest per food cell).
10. **Pitfalls** (all animals on pitfall cells).
11. **Starvation/pitfall deaths** from post-interact energy.
12. **Reproduction:** every alive animal with `age == repro_age` and not yet `has_reproduced` produces 0/1/2 offspring; set `has_reproduced`.
13. **Metrics epoch** if `tick % interval == 0` (CSV, optional cull, optional snapshot).
14. **Stress** auto trigger/deactivate.
15. Increment tick. Watch-UI callback if attached.

Deterministic given config + seed (one RNG stream).

---

## 11. Metrics (end of each metrics epoch)

Keep v2 KPIs that still apply, and add:

| KPI | Meaning |
|---|---|
| deaths_max_age | Hard age cap |
| deaths_cull | Optional epoch cull |
| deaths_emergency | Unchanged |
| births_count | Offspring created this epoch |
| avg_repro_age | Mean `repro_age` of living animals (evolving trait) |
| avg_age, median_age, max_age_alive | Of living animals |
| avg_mobility, avg_food_absorption | Age modifiers of living |
| avg_move_probability | `speed * age_mobility` |

Death cause `"age"` from v2 is **not** used. Use `max_age` and `cull`.

---

## 12. Output, CLI, and watch UI

- Output: `runs/{timestamp}/` with copied config, `metrics.csv`, optional snapshots.
- CLI:
  - `python main.py budget --config ...`
  - `python main.py run --config ... --seed ... --max-ticks ...` (headless)
  - `python main.py watch --config ...` (live window)
  - `python main.py sweep --sweep-config ...`
- Sweep: Cartesian product × seeds; local process pool first.

### 12.1 Watch mode (first build)

Purpose: **see** what is happening — not a research dashboard.

- Local window (default toolkit: **Pygame**; can be swapped later).
- Grid: animals (color = energy, size = weight), food (green), pitfalls (red).
- HUD: tick, alive, mean energy, births/deaths this epoch, stress on/off.
- Controls: pause, step one tick, speed (and render every N ticks so the sim can run faster than the display).
- The engine does not import the viewer. `watch` runs ticks and asks the viewer to draw. Sweeps stay headless.

Optional soon after: replay a `runs/` folder from snapshots with a tick slider.

---

## 13. Validation (minimum)

- Aging plateau: `food_absorption(200) == food_absorption(500) == food_absorption(onset) == 1.0`.
- After onset: absorption decreases; just below `max_age` ≈ `absorption_end`.
- `max_age`: energy 1.0 still dies, cause `max_age`.
- Genetic timing: all-zero fertility bits → `repro_age_min`; all-one → `repro_age_max`.
- Animal with `repro_age=70` on a short test clock produces children at age 70; parent still alive.
- Energy 0.4 at fertility tick → 0 offspring but `has_reproduced` true (no second chance).
- Emergency, torus, pitfall bits, coding-only mutation, same-seed determinism.
- Livability smoke: default config, seed 42, at least 10 metrics epochs, not extinct.

---

## 14. Parameters (v3 defaults)

See also `config/default_config.json` when it is added. New block:

```json
"aging": {
  "onset": 1000,
  "max_age": 1800,
  "mobility_end": 0.05,
  "absorption_end": 0.20,
  "curve": "linear"
}
```

```json
"reproduction": {
  "timing": "genetic",
  "repro_age_min": 700,
  "repro_age_max": 1100,
  "repro_energy_low": 0.50,
  "repro_energy_high": 0.75
}
```

```json
"metrics": {
  "interval": 1000,
  "cull_enabled": false,
  "survival_threshold": 0.50
}
```
