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
- **Overlapping generations** with reproduction checkpoints
- **Biological aging** (mobility, food absorption, hard max age)
- Emergency death, starvation, pitfall death, max-age death
- Headless CLI, JSON config, per-generation CSV, optional snapshots
- Parameter sweep (multiple configs × seeds)
- Livability budget printed before a run

### 1.3 Out of scope (v3 first build)

- UI / web dashboard
- GPU kernels (design allows them later)
- Splitting one world across multiple machines
- Multi-species, pathfinding, 3D, video export

### 1.4 Compatibility with v2

Same **scientific rules** where they still make sense. **Not** bit-identical to v2 runs:

- Tick order is synchronous (not shuffled per-agent)
- Aging is new
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

**Livability check (required):** before `run()`, compute and print (or log) expected drain, expected food per animal if evenly shared, ticks-to-emergency with zero food, food density vs eyesight area. Warn if ticks-to-emergency < 0.5 × first reproduction tick.

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
| generation | Generation index of birth (informational) |
| alive | Bool |

**Age** (ticks) = `current_tick - birth_tick`. Computed, not stored twice.

Initial DNA is random bits. Weight/speed: map raw `[0,1]` DNA decode into **init range** `[0.2, 0.8]` at birth of gen-0. Offspring use full **limits** `[0.1, 1.0]` via the same decode (evolution may leave the init band). v2 mapped everything to limits and ignored init range; v3 honors init range for generation 0.

---

## 4. DNA

- Length default **2048** bits.
- Coding regions (default): `[0,64)` reserved (weight+speed), `[64,128)` reserved, `[128,160)` defense (32 bits).
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

`food_absorption(age)` is in §7. Young animals get the full `food_gain`.

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

v3 aging is biological and continuous.

### 7.1 Parameters

| Name | Default | Meaning |
|---|---|---|
| `aging.max_age` | **1800** ticks | Hard death, any energy |
| `aging.onset` | **800** ticks | No senescence before this |
| `aging.mobility_end` | **0.05** | Mobility at `max_age` (just before death) |
| `aging.absorption_end` | **0.20** | Food absorption at `max_age` |
| `aging.curve` | `"linear"` | Interpolation from onset → max_age |

Constraints: `0 <= onset < max_age`. `mobility_end` and `absorption_end` in `[0, 1]`.

### 7.2 Curves

Let `age = current_tick - birth_tick`.

If `age >= max_age`: die, cause `max_age` (checked at start of animal phase, before drain).

If `age <= onset`:

```
age_mobility = 1.0
food_absorption = 1.0
```

If `onset < age < max_age`:

```
t = (age - onset) / (max_age - onset)   # 0 → 1
age_mobility     = 1.0 + t * (mobility_end - 1.0)
food_absorption  = 1.0 + t * (absorption_end - 1.0)
```

Linear is the v3 default. `curve: "quadratic"` may be added later (`t^2`) without changing the rest of the spec.

### 7.3 Why this removes old animals

They do not need a special “generation cull” to vanish:

- They skip more moves → miss food.
- They get less energy per meal.
- Drain continues (weight × speed still apply).
- Emergency death can finish them if energy falls below 0.10 with no food in sight.
- If they still live, `max_age` kills them.

Typical lifetime with defaults: useful adult ~ ticks 0–800, decline 800–1800, gone by 1800. They can hit the 70% checkpoint of a 1000-tick generation clock (~tick 700 of *world* time is not the same as age 700 — each animal has its own age). Generation checkpoints are **world-clock** events; age is **per animal**.

### 7.4 Generation 100% energy cull (optional)

v2 killed everyone with `energy <= survival_threshold` at 100% of `gen_length`.

v3: **`generation.survival_cull_enabled: false` by default.** Aging is the intended removal of old/weak animals. The cull can be enabled for experiments.

If enabled: at the survival checkpoint, `energy <= survival_threshold` → die, cause `cull` (not `age`, to avoid mixing with `max_age`).

---

## 8. Overlapping generations and reproduction

Generations **overlap**. Parents are not replaced by children. Children are added. Parents keep living until they die of starvation, emergency, pitfall, max_age, or optional cull.

### 8.1 World generation clock

A **generation cycle** is a world-time window, not an animal’s lifetime.

| Event | Default | Tick offset from cycle start |
|---|---|---|
| Primary reproduction | 70% of `gen_length` | 700 if `gen_length=1000` |
| Optional survival cull | 100% | 1000 |
| Bonus reproduction | 120% | 1200 |
| Cycle advances | after bonus | `gen_start = now` |

`gen_length` default 1000. All percentages configurable.

After bonus reproduction, generation index increments, cycle restarts. Animals already alive keep their `birth_tick` and age.

### 8.2 Offspring count (per alive animal, at a repro checkpoint)

```
energy < repro_energy_low   → 0     # default 0.50
energy < repro_energy_high  → 1     # default 0.75
else                        → 2
```

### 8.3 Offspring

- DNA: copy + mutate (stress rate if stress mode, else base).
- Energy: 1.0.
- Position: uniform in 3×3 around parent, toroidal (including parent cell).
- `birth_tick`: current tick.
- `generation`: current world generation index + 1 (child of this cycle).

Parents do **not** lose energy from reproducing (v2 behavior, kept unless we decide otherwise later).

Reproduction uses energy **after** that tick’s drain/eat/pitfall (checkpoint runs at end of tick).

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
12. **Generation checkpoints** if this tick matches (repro / optional cull / bonus / advance).
13. **Stress** auto trigger/deactivate.
14. Increment tick. Callbacks/metrics as configured.

Deterministic given config + seed (one RNG stream).

---

## 11. Metrics (per generation cycle end)

Keep v2 KPIs, and add:

| KPI | Meaning |
|---|---|
| deaths_max_age | Hard age cap |
| deaths_cull | Optional 100% energy cull |
| deaths_emergency | Unchanged |
| avg_age, median_age, max_age_alive | Of living animals |
| avg_mobility, avg_food_absorption | Age modifiers of living |
| avg_move_probability | `speed * age_mobility` |

Death cause `"age"` from v2 is **not** used. Use `max_age` and `cull`.

---

## 12. Output and CLI

- Output: `runs/{timestamp}/` with copied config, `metrics.csv`, optional snapshots.
- CLI (first build): `--config`, `--mode single|sweep`, `--seed`, `--max-generations`, `--headless` (always headless for now).
- Sweep: Cartesian product of variable params × `runs_per_set` seeds. Stability band as in v2. Parallel processes on local CPU first.

---

## 13. Validation (minimum)

- Aging: at `onset` mobility=1; just below `max_age` mobility ≈ `mobility_end`; at `max_age` death regardless of energy=1.0.
- Food absorption multiplies gain; a young vs old animal on the same food get different energy.
- Move skip: mobility 0 → never moves (force via config in tests).
- Emergency: energy 0.05, no food in range → dead; food in range → lives.
- Overlap: after primary repro, `alive = parents + children` (parents still present).
- Torus wrap, pitfall bitwise damage, mutation coding-only, determinism (same seed → same metrics).
- Livability: default config survives **at least 10 generation cycles** without extinction in a smoke test (not necessarily all seeds — document the seed used).

---

## 14. Parameters (v3 defaults)

See also `config/default_config.json` when it is added. New block:

```json
"aging": {
  "max_age": 1800,
  "onset": 800,
  "mobility_end": 0.05,
  "absorption_end": 0.20,
  "curve": "linear"
}
```

```json
"generation": {
  "gen_length": 1000,
  "repro_checkpoint_pct": 0.70,
  "survival_check_pct": 1.00,
  "bonus_repro_pct": 1.20,
  "survival_threshold": 0.50,
  "survival_cull_enabled": false,
  "repro_energy_low": 0.50,
  "repro_energy_high": 0.75
}
```
