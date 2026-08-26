# How to use v3

Run everything from this `v3/` folder, with a local Python 3.12+ install (not Docker). Docker in the repo root is for v2 only.

## Setup (once)

```
cd v3
pip install -r requirements.txt
```

That installs NumPy, Numba, pygame (watch window), and pytest. GPU drivers are optional.

## Commands

Print whether the default economy looks livable (no simulation):

```
python main.py budget
```

Headless run, 10 metrics epochs (10 000 ticks with the default interval of 1000). Writes `runs/<timestamp>/` (`metrics.csv`, `config.json`, snapshots, `summary.json`):

```
python main.py run --max-epochs 10
python main.py run --config config/default_config.json --seed 7 --max-epochs 20 --output-dir runs
```

Watch the grid live (needs a desktop session):

```
python main.py watch
```

Keys: **space** pause, **.** step one tick, **+ / −** speed, **q** or **Esc** quit. Green = food, red = pitfalls, animals are colored by energy.

Local parameter sweep (process pool):

```
python main.py sweep --sweep-config config/sweep_mini.json
python main.py sweep --sweep-config config/sweep_template.json --workers 4
```

Output is `runs/sweeps/<timestamp>/` (`summary.csv`, `detailed.csv`, `stability_report.json`).

Speed check:

```
python main.py bench --backend numba
```

Tests:

```
python -m pytest
```

Cluster (one machine exports jobs; each worker runs one index; merge CSVs at the end):

```
python main.py export-jobs --sweep-config config/sweep_template.json --output-dir jobs/exp1
python main.py run-job --jobs-dir jobs/exp1 --index 0 --out jobs/exp1/results/job_000000.json
python main.py merge-sweep --jobs-dir jobs/exp1 --results-dir jobs/exp1/results --output-dir jobs/exp1/merged
```

## Config: copy, then edit

Do not treat `config/default_config.json` as throwaway. Copy it:

```
copy config\default_config.json config\my_run.json
python main.py budget --config config/my_run.json
python main.py run --config config/my_run.json --max-epochs 10
python main.py watch --config config/my_run.json
```

Always run `budget` after big economy changes. If it warns that food in eyesight is sparse, animals will starve before they breed.

Set `"backend": "numba"` in `perf` for faster runs. `"cuda"` uses a GPU if Numba sees one, otherwise it falls back to Numba then NumPy. Leave `"numpy"` if you want identical CPU tests / no JIT delay.

## Knobs and what they do

| Setting | Default | If you raise it | If you lower it |
|---|---|---|---|
| `world.width` / `height` | 80 | Grid is sparser; harder to find food unless you also raise `food_rate` | Denser, easier to live, watch window smaller |
| `population.initial_count` | 80 | More founders; more food competition | Easier per animal, more extinction risk from unlucky deaths |
| `resources.food_rate` | 4.0 | Higher carrying capacity, less emergency death | Population collapses (the v2 failure mode) |
| `resources.pitfall_rate` | 0.5 | More damage / selection on defense bits | Gentler world |
| `properties.eyesight_radius` | 10 | Animals see food farther | Isolated animals die via emergency death |
| `energy.base_metabolism` | 0.001 | Faster drain, shorter lives | Slower drain, easier survival |
| `energy.k_weight_speed` | 0.01 | Heavy+fast is costlier (speed is already a move-chance bonus) | Speed/weight cost weaker |
| `energy.food_gain` | 0.2 | Bigger meals | Harder to recover after a miss |
| `energy.low_energy_death_threshold` | 0.10 | Emergency death triggers earlier | More time to reach food while almost empty |
| `aging.onset` | 1000 | Longer prime of life | Senescence starts earlier |
| `aging.max_age` | 1800 | Animals live longer after decline | Hard death sooner (must stay **> onset**) |
| `reproduction.repro_age_min` / `max` | 700 / 1100 | Later first clutch (more time to starve first) | Earlier breeding; can overlap senescence if max > onset |
| `reproduction.repro_energy_low` / `high` | 0.50 / 0.75 | Harder to get 1 or 2 offspring | More births at the fertility tick |
| `genetics.base_mutation_rate` | 0.01 | More drift every birth | More faithful inheritance |
| `genetics.stress_mutation_rate` | 0.20 | Stronger hypermutation during stress | Weaker adaptation pulse |
| `stress.trigger_tick` | `null` (off) | Set e.g. `5000` to fire the experiment (new pitfalls + high mutation) | Leave `null` for a calm world |
| `perf.max_animals` | 800 | Allows bigger booms | Extra births are skipped when the cap is hit |
| `metrics.interval` | 1000 | Rarer CSV rows (an “epoch”) | More frequent logging; `--max-epochs` means this many intervals |
| `viz.cell_size` | 8 | Larger watch pixels | Smaller window |

Reproduction count at the fertility tick: energy **< 0.50** → 0 offspring (still marks “has reproduced”), **0.50–0.75** → 1, **≥ 0.75** → 2. Parents stay in the world.

Sweep files (`config/sweep_template.json`) override dotted keys. `variable_params` is a Cartesian product, e.g. two population sizes × two food rates × `runs_per_set` seeds.

Example scientific run: copy the default config, set `stress.trigger_tick` to `8000` and `stress.duration_ticks` to `2000`, keep pitfall type B as the new environment, then compare survival and defense match with and without a high `stress_mutation_rate`.
