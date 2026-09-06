# How to use v3

Run everything from this `v3/` folder, inside the local virtualenv (not Docker, not the system-wide Python). Docker in the repo root is for v2 only.

## Setup (once)

Python 3.12+ on PATH. From the repo, in PowerShell:

```
cd v3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

That creates `v3/.venv/` (gitignored) and installs NumPy, Numba, pygame (watch window), and pytest. GPU drivers are optional.

**Every new terminal:** `cd v3` then `.\.venv\Scripts\Activate.ps1`. The prompt should show `(.venv)`. Then `python` and `pip` are the venv ones.

If PowerShell blocks the script (`execution of scripts is disabled`), either:

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

or skip activation and call the venv Python directly:

```
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py budget
.\.venv\Scripts\python.exe -m pytest
```

cmd.exe: `.venv\Scripts\activate.bat`. macOS/Linux: `source .venv/bin/activate`.

In Cursor/VS Code: pick the interpreter `v3/.venv/Scripts/python.exe`.

## Commands

(venv activated, still in `v3/`)

Print whether the default economy looks livable (no simulation):

```
python main.py budget
```

Headless run, 10 metrics epochs (10 000 ticks with the default interval of 1000). Writes `runs/<timestamp>/` (`metrics.csv`, `config.json`, snapshots, `summary.json`):

```
python main.py run --max-epochs 10
python main.py run --config config/default_config.json --seed 7 --max-epochs 20 --output-dir runs
```

Control GUI (results only until you open the grid). Pause at a tick or epoch, save/load full states (including DNA), fork a save with a different config:

```
python main.py studio
python main.py studio --config config/default_config.json --saves-dir saves
```

Set **Pause after epochs** (default 50) or **Pause at tick** before Start. Check **Random seed** if you want **Start new** to draw a fresh seed instead of `world.seed` from the JSON (the JSON file is not rewritten; the HUD shows the seed actually used so you can copy it back for a replay). Load / continue / fork always keep the checkpoint seed. The table and totals refresh every `metrics.interval` ticks (1000). **Export epochs…** writes every completed epoch to CSV (opens in Excel) or `.xlsx`. Keep the grid closed while you just want numbers — open it any time, close it to go fast again. Saves go in `saves/<timestamp>_<name>/`.

Scientific batch runs (studies): run many re-seeded replicates from a saved checkpoint and compare conditions.

```
python main.py study
python main.py study-run --study-config config/study_template.json
python main.py merge-study --study-config config/study_template.json --results-dir <study_dir> --output-dir <out>
```

A **study** takes one saved checkpoint as the common starting point and runs `replicates_per_arm` copies of it, each with a **different random seed**, so their futures diverge from the identical start. Replicates are grouped into **arms** (one arm = optional config overrides): 1 arm = a single condition, 2 arms = an A/B test (e.g. stress ON vs OFF), N arms = a checkpoint-rooted mini-sweep. Runs are spread across **all CPU cores** (a process pool sized to `os.cpu_count()`); in the GUI a background thread keeps the window responsive and a progress bar shows completed replicates. You can Cancel mid-run and Add replicates (extend) to an existing study.

Results are aggregated into mean +/- 95% CI (and bootstrap CI + quantile bands) trajectories per KPI, a survival curve (fraction of replicates still alive over time), per-arm final summaries, and a pairwise comparison (mean difference, Cohen's d effect size, permutation-test p-value, plus a Welch t-test if `scipy` is installed, and a sample-size hint for 80% power). Set **burn-in epochs** to ignore the transient right after the start when computing the comparison.

Output nests **inside the origin checkpoint**: `saves/<checkpoint>/studies/<timestamp>_<name>/`, containing `report.json` + `report.txt` (the scientific outcome and a plain-language verdict), `manifest.json` (origin checkpoint id + hash, base seed, every replicate seed, backend, timings -- so the study replays exactly), `arm_summary.csv`, `replicates.csv` (+ per-arm detail under `arms/`), `comparison.json`, and `plots/*.png` (labelled via matplotlib if installed, else a built-in PNG writer). Config-origin studies fall back to `runs/studies/...`.

Headless `study-run` reads a study-config JSON (see `config/study_template.json`) and writes the same tree with no display; `merge-study` re-aggregates a directory of replicate results (a study's `replicates.jsonl`, or many `*.json` from a cluster) into a fresh report.

### How the stress event works (important)

Triggering stress does **two things at once**: it introduces pitfall type **B** (adds it and spawns a burst) *and* raises the mutation rate from `genetics.base_mutation_rate` (0.01) to `genetics.stress_mutation_rate` (0.20). The trigger is checked *after* the tick counter advances, so `stress.trigger_tick` must be **>= 1**, and for a checkpoint origin it must be **after the checkpoint's current tick** (otherwise the event never fires). To get the pitfall change without hypermutation, trigger stress but set `genetics.stress_mutation_rate` equal to the base rate for that arm.

### Example A: 20 runs, pitfalls A->B, 10 normal vs 10 stress

Two arms x 10 replicates. Both arms flip to B; the "normal" arm keeps the base mutation rate, the "stress" arm uses hypermutation. In the GUI (`python main.py study`): pick your starting checkpoint, set Replicates/arm = 10, add two arms:

- `normal (A->B)` -> overrides `stress.trigger_tick=<tick>, genetics.stress_mutation_rate=0.01`
- `stress (A->B)` -> overrides `stress.trigger_tick=<tick>, genetics.stress_mutation_rate=0.20`

where `<tick>` is just after your checkpoint's tick (shown as `t=...` in the checkpoint list). Set Compare metric = `adaptation_score`, Run study. The report's verdict tells you whether hypermutation adapted defenses to B significantly better. (If instead you want "stress ON vs OFF", make the normal arm have no overrides so pitfalls stay A.)

### Example B: different mutation rates under stress

To find how strong hypermutation needs to be, compare several `stress_mutation_rate` values, all under the same A->B event. In the GUI click **"Mutation sweep under stress…"**, enter the trigger tick and a list of rates (e.g. `0.01, 0.05, 0.10, 0.20, 0.40`), and optionally add a no-stress baseline. That creates one arm per rate.

Headless equivalent (runs immediately, fresh from the default config at tick 1):

```
python main.py study-run --study-config config/study_mutation_sweep.json
```

Edit `config/study_mutation_sweep.json` to change the rates, `replicates_per_arm`, `max_epochs`, or to start from a checkpoint (set `"origin_kind": "checkpoint"`, `"origin_path": "saves/<folder>"`, and a `stress.trigger_tick` after that checkpoint's tick). Each arm's final `adaptation_score` (mean +/- CI) and the survival curve show which mutation rate adapts fastest to B without collapsing the population; `arm_summary.csv` and `plots/` have the full trajectories.

Watch the grid live without the studio (needs a desktop session):

```
python main.py watch
```

Keys: **space** pause, **.** step one tick, **+ / −** speed, **q** or **Esc** quit. Green = food, red = pitfalls, animals are colored by energy. The window is sized to the current screen (a 200×200 world will shrink the cell size instead of overflowing); you can also resize it.

The HUD shows **run totals**, not this-tick flashes. Second row: live pitfall counts by type name, cumulative encounters, pitfall deaths, and adaptation (`adapt` = mean coverage of dangerous bits on all encounters so far; `full` = % of encounters with complete coverage; `now` = this tick only). The sparkline is cumulative adaptation over time (0 at the bottom, 1 at the top). Quitting watch writes `runs/watch_<timestamp>/adaptation.csv`. Headless `run` writes `adaptation_score` and `adapted_frac` each epoch in `metrics.csv`.

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
| `viz.cell_size` | 8 | Preferred max pixels per cell in **watch** (window still capped to the screen; you can resize it) | Smaller default window on small worlds |

Reproduction count at the fertility tick: energy **< 0.50** → 0 offspring (still marks “has reproduced”), **0.50–0.75** → 1, **≥ 0.75** → 2. Parents stay in the world.

Sweep files (`config/sweep_template.json`) override dotted keys. `variable_params` is a Cartesian product, e.g. two population sizes × two food rates × `runs_per_set` seeds.

Example scientific run: copy the default config, set `stress.trigger_tick` to `8000` and `stress.duration_ticks` to `2000`, keep pitfall type B as the new environment, then compare survival and defense match with and without a high `stress_mutation_rate`.

## Authorship (for agents)

Python sources under `v3/` carry `Author: <model name>` in the module docstring. If you create a `.py` file, that is you. If you substantially change one, keep the original `Author:` and append `Edited on <date> by <your model name>` beneath it. See [AGENTS.md](AGENTS.md).
