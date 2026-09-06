# Evolution Simulator v3

Array-based rewrite of the HiddenCode evolution simulator.

**Status:** Phases 1–11 complete. Default world stays alive; local sweep, Numba/CUDA backends, and cluster job files work.  
**Branch:** `v3`  
**Parent:** v2 on `main` (Python objects, Streamlit UI, Phases 1–10)

This folder is a **new codebase**, not a patch of `src/`. v2 stays on `main` for reference. v3 keeps the scientific goal and most mechanical rules, and changes the data layout, tick order, aging, and default energy economy.

## Read these in order

1. [HOWTO.md](HOWTO.md) — run commands, watch keys, which settings to change
2. [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) — rules of the world (what must be true)
3. [ARCHITECTURE.md](ARCHITECTURE.md) — how it is built (CPU/GPU, arrays, distribution)
4. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — phased build order and tests
5. [AGENTS.md](AGENTS.md) — conventions for future coding agents (including authorship)

## Scientific goal (unchanged)

Test whether **stress-triggered hypermutation** (coding-region mutation rate jumping under a new pitfall environment) lets a population adapt faster than constant low-rate mutation.

Metrics that matter: survival/recovery after a stress event, defense-bit match to new pitfalls, genetic diversity, population trajectory.

## Why v3 exists

v2 collapsed on default (and most) configs. The inner loop was one Python object per animal, with order-dependent ticks. GPU was never realistic on that layout.

v3 goals:

1. **Livable default world** — population can persist for many overlapping generations.
2. **True aging** — full adult life, then senescence (slower, less food), then a hard max age.
3. **Age-based reproduction** — one clutch per animal in a fertility window (not a global alarm clock).
4. **Performance** — NumPy/Numba arrays, not per-agent Python objects.
5. **Honest rules** — synchronous ticks (perceive → move → interact → resolve).
6. **Watch a run** — local viewer so you can see the grid and stats; headless CLI still exists for sweeps.

## What changed vs v2 (summary)

| Topic | v2 | v3 |
|---|---|---|
| Inner loop | One `Animal` object, Python `for` | Structure of Arrays (NumPy) |
| Tick order | Shuffled, one agent at a time | Synchronous phases |
| Aging | Stub (`age` always 0). “Age death” = energy cull at 100% of generation | Full food/mobility until `onset` (default **1000**), then linear decline, then hard `max_age` |
| Reproduction | Everyone on the same world-clock (70% / 120% of gen length) | Each animal once, at an age in **[700, 1100]** (genetic, or random) |
| Speed gene | Energy cost only; does not affect movement | Energy cost **and** move chance (tradeoff) |
| Default world | 500×500, 200 agents (unlivable) | Small dense world first; scale after it holds |
| UI | Streamlit (config/sweep, weak live grid) | Local **watch** window (pause/step/speed) + headless CLI |
| GPU | Stub | Optional CUDA kernels; falls back to Numba then NumPy |

## GPU and servers (short answer)

- **Python + NumPy is CPU.** The large speedup vs v2 comes from arrays + Numba on the CPU, not from a GPU.
- **GPU is optional.** Set `perf.backend` to `"cuda"` (or `"numba_cuda"`). If there is no NVIDIA GPU, the engine falls back to Numba, then NumPy.
- **Many independent runs** scale across processes (`sweep`, and `study` for re-seeded replicates from a checkpoint) or machines (`export-jobs` / `run-job` / `merge-sweep`, `merge-study`).
- **One world split across many servers** is out of scope.

Details: [ARCHITECTURE.md](ARCHITECTURE.md) §4–5.

## Design choices (locked)

1. Aging onset = 1000, then linear decline; ages 200 and 500 get identical full food energy.
2. One clutch per life at a genetically encoded age in [700, 1100].
3. Offspring count 0/1/2 from energy — kept.
4. Late breeders overlapping senescence are OK.
5. Watch UI via Pygame; engine stays UI-agnostic.

## How to use

**Local Python + `v3/.venv`** — not Docker (the root Dockerfile is v2/Streamlit). Details: **[HOWTO.md](HOWTO.md)**.

```
cd v3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py budget
python main.py studio
python main.py study
python main.py run --max-epochs 10
python main.py watch
```

Copy `config/default_config.json` before experiments. After changing food, grid size, or metabolism, run `budget` first. Set `perf.backend` to `"numba"` for faster runs.

## Authorship (for agents)

Every Python source file under `v3/` (`src/`, `tests/`, `main.py`) must name its author in the module docstring:

```
Author: <model name>
```

If you **create** a source file, write **yourself** as that author (your model name, not a previous agent's). If you **substantially change** an existing file, keep the original `Author:` line and append `Edited on <date> by <your model name>` beneath it. JSON configs and generated run output are not stamped. Details: [AGENTS.md](AGENTS.md).

