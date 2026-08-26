# Evolution Simulator v3

Array-based rewrite of the HiddenCode evolution simulator.

**Status:** design review (no simulation code yet)  
**Branch:** `v3`  
**Parent:** v2 on `main` (Python objects, Streamlit UI, Phases 1–10)

This folder is a **new codebase**, not a patch of `src/`. v2 stays on `main` for reference. v3 keeps the scientific goal and most mechanical rules, and changes the data layout, tick order, aging, and default energy economy.

## Read these in order

1. [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) — rules of the world (what must be true)
2. [ARCHITECTURE.md](ARCHITECTURE.md) — how it is built (CPU/GPU, arrays, distribution)
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — phased build order and tests

## Scientific goal (unchanged)

Test whether **stress-triggered hypermutation** (coding-region mutation rate jumping under a new pitfall environment) lets a population adapt faster than constant low-rate mutation.

Metrics that matter: survival/recovery after a stress event, defense-bit match to new pitfalls, genetic diversity, population trajectory.

## Why v3 exists

v2 collapsed on default (and most) configs. The inner loop was one Python object per animal, with order-dependent ticks. GPU was never realistic on that layout.

v3 goals:

1. **Livable default world** — population can persist for many overlapping generations.
2. **True aging** — old animals slow down, absorb less food, then hit a hard max age.
3. **Performance** — NumPy/Numba arrays, not per-agent Python objects.
4. **Honest rules** — synchronous ticks (perceive → move → interact → resolve).
5. **Headless first** — CLI + CSV. UI later, if at all.

## What changed vs v2 (summary)

| Topic | v2 | v3 |
|---|---|---|
| Inner loop | One `Animal` object, Python `for` | Structure of Arrays (NumPy) |
| Tick order | Shuffled, one agent at a time | Synchronous phases |
| Aging | Stub (`age` always 0). “Age death” = energy cull at 100% of generation | Senescence (mobility + food absorption) + hard max age |
| Speed gene | Energy cost only; does not affect movement | Energy cost **and** move chance (tradeoff) |
| Default world | 500×500, 200 agents (unlivable) | Small dense world first; scale after it holds |
| UI | Streamlit | None in the first build |
| GPU | Stub | Optional later; CPU is the default path |

## GPU and servers (short answer)

- **Python + NumPy is CPU.** The large speedup vs v2 comes from arrays + Numba on the CPU, not from a GPU.
- **GPU is optional later** for large populations (DNA bitwise ops, energy, some distance work). It is not automatic and not required to start.
- **Many independent runs** (parameter sweeps) scale well across many CPUs or machines. Design for that from day one.
- **One world split across many servers** is a different, much harder problem. Not in v3.

Details: [ARCHITECTURE.md](ARCHITECTURE.md) §4–5.

## Open questions for review

Please confirm or correct these before implementation:

1. **Aging curves** in the spec (onset, max age, linear decline of mobility and food absorption).
2. **100% generation energy cull** — v3 default **off** (aging + emergency + starvation already remove weak animals). Can be turned on.
3. **Genetic speed affects move probability** so “getting slower” is a real competitive disadvantage. If you want speed to stay cost-only, say so.
4. **Default small world** sizes in the spec.

## License / authorship

Same project as v2. Implementation starts only after this design review.
