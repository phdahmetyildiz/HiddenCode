# Agent notes for Evolution Simulator v3

Work from this `v3/` folder. Do not import v2 `src.*`. Scientific rules: [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md). Layout: [ARCHITECTURE.md](ARCHITECTURE.md).

## Authorship (required)

When you **create** or **substantially change** a source file under `v3/` (Python in `src/`, `tests/`, or `main.py`), record yourself in that file.

**Creating a new file** — original author on the last line of the module docstring:

```
Author: <your model name>
```

**Substantially changing an existing file** — keep the original `Author:` line and **append** an edit line beneath it:

```
Author: <original author>
Edited on <YYYY-MM-DD> by <your model name>
```

Add one edit line per contributor. If you edit again later, update your own date rather than adding duplicates. Trivial changes (typos, formatting) do not need an edit line.

If the file has no docstring (comment-only `__init__.py`), use `# Author: <your model name>` / `# Edited on <date> by <your model name>` comment lines instead.

- Do **not** stamp JSON configs, CSVs, or generated run output.
- Docs (this file, README, HOWTO, ARCHITECTURE, IMPLEMENTATION_PLAN) should keep this rule so the next agent sees it.

A Cursor rule also lives at `../.cursor/rules/v3-authorship.mdc`.
