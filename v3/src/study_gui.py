"""
Tkinter GUI for scientific batch runs (studies).

Pick a starting checkpoint, define arms (each = optional config overrides),
run N re-seeded replicates per arm across all cores, and view mean +/- CI
trajectories, a survival curve, per-arm summaries, and pairwise comparisons.
The process pool runs on a background thread; progress is marshalled to the Tk
main loop via a queue drained in an after() pump.

Author: Cursor Claude Opus 4.8 High
Edited on 2026-09-06 by Cursor Claude Opus 4.8 High: live per-epoch progress
(smooth progress bar + "arm/rep epoch e/E" status) streamed from workers.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Optional

import numpy as np

from src.checkpoint import default_saves_dir, list_checkpoints
from src import plots
from src.study import (
    KEY_KPIS,
    Arm,
    Study,
    StudySpec,
    StudyResult,
    aggregate_existing,
    export_results,
    study_output_dir,
)


PRESETS: dict[str, dict[str, Any]] = {
    "Baseline (no change)": {},
    "Stress ON now (A->B)": {"stress.trigger_tick": 1},
    "Stress at 2000": {"stress.trigger_tick": 2000, "stress.duration_ticks": 2000},
    "Stress + normal mutation (0.01)": {"stress.trigger_tick": 1, "genetics.stress_mutation_rate": 0.01},
    "Stress + hypermutation (0.20)": {"stress.trigger_tick": 1, "genetics.stress_mutation_rate": 0.20},
    "Stress + high mutation (0.40)": {"stress.trigger_tick": 1, "genetics.stress_mutation_rate": 0.40},
    "More food": {"resources.food_rate": 8.0},
    "Harsher pitfalls": {"resources.pitfall_rate": 6.0, "energy.max_pitfall_loss_pct": 1.0},
}

# Default mutation rates offered by the "Mutation sweep" helper.
DEFAULT_MUTATION_RATES = "0.01, 0.05, 0.10, 0.20, 0.40"


def _photo_from_rgb(rgb: np.ndarray, master: tk.Misc) -> tk.PhotoImage:
    h, w, _ = rgb.shape
    ppm = b"P6\n%d %d\n255\n" % (w, h) + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    return tk.PhotoImage(master=master, data=base64.b64encode(ppm).decode("ascii"))


def parse_overrides(text: str) -> dict[str, Any]:
    """Parse 'key=value, key2=value2' or JSON into a dotted-key override dict."""
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        return dict(json.loads(text))
    out: dict[str, Any] = {}
    for part in text.replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected key=value, got '{part}'")
        key, raw = part.split("=", 1)
        out[key.strip()] = _coerce(raw.strip())
    return out


def _coerce(raw: str) -> Any:
    low = raw.lower()
    if low in ("null", "none"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


class StudyApp:
    def __init__(self, saves_dir: Path, config_path: Optional[str] = None):
        self.saves_dir = Path(saves_dir)
        self.config_path = config_path
        self.arms: list[Arm] = [Arm("Baseline", {})]
        self.result: Optional[StudyResult] = None
        self._all_replicates: list = []
        self._q: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._thread_result: dict[str, Any] = {}
        self._reps_done: int = 0
        self._reps_total: int = 0
        self._plot_photo: Optional[tk.PhotoImage] = None
        self._surv_photo: Optional[tk.PhotoImage] = None
        self._last_dest: Optional[Path] = None

        self.root = tk.Tk()
        self.root.title("Evolution Simulator v3 — Study (scientific batch runs)")
        self.root.geometry("1240x820")
        self.root.minsize(1040, 680)

        self.study_name = tk.StringVar(value="study")
        self.origin_kind = tk.StringVar(value="checkpoint")
        self.replicates = tk.StringVar(value="10")
        self.max_epochs = tk.StringVar(value="20")
        self.base_seed = tk.StringVar(value="1234")
        self.random_seed = tk.BooleanVar(value=True)
        self.burn_in = tk.StringVar(value="0")
        self.workers = tk.StringVar(value=str(os.cpu_count() or 1))
        self.bootstrap = tk.BooleanVar(value=True)
        self.save_end = tk.BooleanVar(value=False)
        self.early_stop = tk.BooleanVar(value=True)
        self.compare_metric = tk.StringVar(value="adaptation_score")
        self.plot_metric = tk.StringVar(value="adaptation_score")
        self.status = tk.StringVar(value="Pick a starting checkpoint, define arms, then Run study.")
        self.config_origin_path = tk.StringVar(value=config_path or "")

        self._build()
        self._refresh_checkpoints()
        self._refresh_arms()
        self.root.after(80, self._pump)

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Study name").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.study_name, width=24).pack(side=tk.LEFT, padx=6)
        ttk.Label(top, text="Origin:").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Radiobutton(top, text="Checkpoint", variable=self.origin_kind, value="checkpoint",
                        command=self._refresh_checkpoints).pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Config file", variable=self.origin_kind, value="config").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.config_origin_path, width=40).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Browse config…", command=self._browse_config).pack(side=tk.LEFT)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=3)

        # --- left: checkpoints, arms, settings
        cp = ttk.LabelFrame(left, text="Starting checkpoint", padding=6)
        cp.pack(fill=tk.BOTH, expand=False)
        self.cp_list = tk.Listbox(cp, height=7)
        self.cp_list.pack(fill=tk.BOTH, expand=True)
        ttk.Button(cp, text="Refresh", command=self._refresh_checkpoints).pack(side=tk.LEFT, pady=2)

        arm_f = ttk.LabelFrame(left, text="Arms (conditions to compare)", padding=6)
        arm_f.pack(fill=tk.BOTH, expand=True, pady=4)
        self.arm_list = tk.Listbox(arm_f, height=7)
        self.arm_list.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(arm_f)
        row.pack(fill=tk.X, pady=2)
        ttk.Button(row, text="Add", command=self._add_arm).pack(side=tk.LEFT)
        ttk.Button(row, text="Add preset…", command=self._add_preset).pack(side=tk.LEFT, padx=3)
        ttk.Button(row, text="Edit", command=self._edit_arm).pack(side=tk.LEFT)
        ttk.Button(row, text="Remove", command=self._remove_arm).pack(side=tk.LEFT, padx=3)
        row2 = ttk.Frame(arm_f)
        row2.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(row2, text="Mutation sweep under stress…",
                   command=self._add_mutation_sweep).pack(side=tk.LEFT)
        ttk.Button(row2, text="Clear arms", command=self._clear_arms).pack(side=tk.LEFT, padx=3)

        cfg = ttk.LabelFrame(left, text="Settings", padding=6)
        cfg.pack(fill=tk.X)
        self._labeled(cfg, "Replicates / arm", self.replicates, 0)
        self._labeled(cfg, "Max epochs", self.max_epochs, 1)
        self._labeled(cfg, "Base seed", self.base_seed, 2)
        self._labeled(cfg, "Burn-in epochs", self.burn_in, 3)
        self._labeled(cfg, "Workers (cores)", self.workers, 4)
        ttk.Checkbutton(cfg, text="Random base seed", variable=self.random_seed).grid(row=5, column=0, columnspan=2, sticky=tk.W)
        ttk.Checkbutton(cfg, text="Bootstrap CI", variable=self.bootstrap).grid(row=6, column=0, columnspan=2, sticky=tk.W)
        ttk.Checkbutton(cfg, text="Early-stop when all extinct", variable=self.early_stop).grid(row=7, column=0, columnspan=2, sticky=tk.W)
        ttk.Checkbutton(cfg, text="Save end checkpoint per replicate", variable=self.save_end).grid(row=8, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(cfg, text="Compare metric").grid(row=9, column=0, sticky=tk.W)
        ttk.Combobox(cfg, textvariable=self.compare_metric, values=list(KEY_KPIS),
                     width=18, state="readonly").grid(row=9, column=1, sticky=tk.W)

        runrow = ttk.Frame(left)
        runrow.pack(fill=tk.X, pady=6)
        self.run_btn = ttk.Button(runrow, text="Run study", command=self._run_study)
        self.run_btn.pack(side=tk.LEFT)
        self.extend_btn = ttk.Button(runrow, text="Add replicates", command=self._extend_study, state=tk.DISABLED)
        self.extend_btn.pack(side=tk.LEFT, padx=4)
        self.cancel_btn = ttk.Button(runrow, text="Cancel", command=self._cancel_study, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill=tk.X)

        # --- right: results
        head = ttk.Frame(right)
        head.pack(fill=tk.X)
        ttk.Label(head, text="Plot metric").pack(side=tk.LEFT)
        ttk.Combobox(head, textvariable=self.plot_metric, values=list(KEY_KPIS), width=18,
                     state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Button(head, text="Redraw", command=self._draw_plots).pack(side=tk.LEFT)
        ttk.Button(head, text="Export…", command=self._export).pack(side=tk.RIGHT)
        ttk.Button(head, text="Open folder", command=self._open_folder).pack(side=tk.RIGHT, padx=4)

        self.traj_label = tk.Label(right, bg="#121218")
        self.traj_label.pack(fill=tk.X, pady=2)
        self.surv_label = tk.Label(right, bg="#121218")
        self.surv_label.pack(fill=tk.X, pady=2)

        cols = ("arm", "n", "survival", "extinct", "final_mean", "ci")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=6)
        widths = (150, 40, 70, 60, 90, 140)
        for c, wd in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=wd, anchor=tk.W)
        self.tree.pack(fill=tk.X, pady=4)

        ttk.Label(right, text="Report").pack(anchor=tk.W)
        self.report_box = tk.Text(right, height=10, wrap=tk.WORD, bg="#0e0e14", fg="#d0d0da")
        self.report_box.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.root, textvariable=self.status, padding=8).pack(fill=tk.X)

    def _labeled(self, parent, text, var, row) -> None:
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1, sticky=tk.W, padx=4)

    # ------------------------------------------------------------- actions
    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(title="Base config", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.config_origin_path.set(path)
            self.origin_kind.set("config")

    def _refresh_checkpoints(self) -> None:
        self.cp_list.delete(0, tk.END)
        for row in list_checkpoints(self.saves_dir):
            self.cp_list.insert(
                tk.END,
                f"t={row.get('tick','?')} n={row.get('alive','?')} {row.get('name','')} | {row.get('path','')}",
            )

    def _selected_checkpoint(self) -> Optional[str]:
        sel = self.cp_list.curselection()
        if not sel:
            return None
        return self.cp_list.get(sel[0]).split(" | ")[-1].strip() or None

    def _selected_checkpoint_tick(self) -> Optional[int]:
        """Parse 't=<tick> ...' from the selected checkpoint line, if any."""
        sel = self.cp_list.curselection()
        if not sel:
            return None
        line = self.cp_list.get(sel[0])
        try:
            token = line.split("t=", 1)[1].split()[0]
            return int(token)
        except (IndexError, ValueError):
            return None

    def _refresh_arms(self) -> None:
        self.arm_list.delete(0, tk.END)
        for arm in self.arms:
            ov = ", ".join(f"{k}={v}" for k, v in arm.overrides.items()) or "(no change)"
            self.arm_list.insert(tk.END, f"{arm.label}  [{ov}]")

    def _add_arm(self) -> None:
        label = simpledialog.askstring("Add arm", "Arm label:", parent=self.root)
        if not label:
            return
        ov = simpledialog.askstring("Overrides", "key=value, comma-separated (blank = none):", parent=self.root)
        try:
            overrides = parse_overrides(ov or "")
        except Exception as exc:
            messagebox.showerror("Overrides", str(exc))
            return
        self.arms.append(Arm(label, overrides))
        self._refresh_arms()

    def _add_preset(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Add preset arm")
        ttk.Label(win, text="Choose a preset:").pack(padx=10, pady=6)
        choice = tk.StringVar(value=next(iter(PRESETS)))
        ttk.Combobox(win, textvariable=choice, values=list(PRESETS), state="readonly",
                     width=28).pack(padx=10)

        def add() -> None:
            name = choice.get()
            self.arms.append(Arm(name, dict(PRESETS[name])))
            self._refresh_arms()
            win.destroy()

        ttk.Button(win, text="Add", command=add).pack(pady=8)

    def _edit_arm(self) -> None:
        sel = self.arm_list.curselection()
        if not sel:
            return
        arm = self.arms[sel[0]]
        ov = simpledialog.askstring(
            "Edit overrides", f"Overrides for '{arm.label}':",
            initialvalue=", ".join(f"{k}={v}" for k, v in arm.overrides.items()), parent=self.root,
        )
        if ov is None:
            return
        try:
            arm.overrides = parse_overrides(ov)
        except Exception as exc:
            messagebox.showerror("Overrides", str(exc))
            return
        self._refresh_arms()

    def _remove_arm(self) -> None:
        sel = self.arm_list.curselection()
        if sel:
            self.arms.pop(sel[0])
            self._refresh_arms()

    def _clear_arms(self) -> None:
        self.arms = []
        self._refresh_arms()

    def _add_mutation_sweep(self) -> None:
        """Add one stress arm per mutation rate: all trigger A->B, differ only in
        genetics.stress_mutation_rate. Optionally add a no-stress baseline."""
        # A sensible default trigger tick: just after the checkpoint's tick.
        ck_tick = self._selected_checkpoint_tick() if self.origin_kind.get() == "checkpoint" else None
        default_tick = (ck_tick + 1) if ck_tick is not None else 1
        tick = simpledialog.askinteger(
            "Mutation sweep under stress",
            "Stress trigger tick (when pitfalls change A->B).\n"
            "Must be after the checkpoint's current tick.",
            parent=self.root, minvalue=1, initialvalue=default_tick,
        )
        if tick is None:
            return
        if ck_tick is not None and tick <= ck_tick:
            messagebox.showerror(
                "Trigger tick",
                f"Trigger tick {tick} is not after the checkpoint tick {ck_tick};\n"
                "the stress event would never fire. Pick a larger tick.",
            )
            return
        raw = simpledialog.askstring(
            "Mutation sweep under stress",
            "stress_mutation_rate values to compare (comma-separated):",
            initialvalue=DEFAULT_MUTATION_RATES, parent=self.root,
        )
        if not raw:
            return
        try:
            rates = [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]
        except ValueError as exc:
            messagebox.showerror("Mutation sweep", f"Invalid rate list: {exc}")
            return
        if not rates or any(not (0.0 <= r <= 1.0) for r in rates):
            messagebox.showerror("Mutation sweep", "Each rate must be a number in [0, 1].")
            return
        if messagebox.askyesno("Baseline arm",
                               "Also add a no-stress baseline arm (pitfalls stay A)?"):
            self.arms.append(Arm("no stress (A)", {}))
        for r in rates:
            self.arms.append(Arm(
                f"stress mut={r:g}",
                {"stress.trigger_tick": tick, "genetics.stress_mutation_rate": r},
            ))
        self._refresh_arms()
        self.compare_metric.set("adaptation_score")
        self.status.set(
            f"Added stress arms at tick {tick} for mutation rates: "
            + ", ".join(f"{r:g}" for r in rates)
        )

    def _build_spec(self) -> Optional[StudySpec]:
        kind = self.origin_kind.get()
        if kind == "checkpoint":
            origin = self._selected_checkpoint()
            if not origin:
                messagebox.showinfo("Study", "Select a starting checkpoint from the list.")
                return None
        else:
            origin = self.config_origin_path.get().strip()
            if not origin:
                messagebox.showinfo("Study", "Choose a base config file for a config-origin study.")
                return None
        try:
            spec = StudySpec(
                name=self.study_name.get().strip() or "study",
                origin_path=origin,
                origin_kind=kind,
                arms=[Arm(a.label, dict(a.overrides)) for a in self.arms],
                replicates_per_arm=int(self.replicates.get()),
                max_epochs=int(self.max_epochs.get()),
                base_seed=int(self.base_seed.get()),
                random_base_seed=self.random_seed.get(),
                burn_in_epochs=int(self.burn_in.get()),
                compare_metric=self.compare_metric.get(),
                bootstrap=self.bootstrap.get(),
                workers=int(self.workers.get()),
                early_stop_all_extinct=self.early_stop.get(),
                save_end_checkpoints=self.save_end.get(),
            )
        except ValueError as exc:
            messagebox.showerror("Settings", f"Invalid number: {exc}")
            return None
        errors = spec.validate()
        if errors:
            messagebox.showerror("Study spec", "\n".join(errors))
            return None
        return spec

    def _run_study(self, start_index: int = 0, keep_existing: bool = False) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        spec = self._build_spec()
        if spec is None:
            return
        if not keep_existing:
            self._all_replicates = []
        self._cancel.clear()
        self._thread_result = {}
        total = len(spec.arms) * spec.replicates_per_arm
        self._reps_done, self._reps_total = 0, total
        self.progress.configure(maximum=total, value=0)
        self.run_btn.configure(state=tk.DISABLED)
        self.extend_btn.configure(state=tk.DISABLED)
        self.cancel_btn.configure(state=tk.NORMAL)
        self.base_seed.set(str(spec.base_seed))  # reflect a drawn random seed
        self.status.set(f"Running {total} replicates on {spec.resolved_workers()} cores…")

        dest = study_output_dir(spec)
        end_ckpt = (dest / "end_states") if spec.save_end_checkpoints else None

        def work() -> None:
            try:
                study = Study(spec)

                def cb(done: int, tot: int) -> None:
                    self._q.put(("progress", done, tot))

                def ecb(done_u: int, tot_u: int, label: str) -> None:
                    self._q.put(("epoch", done_u, tot_u, label))

                result = study.run(
                    progress_callback=cb,
                    epoch_callback=ecb,
                    cancel_event=self._cancel,
                    end_ckpt_dir=end_ckpt,
                    start_index=start_index,
                )
                self._thread_result["result"] = result
                self._thread_result["dest"] = dest
            except Exception as exc:  # surface to the UI thread
                self._thread_result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                self._q.put(("done", 0, 0))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _extend_study(self) -> None:
        if self.result is None:
            return
        extra = simpledialog.askinteger("Add replicates", "How many more replicates per arm?",
                                        parent=self.root, minvalue=1, initialvalue=5)
        if not extra:
            return
        existing = self.result.spec.replicates_per_arm
        self.replicates.set(str(extra))
        self.random_seed.set(False)  # keep the same base seed so new seeds don't collide
        self._run_study(start_index=existing, keep_existing=True)

    def _cancel_study(self) -> None:
        self._cancel.set()
        self.status.set("Cancelling — terminating workers…")

    # -------------------------------------------------------------- pump
    def _pump(self) -> None:
        try:
            while True:
                item = self._q.get_nowait()
                kind = item[0]
                if kind == "epoch":
                    _, done_u, tot_u, label = item
                    self.progress.configure(value=done_u, maximum=max(tot_u, 1))
                    self.status.set(
                        f"{label}   "
                        f"({done_u}/{tot_u} epochs, {self._reps_done}/{self._reps_total} reps done)"
                    )
                elif kind == "progress":
                    _, a, b = item
                    self._reps_done, self._reps_total = a, b
                elif kind == "done":
                    self._finish_run()
        except queue.Empty:
            pass
        self.root.after(80, self._pump)

    def _finish_run(self) -> None:
        self.run_btn.configure(state=tk.NORMAL)
        self.cancel_btn.configure(state=tk.DISABLED)
        # Hard cancel: discard whatever partial results came back and reset.
        if self._cancel.is_set():
            self._cancel.clear()
            self.progress.configure(value=0)
            self.status.set("Cancelled. No results kept.")
            return
        if "error" in self._thread_result:
            self.status.set("Study failed.")
            messagebox.showerror("Study failed", self._thread_result["error"])
            return
        result: Optional[StudyResult] = self._thread_result.get("result")
        if result is None:
            self.status.set("Study cancelled or produced no result.")
            return
        # merge with any prior replicates (extend)
        self._all_replicates = list(self._all_replicates) + list(result.replicates)
        merged_spec = result.spec
        merged_spec.replicates_per_arm = max(
            1, len(self._all_replicates) // max(1, len(merged_spec.arms))
        )
        self.result = aggregate_existing(merged_spec, self._all_replicates)
        self._last_dest = self._thread_result.get("dest")
        self.extend_btn.configure(state=tk.NORMAL)
        errs = [r for r in self.result.replicates if r.error]
        note = f"  ({len(errs)} replicate error(s))" if errs else ""
        self.status.set(
            f"Done: {len(self.result.replicates)} replicates, "
            f"{len(self.result.arms)} arm(s).{note} Review, then Export."
        )
        self.plot_metric.set(self.compare_metric.get())
        self._draw_plots()
        self._fill_table()
        self._fill_report()

    # ------------------------------------------------------------ results
    def _draw_plots(self) -> None:
        if self.result is None:
            return
        metric = self.plot_metric.get()
        try:
            traj = plots.trajectory_rgb(self.result.arms, metric, size=(700, 300))
            surv = plots.survival_rgb(self.result.arms, size=(700, 200))
        except Exception as exc:
            self.status.set(f"Plot error: {exc}")
            return
        self._plot_photo = _photo_from_rgb(traj, self.traj_label)
        self.traj_label.configure(image=self._plot_photo)
        self._surv_photo = _photo_from_rgb(surv, self.surv_label)
        self.surv_label.configure(image=self._surv_photo)

    def _fill_table(self) -> None:
        for kid in self.tree.get_children():
            self.tree.delete(kid)
        if self.result is None:
            return
        metric = self.compare_metric.get()
        for arm in self.result.arms:
            fin = arm.final.get(metric, {})
            mean = fin.get("mean", float("nan"))
            cil = fin.get("ci_low", float("nan"))
            cih = fin.get("ci_high", float("nan"))
            ci = "—" if (mean != mean) else f"[{cil:.3f}, {cih:.3f}]"
            self.tree.insert("", tk.END, values=(
                arm.label, arm.n_replicates, f"{arm.survival_prob:.0%}",
                f"{arm.extinction_rate:.0%}",
                "—" if mean != mean else f"{mean:.3f}", ci,
            ))

    def _fill_report(self) -> None:
        self.report_box.delete("1.0", tk.END)
        if self.result is None:
            return
        from src.study import build_report
        _, text = build_report(self.result)
        self.report_box.insert(tk.END, text)

    def _export(self) -> None:
        if self.result is None:
            messagebox.showinfo("Export", "Run a study first.")
            return
        dest = self._last_dest or study_output_dir(self.result.spec)
        try:
            paths = export_results(self.result, dest)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self._last_dest = dest
        self.status.set(f"Exported to {dest}")
        messagebox.showinfo("Export", f"Wrote {len(paths)} artifacts to:\n{dest}")

    def _open_folder(self) -> None:
        dest = self._last_dest
        if dest is None and self.result is not None:
            dest = study_output_dir(self.result.spec)
        if dest is None:
            return
        Path(dest).mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(dest))  # type: ignore[attr-defined]
        except Exception:
            self.status.set(f"Folder: {dest}")

    def run(self) -> None:
        self.root.mainloop()


def run_study_gui(saves_dir: str | Path | None = None, config_path: Optional[str] = None) -> None:
    app = StudyApp(Path(saves_dir) if saves_dir else default_saves_dir(), config_path=config_path)
    app.run()
