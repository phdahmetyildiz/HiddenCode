"""Tkinter control studio: results without a grid, optional live view, checkpoints.

Author: Cursor Grok 4.6 High Fast
"""

from __future__ import annotations

import base64
import secrets
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from pathlib import Path

import numpy as np

from src.checkpoint import (
    default_saves_dir,
    delete_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from src.config import SimConfig, load_config
from src.engine import SimulationEngine
from src.livability import evaluate
from src.render import world_rgb
from src.watch import format_pitfall_line, format_totals_line


def choose_world_seed(config_seed: int, randomize: bool) -> tuple[int, str]:
    """Pick the seed for Start new. Does not write the config file."""
    if randomize:
        seed = secrets.randbelow(2**31 - 1) or 1
        return seed, f"seed {seed} (random)"
    seed = int(config_seed)
    return seed, f"seed {seed} (from config)"


def _photo_from_rgb(rgb: np.ndarray, master, zoom: int) -> tk.PhotoImage:
    if zoom > 1:
        rgb = np.repeat(np.repeat(rgb, zoom, axis=0), zoom, axis=1)
    h, w, _ = rgb.shape
    ppm = b"P6\n%d %d\n255\n" % (w, h) + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    return tk.PhotoImage(master=master, data=base64.b64encode(ppm).decode("ascii"))


class StudioApp:
    def __init__(self, config: SimConfig, config_path: str | None, saves_dir: Path):
        self.config = config
        self.config_path = config_path or ""
        self.saves_dir = Path(saves_dir)
        self.engine: SimulationEngine | None = None
        self.running = False
        self.paused = True
        self.grid_win: tk.Toplevel | None = None
        self._grid_label: tk.Label | None = None
        self._grid_photo: tk.PhotoImage | None = None
        self._parent_save: str | None = None

        self.root = tk.Tk()
        self.root.title("Evolution Simulator v3 — Studio")
        self.root.geometry("1100x720")
        self.root.minsize(900, 560)

        self.stop_mode = tk.StringVar(value="epochs")
        self.stop_value = tk.StringVar(value="50")
        self.random_seed = tk.BooleanVar(value=False)
        self.seed_display = tk.StringVar(value=f"seed {config.world.seed} (from config)")
        self.status = tk.StringVar(value="Load a config, set a stop point, then Start.")
        self.kpi = tk.StringVar(value="No run yet. Results update every metrics epoch (default 1000 ticks).")
        self.totals = tk.StringVar(value="")
        self.pits = tk.StringVar(value="")
        self.config_var = tk.StringVar(value=self.config_path)

        self._build()
        self._refresh_saves()
        self._refresh_seed_hint()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(30, self._pump)

    def _refresh_seed_hint(self) -> None:
        if self.engine is not None:
            return
        if self.random_seed.get():
            self.seed_display.set("seed will be chosen on Start new")
        else:
            self.seed_display.set(f"seed {self.config.world.seed} (from config)")

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Config").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.config_var, width=70).pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        ttk.Button(top, text="Browse…", command=self._browse_config).pack(side=tk.LEFT)

        stop = ttk.LabelFrame(self.root, text="Stop / pause before start", padding=8)
        stop.pack(fill=tk.X, padx=8, pady=4)
        stop_row = ttk.Frame(stop)
        stop_row.pack(fill=tk.X)
        ttk.Radiobutton(stop_row, text="Never (run until Pause)", variable=self.stop_mode, value="never").pack(side=tk.LEFT)
        ttk.Radiobutton(stop_row, text="Pause at tick", variable=self.stop_mode, value="tick").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Radiobutton(stop_row, text="Pause after epochs", variable=self.stop_mode, value="epochs").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(stop_row, textvariable=self.stop_value, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Label(stop_row, text="(1 epoch = metrics.interval ticks, default 1000)").pack(side=tk.LEFT)
        seed_row = ttk.Frame(stop)
        seed_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Checkbutton(
            seed_row,
            text="Random seed",
            variable=self.random_seed,
            command=self._refresh_seed_hint,
        ).pack(side=tk.LEFT)
        ttk.Label(seed_row, textvariable=self.seed_display).pack(side=tk.LEFT, padx=8)

        btns = ttk.Frame(self.root, padding=(8, 4))
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Start new", command=self._start_new).pack(side=tk.LEFT)
        ttk.Button(btns, text="Resume", command=self._resume).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Pause", command=self._pause).pack(side=tk.LEFT)
        ttk.Button(btns, text="Open grid", command=self._open_grid).pack(side=tk.LEFT, padx=(16, 4))
        ttk.Button(btns, text="Close grid", command=self._close_grid).pack(side=tk.LEFT)
        ttk.Label(btns, text="Grid off = faster.", foreground="#555").pack(side=tk.LEFT, padx=8)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        head = ttk.Frame(left)
        head.pack(fill=tk.X)
        ttk.Label(head, text="Results (refresh each epoch)").pack(side=tk.LEFT)
        ttk.Button(head, text="Export epochs…", command=self._export_epochs).pack(side=tk.RIGHT)
        ttk.Label(left, textvariable=self.kpi, justify=tk.LEFT, wraplength=620).pack(anchor=tk.W, pady=4)
        ttk.Label(left, textvariable=self.totals, justify=tk.LEFT).pack(anchor=tk.W)
        ttk.Label(left, textvariable=self.pits, justify=tk.LEFT, wraplength=620).pack(anchor=tk.W, pady=2)

        cols = ("epoch", "tick", "alive", "births", "d.em", "d.pit", "adapt", "full%")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (56, 70, 60, 60, 60, 60, 70, 60)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor=tk.E)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=6)

        sav = ttk.LabelFrame(right, text="Saved states", padding=6)
        sav.pack(fill=tk.BOTH, expand=True)
        self.save_list = tk.Listbox(sav, height=18)
        self.save_list.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(sav)
        row.pack(fill=tk.X, pady=4)
        ttk.Button(row, text="Save current…", command=self._save_now).pack(side=tk.LEFT)
        ttk.Button(row, text="Load / continue", command=self._load_selected).pack(side=tk.LEFT, padx=4)
        row2 = ttk.Frame(sav)
        row2.pack(fill=tk.X)
        ttk.Button(row2, text="Fork with other config…", command=self._fork_selected).pack(side=tk.LEFT)
        ttk.Button(row2, text="Delete", command=self._delete_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="Refresh", command=self._refresh_saves).pack(side=tk.LEFT)

        ttk.Label(self.root, textvariable=self.status, padding=8).pack(fill=tk.X)

    def _export_epochs(self) -> None:
        if self.engine is None or not self.engine.epoch_history:
            messagebox.showinfo("Export", "No completed epochs yet. Wait until the first 1000 ticks finish.")
            return
        tick = self.engine.world.tick
        path = filedialog.asksaveasfilename(
            title="Export epoch statistics",
            defaultextension=".csv",
            initialfile=f"epochs_tick{tick}.csv",
            filetypes=[
                ("CSV (Excel)", "*.csv"),
                ("Excel workbook", "*.xlsx"),
                ("All", "*.*"),
            ],
        )
        if not path:
            return
        from src.logging_io import export_epochs

        try:
            out = export_epochs(path, self.engine.epoch_history)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status.set(f"Exported {len(self.engine.epoch_history)} epochs → {out}")

    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Simulation config",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            initialdir=str(Path(self.config_path).parent) if self.config_path else ".",
        )
        if path:
            self.config_var.set(path)
            try:
                self.config = load_config(path)
                self.config_path = path
                if self.engine is None:
                    self._refresh_seed_hint()
            except Exception as exc:
                messagebox.showerror("Config", str(exc))

    def _stop_target(self) -> tuple[str, int | None]:
        mode = self.stop_mode.get()
        if mode == "never":
            return mode, None
        raw = self.stop_value.get().strip()
        try:
            value = int(raw)
        except ValueError:
            raise ValueError("Stop value must be an integer") from None
        if value < 1:
            raise ValueError("Stop value must be >= 1")
        return mode, value

    def _should_pause(self) -> bool:
        if self.engine is None:
            return True
        try:
            mode, value = self._stop_target()
        except ValueError:
            return False
        if mode == "never" or value is None:
            return False
        if mode == "tick" and self.engine.world.tick >= value:
            return True
        if mode == "epochs" and self.engine.epochs_completed >= value:
            return True
        return False

    def _bind_engine(self, engine: SimulationEngine) -> None:
        self.engine = engine
        engine.on_epoch = self._on_epoch

    def _start_new(self) -> None:
        path = self.config_var.get().strip()
        try:
            if path:
                self.config = load_config(path)
                self.config_path = path
            self._stop_target()
        except Exception as exc:
            messagebox.showerror("Cannot start", str(exc))
            return
        report = evaluate(self.config)
        run_cfg = self.config.copy()
        run_cfg.world.seed, label = choose_world_seed(run_cfg.world.seed, self.random_seed.get())
        self.seed_display.set(label)
        engine = SimulationEngine(run_cfg)
        engine.initialize()
        self._bind_engine(engine)
        self._parent_save = None
        self._reload_tree()
        self._update_labels()
        self.paused = False
        self.running = True
        self.status.set("Running. Grid is closed (faster). Open grid if you want to see the world.")
        if report.warns:
            messagebox.showwarning("Livability", report.as_text())

    def _resume(self) -> None:
        if self.engine is None:
            messagebox.showinfo("Resume", "Start a new run or load a saved state first.")
            return
        try:
            self._stop_target()
        except ValueError as exc:
            messagebox.showerror("Stop value", str(exc))
            return
        if self._should_pause():
            messagebox.showinfo(
                "Stop point",
                "Already at or past the stop value. Raise it, then Resume.",
            )
            return
        self.paused = False
        self.running = True
        self.status.set("Running.")

    def _pause(self) -> None:
        self.paused = True
        self.status.set("Paused. You can save the state, open the grid, or change the stop value.")
        self._update_labels()
        if self.grid_win is not None:
            self._draw_grid()

    def _open_grid(self) -> None:
        if self.engine is None:
            messagebox.showinfo("Grid", "Start or load a run first.")
            return
        if self.grid_win is not None and self.grid_win.winfo_exists():
            self.grid_win.lift()
            return
        self.grid_win = tk.Toplevel(self.root)
        self.grid_win.title("Simulation grid")
        self.grid_win.protocol("WM_DELETE_WINDOW", self._close_grid)
        self._grid_label = tk.Label(self.grid_win, bg="#121218")
        self._grid_label.pack(fill=tk.BOTH, expand=True)
        self._draw_grid()

    def _close_grid(self) -> None:
        if self.grid_win is not None:
            try:
                self.grid_win.destroy()
            except tk.TclError:
                pass
        self.grid_win = None
        self._grid_label = None
        self._grid_photo = None

    def _draw_grid(self) -> None:
        if self.engine is None or self.grid_win is None or self._grid_label is None:
            return
        if not self.grid_win.winfo_exists():
            self._close_grid()
            return
        rgb = world_rgb(self.engine.world)
        h, w, _ = rgb.shape
        max_w = max(200, self.grid_win.winfo_width() or 640)
        max_h = max(200, self.grid_win.winfo_height() or 640)
        zoom = max(1, min(max_w // w, max_h // h, 8))
        self._grid_photo = _photo_from_rgb(rgb, self.grid_win, zoom)
        self._grid_label.configure(image=self._grid_photo)

    def _save_now(self) -> None:
        if self.engine is None:
            messagebox.showinfo("Save", "Nothing to save yet.")
            return
        self._pause()
        name = simpledialog.askstring("Save state", "Name for this checkpoint:", parent=self.root)
        if not name:
            return
        try:
            path = save_checkpoint(
                self.engine, self.saves_dir, name, parent=self._parent_save,
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.status.set(f"Saved {path}")
        self._refresh_saves()

    def _selected_path(self) -> str | None:
        sel = self.save_list.curselection()
        if not sel:
            return None
        line = self.save_list.get(sel[0])
        # "tick=.. | name | path"
        path = line.split(" | ")[-1].strip()
        return path or None

    def _load_selected(self) -> None:
        path = self._selected_path()
        if not path:
            messagebox.showinfo("Load", "Select a saved state in the list.")
            return
        try:
            engine = load_checkpoint(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self._bind_engine(engine)
        self.config = engine.config
        self._parent_save = Path(path).name
        self.paused = True
        self.running = True
        self.seed_display.set(f"seed {engine.config.world.seed} (from save)")
        self._reload_tree()
        self._update_labels()
        self.status.set(f"Loaded {path} (paused). Resume to continue, or Fork with another config.")
        if self.grid_win is not None:
            self._draw_grid()

    def _fork_selected(self) -> None:
        path = self._selected_path()
        if not path:
            messagebox.showinfo("Fork", "Select a saved state, then choose a config with the new rules.")
            return
        cfg_path = filedialog.askopenfilename(
            title="Config to apply on top of this save (grid size must match)",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if not cfg_path:
            return
        try:
            override = load_config(cfg_path)
            engine = load_checkpoint(path, config_override=override)
        except Exception as exc:
            messagebox.showerror("Fork failed", str(exc))
            return
        self._bind_engine(engine)
        self.config = engine.config
        self.config_var.set(cfg_path)
        self._parent_save = Path(path).name
        self.paused = True
        self.running = True
        self.seed_display.set(f"seed {engine.config.world.seed} (from save)")
        self._reload_tree()
        self._update_labels()
        self.status.set(f"Forked {Path(path).name} with {cfg_path} (paused).")

    def _delete_selected(self) -> None:
        path = self._selected_path()
        if not path:
            return
        if not messagebox.askyesno("Delete", f"Delete checkpoint\n{path}?"):
            return
        try:
            delete_checkpoint(path)
        except Exception as exc:
            messagebox.showerror("Delete failed", str(exc))
            return
        self._refresh_saves()

    def _refresh_saves(self) -> None:
        self.save_list.delete(0, tk.END)
        for row in list_checkpoints(self.saves_dir):
            line = (
                f"t={row.get('tick', '?')}  n={row.get('alive', '?')}  "
                f"{row.get('name', '')}  | {row.get('path', '')}"
            )
            self.save_list.insert(tk.END, line)

    def _on_epoch(self, metrics, _eng=None) -> None:
        self._append_epoch_row(metrics)
        self._update_labels()

    def _append_epoch_row(self, metrics) -> None:
        adapt = getattr(metrics, "adaptation_score", 0.0)
        full = 100.0 * getattr(metrics, "adapted_frac", 0.0)
        self.tree.insert(
            "",
            0,
            values=(
                metrics.epoch,
                metrics.tick,
                metrics.alive_count,
                metrics.births_count,
                metrics.deaths_emergency,
                metrics.deaths_pitfall,
                f"{adapt:.3f}",
                f"{full:.0f}",
            ),
        )
        kids = self.tree.get_children()
        if len(kids) > 80:
            self.tree.delete(kids[-1])

    def _reload_tree(self) -> None:
        for kid in self.tree.get_children():
            self.tree.delete(kid)
        if self.engine is None:
            return
        for metrics in self.engine.epoch_history[-80:]:
            self._append_epoch_row(metrics)

    def _update_labels(self) -> None:
        if self.engine is None:
            return
        e = self.engine
        w = e.world
        life = e.lifetime
        last = e.epoch_history[-1] if e.epoch_history else None
        enc = life.pitfall_encounters
        adapt = "—" if enc == 0 else f"{life.pitfall_adapt_sum / enc:.3f}"
        full = "—" if enc == 0 else f"{100.0 * life.pitfall_zero_damage / enc:.0f}%"
        self.kpi.set(
            f"tick {w.tick}   alive {w.n}   epochs {e.epochs_completed}   "
            f"seed {e.config.world.seed}   backend {e.backend}   "
            f"stress {'ON' if w.stress_mode else 'off'}\n"
            f"last epoch adapt {getattr(last, 'adaptation_score', 0):.3f}   "
            f"lifetime adapt {adapt}  full {full}"
        )
        self.totals.set(format_totals_line(life) + f"  d.pit {life.deaths_pitfall}")
        self.pits.set(format_pitfall_line(w.pitfall_counts_by_name(), life, e.tick_stats))

    def _pump(self) -> None:
        if self.running and not self.paused and self.engine is not None:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 0.04:
                if self._should_pause():
                    self.paused = True
                    self.status.set(
                        f"Paused at tick {self.engine.world.tick}, "
                        f"epoch {self.engine.epochs_completed} (stop point reached). Save if you want."
                    )
                    self._update_labels()
                    break
                self.engine.tick()
                if self.engine.world.is_extinct:
                    self.paused = True
                    self.status.set("Population extinct. State can still be saved.")
                    self._update_labels()
                    break
            if self.grid_win is not None:
                self._draw_grid()
            elif self.engine is not None:
                self.status.set(
                    f"Running  tick {self.engine.world.tick}  alive {self.engine.world.n}  "
                    f"epochs {self.engine.epochs_completed}  (grid closed)"
                )
        self.root.after(1, self._pump)

    def _on_close(self) -> None:
        self._close_grid()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_studio(config: SimConfig, config_path: str | None = None, saves_dir: str | Path | None = None) -> None:
    app = StudioApp(config, config_path, Path(saves_dir) if saves_dir else default_saves_dir())
    app.run()
