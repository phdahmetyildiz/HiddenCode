"""
Lightweight plotting for study results: mean +/- CI trajectories, survival
curves, and per-arm comparison bars.

Two backends:
  - numpy software renderer -> RGB uint8 array (no dependency; used for the Tk
    canvas live display and for PNG export when matplotlib is absent).
  - matplotlib (optional) -> labelled PNG files, used only if importable.

Author: Cursor Claude Opus 4.8 High
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

BG = (18, 18, 24)
AXIS = (120, 120, 135)
GRID = (44, 44, 56)
TEXT = (210, 210, 220)

# Distinct arm colors (R, G, B)
PALETTE = (
    (90, 170, 255),
    (255, 140, 90),
    (120, 210, 120),
    (220, 120, 220),
    (240, 210, 90),
    (120, 220, 220),
)


def arm_color(i: int) -> tuple[int, int, int]:
    return PALETTE[i % len(PALETTE)]


# ---------------------------------------------------------------------------
# numpy software renderer
# ---------------------------------------------------------------------------

class _Canvas:
    def __init__(self, w: int, h: int, bg=BG):
        self.w = w
        self.h = h
        self.img = np.empty((h, w, 3), dtype=np.uint8)
        self.img[:] = bg
        # plot area margins
        self.left = 52
        self.right = 14
        self.top = 14
        self.bottom = 34
        self.x0 = self.left
        self.y0 = self.top
        self.pw = max(1, w - self.left - self.right)
        self.ph = max(1, h - self.top - self.bottom)

    def px(self, frac: float) -> int:
        return int(round(self.x0 + frac * self.pw))

    def py(self, frac: float) -> int:
        # frac 0 at bottom, 1 at top
        return int(round(self.y0 + (1.0 - frac) * self.ph))

    def _set(self, x: int, y: int, color) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            self.img[y, x] = color

    def hline(self, y: int, x1: int, x2: int, color) -> None:
        if 0 <= y < self.h:
            a, b = sorted((max(0, x1), min(self.w - 1, x2)))
            self.img[y, a:b + 1] = color

    def vline(self, x: int, y1: int, y2: int, color) -> None:
        if 0 <= x < self.w:
            a, b = sorted((max(0, y1), min(self.h - 1, y2)))
            self.img[a:b + 1, x] = color

    def line(self, x1: int, y1: int, x2: int, y2: int, color, width: int = 1) -> None:
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        x, y = x1, y1
        while True:
            for ox in range(-(width // 2), width // 2 + 1):
                for oy in range(-(width // 2), width // 2 + 1):
                    self._set(x + ox, y + oy, color)
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def frame(self) -> None:
        self.hline(self.y0 + self.ph, self.x0, self.x0 + self.pw, AXIS)
        self.vline(self.x0, self.y0, self.y0 + self.ph, AXIS)
        for g in (0.25, 0.5, 0.75, 1.0):
            y = self.py(g)
            self.hline(y, self.x0, self.x0 + self.pw, GRID)


def _finite_range(series_list: Sequence[np.ndarray]) -> tuple[float, float]:
    lo, hi = np.inf, -np.inf
    for s in series_list:
        v = s[np.isfinite(s)]
        if v.size:
            lo = min(lo, float(v.min()))
            hi = max(hi, float(v.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _interp(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    if xs.size == 0:
        return float("nan")
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    j = int(np.searchsorted(xs, x))
    x1, x2 = xs[j - 1], xs[j]
    y1, y2 = ys[j - 1], ys[j]
    if x2 == x1:
        return float(y1)
    t = (x - x1) / (x2 - x1)
    return float(y1 + t * (y2 - y1))


def trajectory_rgb(arms, metric: str, size: tuple[int, int] = (720, 420),
                   y_range: Optional[tuple[float, float]] = None) -> np.ndarray:
    """Mean line + CI band per arm for one metric over epoch index."""
    w, h = size
    c = _Canvas(w, h)
    c.frame()

    means = []
    lows = []
    highs = []
    xs_list = []
    for arm in arms:
        m = arm.metrics.get(metric)
        if not m:
            means.append(np.array([]))
            lows.append(np.array([]))
            highs.append(np.array([]))
            xs_list.append(np.array([]))
            continue
        means.append(np.array(m["mean"], dtype=float))
        lows.append(np.array(m["ci_low"], dtype=float))
        highs.append(np.array(m["ci_high"], dtype=float))
        xs_list.append(np.arange(len(m["mean"]), dtype=float))

    if y_range is None:
        lo, hi = _finite_range(means + lows + highs)
    else:
        lo, hi = y_range
    span = hi - lo if hi > lo else 1.0
    max_x = max((x.max() if x.size else 0.0) for x in xs_list) or 1.0

    def to_fy(val: float) -> float:
        return (val - lo) / span

    # bands first (alpha blend), then mean lines on top
    for i, arm in enumerate(arms):
        xs = xs_list[i]
        if xs.size < 1:
            continue
        col = np.array(arm_color(i), dtype=float)
        lo_s, hi_s = lows[i], highs[i]
        for px in range(c.x0, c.x0 + c.pw + 1):
            xf = (px - c.x0) / c.pw * max_x
            yl = _interp(xs, lo_s, xf)
            yh = _interp(xs, hi_s, xf)
            if not (np.isfinite(yl) and np.isfinite(yh)):
                continue
            pyl = c.py(to_fy(yl))
            pyh = c.py(to_fy(yh))
            a, b = sorted((pyh, pyl))
            a = max(c.y0, a)
            b = min(c.y0 + c.ph, b)
            if a <= b:
                block = c.img[a:b + 1, px].astype(float)
                c.img[a:b + 1, px] = (0.75 * block + 0.25 * col).astype(np.uint8)

    for i, arm in enumerate(arms):
        xs = xs_list[i]
        if xs.size < 1:
            continue
        col = arm_color(i)
        mean_s = means[i]
        prev = None
        for j in range(xs.size):
            if not np.isfinite(mean_s[j]):
                prev = None
                continue
            px = c.px(xs[j] / max_x)
            py = c.py(to_fy(mean_s[j]))
            if prev is not None:
                c.line(prev[0], prev[1], px, py, col, width=2)
            prev = (px, py)
    return c.img


def survival_rgb(arms, size: tuple[int, int] = (720, 300)) -> np.ndarray:
    w, h = size
    c = _Canvas(w, h)
    c.frame()
    max_x = 1.0
    for arm in arms:
        max_x = max(max_x, float(len(arm.survival_curve)))
    for i, arm in enumerate(arms):
        col = arm_color(i)
        sc = arm.survival_curve
        prev = None
        for j, val in enumerate(sc):
            px = c.px(j / max_x)
            py = c.py(float(val))  # survival already in [0,1]
            if prev is not None:
                c.line(prev[0], prev[1], px, py, col, width=2)
            prev = (px, py)
    return c.img


def bars_rgb(arms, metric: str, size: tuple[int, int] = (520, 360)) -> np.ndarray:
    """Final mean +/- CI per arm as bars with error whiskers."""
    w, h = size
    c = _Canvas(w, h)
    c.frame()
    finals = []
    for arm in arms:
        s = arm.final.get(metric, {})
        finals.append((s.get("mean", float("nan")), s.get("ci_low", float("nan")),
                       s.get("ci_high", float("nan"))))
    vals = [f[0] for f in finals if np.isfinite(f[0])]
    lows = [f[1] for f in finals if np.isfinite(f[1])]
    highs = [f[2] for f in finals if np.isfinite(f[2])]
    if not vals:
        return c.img
    lo = min(0.0, min(lows) if lows else 0.0)
    hi = max(highs) if highs else max(vals)
    span = hi - lo if hi > lo else 1.0
    n = len(arms)
    slot = c.pw / max(1, n)
    bar_w = int(slot * 0.5)
    for i, (mean, cl, ch) in enumerate(finals):
        if not np.isfinite(mean):
            continue
        col = arm_color(i)
        cx = int(c.x0 + slot * (i + 0.5))
        y_base = c.py((0.0 - lo) / span)
        y_top = c.py((mean - lo) / span)
        a, b = sorted((y_base, y_top))
        for x in range(cx - bar_w // 2, cx + bar_w // 2 + 1):
            c.vline(x, a, b, col)
        if np.isfinite(cl) and np.isfinite(ch):
            yl = c.py((cl - lo) / span)
            yh = c.py((ch - lo) / span)
            c.vline(cx, min(yl, yh), max(yl, yh), TEXT)
            c.hline(yl, cx - 6, cx + 6, TEXT)
            c.hline(yh, cx - 6, cx + 6, TEXT)
    return c.img


# ---------------------------------------------------------------------------
# PNG export (minimal, no dependency)
# ---------------------------------------------------------------------------

def save_png(path: str | Path, rgb: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = rgb.shape
    raw = bytearray()
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    for y in range(h):
        raw.append(0)  # filter type 0 (None)
        raw.extend(arr[y].tobytes())
    compressed = zlib.compress(bytes(raw), 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit, truecolor RGB
    png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.write_bytes(png)
    return path


# ---------------------------------------------------------------------------
# High-level: write all study plots (matplotlib if available, else numpy)
# ---------------------------------------------------------------------------

def save_study_plots(result, out_dir: str | Path,
                     metrics: Sequence[str] = ("adaptation_score", "alive_count", "genetic_diversity")) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        return _save_with_matplotlib(result, out, metrics)
    except Exception:
        pass
    written: list[Path] = []
    for metric in metrics:
        if not result.arms or metric not in result.arms[0].metrics:
            continue
        written.append(save_png(out / f"trajectory_{metric}.png",
                                trajectory_rgb(result.arms, metric)))
    written.append(save_png(out / "survival.png", survival_rgb(result.arms)))
    if result.arms and result.spec.compare_metric in result.arms[0].final:
        written.append(save_png(out / f"bars_{result.spec.compare_metric}.png",
                                bars_rgb(result.arms, result.spec.compare_metric)))
    return written


def _save_with_matplotlib(result, out: Path, metrics: Sequence[str]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    written: list[Path] = []
    for metric in metrics:
        if not result.arms or metric not in result.arms[0].metrics:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=110)
        for i, arm in enumerate(result.arms):
            m = arm.metrics[metric]
            x = np.arange(len(m["mean"]))
            col = np.array(arm_color(i)) / 255.0
            ax.plot(x, m["mean"], color=col, label=arm.label, lw=2)
            ax.fill_between(x, m["ci_low"], m["ci_high"], color=col, alpha=0.2)
        ax.set_xlabel("epoch index")
        ax.set_ylabel(metric)
        ax.set_title(f"{result.spec.name}: {metric} (mean +/- 95% CI)")
        ax.legend()
        fig.tight_layout()
        p = out / f"trajectory_{metric}.png"
        fig.savefig(p)
        plt.close(fig)
        written.append(p)

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=110)
    for i, arm in enumerate(result.arms):
        col = np.array(arm_color(i)) / 255.0
        ax.plot(np.arange(len(arm.survival_curve)), arm.survival_curve,
                color=col, label=arm.label, lw=2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("epoch index")
    ax.set_ylabel("fraction of replicates still running")
    ax.set_title(f"{result.spec.name}: survival")
    ax.legend()
    fig.tight_layout()
    p = out / "survival.png"
    fig.savefig(p)
    plt.close(fig)
    written.append(p)
    return written
