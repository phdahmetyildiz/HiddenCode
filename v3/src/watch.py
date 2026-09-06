"""
Local watch window. Optional pygame — engine does not import this.

Author: Cursor Grok 4.6 High Fast
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import SimConfig
from src.engine import SimulationEngine
from src.livability import evaluate

HUD_H = 112
SPARK_H = 34
# Leave room for title bar, taskbar, and desktop margins.
_MARGIN_X = 80
_MARGIN_Y = 120


@dataclass(frozen=True)
class WatchLayout:
    window_w: int
    window_h: int
    cell: int


def fit_watch_geometry(
    world_w: int,
    world_h: int,
    preferred_cell: int,
    display_w: int,
    display_h: int,
    hud: int = HUD_H,
    margin_x: int = _MARGIN_X,
    margin_y: int = _MARGIN_Y,
) -> WatchLayout:
    """Size the watch window to the desktop, using `preferred_cell` as a maximum."""
    usable_w = max(32, int(display_w) - margin_x)
    usable_h = max(32, int(display_h) - margin_y - hud)
    ww = max(1, int(world_w))
    hh = max(1, int(world_h))
    scale = min(float(preferred_cell), usable_w / ww, usable_h / hh)
    if scale >= 1.0:
        cell = max(1, int(scale))
        grid_w = ww * cell
        grid_h = hh * cell
    else:
        cell = 0
        grid_w = max(1, int(ww * scale))
        grid_h = max(1, int(hh * scale))
    return WatchLayout(window_w=grid_w, window_h=grid_h + hud, cell=cell)


def _desktop_size(pygame_mod) -> tuple[int, int]:
    info = pygame_mod.display.Info()
    dw = int(info.current_w or 0)
    dh = int(info.current_h or 0)
    if dw < 64 or dh < 64:
        return 1280, 720
    return dw, dh


def format_totals_line(life) -> str:
    return (
        f"births {life.births}  "
        f"d.em {life.deaths_emergency}  d.st {life.deaths_starvation}  "
        f"d.age {life.deaths_max_age}"
    )


def format_pitfall_line(counts: dict[str, int], life, tick) -> str:
    pits = " ".join(f"{name}:{n}" for name, n in counts.items()) or "none"
    enc = life.pitfall_encounters
    if enc == 0:
        adapt = "adapt —"
    else:
        cum = life.pitfall_adapt_sum / enc
        full = 100.0 * life.pitfall_zero_damage / enc
        adapt = f"adapt {cum:.2f}  full {full:.0f}%"
        if tick.pitfall_encounters > 0:
            now = tick.pitfall_adapt_sum / tick.pitfall_encounters
            adapt += f"  now {now:.2f}"
    return f"pits {pits}  enc {enc}  d.pit {life.deaths_pitfall}  {adapt}"


def write_adaptation_csv(path, series: list[float], sample_every: int = 10) -> None:
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample", "tick", "adaptation_cum"])
        for i, value in enumerate(series):
            tick = (i + 1) * sample_every
            cell = "" if value != value else f"{float(value):.6f}"
            w.writerow([i, tick, cell])


def _draw_sparkline(surface, pygame_mod, series: list[float], x: int, y: int, width: int, height: int) -> None:
    pygame_mod.draw.rect(surface, (28, 28, 36), (x, y, width, height), border_radius=2)
    pygame_mod.draw.line(
        surface, (50, 50, 60),
        (x, y + height // 2), (x + width, y + height // 2), 1,
    )
    pts = [v for v in series if v == v]
    if len(pts) < 2 or width < 4 or height < 4:
        return
    last = pts[-min(len(pts), width):]
    n = len(last)
    coords = []
    for i, v in enumerate(last):
        px = x + int(i * (width - 1) / max(1, n - 1))
        py = y + height - 1 - int(max(0.0, min(1.0, v)) * (height - 1))
        coords.append((px, py))
    if len(coords) >= 2:
        pygame_mod.draw.lines(surface, (90, 190, 220), False, coords, 2)


def run_watch(config: SimConfig) -> None:
    try:
        import os

        import pygame
    except ImportError as exc:
        raise SystemExit(
            "Watch mode needs pygame. Install with: pip install pygame"
        ) from exc

    engine = SimulationEngine(config)
    engine.initialize()
    report = evaluate(config)
    print(report.as_text())

    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    pygame.init()
    display_w, display_h = _desktop_size(pygame)
    layout = fit_watch_geometry(
        config.world.width,
        config.world.height,
        config.viz.cell_size,
        display_w,
        display_h,
    )
    screen_w, screen_h = layout.window_w, layout.window_h
    if layout.cell and layout.cell < config.viz.cell_size:
        print(
            f"Watch: scaled cell {config.viz.cell_size} → {layout.cell} px "
            f"to fit {display_w}×{display_h} ({screen_w}×{screen_h} window)"
        )
    elif layout.cell == 0:
        print(
            f"Watch: world larger than one pixel per cell; "
            f"window {screen_w}×{screen_h} on {display_w}×{display_h}"
        )

    screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)
    pygame.display.set_caption("Evolution Simulator v3")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    paused = False
    speed = 1
    running = True
    render_every = config.viz.render_every_n_ticks

    def draw() -> None:
        w = engine.world
        grid = pygame.Surface((config.world.width, config.world.height))
        grid.fill((18, 18, 24))
        fx, fy = w.food_positions()
        for x, y in zip(fx.tolist(), fy.tolist()):
            grid.set_at((int(x), int(y)), (40, 160, 70))
        px, py = w.pitfall_positions()
        for x, y in zip(px.tolist(), py.tolist()):
            grid.set_at((int(x), int(y)), (180, 40, 40))
        n = w.n
        if n:
            e = w.energy[:n]
            for i in range(n):
                gcol = int(40 + 200 * float(e[i]))
                rcol = int(220 * (1.0 - float(e[i])))
                grid.set_at((int(w.x[i]), int(w.y[i])), (rcol, gcol, 40))
        area_h = max(1, screen_h - HUD_H)
        scale = min(screen_w / config.world.width, area_h / config.world.height)
        dw = max(1, int(config.world.width * scale))
        dh = max(1, int(config.world.height * scale))
        scaled = pygame.transform.scale(grid, (dw, dh))
        screen.fill((12, 12, 16))
        screen.blit(scaled, ((screen_w - dw) // 2, (area_h - dh) // 2))
        life = engine.lifetime
        hud_y = screen_h - HUD_H
        pygame.draw.rect(screen, (16, 16, 22), (0, hud_y, screen_w, HUD_H))
        line0 = (
            f"tick {w.tick}  alive {w.n}  "
            f"{'PAUSE' if paused else 'RUN'}  speed {speed}x  "
            f"{'STRESS' if w.stress_mode else ''}"
        )
        line1 = format_totals_line(life)
        line2 = format_pitfall_line(w.pitfall_counts_by_name(), life, engine.tick_stats)
        screen.blit(font.render(line0, True, (230, 230, 230)), (8, hud_y + 4))
        screen.blit(font.render(line1, True, (210, 210, 220)), (8, hud_y + 22))
        screen.blit(font.render(line2, True, (210, 210, 220)), (8, hud_y + 40))
        _draw_sparkline(
            screen, pygame,
            engine.adaptation_series,
            8, hud_y + 58,
            max(32, screen_w - 16), SPARK_H,
        )
        screen.blit(
            font.render("space pause  . step  +/- speed  q quit   (adapt 0–1 over time)", True, (140, 140, 150)),
            (8, screen_h - 18),
        )
        pygame.display.flip()

    draw()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen_w = max(64, int(event.w))
                screen_h = max(HUD_H + 32, int(event.h))
                screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)
                draw()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_PERIOD:
                    engine.tick()
                    draw()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(64, speed * 2)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(1, speed // 2)

        if not paused:
            for _ in range(speed):
                engine.tick()
                if engine.world.is_extinct:
                    paused = True
                    break
            if engine.world.tick % render_every == 0:
                draw()

        clock.tick(30)

    from datetime import datetime
    from pathlib import Path

    out = Path(config.viz.output_dir) / f"watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    write_adaptation_csv(out / "adaptation.csv", engine.adaptation_series)
    print(f"Adaptation series: {out / 'adaptation.csv'}")
    pygame.quit()
