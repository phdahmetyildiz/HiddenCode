"""Local watch window. Optional pygame — engine does not import this."""

from __future__ import annotations

from src.config import SimConfig
from src.engine import SimulationEngine
from src.livability import evaluate


def run_watch(config: SimConfig) -> None:
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit(
            "Watch mode needs pygame. Install with: pip install pygame"
        ) from exc

    engine = SimulationEngine(config)
    engine.initialize()
    report = evaluate(config)
    print(report.as_text())

    cell = config.viz.cell_size
    hud = 48
    screen_w = config.world.width * cell
    screen_h = config.world.height * cell + hud
    pygame.init()
    screen = pygame.display.set_mode((screen_w, screen_h))
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
            wt = w.weight[:n]
            for i in range(n):
                gcol = int(40 + 200 * float(e[i]))
                rcol = int(220 * (1.0 - float(e[i])))
                grid.set_at((int(w.x[i]), int(w.y[i])), (rcol, gcol, 40))
                _ = wt
        scaled = pygame.transform.scale(grid, (screen_w, screen_h - hud))
        screen.fill((12, 12, 16))
        screen.blit(scaled, (0, 0))
        stats = engine.tick_stats
        line = (
            f"tick {w.tick}  alive {w.n}  "
            f"{'PAUSE' if paused else 'RUN'}  speed {speed}x  "
            f"births {stats.births}  "
            f"d.em {stats.deaths_emergency} d.st {stats.deaths_starvation} "
            f"d.age {stats.deaths_max_age}  "
            f"{'STRESS' if w.stress_mode else ''}"
        )
        screen.blit(font.render(line, True, (230, 230, 230)), (8, screen_h - 36))
        screen.blit(
            font.render("space pause  . step  +/- speed  q quit", True, (140, 140, 150)),
            (8, screen_h - 18),
        )
        pygame.display.flip()

    draw()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
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

    pygame.quit()
