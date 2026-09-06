"""
Watch window sizing (no pygame display required).

Author: Cursor Grok 4.6 High Fast
"""

from src.watch import HUD_H, fit_watch_geometry


def test_small_world_keeps_preferred_cell():
    layout = fit_watch_geometry(80, 80, preferred_cell=8, display_w=1920, display_h=1080)
    assert layout.cell == 8
    assert layout.window_w == 640
    assert layout.window_h == 640 + HUD_H
    assert layout.window_h < 1080
    assert layout.window_w < 1920


def test_200_world_shrinks_to_fit_1080p():
    layout = fit_watch_geometry(200, 200, preferred_cell=8, display_w=1920, display_h=1080)
    assert layout.window_w < 1920
    assert layout.window_h < 1080
    assert layout.cell >= 1
    assert layout.cell < 8
    assert layout.window_w == 200 * layout.cell


def test_huge_world_never_exceeds_usable_desktop():
    layout = fit_watch_geometry(2000, 2000, preferred_cell=8, display_w=1920, display_h=1080)
    assert layout.window_w <= 1920 - 80
    assert layout.window_h <= 1080 - 120
    assert layout.window_w >= 1
    assert layout.window_h > HUD_H


def test_cell_size_is_a_maximum_not_a_minimum():
    small = fit_watch_geometry(80, 80, preferred_cell=4, display_w=1920, display_h=1080)
    assert small.cell == 4
    assert small.window_w == 320
