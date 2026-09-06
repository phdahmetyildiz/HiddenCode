"""
Studio seed picker: config seed vs one-shot random (does not rewrite JSON).

Author: Cursor Grok 4.6 High Fast
"""

from src.studio import choose_world_seed


def test_choose_world_seed_from_config():
    seed, label = choose_world_seed(42, False)
    assert seed == 42
    assert "42" in label
    assert "from config" in label


def test_choose_world_seed_random_in_range_and_not_always_config():
    seeds = [choose_world_seed(42, True)[0] for _ in range(12)]
    assert all(1 <= s < 2**31 for s in seeds)
    assert any(s != 42 for s in seeds)
    labels = [choose_world_seed(1, True)[1] for _ in range(3)]
    assert all(text.startswith("seed ") and "(random)" in text for text in labels)
