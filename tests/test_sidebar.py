"""Tests for the filter-chip labels drawn above the results."""

from __future__ import annotations

from datetime import date, time

from ui.sidebar import CLICKED, DEFAULTS, START_KEY, chip_labels


def test_default_labels():
    labels = chip_labels(DEFAULTS)

    assert labels["what"] == "Any wine · full · ≤ €50"
    assert labels["where"] == "2 km from Puerta del Sol"
    assert labels["when"] == "Open now"
    assert labels["ranking"] == "Ranking" and labels["solver"] == "Solver"


def test_labels_follow_the_widget_values():
    state = {
        **DEFAULTS,
        "styles": ["Red", "White"],
        "where": "Shop",
        "food": "Paella",
        START_KEY: CLICKED,
        "max_distance": 1500,
        "when": "Pick a time",
        "pick_date": date(2026, 12, 5),
        "pick_time": time(22, 0),
        "w_quality": 9,
        "use_ac3": False,
    }

    labels = chip_labels(state)

    assert labels["what"] == "Red / White · full · ≤ €50 · shop · with paella"
    assert labels["where"] == "1.5 km from clicked point"
    assert labels["when"] == "Open Sat 05 Dec, 22:00"
    assert labels["ranking"] == "Ranking · custom"
    assert labels["solver"] == "Solver · 1 off"
    assert chip_labels({**state, "open_only": False})["when"] == "Any time"
