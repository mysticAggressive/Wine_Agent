"""Tests for the one-line filter summary shown under the preference panel."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from agent.models import Style
from agent.preferences import Preferences
from ui.sidebar import WHERE, filter_summary

MADRID = ZoneInfo("Europe/Madrid")


def make_prefs(**overrides) -> Preferences:
    fields = {
        "styles": frozenset(),
        "budget_eur": 50.0,
        "venue_kinds": frozenset(),
        "dish_id": None,
        "want_body": 4,
        "user_lat": 40.4168,
        "user_lon": -3.7038,
        "max_distance_m": 2000.0,
        "at_time": datetime(2026, 11, 27, 13, 0, tzinfo=MADRID),
        "open_only": True,
        "exclude_unknown_hours": False,
        "weights": {},
    }
    fields.update(overrides)
    return Preferences(**fields)


def test_default_summary_mentions_any_style_budget_distance_and_open():
    text = filter_summary(make_prefs(), "Puerta del Sol", "Surprise me")

    assert text.startswith("Any style · ≤ €50")
    assert "2 km from Puerta del Sol" in text
    assert "open " in text


def test_styles_venue_and_food_appear_and_open_is_omitted():
    prefs = make_prefs(
        styles=frozenset({Style.RED, Style.WHITE}),
        venue_kinds=WHERE["Shop"],
        open_only=False,
    )

    text = filter_summary(prefs, "Puerta del Sol", "Paella")

    assert "Red / White" in text
    assert "shop" in text
    assert "with paella" in text
    assert "open " not in text
