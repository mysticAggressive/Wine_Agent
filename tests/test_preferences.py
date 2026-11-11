"""Tests for agent/preferences.py (task 2.2)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from agent.config import CENTER, FEATURES
from agent.preferences import Preferences, default_preferences, haversine_m, validate

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=ZoneInfo("Europe/Madrid"))


@pytest.fixture
def defaults() -> Preferences:
    return default_preferences(NOW)


def test_defaults_match_the_plan(defaults):
    assert defaults.styles == frozenset()
    assert defaults.budget_eur == 50
    assert defaults.venue_kinds == frozenset()
    assert defaults.dish_id is None
    assert defaults.want_body == 4
    assert (defaults.user_lat, defaults.user_lon) == CENTER
    assert defaults.max_distance_m == 2000
    assert defaults.at_time == NOW
    assert defaults.open_only is True
    assert defaults.exclude_unknown_hours is False
    assert defaults.weights == {name: 5 for name in FEATURES}
    assert defaults.top_k == 10
    assert not hasattr(defaults, "want_acidity")


def test_defaults_validate_clean(defaults):
    assert validate(defaults) == []


def test_boundary_values_are_valid(defaults):
    prefs = replace(
        defaults,
        want_body=2,
        max_distance_m=200,
        top_k=50,
        weights={name: 0 for name in FEATURES},
    )
    assert validate(prefs) == []
    assert validate(replace(defaults, want_body=5, max_distance_m=4000, top_k=1)) == []


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"budget_eur": 0}, "budget_eur"),
        ({"budget_eur": -5.0}, "budget_eur"),
        ({"want_body": 1}, "want_body"),
        ({"want_body": 6}, "want_body"),
        ({"max_distance_m": 199}, "max_distance_m"),
        ({"max_distance_m": 4001}, "max_distance_m"),
        ({"at_time": datetime(2026, 11, 11, 12, 0)}, "at_time"),
        ({"weights": {"taste": 5.0}}, "weights"),
        ({"weights": {**{n: 5.0 for n in FEATURES}, "extra": 1.0}}, "weights"),
        ({"weights": {**{n: 5.0 for n in FEATURES}, "taste": 11.0}}, "taste"),
        ({"weights": {**{n: 5.0 for n in FEATURES}, "distance": -1.0}}, "distance"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": 51}, "top_k"),
        ({"user_lat": 40.47}, "user position"),  # about 5.9 km north of the centre
    ],
)
def test_each_rule_has_a_failing_case(defaults, changes, expected):
    problems = validate(replace(defaults, **changes))
    assert len(problems) == 1
    assert expected in problems[0]


def test_several_problems_are_all_reported(defaults):
    problems = validate(replace(defaults, budget_eur=0, want_body=9, top_k=0))
    assert len(problems) == 3


def test_haversine():
    assert haversine_m(*CENTER, *CENTER) == 0
    # one degree of latitude is about 111.2 km
    assert haversine_m(40.0, -3.7, 41.0, -3.7) == pytest.approx(111_195, rel=1e-3)
    assert haversine_m(40.4168, -3.7038, 40.4155, -3.7074) == pytest.approx(337, abs=5)
