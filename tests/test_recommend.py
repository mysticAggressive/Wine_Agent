"""Tests for agent/recommend.py (DEVELOPMENT_PLAN.md task 5.3)."""

from __future__ import annotations

import dataclasses
import itertools
import time
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from agent.config import DATA, FEATURES, MAX_PER_VENUE
from agent.csp_solver import SolverOptions
from agent.data_loader import load_all
from agent.models import Candidate, Style
from agent.preferences import default_preferences
from agent.recommend import cap_per_venue, recommend, solve_candidates
from agent.scoring import evaluate
from tests.test_wine_csp import brute_force

ALL_OPTIONS = [SolverOptions(*f) for f in itertools.product((True, False), repeat=4)]


def keys(result):
    return [(c.wine_id, c.venue_id, c.dish_id, c.score) for c in result.candidates]


def test_ranked_list_is_sorted(prefs, agent_data):
    result = recommend(prefs, agent_data)
    assert result.candidates
    order = [(-c.score, c.wine_id, c.venue_id, c.dish_id) for c in result.candidates]
    assert order == sorted(order)
    for c in result.candidates:
        assert c.score == evaluate(c.features, prefs.weights)
        assert c.price_at_venue <= prefs.budget_eur
        assert c.distance_m <= prefs.max_distance_m
        assert c.reasons


def test_candidates_are_every_solution(prefs, agent_data):
    candidates, diagnostics = solve_candidates(prefs, agent_data)
    expected = brute_force(prefs, agent_data)
    assert {(c.wine_id, c.venue_id, c.dish_id) for c in candidates} == expected
    assert diagnostics.solutions_found == len(expected)


def test_limit_keeps_the_best(prefs, agent_data):
    full, _ = solve_candidates(prefs, agent_data)
    top3, diagnostics = solve_candidates(prefs, agent_data, limit=3)
    assert top3 == full[:3]
    assert diagnostics.solutions_found == len(full)


def test_per_venue_cap_and_one_row_per_wine_venue(prefs, agent_data):
    p = dataclasses.replace(prefs, open_only=False, top_k=50)
    result = recommend(p, agent_data)
    per_venue = Counter(c.venue_id for c in result.candidates)
    assert max(per_venue.values()) <= MAX_PER_VENUE
    pairs = [(c.wine_id, c.venue_id) for c in result.candidates]
    assert len(pairs) == len(set(pairs))
    # recommend == cap_per_venue over the full ranked list.
    full, _ = solve_candidates(p, agent_data)
    assert result.candidates == cap_per_venue(full, p.top_k)


def test_top_k_limits_rows(prefs, agent_data):
    result = recommend(dataclasses.replace(prefs, top_k=2), agent_data)
    assert len(result.candidates) == 2


def test_cap_per_venue_keeps_best_dish():
    def cand(w, v, d, s):
        return Candidate(w, v, d, 10.0, 100.0, dict.fromkeys(FEATURES, s), s, ())

    rows = [
        cand("w1", "v1", "a", 0.9),
        cand("w1", "v1", "b", 0.8),
        cand("w2", "v1", "a", 0.7),
        cand("w3", "v1", "a", 0.6),
        cand("w3", "v2", "a", 0.5),
    ]
    kept = cap_per_venue(rows, top_k=10, max_per_venue=2)
    assert [(c.wine_id, c.venue_id, c.dish_id) for c in kept] == [
        ("w1", "v1", "a"),
        ("w2", "v1", "a"),
        ("w3", "v2", "a"),
    ]


def test_zero_solutions_gives_relaxation_hints(prefs, agent_data):
    result = recommend(dataclasses.replace(prefs, budget_eur=5), agent_data)
    assert result.candidates == []
    assert result.diagnostics.solutions_found == 0
    hints = result.diagnostics.relaxation_hints
    assert hints and hints[0].startswith("Removing the budget limit gives")
    assert "cheapest: €8.00" in hints[0]


def test_hints_name_the_style_when_style_is_the_problem(prefs, agent_data):
    # The only white (w0003) is over a 10 euro budget everywhere.
    p = dataclasses.replace(prefs, styles=frozenset({Style.WHITE}), budget_eur=10)
    hints = recommend(p, agent_data).diagnostics.relaxation_hints
    assert any(h.startswith("Accepting any wine style") for h in hints)
    assert any(h.startswith("Removing the budget limit") for h in hints)


def test_no_hints_when_there_are_solutions(prefs, agent_data):
    assert recommend(prefs, agent_data).diagnostics.relaxation_hints == []


def test_results_identical_across_runs(prefs, agent_data):
    assert keys(recommend(prefs, agent_data)) == keys(recommend(prefs, agent_data))


@pytest.mark.parametrize("options", ALL_OPTIONS)
def test_results_identical_under_every_solver_option(prefs, agent_data, options):
    p = dataclasses.replace(prefs, open_only=False, top_k=50)
    assert keys(recommend(p, agent_data, options)) == keys(recommend(p, agent_data))


def test_diagnostics_filled(prefs, agent_data):
    d = recommend(prefs, agent_data).diagnostics
    assert d.initial_domain_sizes == {"wine": 5, "venue": 4, "dish": 3}
    assert set(d.pruned_by_constraint) >= {"style", "budget", "stocks", "pairs"}
    assert d.ac3_revisions > 0
    assert d.assignments > 0
    assert sorted(d.variable_order) == ["dish", "venue", "wine"]
    assert d.elapsed_ms >= 0
    # node/4 (bar) only opens at 18:00; it is noon.
    assert d.venue_exclusions == {"node/4": "open_now"}


def test_invalid_preferences_raise(prefs, agent_data):
    with pytest.raises(ValueError, match="Invalid preferences"):
        recommend(dataclasses.replace(prefs, budget_eur=0), agent_data)
    with pytest.raises(ValueError, match="unknown dish_id"):
        recommend(dataclasses.replace(prefs, dish_id="haggis"), agent_data)


# --------------------------------------------------------------------------- #
# Real data (skipped until the OSM venues and the inventory are built)
# --------------------------------------------------------------------------- #

REAL_FILES = ("wines.csv", "venues_madrid.json", "inventory.csv", "pairing_rules.json")
needs_real_data = pytest.mark.skipif(
    not all((DATA / name).exists() for name in REAL_FILES),
    reason="real venues / inventory not built yet",
)


@needs_real_data
def test_performance_on_real_data():
    data = load_all()
    prefs = default_preferences(datetime(2026, 11, 11, 20, 0, tzinfo=ZoneInfo("Europe/Madrid")))
    recommend(prefs, data)  # warm the static indexes, as the app does once
    started = time.perf_counter()
    result = recommend(prefs, data)
    elapsed = time.perf_counter() - started
    assert result.candidates
    assert elapsed < 2.0, f"recommend took {elapsed:.2f} s"
