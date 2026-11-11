"""Tests for agent/wine_csp.py (DEVELOPMENT_PLAN.md task 5.1)."""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from agent.config import MIN_PAIRING_SCORE
from agent.csp_solver import SolverOptions, solve_all
from agent.data_loader import AgentData
from agent.models import OpenState, Style, VenueKind
from agent.opening import open_state
from agent.pairing import dishes_for_venue, pairing_score
from agent.preferences import Preferences, haversine_m
from agent.wine_csp import (
    BUDGET,
    CONSTRAINT_IDS,
    DISH,
    OPEN_NOW,
    VENUE,
    WINE,
    build_wine_csp,
)
from tests.conftest import MONDAY_NOON

ALL_OPTIONS = [SolverOptions(*flags) for flags in itertools.product((True, False), repeat=4)]


def brute_force(prefs: Preferences, data: AgentData) -> set[tuple[str, str, str]]:
    """Every (wine, venue, dish) triple passing all eight hard constraints, checked
    directly against the data without any CSP machinery."""
    rules = data.pairing_rules
    dishes = [prefs.dish_id] if prefs.dish_id is not None else list(rules.dishes)
    found: set[tuple[str, str, str]] = set()
    for wine_id, wine in data.wines.items():
        if prefs.styles and wine.style not in prefs.styles:
            continue
        for venue_id, venue in data.venues.items():
            if prefs.venue_kinds and venue.kind not in prefs.venue_kinds:
                continue
            if prefs.open_only:
                state = open_state(venue.opening_hours, prefs.at_time)
                if state == OpenState.CLOSED:
                    continue
                if state == OpenState.UNKNOWN and prefs.exclude_unknown_hours:
                    continue
            distance = haversine_m(prefs.user_lat, prefs.user_lon, venue.lat, venue.lon)
            if distance > prefs.max_distance_m:
                continue
            if not data.inventory.stocks(venue_id, wine_id):
                continue
            if data.inventory.price(venue_id, wine_id) > prefs.budget_eur:
                continue
            for dish_id in dishes:
                if dish_id not in dishes_for_venue(venue, rules):
                    continue
                if pairing_score(wine, dish_id, rules) < MIN_PAIRING_SCORE:
                    continue
                found.add((wine_id, venue_id, dish_id))
    return found


def solved(prefs: Preferences, data: AgentData, options: SolverOptions | None = None) -> set:
    csp = build_wine_csp(prefs, data)
    solutions, _ = solve_all(csp, options or SolverOptions())
    return {(s[WINE], s[VENUE], s[DISH]) for s in solutions}


def test_three_variables_and_eight_constraints(prefs, agent_data):
    csp = build_wine_csp(prefs, agent_data)
    assert list(csp.variables) == [WINE, VENUE, DISH]
    assert [c.id for c in csp.constraints] == list(CONSTRAINT_IDS)
    assert csp.domains[WINE] == set(agent_data.wines)
    assert csp.domains[VENUE] == set(agent_data.venues)
    assert csp.domains[DISH] == set(agent_data.pairing_rules.dishes)


def test_open_now_left_out_when_not_open_only(prefs, agent_data):
    csp = build_wine_csp(dataclasses.replace(prefs, open_only=False), agent_data)
    assert OPEN_NOW not in [c.id for c in csp.constraints]


def test_skip_removes_named_constraint_and_rejects_unknown(prefs, agent_data):
    csp = build_wine_csp(prefs, agent_data, skip=(BUDGET,))
    assert BUDGET not in [c.id for c in csp.constraints]
    with pytest.raises(ValueError, match="unknown constraint"):
        build_wine_csp(prefs, agent_data, skip=("nope",))


def test_dish_domain_is_single_value_when_chosen(prefs, agent_data):
    csp = build_wine_csp(dataclasses.replace(prefs, dish_id="shellfish"), agent_data)
    assert csp.domains[DISH] == {"shellfish"}


def test_budget_below_cheapest_wine_empties_a_domain(prefs, agent_data):
    cheapest = min(w.retail_price for w in agent_data.wines.values())
    csp = build_wine_csp(dataclasses.replace(prefs, budget_eur=cheapest - 1), agent_data)
    solutions, stats = solve_all(csp, SolverOptions())
    assert solutions == []
    assert 0 in stats.domain_sizes_after_ac3.values()
    assert stats.pruned_by_constraint[BUDGET] > 0


def test_restaurant_with_nothing_in_budget_and_style_is_removed_by_ac3(prefs, agent_data):
    # way/3 is the only restaurant; its only white (w0003) costs 45 there.
    white_40 = dataclasses.replace(prefs, styles=frozenset({Style.WHITE}), budget_eur=40)
    csp = build_wine_csp(white_40, agent_data)
    _, stats = solve_all(csp, SolverOptions())
    assert stats.pruned_by_constraint[BUDGET] >= 1
    assert "way/3" not in csp.domains[VENUE]
    assert csp.venue_exclusions()["way/3"] == BUDGET
    # It survived the unary constraints, so it was AC-3 that removed it.
    assert "way/3" not in csp.unary_venue_exclusions


def test_closed_venue_excluded_by_open_now(prefs, agent_data):
    # node/1 is closed on Mondays.
    csp = build_wine_csp(dataclasses.replace(prefs, at_time=MONDAY_NOON), agent_data)
    solve_all(csp, SolverOptions())
    assert csp.venue_exclusions()["node/1"] == OPEN_NOW


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"styles": frozenset({Style.RED})},
        {"styles": frozenset({Style.WHITE, Style.SPARKLING}), "budget_eur": 20.0},
        {"venue_kinds": frozenset({VenueKind.SHOP})},
        {"venue_kinds": frozenset({VenueKind.RESTAURANT, VenueKind.BAR})},
        {"open_only": False},
        {"exclude_unknown_hours": True},
        {"max_distance_m": 500.0},
        {"dish_id": "shellfish"},
        {"dish_id": "chocolate_dessert"},
        {"budget_eur": 5.0},
        {"at_time": MONDAY_NOON},
    ],
)
def test_solver_matches_brute_force(prefs, agent_data, change):
    p = dataclasses.replace(prefs, **change)
    expected = brute_force(p, agent_data)
    for options in ALL_OPTIONS:
        assert solved(p, agent_data, options) == expected


def test_value_key_prefers_widely_stocked_wines(prefs, agent_data):
    csp = build_wine_csp(dataclasses.replace(prefs, open_only=False), agent_data)
    assert csp.value_key is not None
    # w0002 is carried by all four venues, w0005 only by node/1.
    assert csp.value_key(csp, WINE, "w0002") < csp.value_key(csp, WINE, "w0005")
    assert csp.value_key(csp, DISH, "shellfish") == 0
