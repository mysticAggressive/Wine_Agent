"""Tests for agent/models.py (task 2.1)."""

from __future__ import annotations

import dataclasses

import pytest

from agent.models import (
    Candidate,
    Diagnostics,
    Dish,
    OpenState,
    RecommendationResult,
    Style,
    Venue,
    VenueKind,
    Wine,
)


def _wine() -> Wine:
    return Wine(
        wine_id="w0001",
        winery="Bodegas Prueba",
        name="Gran Reserva",
        year=None,
        rating=4.6,
        num_reviews=900,
        region="Ribera del Duero",
        retail_price=60.0,
        wine_type="Ribera Del Duero Red",
        style=Style.RED,
        body=5,
        acidity=3,
        imputed=False,
    )


def _candidate() -> Candidate:
    return Candidate(
        wine_id="w0001",
        venue_id="node/1",
        dish_id="grilled_red_meat",
        price_at_venue=60.0,
        distance_m=120.5,
        features={"taste": 0.9, "quality": 0.5, "price_fit": 0.1, "pairing": 1.0, "distance": 0.8},
        score=0.66,
        reasons=("Full-bodied red", "Within budget"),
    )


@pytest.mark.parametrize("enum_cls", [Style, VenueKind, OpenState])
def test_enums_round_trip(enum_cls):
    for member in enum_cls:
        assert enum_cls(member.value) is member
        assert member == member.value


def test_style_values_have_no_rose():
    assert {s.value for s in Style} == {"red", "white", "sparkling", "fortified", "unknown"}
    with pytest.raises(ValueError):
        Style("rose")


def test_venue_kind_and_open_state_values():
    assert {k.value for k in VenueKind} == {"shop", "restaurant", "bar"}
    assert {s.value for s in OpenState} == {"open", "closed", "unknown"}


def test_wine_venue_dish_construct_and_are_frozen():
    wine = _wine()
    venue = Venue(
        venue_id="node/1",
        name="Vinoteca Sol",
        kind=VenueKind.SHOP,
        lat=40.417,
        lon=-3.704,
        address="",
        opening_hours="",
        cuisines=(),
        cuisine_assumed=False,
        website="",
    )
    dish = Dish(dish_id="shellfish", label="Shellfish")
    assert wine.year is None and wine.style is Style.RED
    assert venue.kind is VenueKind.SHOP
    assert dish.label == "Shellfish"
    assert len({wine, venue, dish}) == 3  # hashable
    with pytest.raises(dataclasses.FrozenInstanceError):
        wine.body = 2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        venue.name = "x"  # type: ignore[misc]


def test_candidate_as_row_is_flat():
    row = _candidate().as_row()
    assert row["wine_id"] == "w0001"
    assert row["venue_id"] == "node/1"
    assert row["dish_id"] == "grilled_red_meat"
    assert row["price_at_venue"] == 60.0
    assert row["distance_m"] == 120.5
    assert row["score"] == 0.66
    assert row["pairing"] == 1.0 and row["taste"] == 0.9
    assert row["reasons"] == "Full-bodied red; Within budget"
    assert all(not isinstance(v, (dict, list, tuple)) for v in row.values())


def test_diagnostics_and_result_construct():
    diagnostics = Diagnostics(
        initial_domain_sizes={"wine": 5, "venue": 4, "dish": 3},
        pruned_by_constraint={"style": 2},
        domain_sizes_after_node_consistency={"wine": 3, "venue": 4, "dish": 3},
        domain_sizes_after_ac3={"wine": 2, "venue": 3, "dish": 2},
        ac3_revisions=7,
        assignments=10,
        backtracks=1,
        forward_check_prunes=4,
        solutions_found=1,
        variable_order=["dish", "wine", "venue"],
        elapsed_ms=1.5,
        relaxation_hints=[],
    )
    result = RecommendationResult(candidates=[_candidate()], diagnostics=diagnostics)
    assert result.candidates[0].score == 0.66
    diagnostics.backtracks += 1  # mutable on purpose
    assert result.diagnostics.backtracks == 2
