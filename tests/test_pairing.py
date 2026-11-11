"""Tests for ``agent/pairing.py`` on the miniature world (DEVELOPMENT_PLAN.md task 3.2)."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

import agent.pairing as pairing
from agent.data_loader import PairingRules
from agent.models import Style, Venue, VenueKind, Wine
from agent.pairing import (
    clear_pairing_cache,
    dishes_for_venue,
    pairing_reason,
    pairing_score,
)


def test_full_red_beats_light_white_for_red_meat(
    wines: dict[str, Wine], pairing_rules: PairingRules
) -> None:
    red = pairing_score(wines["w0001"], "grilled_red_meat", pairing_rules)
    white = pairing_score(wines["w0003"], "grilled_red_meat", pairing_rules)
    assert red > white
    assert red == pytest.approx(1.0)


def test_style_absent_from_rule_scores_exactly_zero(
    wines: dict[str, Wine], pairing_rules: PairingRules
) -> None:
    assert pairing_score(wines["w0003"], "grilled_red_meat", pairing_rules) == 0.0
    assert pairing_score(wines["w0004"], "chocolate_dessert", pairing_rules) == 0.0
    assert pairing_score(wines["w0001"], "shellfish", pairing_rules) == 0.0


def test_score_follows_the_formula(wines: dict[str, Wine], pairing_rules: PairingRules) -> None:
    # PX (fortified, body 5, acidity 1) with red meat: style 0.2, body in range,
    # acidity one step below [2, 3] -> 0.2 * (0.5 + 0.3 * 1 + 0.2 * 0.5), no bonus.
    assert pairing_score(wines["w0005"], "grilled_red_meat", pairing_rules) == pytest.approx(0.18)
    # Cava (sparkling 0.9, body 2, acidity 3) with shellfish: everything in range.
    assert pairing_score(wines["w0004"], "shellfish", pairing_rules) == pytest.approx(0.9)
    # Ribera red (style 0.3, body 5, acidity 3 = one step above [1, 2]) with chocolate.
    assert pairing_score(wines["w0001"], "chocolate_dessert", pairing_rules) == pytest.approx(0.27)
    # Rioja red (style 0.3, body 4 = one step below [5, 5], acidity one step out).
    assert pairing_score(wines["w0002"], "chocolate_dessert", pairing_rules) == pytest.approx(0.225)


def test_type_bonus_adds_and_caps_at_one(
    wines: dict[str, Wine], pairing_rules: PairingRules
) -> None:
    albarino = wines["w0003"]
    plain_white = replace(albarino, wine_id="x1", wine_type="Verdejo")
    assert pairing_score(plain_white, "shellfish", pairing_rules) == pytest.approx(1.0)
    assert pairing_score(albarino, "shellfish", pairing_rules) == 1.0  # capped
    px = wines["w0005"]
    sherry = replace(px, wine_id="x2", wine_type="Sherry", body=3)
    with_bonus = pairing_score(replace(sherry, wine_id="x3", wine_type="Pedro Ximenez"),
                               "chocolate_dessert", pairing_rules)  # fmt: skip
    without = pairing_score(sherry, "chocolate_dessert", pairing_rules)
    assert without == pytest.approx(0.7)  # 1.0 * (0.5 + 0.3 * 0 + 0.2 * 1)
    assert with_bonus == pytest.approx(0.85)


def test_missing_wine_type_never_gets_bonus(pairing_rules: PairingRules) -> None:
    rule = pairing_rules.dishes["shellfish"]
    rules = PairingRules(
        dishes={"shellfish": replace(rule, type_bonus=("", "Albarino"), styles={"unknown": 0.5})},
        cuisine_to_dishes={},
    )
    wine = Wine("x9", "W", "N", None, 4.2, 10, "R", 10.0, "", Style.UNKNOWN, 3, 3, True)
    assert pairing_score(wine, "shellfish", rules) == pytest.approx(0.5)


def test_score_always_within_unit_interval(
    wines: dict[str, Wine], pairing_rules: PairingRules
) -> None:
    variants = [
        replace(wine, wine_id=f"{wine.wine_id}-{body}-{acidity}", body=body, acidity=acidity)
        for wine in wines.values()
        for body in range(1, 7)
        for acidity in range(1, 5)
    ]
    for wine in [*wines.values(), *variants]:
        for dish_id in pairing_rules.dishes:
            assert 0.0 <= pairing_score(wine, dish_id, pairing_rules) <= 1.0


def test_unknown_dish_raises_key_error(wines: dict[str, Wine], pairing_rules: PairingRules) -> None:
    with pytest.raises(KeyError):
        pairing_score(wines["w0001"], "no_such_dish", pairing_rules)
    with pytest.raises(KeyError):
        pairing_reason(wines["w0001"], "no_such_dish", pairing_rules)


def test_scores_are_cached_per_wine_and_dish(
    wines: dict[str, Wine], pairing_rules: PairingRules, monkeypatch: pytest.MonkeyPatch
) -> None:
    clear_pairing_cache()
    calls: list[tuple[str, str]] = []
    original = pairing._score

    def counting(wine, rule):
        calls.append((wine.wine_id, rule.dish_id))
        return original(wine, rule)

    monkeypatch.setattr(pairing, "_score", counting)
    first = pairing_score(wines["w0001"], "grilled_red_meat", pairing_rules)
    second = pairing_score(wines["w0001"], "grilled_red_meat", pairing_rules)
    pairing_score(wines["w0001"], "shellfish", pairing_rules)
    assert first == second
    assert calls == [("w0001", "grilled_red_meat"), ("w0001", "shellfish")]


def test_cache_does_not_mix_up_rules_or_wines(
    wines: dict[str, Wine], pairing_rules: PairingRules
) -> None:
    red = wines["w0001"]
    assert pairing_score(red, "grilled_red_meat", pairing_rules) == pytest.approx(1.0)
    # Same wine id, different wine.
    impostor = replace(wines["w0003"], wine_id="w0001")
    assert pairing_score(impostor, "grilled_red_meat", pairing_rules) == 0.0
    # Same wine and dish id, different rules object.
    rule = pairing_rules.dishes["grilled_red_meat"]
    other = PairingRules(
        dishes={"grilled_red_meat": replace(rule, styles={"white": 1.0})},
        cuisine_to_dishes=dict(pairing_rules.cuisine_to_dishes),
    )
    assert pairing_score(red, "grilled_red_meat", other) == 0.0
    assert pairing_score(red, "grilled_red_meat", pairing_rules) == pytest.approx(1.0)


def test_pairing_reason_sentence(wines: dict[str, Wine], pairing_rules: PairingRules) -> None:
    assert pairing_reason(wines["w0001"], "grilled_red_meat", pairing_rules) == (
        "Very full-bodied Ribera del Duero red — tannin and body stand up to charred, fatty meat."
    )
    assert pairing_reason(wines["w0002"], "grilled_red_meat", pairing_rules) == (
        "Full-bodied Rioja red — tannin and body stand up to charred, fatty meat."
    )
    assert pairing_reason(wines["w0003"], "shellfish", pairing_rules) == (
        "Light-bodied Albarino — high acidity and a light body lift briny shellfish."
    )


def test_pairing_reason_for_a_mismatch_and_unknown_type(
    wines: dict[str, Wine], pairing_rules: PairingRules
) -> None:
    mismatch = pairing_reason(wines["w0003"], "grilled_red_meat", pairing_rules)
    assert mismatch == "Light-bodied Albarino — not a recommended style for grilled red meat."
    untyped = replace(wines["w0002"], wine_id="x4", wine_type="", style=Style.RED)
    assert pairing_reason(untyped, "grilled_red_meat", pairing_rules).startswith(
        "Full-bodied red — "
    )
    generic = replace(wines["w0002"], wine_id="x5", wine_type="Red")
    assert pairing_reason(generic, "grilled_red_meat", pairing_rules).startswith(
        "Full-bodied red — "
    )


def test_shop_serves_every_dish(venues: dict[str, Venue], pairing_rules: PairingRules) -> None:
    for venue in venues.values():
        if venue.kind == VenueKind.SHOP:
            assert dishes_for_venue(venue, pairing_rules) == frozenset(pairing_rules.dishes)


def test_restaurant_serves_its_cuisine_dishes(
    venues: dict[str, Venue], pairing_rules: PairingRules
) -> None:
    served = dishes_for_venue(venues["way/3"], pairing_rules)
    assert served == frozenset({"grilled_red_meat", "shellfish"})
    assert "chocolate_dessert" not in served


def test_venue_without_cuisine_gets_spanish_set(
    venues: dict[str, Venue], pairing_rules: PairingRules
) -> None:
    bar = venues["node/4"]
    assert bar.cuisines == ()
    assert dishes_for_venue(bar, pairing_rules) == frozenset({"grilled_red_meat", "shellfish"})


def test_cuisine_union_and_default_fallback(
    venues: dict[str, Venue], pairing_rules: PairingRules
) -> None:
    restaurant = venues["way/3"]
    seafood = replace(restaurant, cuisines=("seafood",))
    assert dishes_for_venue(seafood, pairing_rules) == frozenset({"shellfish"})
    unlisted = replace(restaurant, cuisines=("klingon",))
    assert dishes_for_venue(unlisted, pairing_rules) == frozenset({"grilled_red_meat"})
    both = replace(restaurant, cuisines=("seafood", "klingon"))
    assert dishes_for_venue(both, pairing_rules) == frozenset({"grilled_red_meat", "shellfish"})
    shouting = replace(restaurant, cuisines=(" Seafood ",))
    assert dishes_for_venue(shouting, pairing_rules) == frozenset({"shellfish"})


def test_dishes_for_venue_ignores_unknown_dish_ids(
    venues: dict[str, Venue], pairing_rules: PairingRules
) -> None:
    rules = PairingRules(
        dishes=pairing_rules.dishes,
        cuisine_to_dishes={"spanish": ("shellfish", "ghost_dish")},
    )
    assert dishes_for_venue(venues["way/3"], rules) == frozenset({"shellfish"})


def test_module_does_not_import_ui_libraries() -> None:
    tree = ast.parse(Path(pairing.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"streamlit", "folium"}
