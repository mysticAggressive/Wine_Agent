"""Checks on the committed ``data/pairing_rules.json`` (DEVELOPMENT_PLAN.md task 3.1)."""

from __future__ import annotations

import json
from collections import defaultdict

import pytest

from agent.config import DATA, MIN_PAIRING_SCORE
from agent.data_loader import PairingRules, load_pairing_rules, load_wines
from agent.models import Style, Wine
from agent.pairing import pairing_score

RULES_PATH = DATA / "pairing_rules.json"
MIN_WINES_PER_DISH = 20


@pytest.fixture(scope="module")
def real_rules() -> PairingRules:
    """The committed pairing rules."""
    return load_pairing_rules(RULES_PATH)


@pytest.fixture(scope="module")
def real_wines() -> dict[str, Wine]:
    """The committed cleaned wines."""
    return load_wines(DATA / "wines.csv")


@pytest.fixture(scope="module")
def payload() -> dict:
    """The raw JSON payload."""
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def _mean_score_by_type(wines: dict[str, Wine], dish_id: str, rules: PairingRules):
    scores: dict[str, list[float]] = defaultdict(list)
    for wine in wines.values():
        scores[wine.wine_type].append(pairing_score(wine, dish_id, rules))
    return {wine_type: sum(vals) / len(vals) for wine_type, vals in scores.items()}


def test_json_loads_with_expected_structure(payload: dict, real_rules: PairingRules) -> None:
    assert set(payload) == {"dishes", "cuisine_to_dishes"}
    assert 15 <= len(real_rules.dishes) <= 22
    for item in payload["dishes"]:
        assert set(item) == {
            "dish_id",
            "label",
            "styles",
            "body",
            "acidity",
            "type_bonus",
            "why",
        }


def test_dish_ids_are_unique(payload: dict) -> None:
    ids = [item["dish_id"] for item in payload["dishes"]]
    assert len(ids) == len(set(ids))


def test_rule_values_are_well_formed(real_rules: PairingRules) -> None:
    valid_styles = {style.value for style in Style}
    for rule in real_rules.dishes.values():
        assert rule.label
        assert rule.why.endswith("."), rule.dish_id
        assert rule.styles, rule.dish_id
        assert set(rule.styles) <= valid_styles, rule.dish_id
        assert all(0.0 < weight <= 1.0 for weight in rule.styles.values()), rule.dish_id
        assert 2 <= rule.body[0] <= rule.body[1] <= 5, rule.dish_id
        assert 1 <= rule.acidity[0] <= rule.acidity[1] <= 3, rule.dish_id


def test_type_bonus_strings_exist_in_wines(
    real_rules: PairingRules, real_wines: dict[str, Wine]
) -> None:
    wine_types = {wine.wine_type for wine in real_wines.values()} - {""}
    for rule in real_rules.dishes.values():
        assert set(rule.type_bonus) <= wine_types, rule.dish_id


def test_cuisine_map_references_existing_dishes(real_rules: PairingRules) -> None:
    for cuisine, dish_ids in real_rules.cuisine_to_dishes.items():
        assert dish_ids, cuisine
        assert len(dish_ids) == len(set(dish_ids)), cuisine
        assert set(dish_ids) <= set(real_rules.dishes), cuisine


def test_cuisine_map_has_required_keys(real_rules: PairingRules) -> None:
    mapping = real_rules.cuisine_to_dishes
    assert "spanish" in mapping
    assert "_default" in mapping
    assert mapping["regional"] == mapping["spanish"]
    assert len(mapping) >= 26  # at least 25 cuisines + _default
    assert all(key == key.strip().lower() for key in mapping)
    assert real_rules.dishes_for_cuisine("no_such_cuisine") == mapping["_default"]


def test_every_dish_is_served_by_some_cuisine(real_rules: PairingRules) -> None:
    served = {d for dish_ids in real_rules.cuisine_to_dishes.values() for d in dish_ids}
    assert served == set(real_rules.dishes)


def test_every_dish_has_enough_matching_wines(
    real_rules: PairingRules, real_wines: dict[str, Wine]
) -> None:
    for dish_id in real_rules.dishes:
        matching = sum(
            pairing_score(wine, dish_id, real_rules) >= MIN_PAIRING_SCORE
            for wine in real_wines.values()
        )
        assert matching >= MIN_WINES_PER_DISH, f"{dish_id}: only {matching} wines pair"


def test_every_wine_type_pairs_with_some_dish(
    real_rules: PairingRules, real_wines: dict[str, Wine]
) -> None:
    for wine_type in {wine.wine_type for wine in real_wines.values()}:
        best = max(
            _mean_score_by_type(real_wines, dish_id, real_rules)[wine_type]
            for dish_id in real_rules.dishes
        )
        assert best >= MIN_PAIRING_SCORE, wine_type


@pytest.mark.parametrize(
    ("dish_id", "expected_top"),
    [
        ("shellfish", {"Albarino", "Verdejo", "Cava"}),
        ("chocolate_dessert", {"Pedro Ximenez"}),
        ("blue_cheese", {"Pedro Ximenez"}),
        ("jamon_iberico", {"Sherry"}),
        ("roast_lamb", {"Ribera Del Duero Red", "Toro Red"}),
        ("grilled_red_meat", {"Ribera Del Duero Red", "Toro Red", "Rioja Red"}),
    ],
)
def test_classic_pairings_come_out_on_top(
    dish_id: str, expected_top: set[str], real_rules: PairingRules, real_wines: dict[str, Wine]
) -> None:
    means = _mean_score_by_type(real_wines, dish_id, real_rules)
    best = max(means.values())
    top = {wine_type for wine_type, mean in means.items() if mean == pytest.approx(best)}
    assert top == expected_top


def test_dishes_prefer_different_wines(
    real_rules: PairingRules, real_wines: dict[str, Wine]
) -> None:
    def passing_styles(dish_id: str) -> set[Style]:
        return {
            wine.style
            for wine in real_wines.values()
            if pairing_score(wine, dish_id, real_rules) >= MIN_PAIRING_SCORE
        }

    assert Style.RED not in passing_styles("shellfish")
    assert Style.RED not in passing_styles("white_fish")
    assert passing_styles("chocolate_dessert") == {Style.FORTIFIED}
    assert passing_styles("roast_lamb") == {Style.RED}
    assert {Style.RED, Style.FORTIFIED} <= passing_styles("jamon_iberico")
    assert Style.WHITE not in passing_styles("grilled_red_meat")
