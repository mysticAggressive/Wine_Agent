"""Tests for agent/data_loader.py (task 2.3)."""

from __future__ import annotations

import pytest

from agent.config import DATA
from agent.data_loader import (
    AgentData,
    Inventory,
    load_all,
    load_inventory,
    load_pairing_rules,
    load_venues,
    load_wines,
)
from agent.models import Dish, OpenState, Style, VenueKind
from agent.opening import open_state
from tests.conftest import MONDAY_NOON, WEDNESDAY_NOON


def _real(name: str):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"data/{name} has not been built yet")
    return path


# --- fixtures ---------------------------------------------------------------------


def test_fixture_wines(wines):
    assert len(wines) == 5
    assert list(wines) == ["w0001", "w0002", "w0003", "w0004", "w0005"]
    styles = [w.style for w in wines.values()]
    assert styles.count(Style.RED) == 2
    assert styles.count(Style.WHITE) == 1
    assert styles.count(Style.SPARKLING) == 1
    assert styles.count(Style.FORTIFIED) == 1
    assert all(8 <= w.retail_price <= 60 for w in wines.values())
    first, cava = wines["w0001"], wines["w0004"]
    assert first.year == 2015 and isinstance(first.year, int)
    assert first.wine_type == "Ribera Del Duero Red"
    assert isinstance(first.body, int) and isinstance(first.rating, float)
    assert first.imputed is False
    assert cava.year is None and cava.imputed is True


def test_fixture_venues(venues):
    assert len(venues) == 4
    kinds = [v.kind for v in venues.values()]
    assert kinds.count(VenueKind.SHOP) == 2
    assert kinds.count(VenueKind.RESTAURANT) == 1
    assert kinds.count(VenueKind.BAR) == 1
    restaurant = venues["way/3"]
    assert restaurant.cuisines == ("spanish",)
    assert restaurant.cuisine_assumed is False
    assert venues["node/4"].cuisine_assumed is True
    assert venues["node/2"].opening_hours == ""
    assert not hasattr(venues["node/1"], "osm_shop")  # unknown keys are ignored


def test_fixture_opening_hours(venues):
    assert open_state(venues["node/1"].opening_hours, MONDAY_NOON) is OpenState.CLOSED
    assert open_state(venues["node/1"].opening_hours, WEDNESDAY_NOON) is OpenState.OPEN
    assert open_state(venues["node/2"].opening_hours, WEDNESDAY_NOON) is OpenState.UNKNOWN


def test_fixture_inventory(inventory, wines, venues):
    assert len(inventory) == 11
    assert inventory.wines_at("node/2") == frozenset({"w0002", "w0004"})
    assert inventory.venues_with("w0002") == frozenset(venues)
    assert inventory.venues_with("w0005") == frozenset({"node/1"})
    assert inventory.price("way/3", "w0002") == 30.0
    assert inventory.stocks("node/1", "w0001")
    assert not inventory.stocks("node/2", "w0001")
    assert inventory.venue_ids() == frozenset(venues)
    assert isinstance(inventory.wines_at("node/1"), frozenset)
    # at least one, but not every, combination is valid
    pairs = [(v, w) for v in venues for w in wines]
    stocked = [inventory.stocks(v, w) for v, w in pairs]
    assert any(stocked) and not all(stocked)


def test_inventory_unknown_ids(inventory):
    assert inventory.wines_at("node/999") == frozenset()
    assert inventory.venues_with("w9999") == frozenset()
    with pytest.raises(KeyError):
        inventory.price("node/2", "w0001")  # venue exists, wine not stocked
    with pytest.raises(KeyError):
        inventory.price("node/999", "w0001")


def test_inventory_from_rows():
    inventory = Inventory([("node/1", "w0001", 10), ("node/1", "w0002", 12.5)])
    assert inventory.price("node/1", "w0001") == 10.0
    assert len(inventory) == 2
    assert len(Inventory()) == 0


def test_fixture_pairing_rules(pairing_rules):
    assert list(pairing_rules.dishes) == ["grilled_red_meat", "shellfish", "chocolate_dessert"]
    rule = pairing_rules.dishes["grilled_red_meat"]
    assert rule.label == "Grilled red meat"
    assert rule.styles == {"red": 1.0, "fortified": 0.2}
    assert rule.styles.get(Style.RED, 0.0) == 1.0  # Style members work as keys
    assert rule.styles.get(Style.WHITE, 0.0) == 0.0
    assert rule.body == (4, 5)
    assert rule.acidity == (2, 3)
    assert "Rioja Red" in rule.type_bonus
    assert rule.why.startswith("Tannin")
    assert rule.dish == Dish("grilled_red_meat", "Grilled red meat")
    assert pairing_rules.cuisine_to_dishes["spanish"] == ("grilled_red_meat", "shellfish")
    assert pairing_rules.dishes_for_cuisine("seafood") == ("shellfish",)
    assert pairing_rules.dishes_for_cuisine("martian") == ("grilled_red_meat",)


def test_duplicate_dish_id_raises(tmp_path):
    path = tmp_path / "rules.json"
    dish = '{"dish_id": "a", "label": "A", "styles": {"red": 1}, "body": [2, 5], "acidity": [1, 3]}'
    path.write_text(f'{{"dishes": [{dish}, {dish}], "cuisine_to_dishes": {{}}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_pairing_rules(path)


def test_load_all_on_fixture_dir(data_dir):
    data = load_all(data_dir)
    assert isinstance(data, AgentData)
    assert len(data.wines) == 5
    assert len(data.venues) == 4
    assert len(data.inventory) == 11
    assert len(data.pairing_rules.dishes) == 3


def test_agent_data_fixture(agent_data):
    for venue_id in agent_data.inventory.venue_ids():
        assert venue_id in agent_data.venues
        assert agent_data.inventory.wines_at(venue_id) <= set(agent_data.wines)


# --- real files -------------------------------------------------------------------


def test_real_wines():
    wines = load_wines(_real("wines.csv"))
    assert len(wines) == 2024
    assert all(key == wine.wine_id for key, wine in wines.items())
    assert any(wine.year is None for wine in wines.values())
    assert all(isinstance(wine.style, Style) for wine in wines.values())
    assert load_wines() == wines  # default path


def test_real_venues():
    venues = load_venues(_real("venues_madrid.json"))
    assert len(venues) > 0
    assert all(key == venue.venue_id for key, venue in venues.items())
    assert all(isinstance(venue.kind, VenueKind) for venue in venues.values())
    assert all(isinstance(venue.cuisines, tuple) for venue in venues.values())


def test_real_inventory():
    inventory = load_inventory(_real("inventory.csv"))
    assert len(inventory) > 0
    venue_id = sorted(inventory.venue_ids())[0]
    wine_id = sorted(inventory.wines_at(venue_id))[0]
    assert inventory.price(venue_id, wine_id) > 0
    assert venue_id in inventory.venues_with(wine_id)


def test_real_pairing_rules():
    rules = load_pairing_rules(_real("pairing_rules.json"))
    assert len(rules.dishes) > 0
    for dish_ids in rules.cuisine_to_dishes.values():
        assert set(dish_ids) <= set(rules.dishes)


def test_real_load_all():
    for name in ("wines.csv", "venues_madrid.json", "inventory.csv", "pairing_rules.json"):
        _real(name)
    data = load_all()
    assert len(data.wines) == 2024
    assert data.inventory.venue_ids() <= set(data.venues)
