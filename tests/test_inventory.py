"""Tests for data/build_inventory.py and the committed data/inventory.csv (task 1.4)."""

from __future__ import annotations

import csv
import json
from collections import Counter

import pytest

from agent.config import DATA
from data import build_inventory

INVENTORY = DATA / "inventory.csv"
VENUES = DATA / "venues_madrid.json"


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    if not INVENTORY.exists():
        pytest.skip("data/inventory.csv has not been built yet")
    with open(INVENTORY, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sources() -> tuple[list[dict], list[dict]]:
    with open(DATA / "wines.csv", newline="", encoding="utf-8") as handle:
        wines = list(csv.DictReader(handle))
    return wines, json.loads(VENUES.read_text(encoding="utf-8"))["venues"]


def test_rebuild_gives_identical_bytes(rows, tmp_path, monkeypatch):
    out = tmp_path / "inventory.csv"
    monkeypatch.setattr(build_inventory, "OUT_PATH", out)
    build_inventory.main()
    assert out.read_bytes() == INVENTORY.read_bytes()


def test_ids_exist_and_no_duplicates(rows):
    wines, venues = _sources()
    wine_ids = {w["wine_id"] for w in wines}
    venue_ids = {v["venue_id"] for v in venues}
    pairs = [(r["venue_id"], r["wine_id"]) for r in rows]
    assert len(pairs) == len(set(pairs))
    assert {r["wine_id"] for r in rows} <= wine_ids
    assert {r["venue_id"] for r in rows} == venue_ids


def test_every_venue_has_at_least_10_wines(rows):
    assert min(Counter(r["venue_id"] for r in rows).values()) >= 10


def test_prices_apply_the_markup(rows):
    wines, venues = _sources()
    retail = {w["wine_id"]: float(w["retail_price"]) for w in wines}
    kind = {v["venue_id"]: v["kind"] for v in venues}
    markup = {"shop": 1.0, "bar": 2.0, "restaurant": 2.5}
    for r in rows:
        expected = round(retail[r["wine_id"]] * markup[kind[r["venue_id"]]], 2)
        assert float(r["price_eur"]) == pytest.approx(expected)


# --- offline ----------------------------------------------------------------------------


def test_build_rows_is_deterministic_and_in_range():
    wines = [
        {"wine_id": f"w{i:04d}", "num_reviews": str(i * 10), "retail_price": "10.00"}
        for i in range(1, 201)
    ]
    venues = [
        {"venue_id": "node/1", "kind": "shop", "osm_shop": "wine"},
        {"venue_id": "node/2", "kind": "shop", "osm_shop": "alcohol"},
        {"venue_id": "way/3", "kind": "restaurant"},
        {"venue_id": "node/4", "kind": "bar"},
    ]
    first = build_inventory.build_rows(wines, venues)
    assert first == build_inventory.build_rows(wines, list(reversed(venues)))
    counts = Counter(r[0] for r in first)
    for venue in venues:
        lo, hi = build_inventory.WINE_COUNT[build_inventory.category(venue)]
        assert lo <= counts[venue["venue_id"]] <= hi
    prices = {r[0]: r[2] for r in first}
    assert prices["way/3"] == "25.00" and prices["node/4"] == "20.00"
