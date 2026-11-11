"""Tests for data/fetch_osm.py and the committed data/venues_madrid.json (task 1.2)."""

from __future__ import annotations

import json

import pytest

from agent.config import CENTER, DATA
from agent.preferences import haversine_m
from data import fetch_osm

VENUES = DATA / "venues_madrid.json"
KINDS = {"shop", "restaurant", "bar"}


@pytest.fixture(scope="module")
def document() -> dict:
    if not VENUES.exists():
        pytest.skip("data/venues_madrid.json has not been fetched yet")
    return json.loads(VENUES.read_text(encoding="utf-8"))


def test_header(document):
    assert document["radius_m"] == 4000
    assert document["center"] == list(CENTER)
    assert "OpenStreetMap" in document["source"]


def test_every_venue_is_valid(document):
    venues = document["venues"]
    ids = [v["venue_id"] for v in venues]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)
    for v in venues:
        assert v["name"].strip()
        assert v["kind"] in KINDS
        assert haversine_m(CENTER[0], CENTER[1], v["lat"], v["lon"]) <= 4100
        assert v["cuisine_assumed"] == (v["kind"] != "shop" and not v["cuisines"])


def test_at_least_20_shops(document):
    assert sum(v["kind"] == "shop" for v in document["venues"]) >= 20


# --- offline helpers ------------------------------------------------------------------


def _el(type_, id_, lat, lon, **tags):
    el = {"type": type_, "id": id_, "tags": tags}
    if type_ == "node":
        el.update(lat=lat, lon=lon)
    else:
        el["center"] = {"lat": lat, "lon": lon}
    return el


def test_element_to_venue_kinds_and_drops():
    shop_bar = fetch_osm.element_to_venue(
        _el(
            "way", 7, 40.4, -3.7, name="Vinos", shop="wine", amenity="bar", cuisine="Tapas; spanish"
        )
    )
    assert shop_bar["venue_id"] == "way/7"
    assert shop_bar["kind"] == "shop"  # shop wins
    assert shop_bar["cuisines"] == ["tapas", "spanish"]
    pub = fetch_osm.element_to_venue(_el("node", 1, 40.4, -3.7, name="Pub", amenity="pub"))
    assert pub["kind"] == "bar" and pub["cuisine_assumed"]
    assert fetch_osm.element_to_venue(_el("node", 2, 40.4, -3.7, amenity="bar")) is None
    assert fetch_osm.element_to_venue({"type": "way", "id": 3, "tags": {"name": "X"}}) is None


def test_deduplicate_keeps_richer_within_30m():
    a = {"venue_id": "node/1", "name": "Bar Uno", "lat": 40.4, "lon": -3.7, "_n_tags": 3}
    b = {"venue_id": "node/2", "name": "bar uno", "lat": 40.4001, "lon": -3.7, "_n_tags": 9}
    c = {"venue_id": "node/3", "name": "Bar Uno", "lat": 40.41, "lon": -3.7, "_n_tags": 1}
    kept = {v["venue_id"] for v in fetch_osm.deduplicate([a, b, c])}
    assert kept == {"node/2", "node/3"}


def test_tiles_cover_the_circle():
    tiles = fetch_osm.tile_bboxes(3)
    assert len(tiles) == 9
    south = min(t[0] for t in tiles)
    west = min(t[1] for t in tiles)
    north = max(t[2] for t in tiles)
    east = max(t[3] for t in tiles)
    lat, lon = CENTER
    assert haversine_m(lat, lon, south, lon) == pytest.approx(4000, rel=0.01)
    assert haversine_m(lat, lon, lat, west) == pytest.approx(4000, rel=0.01)
    assert south < lat < north and west < lon < east


def test_split_queries_and_merge():
    queries = fetch_osm.split_queries(2)
    assert len(queries) == 2 * 4
    assert all("around:4000" in q and "out center tags" in q for q in queries)
    merged = fetch_osm.merge_payloads(
        [
            {"version": 0.6, "elements": [{"type": "node", "id": 1}, {"type": "way", "id": 1}]},
            {"version": 0.6, "elements": [{"type": "node", "id": 1}, {"type": "node", "id": 2}]},
        ]
    )
    assert merged["version"] == 0.6
    assert [(e["type"], e["id"]) for e in merged["elements"]] == [
        ("node", 1),
        ("way", 1),
        ("node", 2),
    ]
