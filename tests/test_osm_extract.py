"""Tests for data/osm_extract.py: reading an OSM extract as Overpass would answer."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("osmium")

from data import osm_extract  # noqa: E402

CENTER = (40.4168, -3.7038)

# ~111 m per 0.001 degrees of latitude.
OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
 <node id="1" version="1" lat="40.4170" lon="-3.7040">
  <tag k="shop" v="wine"/><tag k="name" v="Vinoteca"/>
 </node>
 <node id="2" version="1" lat="40.4900" lon="-3.7040">
  <tag k="amenity" v="bar"/><tag k="name" v="Too far"/>
 </node>
 <node id="3" version="1" lat="40.4160" lon="-3.7030">
  <tag k="amenity" v="cafe"/><tag k="name" v="Not wanted"/>
 </node>
 <node id="10" version="1" lat="40.4100" lon="-3.7000"/>
 <node id="11" version="1" lat="40.4120" lon="-3.7000"/>
 <node id="12" version="1" lat="40.4120" lon="-3.6980"><tag k="highway" v="crossing"/></node>
 <node id="13" version="1" lat="40.4100" lon="-3.6980"/>
 <node id="20" version="1" lat="40.5000" lon="-3.7000"/>
 <node id="21" version="1" lat="40.5010" lon="-3.7000"/>
 <way id="100" version="1">
  <nd ref="10"/><nd ref="11"/><nd ref="12"/><nd ref="13"/><nd ref="10"/>
  <tag k="amenity" v="restaurant"/><tag k="name" v="Casa"/>
 </way>
 <way id="200" version="1">
  <nd ref="10"/><nd ref="11"/><tag k="highway" v="footway"/>
 </way>
 <way id="201" version="1">
  <nd ref="11"/><nd ref="12"/><tag k="highway" v="motorway"/>
 </way>
 <way id="202" version="1">
  <nd ref="12"/><nd ref="13"/><tag k="highway" v="residential"/><tag k="foot" v="no"/>
 </way>
 <way id="203" version="1">
  <nd ref="20"/><nd ref="21"/><tag k="highway" v="footway"/>
 </way>
 <relation id="300" version="1">
  <member type="way" ref="100" role="outer"/>
  <tag k="type" v="multipolygon"/><tag k="amenity" v="pub"/><tag k="name" v="Rel pub"/>
 </relation>
</osm>
"""


@pytest.fixture
def osm_file(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.osm"
    path.write_text(OSM_XML, encoding="utf-8")
    return path


def test_venues_payload_matches_overpass_semantics(osm_file):
    payload = osm_extract.venues_payload(osm_file, CENTER, 4000)
    by_key = {(e["type"], e["id"]): e for e in payload["elements"]}
    assert set(by_key) == {("node", 1), ("way", 100), ("relation", 300)}
    assert by_key[("node", 1)]["lat"] == pytest.approx(40.417)
    assert by_key[("way", 100)]["center"] == {"lat": 40.411, "lon": -3.699}
    assert by_key[("relation", 300)]["center"] == {"lat": 40.411, "lon": -3.699}
    assert by_key[("way", 100)]["tags"]["name"] == "Casa"


def test_parse_and_match_filter():
    clauses = osm_extract.parse_way_filter('["highway"]["foot"!~"no"]["highway"~"^foot"]')
    assert osm_extract.tags_match({"highway": "footway"}, clauses)
    assert not osm_extract.tags_match({"highway": "footway", "foot": "no"}, clauses)
    assert not osm_extract.tags_match({"highway": "residential"}, clauses)
    assert not osm_extract.tags_match({"foot": "yes"}, clauses)
    with pytest.raises(ValueError):
        osm_extract.parse_way_filter('["highway"](around:10)')


def test_network_payload_applies_filter_and_bbox(osm_file):
    walk = '["highway"]["highway"!~"motorway"]["foot"!~"no"]'
    payload = osm_extract.network_payload(osm_file, (40.40, -3.71, 40.42, -3.69), walk)
    ways = [e for e in payload["elements"] if e["type"] == "way"]
    nodes = {e["id"]: e for e in payload["elements"] if e["type"] == "node"}
    assert [w["id"] for w in ways] == [200]  # 201 motorway, 202 foot=no, 203 outside
    assert ways[0]["nodes"] == [10, 11]
    assert set(nodes) == {10, 11}
    assert nodes[10]["lat"] == pytest.approx(40.41)
