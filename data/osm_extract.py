"""Offline route to the OSM data: a regional ``.osm.pbf`` extract instead of Overpass.

When no Overpass server answers (task 1.2 / 1.3 risk in DEVELOPMENT_PLAN.md 17), the
same OpenStreetMap data can be read from a daily regional extract. This module turns
the extract into *the same JSON Overpass would have returned*, so the rest of the
pipeline does not change:

* :func:`venues_payload` mirrors the task 1.2 query (``nwr[shop~wine|alcohol]`` and
  ``nwr[amenity~restaurant|bar|pub]`` around the centre, ``out center tags``) and is
  fed to ``fetch_osm.build_document``.
* :func:`network_payload` mirrors OSMnx's walk-network query (``way[<walk filter>]
  (poly:...); >; out;``) and is fed to OSMnx's own graph builder by
  ``fetch_graph.py --from-pbf``.

Download an extract with ``python data/osm_extract.py`` (checksum-verified, saved
under ``./cache``, git-ignored).

Data © OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import osmium
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import CENTER, RADIUS_M  # noqa: E402

EXTRACT_URL = "https://download.openstreetmap.fr/extracts/europe/spain/comunidad_de_madrid.osm.pbf"
EXTRACT_PATH = ROOT / "cache" / "osm_extract" / "comunidad_de_madrid.osm.pbf"
USER_AGENT = "madrid-wine-agent/0.1 (course project)"

SHOP_RE = re.compile(r"^(wine|alcohol)$")
AMENITY_RE = re.compile(r"^(restaurant|bar|pub)$")

Bbox = tuple[float, float, float, float]  # (south, west, north, east)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    r = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def osm_timestamp(path: Path) -> str:
    """Return the extract's replication timestamp (``osmosis_replication_timestamp``)."""
    header = osmium.io.Reader(str(path), osmium.osm.osm_entity_bits.NOTHING).header()
    return header.get("osmosis_replication_timestamp", "")


# --------------------------------------------------------------------------- #
# Venues (task 1.2)
# --------------------------------------------------------------------------- #


def _wanted(tags: Any) -> bool:
    return bool(SHOP_RE.search(tags.get("shop", ""))) or bool(
        AMENITY_RE.search(tags.get("amenity", ""))
    )


def _bbox_center(points: Iterable[tuple[float, float]]) -> dict[str, float] | None:
    """Overpass ``out center``: the centre of the bounding box of the points."""
    pts = list(points)
    if not pts:
        return None
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    return {
        "lat": round((min(lats) + max(lats)) / 2, 7),
        "lon": round((min(lons) + max(lons)) / 2, 7),
    }


def venues_payload(
    path: Path, center: tuple[float, float] = CENTER, radius_m: float = RADIUS_M
) -> dict[str, Any]:
    """Return an Overpass-shaped payload of the task 1.2 query, read from ``path``.

    ``around`` semantics: a node matches when it is within ``radius_m`` of ``center``;
    a way or relation when any of its nodes is. Ways and relations carry ``center``
    (bounding-box centre), nodes ``lat``/``lon``, as with ``out center tags``.
    """
    lat0, lon0 = center

    def near(lat: float, lon: float) -> bool:
        return haversine_m(lat0, lon0, lat, lon) <= radius_m

    elements: list[dict[str, Any]] = []
    way_points: dict[int, list[tuple[float, float]]] = {}
    relations: list[tuple[int, dict[str, str], list[tuple[str, int]]]] = []

    # Pass 1: matching nodes and ways (with node locations), relations by member ref.
    processor = osmium.FileProcessor(str(path)).with_locations()
    for obj in processor:
        if obj.is_node():
            if _wanted(obj.tags) and obj.location.valid():
                lat, lon = obj.location.lat, obj.location.lon
                if near(lat, lon):
                    elements.append(
                        {
                            "type": "node",
                            "id": obj.id,
                            "lat": round(lat, 7),
                            "lon": round(lon, 7),
                            "tags": dict(obj.tags),
                        }
                    )
        elif obj.is_way():
            if _wanted(obj.tags):
                pts = [(n.lat, n.lon) for n in obj.nodes if n.location.valid()]
                if any(near(*p) for p in pts):
                    elements.append(
                        {
                            "type": "way",
                            "id": obj.id,
                            "center": _bbox_center(pts),
                            "tags": dict(obj.tags),
                        }
                    )
        elif obj.is_relation() and _wanted(obj.tags):
            members = [(m.type, m.ref) for m in obj.members]
            relations.append((obj.id, dict(obj.tags), members))

    if relations:
        node_ids = {ref for _, _, ms in relations for t, ref in ms if t == "n"}
        way_ids = {ref for _, _, ms in relations for t, ref in ms if t == "w"}
        node_points: dict[int, tuple[float, float]] = {}
        for obj in osmium.FileProcessor(str(path)).with_locations():
            if obj.is_node() and obj.id in node_ids and obj.location.valid():
                node_points[obj.id] = (obj.location.lat, obj.location.lon)
            elif obj.is_way() and obj.id in way_ids:
                way_points[obj.id] = [(n.lat, n.lon) for n in obj.nodes if n.location.valid()]
        for rel_id, tags, members in relations:
            pts: list[tuple[float, float]] = []
            for t, ref in members:
                if t == "n" and ref in node_points:
                    pts.append(node_points[ref])
                elif t == "w":
                    pts.extend(way_points.get(ref, ()))
            if any(near(*p) for p in pts):
                elements.append(
                    {"type": "relation", "id": rel_id, "center": _bbox_center(pts), "tags": tags}
                )

    order = {"node": 0, "way": 1, "relation": 2}
    elements.sort(key=lambda e: (order[e["type"]], e["id"]))
    return {"version": 0.6, "generator": f"osm_extract.py from {path.name}", "elements": elements}


# --------------------------------------------------------------------------- #
# Street network (task 1.3)
# --------------------------------------------------------------------------- #

_CLAUSE_RE = re.compile(r'\["([^"]+)"(?:(!?~)"([^"]*)")?\]')


def parse_way_filter(way_filter: str) -> list[tuple[str, str | None, re.Pattern | None]]:
    """Parse an Overpass tag filter such as OSMnx's into ``(key, op, regex)`` clauses.

    Supported clauses (all that OSMnx's presets use): ``["k"]`` (key exists),
    ``["k"~"re"]`` and ``["k"!~"re"]``. Raises ``ValueError`` on anything else.
    """
    clauses = []
    pos = 0
    for match in _CLAUSE_RE.finditer(way_filter):
        if match.start() != pos:
            raise ValueError(f"unsupported filter syntax near: {way_filter[pos:]!r}")
        key, op, regex = match.groups()
        clauses.append((key, op, re.compile(regex) if op else None))
        pos = match.end()
    if pos != len(way_filter):
        raise ValueError(f"unsupported filter syntax near: {way_filter[pos:]!r}")
    return clauses


def tags_match(tags: Any, clauses: list[tuple[str, str | None, re.Pattern | None]]) -> bool:
    """Apply parsed clauses with Overpass semantics (unanchored regex search)."""
    for key, op, regex in clauses:
        value = tags.get(key)
        if op is None:
            if value is None:
                return False
        elif op == "~":
            if value is None or not regex.search(value):
                return False
        elif value is not None and regex.search(value):  # "!~"
            return False
    return True


def network_payload(path: Path, bbox: Bbox, way_filter: str) -> dict[str, Any]:
    """Return what ``way<way_filter>(bbox); >; out;`` would return from Overpass.

    Ways that pass the filter and have at least one node inside ``bbox``, with all
    their tags, followed by every node they reference (location and tags).
    """
    clauses = parse_way_filter(way_filter)
    south, west, north, east = bbox
    ways: list[dict[str, Any]] = []
    node_points: dict[int, tuple[float, float]] = {}

    for obj in (
        osmium.FileProcessor(str(path))
        .with_locations()
        .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
    ):
        if "highway" not in obj.tags or not tags_match(obj.tags, clauses):
            continue
        refs = [n for n in obj.nodes if n.location.valid()]
        if not any(south <= n.lat <= north and west <= n.lon <= east for n in refs):
            continue
        ways.append(
            {"type": "way", "id": obj.id, "nodes": [n.ref for n in refs], "tags": dict(obj.tags)}
        )
        for n in refs:
            node_points[n.ref] = (n.lat, n.lon)

    node_tags: dict[int, dict[str, str]] = {}
    for obj in osmium.FileProcessor(str(path)).with_filter(
        osmium.filter.EntityFilter(osmium.osm.NODE)
    ):
        if obj.id in node_points and len(obj.tags):
            node_tags[obj.id] = dict(obj.tags)

    nodes = [
        {"type": "node", "id": nid, "lat": lat, "lon": lon, "tags": node_tags.get(nid, {})}
        for nid, (lat, lon) in sorted(node_points.items())
    ]
    ways.sort(key=lambda w: w["id"])
    return {
        "version": 0.6,
        "generator": f"osm_extract.py from {path.name}",
        "elements": ways + nodes,
    }


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def download(url: str = EXTRACT_URL, out: Path = EXTRACT_PATH) -> Path:
    """Download the extract and verify it against the ``.md5`` published next to it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    expected = requests.get(url + ".md5", headers=headers, timeout=60).text.split()[0]
    digest = hashlib.md5()
    tmp = out.with_suffix(".part")
    with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as handle:
            for chunk in resp.iter_content(1 << 20):
                handle.write(chunk)
                digest.update(chunk)
    if digest.hexdigest() != expected:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {url}")
    tmp.replace(out)
    return out


def main(argv: list[str] | None = None) -> int:
    """Download the regional extract used by ``--from-pbf``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=EXTRACT_URL)
    parser.add_argument("--out", type=Path, default=EXTRACT_PATH)
    args = parser.parse_args(argv)
    path = download(args.url, args.out)
    print(
        f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB), OSM data as of {osm_timestamp(path)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
