"""Fetch Madrid wine shops, restaurants and bars from OpenStreetMap (task 1.2).

Run by hand, once:  ``python data/fetch_osm.py``

Sends one Overpass query, turns every element into a venue dict that mirrors the
``Venue`` dataclass (plus ``osm_shop``, needed by ``build_inventory.py``), removes
unnamed and duplicate venues, and writes ``data/venues_madrid.json``.

The raw Overpass response is kept in ``./cache`` (git-ignored). ``--offline`` rebuilds
the JSON from that copy without touching the network.

Data © OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import CENTER, DATA, RADIUS_M  # noqa: E402

OUT_PATH = DATA / "venues_madrid.json"
RAW_CACHE_PATH = ROOT / "cache" / "overpass_venues_raw.json"

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    # Further public Overpass instances, tried when the two above are unreachable.
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
USER_AGENT = "madrid-wine-agent/0.1 (course project)"
REQUEST_TIMEOUT_S = 180
RETRY_WAITS_S = (10, 20, 40, 80)
RETRY_STATUSES = frozenset({429, 504})

SHOP_TAGS = ("wine", "alcohol")
AMENITY_KIND = {"restaurant": "restaurant", "bar": "bar", "pub": "bar"}

DUPLICATE_RADIUS_M = 30.0
# Overpass `around` matches a way when any part of it is inside the circle, so the
# centre of a large way can fall slightly outside. Anything beyond this is dropped.
MAX_DISTANCE_M = RADIUS_M + 100
MAX_RESTAURANTS = 2500
SOURCE = "OpenStreetMap contributors (ODbL)"

QUERY = f"""[out:json][timeout:120];
(
  nwr["shop"~"^({"|".join(SHOP_TAGS)})$"](around:{RADIUS_M},{CENTER[0]},{CENTER[1]});
  nwr["amenity"~"^({"|".join(AMENITY_KIND)})$"](around:{RADIUS_M},{CENTER[0]},{CENTER[1]});
);
out center tags;
"""

# Tag clauses of QUERY, used one at a time by the --split fallback.
TAG_CLAUSES = (
    f'["shop"~"^({"|".join(SHOP_TAGS)})$"]',
    f'["amenity"~"^({"|".join(AMENITY_KIND)})$"]',
)
METRES_PER_DEG_LAT = 111_320.0


def tile_bboxes(n: int) -> list[tuple[float, float, float, float]]:
    """Split the bounding box of the search circle into ``n x n`` tiles.

    Returns ``(south, west, north, east)`` tuples. Neighbouring tiles share an edge;
    an element on the edge may come back twice and is merged by id.
    """
    lat0, lon0 = CENTER
    dlat = RADIUS_M / METRES_PER_DEG_LAT
    dlon = RADIUS_M / (METRES_PER_DEG_LAT * math.cos(math.radians(lat0)))
    south, west = lat0 - dlat, lon0 - dlon
    step_lat, step_lon = 2 * dlat / n, 2 * dlon / n
    return [
        (
            round(south + i * step_lat, 6),
            round(west + j * step_lon, 6),
            round(south + (i + 1) * step_lat, 6),
            round(west + (j + 1) * step_lon, 6),
        )
        for i in range(n)
        for j in range(n)
    ]


def split_queries(n: int) -> list[str]:
    """QUERY cut into one query per tag clause and tile, each still limited to the circle.

    Together they return exactly the elements of QUERY (with repeats on tile edges).
    """
    queries = []
    for clause in TAG_CLAUSES:
        for s, w, n_, e in tile_bboxes(n):
            queries.append(
                "[out:json][timeout:120];\n"
                f"nwr{clause}(around:{RADIUS_M},{CENTER[0]},{CENTER[1]})({s},{w},{n_},{e});\n"
                "out center tags;\n"
            )
    return queries


def merge_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge Overpass payloads, keeping each ``(type, id)`` once, in first-seen order."""
    seen: set[tuple[str, int]] = set()
    elements: list[dict[str, Any]] = []
    for payload in payloads:
        for element in payload["elements"]:
            key = (element["type"], element["id"])
            if key not in seen:
                seen.add(key)
                elements.append(element)
    merged = {k: v for k, v in payloads[0].items() if k != "elements"} if payloads else {}
    merged["elements"] = elements
    return merged


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    r = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _try_once(url: str, query: str) -> tuple[dict[str, Any] | None, str]:
    """POST the query once. Return ``(payload, "")`` or ``(None, problem)``."""
    try:
        resp = requests.post(
            url,
            data={"data": query},
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"
    try:
        payload = resp.json()
    except ValueError:
        return None, "response was not JSON"
    if "elements" not in payload:
        return None, "response has no 'elements'"
    if not payload["elements"] and payload.get("remark"):
        return None, f"Overpass remark: {payload['remark']}"
    return payload, ""


def fetch_overpass(query: str = QUERY, urls: tuple[str, ...] = OVERPASS_URLS) -> dict[str, Any]:
    """Run the Overpass query, with back-off and a mirror, and return the JSON payload.

    Each endpoint gets one attempt plus up to four retries (10 s, 20 s, 40 s, 80 s).
    HTTP 429/504 and transport errors (resets, timeouts) are retried; any other HTTP
    status moves straight on to the mirror.
    """
    problems: list[str] = []
    for url in urls:
        for attempt in range(len(RETRY_WAITS_S) + 1):
            payload, problem = _try_once(url, query)
            if payload is not None:
                print(f"Fetched {len(payload['elements'])} elements from {url}")
                return payload
            problems.append(f"{url} attempt {attempt + 1}: {problem}")
            print(f"  {problems[-1]}", file=sys.stderr)
            retryable = not problem.startswith("HTTP ") or int(problem[5:]) in RETRY_STATUSES
            if not retryable or attempt == len(RETRY_WAITS_S):
                break
            time.sleep(RETRY_WAITS_S[attempt])
    raise RuntimeError("Overpass fetch failed on every endpoint:\n  " + "\n  ".join(problems))


def element_to_venue(element: dict[str, Any]) -> dict[str, Any] | None:
    """Turn one Overpass element into a venue dict, or ``None`` if it must be dropped.

    Dropped: no name, no coordinates, or neither a wanted shop nor a wanted amenity.
    The returned dict carries a private ``_n_tags`` key used only for de-duplication.
    """
    tags: dict[str, str] = element.get("tags") or {}
    name = (tags.get("name") or "").strip()
    if not name:
        return None

    point = element if element.get("type") == "node" else element.get("center") or {}
    lat, lon = point.get("lat"), point.get("lon")
    if lat is None or lon is None:
        return None

    shop = (tags.get("shop") or "").strip()
    amenity = (tags.get("amenity") or "").strip()
    if shop in SHOP_TAGS:  # shop wins over amenity
        kind = "shop"
    elif amenity in AMENITY_KIND:
        kind = AMENITY_KIND[amenity]
    else:
        return None

    address = " ".join(
        tags[key].strip()
        for key in ("addr:street", "addr:housenumber", "addr:postcode")
        if tags.get(key, "").strip()
    )
    cuisines = [c.strip().lower() for c in tags.get("cuisine", "").split(";") if c.strip()]
    cuisines = list(dict.fromkeys(cuisines))  # drop repeats, keep order

    return {
        "venue_id": f"{element['type']}/{element['id']}",
        "name": name,
        "kind": kind,
        "lat": round(float(lat), 7),
        "lon": round(float(lon), 7),
        "address": address,
        "opening_hours": tags.get("opening_hours", "").strip(),
        "cuisines": cuisines,
        "cuisine_assumed": kind != "shop" and not cuisines,
        "website": (tags.get("website") or tags.get("contact:website") or "").strip(),
        "osm_shop": shop,
        "_n_tags": len(tags),
    }


def _id_key(venue_id: str) -> tuple[str, int]:
    kind, _, number = venue_id.partition("/")
    return kind, int(number)


def deduplicate(venues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove venues with the same lower-cased name within 30 m of a kept one.

    Venues are considered richest first (most tags, ties by id), so the one with more
    tags survives.
    """
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for venue in venues:
        by_name[venue["name"].casefold()].append(venue)

    kept: list[dict[str, Any]] = []
    for group in by_name.values():
        group.sort(key=lambda v: (-v["_n_tags"], _id_key(v["venue_id"])))
        survivors: list[dict[str, Any]] = []
        for venue in group:
            if all(
                haversine_m(venue["lat"], venue["lon"], s["lat"], s["lon"]) > DUPLICATE_RADIUS_M
                for s in survivors
            ):
                survivors.append(venue)
        kept.extend(survivors)
    return kept


def kind_counts(venues: list[dict[str, Any]]) -> dict[str, int]:
    """Number of venues per kind, always with all three keys."""
    counts = Counter(v["kind"] for v in venues)
    return {kind: counts.get(kind, 0) for kind in ("shop", "restaurant", "bar")}


def build_document(payload: dict[str, Any], fetched_at: str) -> tuple[dict[str, Any], dict]:
    """Build the output document from a raw Overpass payload.

    Returns ``(document, stats)`` where ``stats`` holds the numbers for the summary.
    """
    elements = payload["elements"]
    venues = [v for v in map(element_to_venue, elements) if v is not None]
    n_usable = len(venues)

    venues = [
        v for v in venues if haversine_m(CENTER[0], CENTER[1], v["lat"], v["lon"]) <= MAX_DISTANCE_M
    ]
    n_in_range = len(venues)

    venues = deduplicate(venues)
    n_deduplicated = len(venues)

    filter_note: dict[str, Any] | None = None
    before = kind_counts(venues)
    if before["restaurant"] > MAX_RESTAURANTS:
        venues = [
            v for v in venues if v["kind"] != "restaurant" or v["opening_hours"] or v["cuisines"]
        ]
        filter_note = {
            "rule": (
                f"More than {MAX_RESTAURANTS} restaurants were found, so all shops and "
                "bars were kept and only restaurants with an opening_hours tag or a "
                "cuisine tag were kept."
            ),
            "counts_before": before,
            "counts_after": kind_counts(venues),
        }

    venues.sort(key=lambda v: v["venue_id"])
    for venue in venues:
        del venue["_n_tags"]

    document: dict[str, Any] = {
        "fetched_at": fetched_at,
        "center": list(CENTER),
        "radius_m": RADIUS_M,
        "source": SOURCE,
    }
    if filter_note is not None:
        document["filter_note"] = filter_note
    document["venues"] = venues

    stats = {
        "elements": len(elements),
        "dropped_unnamed_or_no_coords": len(elements) - n_usable,
        "dropped_out_of_range": n_usable - n_in_range,
        "dropped_duplicates": n_in_range - n_deduplicated,
        "dropped_by_restaurant_filter": n_deduplicated - len(venues),
    }
    return document, stats


def print_summary(document: dict[str, Any], stats: dict[str, int]) -> None:
    """Print counts per kind and tag coverage."""
    venues = document["venues"]
    print(f"Overpass elements: {stats['elements']}")
    for key in (
        "dropped_unnamed_or_no_coords",
        "dropped_out_of_range",
        "dropped_duplicates",
        "dropped_by_restaurant_filter",
    ):
        print(f"  {key.replace('_', ' ')}: {stats[key]}")
    print(f"Venues written: {len(venues)}")
    for kind, count in kind_counts(venues).items():
        print(f"  {kind}: {count}")
    shop_tags = Counter(v["osm_shop"] for v in venues if v["kind"] == "shop")
    print(f"  shop tag values: {dict(sorted(shop_tags.items()))}")
    print(f"With opening_hours: {sum(1 for v in venues if v['opening_hours'])}")
    print(f"With cuisine: {sum(1 for v in venues if v['cuisines'])}")
    print(f"Restaurant filter applied: {'yes' if 'filter_note' in document else 'no'}")
    cuisine_counts = Counter(c for v in venues for c in v["cuisines"])
    print("Top 30 cuisine values:")
    for cuisine, count in sorted(cuisine_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:30]:
        print(f"  {cuisine}: {count}")


def main(argv: list[str] | None = None) -> int:
    """Fetch (or reuse the cached raw response), build and write the venues file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--offline",
        action="store_true",
        help="rebuild from the cached raw Overpass response instead of fetching",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        metavar="URL",
        help="Overpass interpreter URL to use instead of the built-in list (repeatable)",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=0,
        metavar="N",
        help="send the query in pieces (per tag clause and N x N tiles) and merge them; "
        "use when servers time out on the full query",
    )
    parser.add_argument(
        "--from-pbf",
        type=Path,
        metavar="PATH",
        help="read a regional .osm.pbf extract (see data/osm_extract.py) instead of "
        "querying Overpass; gives the same elements the query would",
    )
    args = parser.parse_args(argv)

    if args.from_pbf:
        from data.osm_extract import osm_timestamp, venues_payload

        payload = venues_payload(args.from_pbf)
        fetched_at = osm_timestamp(args.from_pbf) or datetime.now(UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        print(f"Read {len(payload['elements'])} elements from {args.from_pbf.name}")
    elif args.offline:
        cached = json.loads(RAW_CACHE_PATH.read_text(encoding="utf-8"))
        payload, fetched_at = cached["payload"], cached["fetched_at"]
    else:
        urls = tuple(args.endpoint or OVERPASS_URLS)
        if args.split > 0:
            parts = split_queries(args.split)
            payloads = []
            for i, query in enumerate(parts, 1):
                print(f"Part {i}/{len(parts)}")
                payloads.append(fetch_overpass(query, urls))
            payload = merge_payloads(payloads)
        else:
            payload = fetch_overpass(urls=urls)
        fetched_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_CACHE_PATH.write_text(
            json.dumps({"fetched_at": fetched_at, "payload": payload}, ensure_ascii=False),
            encoding="utf-8",
        )

    document, stats = build_document(payload, fetched_at)
    retrieved_via = f"regional extract {args.from_pbf.name}" if args.from_pbf else "Overpass API"
    document = {
        **{k: v for k, v in document.items() if k != "venues"},
        "retrieved_via": retrieved_via,
        "venues": document["venues"],
    }
    OUT_PATH.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print_summary(document, stats)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
