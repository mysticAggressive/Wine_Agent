"""Build the derived (simulated) inventory: which wines each venue carries (task 1.4).

Run by hand after ``clean_wines.py`` and ``fetch_osm.py``:  ``python data/build_inventory.py``

No free source says which Madrid venue stocks which wine, so each venue gets a
deterministic sample of dataset wines, seeded by its OSM id (DEVELOPMENT_PLAN.md 3.2).
Well-reviewed wines are more likely to be drawn, so they appear in more venues. The
result is labelled "simulated inventory" wherever it is shown.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import DATA, MARKUP  # noqa: E402

WINES_PATH = DATA / "wines.csv"
VENUES_PATH = DATA / "venues_madrid.json"
OUT_PATH = DATA / "inventory.csv"

#: Inclusive range of how many wines a venue carries, by venue category.
WINE_COUNT = {
    "shop_wine": (60, 120),
    "shop_alcohol": (30, 60),
    "restaurant": (15, 40),
    "bar": (10, 25),
}


def venue_seed(venue_id: str) -> int:
    """Stable 64-bit seed from the venue id."""
    return int(hashlib.sha256(venue_id.encode("utf-8")).hexdigest()[:16], 16)


def category(venue: dict) -> str:
    """``shop_wine``, ``shop_alcohol``, ``restaurant`` or ``bar``."""
    if venue["kind"] == "shop":
        return "shop_wine" if venue.get("osm_shop") == "wine" else "shop_alcohol"
    return venue["kind"]


def weighted_sample(rng: random.Random, ids: list[str], weights: list[float], k: int) -> list[str]:
    """Draw ``k`` distinct ids, each draw proportional to weight (without replacement).

    Uses the Efraimidis-Spirakis method: key = u ** (1 / w), keep the ``k`` largest.
    One random number per id, in a fixed order, so the result depends only on the seed.
    """
    keyed = [(rng.random() ** (1.0 / w), wine_id) for wine_id, w in zip(ids, weights, strict=True)]
    keyed.sort(key=lambda kv: (-kv[0], kv[1]))
    return [wine_id for _, wine_id in keyed[:k]]


def build_rows(wines: list[dict], venues: list[dict]) -> list[tuple[str, str, str]]:
    """Return sorted ``(venue_id, wine_id, price_eur)`` rows, prices as 2-decimal text."""
    wines = sorted(wines, key=lambda w: w["wine_id"])
    ids = [w["wine_id"] for w in wines]
    weights = [math.log10(int(w["num_reviews"]) + 10) for w in wines]
    retail = {w["wine_id"]: float(w["retail_price"]) for w in wines}

    rows: list[tuple[str, str, str]] = []
    for venue in sorted(venues, key=lambda v: v["venue_id"]):
        venue_id = venue["venue_id"]
        rng = random.Random(venue_seed(venue_id))
        lo, hi = WINE_COUNT[category(venue)]
        count = min(rng.randint(lo, hi), len(ids))
        markup = MARKUP[venue["kind"]]
        for wine_id in weighted_sample(rng, ids, weights, count):
            rows.append((venue_id, wine_id, f"{round(retail[wine_id] * markup, 2):.2f}"))
    rows.sort()
    return rows


def main() -> int:
    """Read wines and venues, write ``inventory.csv`` and print a summary."""
    with open(WINES_PATH, newline="", encoding="utf-8") as handle:
        wines = list(csv.DictReader(handle))
    venues = json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]
    rows = build_rows(wines, venues)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["venue_id", "wine_id", "price_eur"])
        writer.writerows(rows)

    per_venue = Counter(r[0] for r in rows)
    per_wine = Counter(r[1] for r in rows)
    cats = Counter(category(v) for v in venues)
    print(f"Venues: {len(venues)} {dict(sorted(cats.items()))}")
    print(f"Rows: {len(rows)}")
    print(f"Wines per venue: min {min(per_venue.values())}, max {max(per_venue.values())}")
    print(f"Distinct wines stocked somewhere: {len(per_wine)} of {len(wines)}")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
