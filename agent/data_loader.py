"""Load the files under ``data/`` into models. See DEVELOPMENT_PLAN.md section 4.4."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from agent.config import DATA
from agent.models import Dish, Style, Venue, VenueKind, Wine

DEFAULT_CUISINE_KEY = "_default"


class Inventory:
    """Which venue stocks which wine, and at what price. Every lookup is O(1)."""

    def __init__(self, rows: Iterable[tuple[str, str, float]] = ()) -> None:
        """Build from ``(venue_id, wine_id, price_eur)`` rows."""
        self._by_venue: dict[str, dict[str, float]] = {}
        wines_by_venue: dict[str, set[str]] = {}
        venues_by_wine: dict[str, set[str]] = {}
        for venue_id, wine_id, price in rows:
            self._by_venue.setdefault(venue_id, {})[wine_id] = float(price)
            wines_by_venue.setdefault(venue_id, set()).add(wine_id)
            venues_by_wine.setdefault(wine_id, set()).add(venue_id)
        self._wines_at = {v: frozenset(w) for v, w in wines_by_venue.items()}
        self._venues_with = {w: frozenset(v) for w, v in venues_by_wine.items()}

    def wines_at(self, venue_id: str) -> frozenset[str]:
        """Wine ids stocked by a venue (empty if the venue is unknown)."""
        return self._wines_at.get(venue_id, frozenset())

    def venues_with(self, wine_id: str) -> frozenset[str]:
        """Venue ids that stock a wine (empty if no venue does)."""
        return self._venues_with.get(wine_id, frozenset())

    def price(self, venue_id: str, wine_id: str) -> float:
        """Price in euros of a wine at a venue. Raises ``KeyError`` if not stocked."""
        return self._by_venue[venue_id][wine_id]

    def stocks(self, venue_id: str, wine_id: str) -> bool:
        """True if the venue carries the wine."""
        return wine_id in self._by_venue.get(venue_id, ())

    def venue_ids(self) -> frozenset[str]:
        """All venue ids that stock at least one wine."""
        return frozenset(self._wines_at)

    def __len__(self) -> int:
        """Number of (venue, wine) pairs."""
        return sum(len(wines) for wines in self._by_venue.values())


@dataclass(frozen=True)
class DishRule:
    """Pairing rule of one dish (one entry of ``pairing_rules.json`` ``dishes``)."""

    dish_id: str
    label: str
    styles: dict[str, float]  # style value ("red", …) -> weight 0–1; Style members work as keys
    body: tuple[int, int]  # inclusive (lo, hi)
    acidity: tuple[int, int]  # inclusive (lo, hi)
    type_bonus: tuple[str, ...]  # exact wine_type strings
    why: str

    @property
    def dish(self) -> Dish:
        """The plain ``Dish`` model for this rule."""
        return Dish(dish_id=self.dish_id, label=self.label)


@dataclass(frozen=True)
class PairingRules:
    """All dishes with their rules, and the cuisine -> dishes map."""

    dishes: dict[str, DishRule]  # dish_id -> rule, in file order
    cuisine_to_dishes: dict[str, tuple[str, ...]]  # includes the "_default" key if present

    def dishes_for_cuisine(self, cuisine: str) -> tuple[str, ...]:
        """Dish ids mapped to a cuisine tag, falling back to ``_default``."""
        if cuisine in self.cuisine_to_dishes:
            return self.cuisine_to_dishes[cuisine]
        return self.cuisine_to_dishes.get(DEFAULT_CUISINE_KEY, ())


@dataclass(frozen=True)
class AgentData:
    """Everything the agent reads from ``data/``."""

    wines: dict[str, Wine]
    venues: dict[str, Venue]
    inventory: Inventory
    pairing_rules: PairingRules


def _parse_bool(text: str) -> bool:
    """Parse the ``True``/``False`` text pandas writes."""
    return text.strip().lower() in {"true", "1", "yes"}


def load_wines(path: str | Path = DATA / "wines.csv") -> dict[str, Wine]:
    """Load the cleaned wines, keyed by ``wine_id`` in file order."""
    wines: dict[str, Wine] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            year_text = row["year"].strip()
            wine = Wine(
                wine_id=row["wine_id"],
                winery=row["winery"],
                name=row["name"],
                year=int(float(year_text)) if year_text else None,
                rating=float(row["rating"]),
                num_reviews=int(row["num_reviews"]),
                region=row["region"],
                retail_price=float(row["retail_price"]),
                wine_type=row["wine_type"],
                style=Style(row["style"]),
                body=int(row["body"]),
                acidity=int(row["acidity"]),
                imputed=_parse_bool(row["imputed"]),
            )
            wines[wine.wine_id] = wine
    return wines


def load_venues(path: str | Path = DATA / "venues_madrid.json") -> dict[str, Venue]:
    """Load the cached OSM venues, keyed by ``venue_id``. Unknown keys are ignored."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    venues: dict[str, Venue] = {}
    for item in payload["venues"]:
        venue = Venue(
            venue_id=item["venue_id"],
            name=item["name"],
            kind=VenueKind(item["kind"]),
            lat=float(item["lat"]),
            lon=float(item["lon"]),
            address=item.get("address") or "",
            opening_hours=item.get("opening_hours") or "",
            cuisines=tuple(item.get("cuisines") or ()),
            cuisine_assumed=bool(item.get("cuisine_assumed", False)),
            website=item.get("website") or "",
        )
        venues[venue.venue_id] = venue
    return venues


def load_inventory(path: str | Path = DATA / "inventory.csv") -> Inventory:
    """Load the derived inventory (columns ``venue_id, wine_id, price_eur``)."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [
            (row["venue_id"], row["wine_id"], float(row["price_eur"]))
            for row in csv.DictReader(handle)
        ]
    return Inventory(rows)


def load_pairing_rules(path: str | Path = DATA / "pairing_rules.json") -> PairingRules:
    """Load the dishes, their pairing rules and the cuisine -> dishes map."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    dishes: dict[str, DishRule] = {}
    for item in payload["dishes"]:
        rule = DishRule(
            dish_id=item["dish_id"],
            label=item["label"],
            styles={str(style): float(weight) for style, weight in item["styles"].items()},
            body=(int(item["body"][0]), int(item["body"][1])),
            acidity=(int(item["acidity"][0]), int(item["acidity"][1])),
            type_bonus=tuple(item.get("type_bonus") or ()),
            why=item.get("why") or "",
        )
        if rule.dish_id in dishes:
            raise ValueError(f"Duplicate dish_id in {path}: {rule.dish_id}")
        dishes[rule.dish_id] = rule
    cuisine_to_dishes = {
        str(cuisine).strip().lower(): tuple(dish_ids)
        for cuisine, dish_ids in payload["cuisine_to_dishes"].items()
    }
    return PairingRules(dishes=dishes, cuisine_to_dishes=cuisine_to_dishes)


def load_all(data_dir: str | Path = DATA) -> AgentData:
    """Load wines, venues, inventory and pairing rules from ``data_dir``."""
    base = Path(data_dir)
    return AgentData(
        wines=load_wines(base / "wines.csv"),
        venues=load_venues(base / "venues_madrid.json"),
        inventory=load_inventory(base / "inventory.csv"),
        pairing_rules=load_pairing_rules(base / "pairing_rules.json"),
    )
