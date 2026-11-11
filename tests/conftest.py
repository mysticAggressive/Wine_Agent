"""A hand-made miniature world shared by all tests (DEVELOPMENT_PLAN.md task 2.3).

5 wines, 4 venues, 3 dishes and an inventory linking them. The files are written to
``tmp_path`` in the same formats as the real files under ``data/`` and read back with
the real loaders, so no test touches the network or depends on the real data.

Useful facts about this world:

* ``node/1`` (shop) is closed on Mondays; ``node/2`` (shop) has no opening hours.
* ``way/3`` is a restaurant with ``cuisine=spanish``; ``node/4`` is a bar with no cuisine
  tag (``cuisine_assumed``), open in the evening across midnight.
* Restaurants/bars do not serve ``chocolate_dessert``; shops serve every dish.
* 2026-11-09 is a Monday, 2026-11-11 a Wednesday.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent.data_loader import (
    AgentData,
    Inventory,
    PairingRules,
    load_inventory,
    load_pairing_rules,
    load_venues,
    load_wines,
)
from agent.models import Venue, Wine
from agent.preferences import Preferences, default_preferences

MADRID = ZoneInfo("Europe/Madrid")
WEDNESDAY_NOON = datetime(2026, 11, 11, 12, 0, tzinfo=MADRID)
MONDAY_NOON = datetime(2026, 11, 9, 12, 0, tzinfo=MADRID)

WINE_COLUMNS = [
    "wine_id",
    "winery",
    "name",
    "year",
    "rating",
    "num_reviews",
    "region",
    "retail_price",
    "wine_type",
    "style",
    "body",
    "acidity",
    "imputed",
]

# wine_id, winery, name, year, rating, num_reviews, region, retail_price, wine_type,
# style, body, acidity, imputed
WINE_ROWS = [
    ("w0001", "Bodegas Prueba", "Gran Reserva", "2015", 4.6, 900, "Ribera del Duero", 60.0,
     "Ribera Del Duero Red", "red", 5, 3, False),
    ("w0002", "Bodegas Prueba", "Joven", "2020", 4.3, 2500, "Rioja", 12.0,
     "Rioja Red", "red", 4, 3, False),
    ("w0003", "Mar de Prueba", "Albarino", "2021", 4.4, 400, "Rias Baixas", 18.0,
     "Albarino", "white", 2, 3, False),
    ("w0004", "Cavas Prueba", "Brut Nature", "", 4.3, 150, "Cava", 8.0,
     "Cava", "sparkling", 2, 3, True),
    ("w0005", "Jerez Prueba", "PX Viejo", "", 4.7, 300, "Jerez-Xeres-Sherry", 30.0,
     "Pedro Ximenez", "fortified", 5, 1, False),
]  # fmt: skip

VENUES_PAYLOAD = {
    "fetched_at": "2026-01-01T00:00:00+00:00",
    "center": [40.4168, -3.7038],
    "radius_m": 4000,
    "source": "OpenStreetMap contributors (ODbL)",
    "venues": [
        {
            "venue_id": "node/1",
            "name": "Vinoteca Sol",
            "kind": "shop",
            "lat": 40.4170,
            "lon": -3.7040,
            "address": "Calle Mayor 1 28013",
            "opening_hours": "Tu-Su 10:00-21:00",
            "cuisines": [],
            "cuisine_assumed": False,
            "website": "https://example.org/vinoteca-sol",
            "osm_shop": "wine",
        },
        {
            "venue_id": "node/2",
            "name": "Licores Norte",
            "kind": "shop",
            "lat": 40.4268,
            "lon": -3.7040,
            "address": "",
            "opening_hours": "",
            "cuisines": [],
            "cuisine_assumed": False,
            "website": "",
            "osm_shop": "alcohol",
        },
        {
            "venue_id": "way/3",
            "name": "Casa Prueba",
            "kind": "restaurant",
            "lat": 40.4155,
            "lon": -3.7074,
            "address": "Plaza Mayor 3 28012",
            "opening_hours": "Mo-Su 12:00-23:30",
            "cuisines": ["spanish"],
            "cuisine_assumed": False,
            "website": "",
        },
        {
            "venue_id": "node/4",
            "name": "Bar Prueba",
            "kind": "bar",
            "lat": 40.4089,
            "lon": -3.7007,
            "address": "",
            "opening_hours": "Mo-Su 18:00-02:00",
            "cuisines": [],
            "cuisine_assumed": True,
            "website": "",
        },
    ],
}

PAIRING_PAYLOAD = {
    "dishes": [
        {
            "dish_id": "grilled_red_meat",
            "label": "Grilled red meat",
            "styles": {"red": 1.0, "fortified": 0.2},
            "body": [4, 5],
            "acidity": [2, 3],
            "type_bonus": ["Ribera Del Duero Red", "Rioja Red"],
            "why": "Tannin and body stand up to charred, fatty meat.",
        },
        {
            "dish_id": "shellfish",
            "label": "Shellfish",
            "styles": {"white": 1.0, "sparkling": 0.9},
            "body": [2, 3],
            "acidity": [3, 3],
            "type_bonus": ["Albarino"],
            "why": "High acidity and a light body lift briny shellfish.",
        },
        {
            "dish_id": "chocolate_dessert",
            "label": "Chocolate dessert",
            "styles": {"fortified": 1.0, "red": 0.3},
            "body": [5, 5],
            "acidity": [1, 2],
            "type_bonus": ["Pedro Ximenez"],
            "why": "A sweet, dense wine matches the richness of chocolate.",
        },
    ],
    "cuisine_to_dishes": {
        "spanish": ["grilled_red_meat", "shellfish"],
        "seafood": ["shellfish"],
        "_default": ["grilled_red_meat"],
    },
}

# venue_id, wine_id, price_eur (= retail x 1.0 shop, x 2.0 bar, x 2.5 restaurant)
INVENTORY_ROWS = [
    ("node/1", "w0001", 60.0),
    ("node/1", "w0002", 12.0),
    ("node/1", "w0003", 18.0),
    ("node/1", "w0005", 30.0),
    ("node/2", "w0002", 12.0),
    ("node/2", "w0004", 8.0),
    ("node/4", "w0002", 24.0),
    ("node/4", "w0004", 16.0),
    ("way/3", "w0001", 150.0),
    ("way/3", "w0002", 30.0),
    ("way/3", "w0003", 45.0),
]


@pytest.fixture
def wines_path(tmp_path: Path) -> Path:
    """Path of a miniature ``wines.csv``."""
    path = tmp_path / "wines.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(WINE_COLUMNS)
        writer.writerows(WINE_ROWS)
    return path


@pytest.fixture
def venues_path(tmp_path: Path) -> Path:
    """Path of a miniature ``venues_madrid.json``."""
    path = tmp_path / "venues_madrid.json"
    path.write_text(json.dumps(VENUES_PAYLOAD, indent=1), encoding="utf-8")
    return path


@pytest.fixture
def inventory_path(tmp_path: Path) -> Path:
    """Path of a miniature ``inventory.csv``."""
    path = tmp_path / "inventory.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["venue_id", "wine_id", "price_eur"])
        writer.writerows(INVENTORY_ROWS)
    return path


@pytest.fixture
def pairing_rules_path(tmp_path: Path) -> Path:
    """Path of a miniature ``pairing_rules.json``."""
    path = tmp_path / "pairing_rules.json"
    path.write_text(json.dumps(PAIRING_PAYLOAD, indent=1), encoding="utf-8")
    return path


@pytest.fixture
def data_dir(
    wines_path: Path, venues_path: Path, inventory_path: Path, pairing_rules_path: Path
) -> Path:
    """Directory holding all four miniature data files (usable with ``load_all``)."""
    return wines_path.parent


@pytest.fixture
def wines(wines_path: Path) -> dict[str, Wine]:
    """The 5 fixture wines."""
    return load_wines(wines_path)


@pytest.fixture
def venues(venues_path: Path) -> dict[str, Venue]:
    """The 4 fixture venues."""
    return load_venues(venues_path)


@pytest.fixture
def inventory(inventory_path: Path) -> Inventory:
    """The fixture inventory."""
    return load_inventory(inventory_path)


@pytest.fixture
def pairing_rules(pairing_rules_path: Path) -> PairingRules:
    """The 3 fixture dishes and the cuisine map."""
    return load_pairing_rules(pairing_rules_path)


@pytest.fixture
def agent_data(
    wines: dict[str, Wine],
    venues: dict[str, Venue],
    inventory: Inventory,
    pairing_rules: PairingRules,
) -> AgentData:
    """The whole miniature world."""
    return AgentData(wines=wines, venues=venues, inventory=inventory, pairing_rules=pairing_rules)


@pytest.fixture
def prefs() -> Preferences:
    """Default preferences on a Wednesday at noon, Madrid time."""
    return default_preferences(WEDNESDAY_NOON)
