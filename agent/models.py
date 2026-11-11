"""Shared data models. See DEVELOPMENT_PLAN.md section 4.2."""

from dataclasses import dataclass
from enum import Enum


class Style(str, Enum):  # noqa: UP042 - contract 4.2
    """Wine style derived from the dataset's ``type`` column."""

    RED = "red"
    WHITE = "white"  # the dataset has no rosé
    SPARKLING = "sparkling"
    FORTIFIED = "fortified"
    UNKNOWN = "unknown"


class VenueKind(str, Enum):  # noqa: UP042 - contract 4.2
    """Kind of venue, from its OpenStreetMap tags."""

    SHOP = "shop"  # OSM shop=wine, shop=alcohol
    RESTAURANT = "restaurant"  # OSM amenity=restaurant
    BAR = "bar"  # OSM amenity=bar, amenity=pub


class OpenState(str, Enum):  # noqa: UP042 - contract 4.2
    """Whether a venue is open at a given time."""

    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Wine:
    """One cleaned wine from ``data/wines.csv``."""

    wine_id: str  # "w0001" …, stable across runs
    winery: str
    name: str
    year: int | None  # None for non-vintage
    rating: float
    num_reviews: int
    region: str
    retail_price: float  # euros
    wine_type: str  # original `type` column, "" if missing
    style: Style
    body: int  # 2–5 in this dataset
    acidity: int  # 1–3
    imputed: bool  # True if body or acidity was filled in


@dataclass(frozen=True)
class Venue:
    """One Madrid venue from OpenStreetMap."""

    venue_id: str  # "node/123456" or "way/123456"
    name: str
    kind: VenueKind
    lat: float
    lon: float
    address: str  # "" if OSM has none
    opening_hours: str  # raw OSM tag, "" if missing
    cuisines: tuple[str, ...]  # lower-case, split on ";"
    cuisine_assumed: bool
    website: str


@dataclass(frozen=True)
class Dish:
    """A dish the user can pair a wine with."""

    dish_id: str  # "grilled_red_meat"
    label: str  # "Grilled red meat"


@dataclass(frozen=True)
class Candidate:  # one CSP solution
    """One ranked recommendation: a (wine, venue, dish) triple with its score."""

    wine_id: str
    venue_id: str
    dish_id: str
    price_at_venue: float
    distance_m: float
    features: dict[str, float]  # taste, quality, price_fit, pairing, distance
    score: float  # Eval
    reasons: tuple[str, ...]  # human-readable "why" lines

    def as_row(self) -> dict[str, object]:
        """Return a flat dict for the results table (one column per feature)."""
        row: dict[str, object] = {
            "wine_id": self.wine_id,
            "venue_id": self.venue_id,
            "dish_id": self.dish_id,
            "price_at_venue": self.price_at_venue,
            "distance_m": self.distance_m,
            "score": self.score,
        }
        for name, value in self.features.items():
            row[name] = value
        row["reasons"] = "; ".join(self.reasons)
        return row


@dataclass
class Diagnostics:
    """Search statistics shown in the result-monitoring panel."""

    initial_domain_sizes: dict[str, int]
    pruned_by_constraint: dict[str, int]  # constraint id -> values removed
    domain_sizes_after_node_consistency: dict[str, int]
    domain_sizes_after_ac3: dict[str, int]
    ac3_revisions: int
    assignments: int
    backtracks: int
    forward_check_prunes: int
    solutions_found: int
    variable_order: list[str]  # order MRV picked
    elapsed_ms: float
    relaxation_hints: list[str]  # filled when there are 0 solutions


@dataclass
class RecommendationResult:
    """What ``recommend`` returns: ranked candidates plus diagnostics."""

    candidates: list[Candidate]  # ranked, best first
    diagnostics: Diagnostics
