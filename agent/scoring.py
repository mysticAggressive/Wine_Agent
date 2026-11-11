"""The evaluation function ``Eval(s) = sum(w_i * f_i) / sum(w_i)``. See plan 3.7, task 5.2.

Five features, each in ``[0, 1]``:

========= ==========================================================
taste     ``1 - |body - want_body| / 3``
quality   ``(rating - min_rating) / (max_rating - min_rating)``
price_fit ``1 - price_at_venue / budget``, clipped
pairing   pairing score from the pairing engine
distance  ``1 - distance_m / max_distance_m``, clipped
========= ==========================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from agent.config import FEATURES, WALK_SPEED_M_PER_MIN
from agent.data_loader import AgentData, DishRule, Inventory, PairingRules
from agent.models import Dish, OpenState, Venue, Wine
from agent.opening import open_state
from agent.pairing import pairing_reason, pairing_score
from agent.preferences import Preferences, haversine_m

BODY_SPAN = 3.0  # body runs 2-5
NEUTRAL_QUALITY = 0.5  # used when every wine has the same rating


@dataclass
class ScoringContext:
    """What ``features`` needs besides the wine, the venue and the preferences.

    * ``min_rating`` / ``max_rating`` — rating range of the whole wine dataset.
    * ``distances`` — venue id -> straight-line metres from the user (the dict the wine
      CSP caches); a venue missing from it is measured on demand and added.
    * ``inventory`` — gives the price of a wine at a venue.
    * ``pairing_rules`` — for the pairing score and its reason.
    """

    min_rating: float
    max_rating: float
    inventory: Inventory
    pairing_rules: PairingRules
    distances: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_data(
        cls, data: AgentData, distances: dict[str, float] | None = None
    ) -> ScoringContext:
        """Build a context from the loaded data, computing the rating range once."""
        ratings = [wine.rating for wine in data.wines.values()]
        return cls(
            min_rating=min(ratings, default=0.0),
            max_rating=max(ratings, default=0.0),
            inventory=data.inventory,
            pairing_rules=data.pairing_rules,
            distances=distances if distances is not None else {},
        )

    def distance_m(self, venue: Venue, prefs: Preferences) -> float:
        """Return the straight-line distance in metres from the user to ``venue``."""
        distance = self.distances.get(venue.venue_id)
        if distance is None:
            distance = haversine_m(prefs.user_lat, prefs.user_lon, venue.lat, venue.lon)
            self.distances[venue.venue_id] = distance
        return distance


def _clip(value: float) -> float:
    """Clip to ``[0, 1]``."""
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _taste(wine: Wine, prefs: Preferences) -> float:
    return _clip(1.0 - abs(wine.body - prefs.want_body) / BODY_SPAN)


def _quality(wine: Wine, ctx: ScoringContext) -> float:
    span = ctx.max_rating - ctx.min_rating
    if span <= 0:
        return NEUTRAL_QUALITY
    return _clip((wine.rating - ctx.min_rating) / span)


def _price_fit(price: float, prefs: Preferences) -> float:
    return _clip(1.0 - price / prefs.budget_eur)


def _distance(distance_m: float, prefs: Preferences) -> float:
    return _clip(1.0 - distance_m / prefs.max_distance_m)


def features(
    wine: Wine, venue: Venue, dish_id: str, prefs: Preferences, ctx: ScoringContext
) -> dict[str, float]:
    """Return the five features of a (wine, venue, dish) triple, each in ``[0, 1]``.

    Keys are in the order of ``config.FEATURES``. Raises ``KeyError`` if the venue does
    not stock the wine (there is no price to judge) or the dish is unknown.
    """
    price = ctx.inventory.price(venue.venue_id, wine.wine_id)
    return {
        "taste": _taste(wine, prefs),
        "quality": _quality(wine, ctx),
        "price_fit": _price_fit(price, prefs),
        "pairing": _clip(pairing_score(wine, dish_id, ctx.pairing_rules)),
        "distance": _distance(ctx.distance_m(venue, prefs), prefs),
    }


def _effective_weights(weights: Mapping[str, float], names: tuple[str, ...]) -> list[float]:
    """Return the weight of each name (missing = 0); all ones if every weight is 0."""
    values = [max(0.0, float(weights.get(name, 0.0))) for name in names]
    if not any(values):
        return [1.0] * len(names)
    return values


def evaluate(features: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Return ``Eval = sum(w_i * f_i) / sum(w_i)``, in ``[0, 1]``.

    A feature with no weight counts as weight 0. If every weight is 0, all features
    weigh the same. The products are added left to right in ``config.FEATURES`` order
    (then any extra features, sorted), so the result is reproducible to the last bit.
    """
    names = tuple(n for n in FEATURES if n in features) + tuple(
        sorted(n for n in features if n not in FEATURES)
    )
    if not names:
        return 0.0
    values = _effective_weights(weights, names)
    total = 0.0
    weight_sum = 0.0
    for name, weight in zip(names, values, strict=True):
        total += weight * features[name]
        weight_sum += weight
    return total / weight_sum


class Scorer:
    """``Eval`` for many triples of one search, by id, with the per-wine, per-venue and
    per-(wine, dish) parts cached.

    ``score`` returns bit for bit what ``evaluate(features(...), prefs.weights)``
    returns (same products, same order of addition); it only avoids rebuilding the
    feature dict for each of the hundreds of thousands of solutions.
    """

    def __init__(
        self,
        prefs: Preferences,
        wines: Mapping[str, Wine],
        venues: Mapping[str, Venue],
        ctx: ScoringContext,
    ) -> None:
        self._prefs = prefs
        self._wines = wines
        self._venues = venues
        self._ctx = ctx
        (
            self._w_taste,
            self._w_quality,
            self._w_price,
            self._w_pairing,
            self._w_distance,
        ) = _effective_weights(prefs.weights, FEATURES)
        self._weight_sum = 0.0
        for weight in _effective_weights(prefs.weights, FEATURES):
            self._weight_sum += weight
        self._wine_part: dict[str, float] = {}  # w_taste*taste + w_quality*quality
        self._venue_part: dict[str, float] = {}  # w_distance*distance
        self._pair_part: dict[tuple[str, str], float] = {}  # w_pairing*pairing

    def score(self, wine_id: str, venue_id: str, dish_id: str) -> float:
        """Return ``Eval`` of the triple. Raises ``KeyError`` if it is not stocked."""
        prefs, ctx = self._prefs, self._ctx
        wine_part = self._wine_part.get(wine_id)
        if wine_part is None:
            wine = self._wines[wine_id]
            wine_part = 0.0
            wine_part += self._w_taste * _taste(wine, prefs)
            wine_part += self._w_quality * _quality(wine, ctx)
            self._wine_part[wine_id] = wine_part
        venue_part = self._venue_part.get(venue_id)
        if venue_part is None:
            distance = ctx.distance_m(self._venues[venue_id], prefs)
            venue_part = self._w_distance * _distance(distance, prefs)
            self._venue_part[venue_id] = venue_part
        pair_key = (wine_id, dish_id)
        pair_part = self._pair_part.get(pair_key)
        if pair_part is None:
            pairing = _clip(pairing_score(self._wines[wine_id], dish_id, ctx.pairing_rules))
            pair_part = self._w_pairing * pairing
            self._pair_part[pair_key] = pair_part
        price_fit = 1.0 - ctx.inventory.price(venue_id, wine_id) / prefs.budget_eur
        if price_fit < 0.0:
            price_fit = 0.0
        elif price_fit > 1.0:
            price_fit = 1.0
        total = wine_part + self._w_price * price_fit
        total += pair_part
        total += venue_part
        return total / self._weight_sum


# --------------------------------------------------------------------------- #
# "Why this pick" lines
# --------------------------------------------------------------------------- #


def _money(amount: float) -> str:
    """``50`` for a whole amount, ``42.50`` otherwise."""
    return f"{amount:.0f}" if float(amount).is_integer() else f"{amount:.2f}"


def format_distance(distance_m: float) -> str:
    """Return ``"650 m"`` below one kilometre and ``"1.2 km"`` from there on."""
    if round(distance_m) < 1000:
        return f"{distance_m:.0f} m"
    return f"{distance_m / 1000:.1f} km"


def _price_line(price: float, budget: float) -> str:
    percent = round((1.0 - price / budget) * 100)
    if percent > 0:
        return f"€{price:.2f} — {percent}% under your €{_money(budget)} budget"
    if percent == 0:
        return f"€{price:.2f} — right at your €{_money(budget)} budget"
    return f"€{price:.2f} — {-percent}% over your €{_money(budget)} budget"


def _origin_line(wine: Wine) -> str:
    region = wine.region or "Spain"
    if wine.wine_type:
        return f"From {region} ({wine.wine_type})"
    return f"From {region}"


def _distance_line(distance_m: float) -> str:
    minutes = max(1, round(distance_m / WALK_SPEED_M_PER_MIN))
    return f"{format_distance(distance_m)} away (about {minutes} min walk)"


def _dish_parts(dish: Dish | DishRule | str, rules: PairingRules | None) -> tuple[str, str]:
    """Return ``(dish_id, label)`` for a dish given as a model, a rule or an id."""
    if isinstance(dish, str):
        if rules is not None and dish in rules.dishes:
            return dish, rules.dishes[dish].label
        return dish, dish.replace("_", " ").capitalize()
    return dish.dish_id, dish.label


def reasons(
    wine: Wine,
    venue: Venue,
    dish: Dish | DishRule | str,
    features: Mapping[str, float],
    prefs: Preferences,
    ctx: ScoringContext | None = None,
) -> tuple[str, ...]:
    """Return the 5-6 short "why this pick" lines of one recommendation.

    In order: price against the budget, the wine's origin (region and type), walking
    distance, the pairing reason, the rating, and — only when the venue's opening hours
    are unknown — a warning to check before going.

    ``dish`` may be a ``Dish``, a ``DishRule`` or a dish id. With ``ctx`` the price and
    distance are exact and the pairing line comes from the pairing engine; without it
    they are recovered from ``features`` (exact for any triple within budget and range).
    """
    rules = ctx.pairing_rules if ctx is not None else None
    dish_id, dish_label = _dish_parts(dish, rules)

    price: float | None = None
    if ctx is not None and ctx.inventory.stocks(venue.venue_id, wine.wine_id):
        price = ctx.inventory.price(venue.venue_id, wine.wine_id)
    if price is None:
        price = (1.0 - features["price_fit"]) * prefs.budget_eur
    if ctx is not None:
        distance_m = ctx.distance_m(venue, prefs)
    else:
        distance_m = (1.0 - features["distance"]) * prefs.max_distance_m

    if rules is not None and dish_id in rules.dishes:
        pairing_line = pairing_reason(wine, dish_id, rules)
    else:
        pairing_line = f"Pairing score {features['pairing']:.2f} with {dish_label.lower()}"

    lines = [
        _price_line(price, prefs.budget_eur),
        _origin_line(wine),
        _distance_line(distance_m),
        pairing_line,
        f"Rated {wine.rating:.1f} by {wine.num_reviews:,} reviewers",
    ]
    if open_state(venue.opening_hours, prefs.at_time) == OpenState.UNKNOWN:
        lines.append("Hours unknown — check before you go")
    return tuple(lines)
