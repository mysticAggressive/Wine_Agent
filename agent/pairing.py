"""Wine-food pairing engine. See DEVELOPMENT_PLAN.md task 3.2.

The rules themselves live in ``data/pairing_rules.json`` (a derived layer, built from
standard sommelier guidance). This module only evaluates them:

* ``pairing_score`` — how well a wine suits a dish, in ``[0, 1]``;
* ``pairing_reason`` — one human-readable sentence for the "why this pick" popup;
* ``dishes_for_venue`` — which dishes a venue can serve, from its cuisine tags.
"""

from __future__ import annotations

from agent.data_loader import DishRule, PairingRules
from agent.models import Style, Venue, VenueKind, Wine

BODY_WEIGHT = 0.3
ACIDITY_WEIGHT = 0.2
BASE_WEIGHT = 0.5
TYPE_BONUS = 0.15
GAP_PENALTY = 0.5  # lost per step outside a rule's body / acidity range
ASSUMED_CUISINE = "spanish"  # dish set of a venue with no cuisine tag (plan 3.5)

_BODY_WORDS = {1: "Very light", 2: "Light-bodied", 3: "Medium-bodied", 4: "Full-bodied"}
_STYLE_WORDS = {
    Style.RED: "red",
    Style.WHITE: "white",
    Style.SPARKLING: "sparkling wine",
    Style.FORTIFIED: "fortified wine",
    Style.UNKNOWN: "wine",
}

# (id(rules), wine_id, dish_id) -> (rules, wine, score). The stored objects keep the
# rules alive (so the id cannot be reused) and let a lookup verify it is the same wine.
_SCORE_CACHE: dict[tuple[int, str, str], tuple[PairingRules, Wine, float]] = {}


def clear_pairing_cache() -> None:
    """Forget every cached pairing score (useful after reloading the rules)."""
    _SCORE_CACHE.clear()


def _range_part(value: int, bounds: tuple[int, int]) -> float:
    """1.0 inside the inclusive range, minus 0.5 per step outside it, never below 0."""
    lo, hi = bounds
    if value < lo:
        gap = lo - value
    elif value > hi:
        gap = value - hi
    else:
        return 1.0
    return max(0.0, 1.0 - GAP_PENALTY * gap)


def _score(wine: Wine, rule: DishRule) -> float:
    """Apply the task 3.2 formula to one wine and one dish rule (no caching)."""
    style_part = float(rule.styles.get(wine.style, 0.0))
    if style_part <= 0.0:
        return 0.0
    body_part = _range_part(wine.body, rule.body)
    acidity_part = _range_part(wine.acidity, rule.acidity)
    bonus = TYPE_BONUS if wine.wine_type and wine.wine_type in rule.type_bonus else 0.0
    score = (
        style_part * (BASE_WEIGHT + BODY_WEIGHT * body_part + ACIDITY_WEIGHT * acidity_part) + bonus
    )
    return max(0.0, min(1.0, score))


def pairing_score(wine: Wine, dish_id: str, rules: PairingRules) -> float:
    """Return how well ``wine`` suits the dish, from 0 (no match) to 1 (ideal).

    A wine whose style is absent from the dish's rule scores exactly 0. Results are
    cached per ``(wine_id, dish_id)`` for each rules object. Raises ``KeyError`` if
    ``dish_id`` is not in ``rules``.
    """
    key = (id(rules), wine.wine_id, dish_id)
    cached = _SCORE_CACHE.get(key)
    if cached is not None and cached[0] is rules and (cached[1] is wine or cached[1] == wine):
        return cached[2]
    score = _score(wine, rules.dishes[dish_id])
    _SCORE_CACHE[key] = (rules, wine, score)
    return score


def _describe(wine: Wine) -> str:
    """Short description of a wine, e.g. ``"Full-bodied Ribera del Duero red"``."""
    body_word = _BODY_WORDS.get(wine.body, "Very full-bodied" if wine.body >= 5 else "Very light")
    if not wine.wine_type:
        kind = _STYLE_WORDS.get(wine.style, "wine")
    else:
        kind = wine.wine_type.replace(" Del ", " del ")
        for suffix in ("Red", "White", "Sparkling"):
            if kind == suffix:
                kind = _STYLE_WORDS[Style(suffix.lower())]
            elif kind.endswith(" " + suffix):
                kind = kind[: -len(suffix)] + suffix.lower()
    return f"{body_word} {kind}"


def pairing_reason(wine: Wine, dish_id: str, rules: PairingRules) -> str:
    """Return one sentence explaining the pairing of ``wine`` with the dish.

    Example: ``"Full-bodied Ribera del Duero red — tannin and body stand up to charred,
    fatty meat."`` When the wine does not suit the dish (score 0) the sentence says so
    instead. Raises ``KeyError`` if ``dish_id`` is not in ``rules``.
    """
    rule = rules.dishes[dish_id]
    description = _describe(wine)
    if pairing_score(wine, dish_id, rules) <= 0.0:
        return f"{description} — not a recommended style for {rule.label.lower()}."
    why = rule.why.strip()
    if not why:
        return f"{description} — a good match for {rule.label.lower()}."
    if not why.endswith((".", "!", "?")):
        why += "."
    return f"{description} — {why[0].lower()}{why[1:]}"


def dishes_for_venue(venue: Venue, rules: PairingRules) -> frozenset[str]:
    """Return the ids of the dishes ``venue`` can serve.

    Shops serve every dish (the user cooks at home). Restaurants and bars serve the
    union of the dishes mapped to each of their cuisine tags (unlisted tags use the
    ``_default`` set); a venue with no cuisine tag gets the ``spanish`` set.
    """
    if venue.kind == VenueKind.SHOP:
        return frozenset(rules.dishes)
    cuisines = [c.strip().lower() for c in venue.cuisines if c.strip()] or [ASSUMED_CUISINE]
    served: set[str] = set()
    for cuisine in cuisines:
        served.update(rules.dishes_for_cuisine(cuisine))
    return frozenset(dish_id for dish_id in served if dish_id in rules.dishes)
