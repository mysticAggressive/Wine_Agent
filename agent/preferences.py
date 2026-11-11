"""User preferences and their validation. See DEVELOPMENT_PLAN.md section 4.3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from agent.config import CENTER, FEATURES
from agent.models import Style, VenueKind

MIN_BODY, MAX_BODY = 2, 5
MIN_DISTANCE_M, MAX_DISTANCE_M = 200.0, 4000.0
MIN_WEIGHT, MAX_WEIGHT = 0.0, 10.0
MIN_TOP_K, MAX_TOP_K = 1, 50
MAX_USER_OFFSET_M = 4500.0
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class Preferences:
    """Everything the user can set in the preference panel."""

    styles: frozenset[Style]  # empty = any style
    budget_eur: float  # > 0
    venue_kinds: frozenset[VenueKind]  # empty = any kind
    dish_id: str | None  # None = let the agent choose the dish
    want_body: int  # 2–5
    user_lat: float
    user_lon: float
    max_distance_m: float  # 200–4000
    at_time: datetime  # timezone-aware, Europe/Madrid
    open_only: bool  # apply the open_now constraint
    exclude_unknown_hours: bool
    weights: dict[str, float]  # keys = the five feature names
    top_k: int = 10


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def default_preferences(now: datetime) -> Preferences:
    """Return the defaults of the preference panel, evaluated at time ``now``."""
    return Preferences(
        styles=frozenset(),
        budget_eur=50.0,
        venue_kinds=frozenset(),
        dish_id=None,
        want_body=4,
        user_lat=CENTER[0],
        user_lon=CENTER[1],
        max_distance_m=2000.0,
        at_time=now,
        open_only=True,
        exclude_unknown_hours=False,
        weights={name: 5.0 for name in FEATURES},
        top_k=10,
    )


def _is_number(value: object) -> bool:
    """True for a finite int or float (bool excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(prefs: Preferences) -> list[str]:
    """Return a list of problems with ``prefs``; the list is empty when it is valid."""
    problems: list[str] = []

    if not _is_number(prefs.budget_eur) or prefs.budget_eur <= 0:
        problems.append("budget_eur must be greater than 0")

    if not _is_number(prefs.want_body) or not MIN_BODY <= prefs.want_body <= MAX_BODY:
        problems.append(f"want_body must be between {MIN_BODY} and {MAX_BODY}")

    if (
        not _is_number(prefs.max_distance_m)
        or not MIN_DISTANCE_M <= prefs.max_distance_m <= MAX_DISTANCE_M
    ):
        problems.append(
            f"max_distance_m must be between {MIN_DISTANCE_M:.0f} and {MAX_DISTANCE_M:.0f}"
        )

    at_time = prefs.at_time
    if not isinstance(at_time, datetime) or at_time.tzinfo is None or at_time.utcoffset() is None:
        problems.append("at_time must be a timezone-aware datetime")

    if not isinstance(prefs.weights, dict) or set(prefs.weights) != set(FEATURES):
        problems.append(f"weights must have exactly these keys: {', '.join(FEATURES)}")
    else:
        for name in FEATURES:
            value = prefs.weights[name]
            if not _is_number(value) or not MIN_WEIGHT <= value <= MAX_WEIGHT:
                problems.append(
                    f"weight '{name}' must be between {MIN_WEIGHT:.0f} and {MAX_WEIGHT:.0f}"
                )

    if (
        not isinstance(prefs.top_k, int)
        or isinstance(prefs.top_k, bool)
        or not MIN_TOP_K <= prefs.top_k <= MAX_TOP_K
    ):
        problems.append(f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}")

    if not _is_number(prefs.user_lat) or not _is_number(prefs.user_lon):
        problems.append("user position must be numeric")
    elif haversine_m(prefs.user_lat, prefs.user_lon, CENTER[0], CENTER[1]) > MAX_USER_OFFSET_M:
        problems.append(
            f"user position must be within {MAX_USER_OFFSET_M:.0f} m of the city centre"
        )

    return problems
