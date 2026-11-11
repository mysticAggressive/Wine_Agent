"""The wine recommendation problem as a CSP. See DEVELOPMENT_PLAN.md 3.7 and task 5.1.

Three variables (``wine``, ``venue``, ``dish``) and eight hard constraints:

========== ================ ====================================================
id         scope            meaning
========== ================ ====================================================
style      wine             wine style is one the user accepts
venue_type venue            venue kind is one the user asked for
open_now   venue            venue is open at the chosen time
max_distance venue          straight-line distance from the user <= the limit
stocks     wine, venue      the venue carries the wine
budget     wine, venue      the wine's price at that venue <= the budget
serves     venue, dish      the venue can serve the dish
pairs      wine, dish       pairing score >= ``MIN_PAIRING_SCORE``
========== ================ ====================================================

The binary constraints never loop over a whole domain: each answers "which values of
the other variable go with this one?" from an index (the inventory, the dishes each
venue serves, the wines that suit each dish), so AC-3 and forward checking stay fast
with ~2,000 wines and ~3,000 venues.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import datetime
from typing import Any

from agent.config import MIN_PAIRING_SCORE
from agent.csp_solver import CSP, Assignment, Constraint, ValueKey
from agent.data_loader import AgentData, Inventory, PairingRules
from agent.models import OpenState, Style, Venue, VenueKind, Wine
from agent.opening import open_state
from agent.pairing import dishes_for_venue, pairing_score
from agent.preferences import Preferences, haversine_m

WINE = "wine"
VENUE = "venue"
DISH = "dish"
VARIABLES: tuple[str, ...] = (WINE, VENUE, DISH)

STYLE = "style"
VENUE_TYPE = "venue_type"
OPEN_NOW = "open_now"
MAX_DISTANCE = "max_distance"
STOCKS = "stocks"
BUDGET = "budget"
SERVES = "serves"
PAIRS = "pairs"
#: The eight constraint ids, in the order they are given to the solver.
CONSTRAINT_IDS: tuple[str, ...] = (
    STYLE,
    VENUE_TYPE,
    OPEN_NOW,
    MAX_DISTANCE,
    STOCKS,
    BUDGET,
    SERVES,
    PAIRS,
)
#: Label of a venue AC-3 removed when no single binary constraint can be named.
ARC_CONSISTENCY = "arc_consistency"

_EMPTY: frozenset[str] = frozenset()


# --------------------------------------------------------------------------- #
# Unary constraints
# --------------------------------------------------------------------------- #


class StyleConstraint(Constraint):
    """``style``: the wine's style is one the user accepts (empty set = any style)."""

    def __init__(self, styles: Collection[Style], wines: Mapping[str, Wine]) -> None:
        super().__init__(STYLE, (WINE,))
        self.styles = frozenset(styles)
        self._wines = wines

    def satisfied(self, assignment: Assignment) -> bool:
        """Return whether the assigned wine has an accepted style."""
        return not self.styles or self._wines[assignment[WINE]].style in self.styles


class VenueTypeConstraint(Constraint):
    """``venue_type``: the venue's kind is one the user asked for (empty set = any)."""

    def __init__(self, kinds: Collection[VenueKind], venues: Mapping[str, Venue]) -> None:
        super().__init__(VENUE_TYPE, (VENUE,))
        self.kinds = frozenset(kinds)
        self._venues = venues

    def satisfied(self, assignment: Assignment) -> bool:
        """Return whether the assigned venue is of an accepted kind."""
        return not self.kinds or self._venues[assignment[VENUE]].kind in self.kinds


class OpenNowConstraint(Constraint):
    """``open_now``: the venue is open at ``at_time``.

    A venue whose hours are unknown passes unless ``exclude_unknown`` is set (plan 3.6).
    """

    def __init__(
        self, at_time: datetime, exclude_unknown: bool, venues: Mapping[str, Venue]
    ) -> None:
        super().__init__(OPEN_NOW, (VENUE,))
        self.at_time = at_time
        self.exclude_unknown = exclude_unknown
        self._venues = venues
        self._states: dict[str, OpenState] = {}

    def state(self, venue_id: str) -> OpenState:
        """Return (and cache) the opening state of a venue at ``at_time``."""
        state = self._states.get(venue_id)
        if state is None:
            state = open_state(self._venues[venue_id].opening_hours, self.at_time)
            self._states[venue_id] = state
        return state

    def satisfied(self, assignment: Assignment) -> bool:
        """Return whether the assigned venue counts as open."""
        state = self.state(assignment[VENUE])
        if state == OpenState.OPEN:
            return True
        return state == OpenState.UNKNOWN and not self.exclude_unknown


class MaxDistanceConstraint(Constraint):
    """``max_distance``: the venue is within ``max_distance_m`` of the user.

    ``distances`` maps venue id -> straight-line metres from the user; the same dict is
    reused by scoring.
    """

    def __init__(self, max_distance_m: float, distances: Mapping[str, float]) -> None:
        super().__init__(MAX_DISTANCE, (VENUE,))
        self.max_distance_m = max_distance_m
        self.distances = distances

    def satisfied(self, assignment: Assignment) -> bool:
        """Return whether the assigned venue is close enough."""
        return self.distances[assignment[VENUE]] <= self.max_distance_m


# --------------------------------------------------------------------------- #
# Binary constraints, index-backed
# --------------------------------------------------------------------------- #


class IndexedConstraint(Constraint):
    """A binary constraint answered from an index instead of a loop over a domain.

    Subclasses implement :meth:`partners`: every value of the other variable that is
    compatible with ``var=value``. ``has_support`` (AC-3), ``supported_values``
    (forward checking) and ``pair_ok`` are then set operations on that index.

    ``unsupported`` records, per variable, the values for which ``has_support``
    answered no. AC-3 removes exactly those values and counts them against this
    constraint, so the record says *which* constraint ruled a value out.
    """

    def __init__(self, id: str, scope: tuple[str, str]) -> None:  # noqa: A002
        super().__init__(id, scope)
        self.unsupported: dict[str, set[str]] = {scope[0]: set(), scope[1]: set()}

    def partners(self, var: str, value: str) -> Collection[str]:
        """Return all values of the other variable compatible with ``var=value``."""
        raise NotImplementedError

    def satisfied(self, assignment: Assignment) -> bool:
        """Return whether the two assigned values are compatible."""
        a, b = self.scope
        return assignment[b] in self.partners(a, assignment[a])

    def pair_ok(self, var: str, value: Any, other_value: Any) -> bool:
        """Return whether ``other_value`` is a partner of ``var=value``."""
        return other_value in self.partners(var, value)

    def has_support(self, var: str, value: Any, other_domain: Collection) -> bool:
        """Return whether some partner of ``var=value`` is in ``other_domain`` (AC-3)."""
        partners = self.partners(var, value)
        if isinstance(other_domain, (set, frozenset)):
            supported = not other_domain.isdisjoint(partners)
        else:
            supported = any(other in partners for other in other_domain)
        if not supported:
            self.unsupported[var].add(value)
        return supported

    def supported_values(self, var: str, value: Any, other_domain: Collection) -> set:
        """Return the partners of ``var=value`` that are in ``other_domain``."""
        partners = self.partners(var, value)
        if isinstance(other_domain, (set, frozenset)):
            return set(other_domain.intersection(partners))
        return {other for other in other_domain if other in partners}


class StocksConstraint(IndexedConstraint):
    """``stocks``: the venue carries the wine. Backed by the ``Inventory`` index."""

    def __init__(self, inventory: Inventory) -> None:
        super().__init__(STOCKS, (WINE, VENUE))
        self._inventory = inventory

    def partners(self, var: str, value: str) -> Collection[str]:
        """Venues that stock a wine, or wines stocked by a venue."""
        if var == WINE:
            return self._inventory.venues_with(value)
        return self._inventory.wines_at(value)


class BudgetConstraint(IndexedConstraint):
    """``budget``: the wine's price at that venue is within the budget.

    A price exists only where the venue stocks the wine, so a pair that is not in the
    inventory does not satisfy this constraint either. ``stocks`` comes first in the
    constraint list, so the diagnostics count "not stocked" against ``stocks`` and "too
    expensive" against ``budget``.
    """

    def __init__(self, budget_eur: float, inventory: Inventory) -> None:
        super().__init__(BUDGET, (WINE, VENUE))
        self.budget_eur = budget_eur
        self._inventory = inventory
        self._venues_for_wine: dict[str, frozenset[str]] = {}
        self._wines_for_venue: dict[str, frozenset[str]] = {}

    def partners(self, var: str, value: str) -> Collection[str]:
        """Venues where a wine is affordable, or affordable wines at a venue (cached)."""
        inventory, budget = self._inventory, self.budget_eur
        if var == WINE:
            found = self._venues_for_wine.get(value)
            if found is None:
                found = frozenset(
                    venue_id
                    for venue_id in inventory.venues_with(value)
                    if inventory.price(venue_id, value) <= budget
                )
                self._venues_for_wine[value] = found
            return found
        found = self._wines_for_venue.get(value)
        if found is None:
            found = frozenset(
                wine_id
                for wine_id in inventory.wines_at(value)
                if inventory.price(value, wine_id) <= budget
            )
            self._wines_for_venue[value] = found
        return found


class ServesConstraint(IndexedConstraint):
    """``serves``: the venue can serve the dish (``dishes_for_venue``, precomputed)."""

    def __init__(
        self,
        dishes_by_venue: Mapping[str, frozenset[str]],
        venues_by_dish: Mapping[str, frozenset[str]],
    ) -> None:
        super().__init__(SERVES, (VENUE, DISH))
        self._dishes_by_venue = dishes_by_venue
        self._venues_by_dish = venues_by_dish

    def partners(self, var: str, value: str) -> Collection[str]:
        """Dishes a venue serves, or venues that serve a dish."""
        if var == VENUE:
            return self._dishes_by_venue.get(value, _EMPTY)
        return self._venues_by_dish.get(value, _EMPTY)


class PairsConstraint(IndexedConstraint):
    """``pairs``: the wine suits the dish (score >= ``MIN_PAIRING_SCORE``, precomputed)."""

    def __init__(
        self,
        dishes_by_wine: Mapping[str, frozenset[str]],
        wines_by_dish: Mapping[str, frozenset[str]],
    ) -> None:
        super().__init__(PAIRS, (WINE, DISH))
        self._dishes_by_wine = dishes_by_wine
        self._wines_by_dish = wines_by_dish

    def partners(self, var: str, value: str) -> Collection[str]:
        """Dishes a wine suits, or wines that suit a dish."""
        if var == WINE:
            return self._dishes_by_wine.get(value, _EMPTY)
        return self._wines_by_dish.get(value, _EMPTY)


# --------------------------------------------------------------------------- #
# Indexes that depend only on the data, not on the preferences
# --------------------------------------------------------------------------- #

_Index = dict[str, frozenset[str]]
# Last (wines, venues, rules) seen and the four indexes built from them. The stored
# objects are compared by identity, so the app (which loads the data once) builds the
# indexes once instead of on every preference change.
_static_cache: tuple[Any, Any, Any, tuple[_Index, _Index, _Index, _Index]] | None = None


def _invert(index: Mapping[str, Collection[str]]) -> _Index:
    """Turn ``key -> values`` into ``value -> keys``."""
    inverse: dict[str, set[str]] = {}
    for key, values in index.items():
        for value in values:
            inverse.setdefault(value, set()).add(key)
    return {value: frozenset(keys) for value, keys in inverse.items()}


def _static_indexes(data: AgentData) -> tuple[_Index, _Index, _Index, _Index]:
    """Return ``(dishes_by_venue, venues_by_dish, dishes_by_wine, wines_by_dish)``."""
    global _static_cache
    cached = _static_cache
    if (
        cached is not None
        and cached[0] is data.wines
        and cached[1] is data.venues
        and cached[2] is data.pairing_rules
    ):
        return cached[3]
    rules: PairingRules = data.pairing_rules
    dishes_by_venue = {
        venue_id: dishes_for_venue(venue, rules) for venue_id, venue in data.venues.items()
    }
    dish_ids = tuple(rules.dishes)
    dishes_by_wine = {
        wine_id: frozenset(
            dish_id
            for dish_id in dish_ids
            if pairing_score(wine, dish_id, rules) >= MIN_PAIRING_SCORE
        )
        for wine_id, wine in data.wines.items()
    }
    indexes = (
        dishes_by_venue,
        _invert(dishes_by_venue),
        dishes_by_wine,
        _invert(dishes_by_wine),
    )
    _static_cache = (data.wines, data.venues, data.pairing_rules, indexes)
    return indexes


# --------------------------------------------------------------------------- #
# The CSP
# --------------------------------------------------------------------------- #


class WineCSP(CSP):
    """The wine CSP plus what the recommender needs around it.

    * ``prefs``, ``data`` — what it was built from.
    * ``distances`` — venue id -> straight-line metres from the user, for every venue
      (also when the ``max_distance`` constraint was left out); scoring reuses it.
    * ``unary_venue_exclusions`` — venue id -> id of the first unary constraint, in
      solver order, that rejects the venue. This is exactly the attribution rule of the
      solver's node-consistency step.
    """

    def __init__(
        self,
        domains: dict[str, Collection[str]],
        constraints: list[Constraint],
        value_key: ValueKey | None,
        prefs: Preferences,
        data: AgentData,
        distances: dict[str, float],
    ) -> None:
        super().__init__(domains, constraints, value_key=value_key)
        self.prefs = prefs
        self.data = data
        self.distances = distances
        self._all_venues: frozenset[str] = frozenset(self.domains[VENUE])
        self.unary_venue_exclusions: dict[str, str] = {}
        unary = self.unary_constraints(VENUE)
        for venue_id in sorted(self._all_venues):
            for constraint in unary:
                if not constraint.satisfied({VENUE: venue_id}):
                    self.unary_venue_exclusions[venue_id] = constraint.id
                    break

    def venue_exclusions(self) -> dict[str, str]:
        """Return ``venue id -> constraint id`` for every venue that was ruled out.

        Call it after ``solve_all``. Venues rejected by a unary constraint carry that
        constraint's id. Venues that passed those but are no longer in the ``venue``
        domain were removed by AC-3; they carry the id of the binary constraint that
        left them without support (``stocks``, ``budget`` or ``serves``), or
        ``"arc_consistency"`` if none can be named. Keys are in sorted order. Venues
        that merely appear in no solution are not listed.
        """
        exclusions: dict[str, str] = {}
        remaining = self.domains[VENUE]
        binary = [c for c in self.binary_constraints(VENUE) if isinstance(c, IndexedConstraint)]
        for venue_id in sorted(self._all_venues):
            unary_id = self.unary_venue_exclusions.get(venue_id)
            if unary_id is not None:
                exclusions[venue_id] = unary_id
            elif venue_id not in remaining:
                exclusions[venue_id] = next(
                    (c.id for c in binary if venue_id in c.unsupported[VENUE]), ARC_CONSISTENCY
                )
        return exclusions


def _make_value_key(inventory: Inventory) -> ValueKey:
    """Build the cheap LCV stand-in of task 4.3.

    A wine carried by many in-domain venues (or a venue carrying many in-domain wines)
    rules out the fewest values of the other variable, so it is tried first. Counts are
    cached for as long as the other variable's domain is the same set object, which is
    the case for all the values ordered at one node of the search and usually for its
    sibling nodes too. Dishes keep their canonical order.
    """
    cache: dict[str, tuple[set, int, dict[str, int]]] = {}

    def value_key(csp: CSP, var: str, value: str) -> int:
        if var == WINE:
            other, index = VENUE, inventory.venues_with
        elif var == VENUE:
            other, index = WINE, inventory.wines_at
        else:
            return 0
        domain = csp.domains[other]
        entry = cache.get(var)
        if entry is None or entry[0] is not domain or entry[1] != len(domain):
            entry = (domain, len(domain), {})
            cache[var] = entry
        counts = entry[2]
        count = counts.get(value)
        if count is None:
            count = counts[value] = len(domain.intersection(index(value)))
        return -count

    return value_key


def user_distances(prefs: Preferences, venues: Mapping[str, Venue]) -> dict[str, float]:
    """Return venue id -> straight-line distance in metres from the user's position."""
    lat, lon = prefs.user_lat, prefs.user_lon
    return {
        venue_id: haversine_m(lat, lon, venue.lat, venue.lon) for venue_id, venue in venues.items()
    }


def build_wine_csp(prefs: Preferences, data: AgentData, skip: Collection[str] = ()) -> WineCSP:
    """Build the wine CSP of plan 3.7 from the user's preferences and the data.

    Variables are declared in the order ``wine``, ``venue``, ``dish`` and constraints in
    the order of ``CONSTRAINT_IDS``. ``open_now`` is left out when ``prefs.open_only`` is
    false. ``skip`` names further constraint ids to leave out; the recommender uses it
    to test which constraint to relax when there is no solution.

    If ``prefs.dish_id`` is set the ``dish`` domain is that single dish (empty if the
    id is not a known dish). The solver's propagation shrinks the domains for good, so
    build a fresh CSP for every solve.
    """
    skipped = frozenset(skip)
    unknown = skipped - set(CONSTRAINT_IDS)
    if unknown:
        raise ValueError(f"unknown constraint ids to skip: {', '.join(sorted(unknown))}")

    rules = data.pairing_rules
    inventory = data.inventory
    distances = user_distances(prefs, data.venues)
    dishes_by_venue, venues_by_dish, dishes_by_wine, wines_by_dish = _static_indexes(data)

    if prefs.dish_id is None:
        dish_domain = set(rules.dishes)
    else:
        dish_domain = {prefs.dish_id} & set(rules.dishes)
    domains: dict[str, Collection[str]] = {
        WINE: set(data.wines),
        VENUE: set(data.venues),
        DISH: dish_domain,
    }

    candidates: list[Constraint | None] = [
        StyleConstraint(prefs.styles, data.wines),
        VenueTypeConstraint(prefs.venue_kinds, data.venues),
        OpenNowConstraint(prefs.at_time, prefs.exclude_unknown_hours, data.venues)
        if prefs.open_only
        else None,
        MaxDistanceConstraint(prefs.max_distance_m, distances),
        StocksConstraint(inventory),
        BudgetConstraint(prefs.budget_eur, inventory),
        ServesConstraint(dishes_by_venue, venues_by_dish),
        PairsConstraint(dishes_by_wine, wines_by_dish),
    ]
    constraints = [c for c in candidates if c is not None and c.id not in skipped]
    return WineCSP(domains, constraints, _make_value_key(inventory), prefs, data, distances)
