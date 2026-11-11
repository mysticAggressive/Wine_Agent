"""The recommender: preferences in, ranked picks and search diagnostics out.

See DEVELOPMENT_PLAN.md task 5.3. Two entry points share one search:

* :func:`solve_candidates` — every CSP solution as a scored ``Candidate``, best first
  (bounded to the best ``limit``); the "sommelier vs. you" game plays over these.
* :func:`recommend` — the same list cut down to ``top_k`` rows with at most
  ``MAX_PER_VENUE`` per venue, which is what the results tab shows.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from agent.config import MAX_PER_VENUE
from agent.csp_solver import Assignment, SearchStats, SolverOptions, solve_all
from agent.data_loader import AgentData
from agent.models import Candidate, Diagnostics, RecommendationResult
from agent.preferences import Preferences, validate
from agent.scoring import Scorer, ScoringContext, evaluate, features, format_distance, reasons
from agent.wine_csp import (
    BUDGET,
    DISH,
    MAX_DISTANCE,
    OPEN_NOW,
    STYLE,
    VENUE,
    VENUE_TYPE,
    WINE,
    build_wine_csp,
)

__all__ = [
    "DEFAULT_CANDIDATE_LIMIT",
    "RELAXABLE",
    "SolverOptions",
    "cap_per_venue",
    "recommend",
    "solve_candidates",
]

#: How many of the best solutions are kept (and ranked) by default.
DEFAULT_CANDIDATE_LIMIT = 20_000
#: Constraints tried, one at a time and in this order, when there is no solution.
RELAXABLE: tuple[str, ...] = (BUDGET, MAX_DISTANCE, OPEN_NOW, STYLE, VENUE_TYPE)
#: Counting the solutions of a relaxed problem stops here ("more than 5,000 options").
HINT_COUNT_CAP = 5_000

# (-score, wine_id, venue_id, dish_id): plain tuple order is the ranking order.
_Ranked = tuple[float, str, str, str]


def _check(prefs: Preferences, data: AgentData, limit: int) -> None:
    """Raise ``ValueError`` listing every problem with the request."""
    problems = validate(prefs)
    if prefs.dish_id is not None and prefs.dish_id not in data.pairing_rules.dishes:
        problems.append(f"unknown dish_id '{prefs.dish_id}'")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        problems.append("limit must be a whole number of at least 1")
    if problems:
        raise ValueError("Invalid preferences: " + "; ".join(problems))


class _BestOf:
    """Keeps the best ``limit`` scored solutions without holding all of them.

    Solutions are buffered; whenever the buffer reaches four times ``limit`` it is
    sorted and cut back to ``limit``, and from then on anything scoring strictly below
    the worst kept solution is dropped on arrival. The outcome is exactly the best
    ``limit`` in ranking order, whatever order the solver finds the solutions in.
    """

    def __init__(self, limit: int, scorer: Scorer) -> None:
        self._limit = limit
        self._capacity = max(4 * limit, limit + 1024)
        self._score = scorer.score
        self._items: list[_Ranked] = []
        self._floor = float("-inf")

    def add(self, assignment: Assignment) -> None:
        """Score one solution (the solver's ``on_solution`` callback) and keep it if it ranks."""
        wine_id, venue_id, dish_id = assignment[WINE], assignment[VENUE], assignment[DISH]
        score = self._score(wine_id, venue_id, dish_id)
        if score < self._floor:
            return
        items = self._items
        items.append((-score, wine_id, venue_id, dish_id))
        if len(items) >= self._capacity:
            items.sort()
            del items[self._limit :]
            self._floor = -items[-1][0]

    def ranked(self) -> list[_Ranked]:
        """Return the kept solutions, best first (score, then wine, venue, dish id)."""
        self._items.sort()
        del self._items[self._limit :]
        return self._items


def _candidate(
    wine_id: str,
    venue_id: str,
    dish_id: str,
    prefs: Preferences,
    data: AgentData,
    ctx: ScoringContext,
) -> Candidate:
    """Turn one solution into a ``Candidate`` with features, score and reasons."""
    wine = data.wines[wine_id]
    venue = data.venues[venue_id]
    feats = features(wine, venue, dish_id, prefs, ctx)
    return Candidate(
        wine_id=wine_id,
        venue_id=venue_id,
        dish_id=dish_id,
        price_at_venue=data.inventory.price(venue_id, wine_id),
        distance_m=ctx.distance_m(venue, prefs),
        features=feats,
        score=evaluate(feats, prefs.weights),
        reasons=reasons(wine, venue, dish_id, feats, prefs, ctx),
    )


def _plural(count: int, capped: bool) -> str:
    if capped:
        return f"more than {HINT_COUNT_CAP:,} options"
    return "1 option" if count == 1 else f"{count:,} options"


def _count_relaxed(
    prefs: Preferences, data: AgentData, constraint_id: str
) -> tuple[int, float, float]:
    """Solve without one constraint; return ``(solutions, cheapest price, nearest metres)``.

    Counting stops at ``HINT_COUNT_CAP`` solutions.
    """
    csp = build_wine_csp(prefs, data, skip=(constraint_id,))
    inventory, distances = data.inventory, csp.distances
    count = 0
    cheapest = nearest = float("inf")

    def note(assignment: Assignment) -> None:
        nonlocal count, cheapest, nearest
        count += 1
        venue_id = assignment[VENUE]
        cheapest = min(cheapest, inventory.price(venue_id, assignment[WINE]))
        nearest = min(nearest, distances[venue_id])

    solve_all(csp, SolverOptions(), on_solution=note, should_stop=lambda: count > HINT_COUNT_CAP)
    return count, cheapest, nearest


def _relaxation_hints(prefs: Preferences, data: AgentData) -> list[str]:
    """Say which single constraint, once removed, would give at least one solution."""
    active = {
        BUDGET: True,
        MAX_DISTANCE: True,
        OPEN_NOW: prefs.open_only,
        STYLE: bool(prefs.styles),
        VENUE_TYPE: bool(prefs.venue_kinds),
    }
    hints: list[str] = []
    for constraint_id in RELAXABLE:
        if not active[constraint_id]:
            continue
        count, cheapest, nearest = _count_relaxed(prefs, data, constraint_id)
        if count == 0:
            continue
        capped = count > HINT_COUNT_CAP
        options = _plural(count, capped)
        if constraint_id == BUDGET:
            detail = "" if capped else f" (cheapest: €{cheapest:.2f})"
            hints.append(f"Removing the budget limit gives {options}{detail}")
        elif constraint_id == MAX_DISTANCE:
            detail = "" if capped else f" (nearest: {format_distance(nearest)})"
            hints.append(f"Removing the distance limit gives {options}{detail}")
        elif constraint_id == OPEN_NOW:
            hints.append(f"Including venues that are closed at that time gives {options}")
        elif constraint_id == STYLE:
            hints.append(f"Accepting any wine style gives {options}")
        else:
            hints.append(f"Accepting any kind of venue gives {options}")
    if not hints:
        hints.append("No single constraint is to blame — try relaxing two or more at once")
    return hints


def _diagnostics(
    stats: SearchStats,
    constraint_ids: Iterable[str],
    elapsed_ms: float,
    hints: list[str],
    venue_exclusions: dict[str, str],
) -> Diagnostics:
    """Copy the solver's statistics into the shared ``Diagnostics`` model."""
    pruned = {constraint_id: 0 for constraint_id in constraint_ids}
    pruned.update(stats.pruned_by_constraint)
    return Diagnostics(
        initial_domain_sizes=dict(stats.initial_domain_sizes),
        pruned_by_constraint=pruned,
        domain_sizes_after_node_consistency=dict(stats.domain_sizes_after_node_consistency),
        domain_sizes_after_ac3=dict(stats.domain_sizes_after_ac3),
        ac3_revisions=stats.ac3_revisions,
        assignments=stats.assignments,
        backtracks=stats.backtracks,
        forward_check_prunes=stats.forward_check_prunes,
        solutions_found=stats.solutions_found,
        variable_order=list(stats.variable_order),
        elapsed_ms=elapsed_ms,
        relaxation_hints=hints,
        venue_exclusions=venue_exclusions,
    )


def _solve_ranked(
    prefs: Preferences, data: AgentData, solver_options: SolverOptions | None, limit: int
) -> tuple[list[_Ranked], Diagnostics, ScoringContext]:
    """Run the search once; return the best ``limit`` solutions, ranked, by id."""
    _check(prefs, data, limit)
    started = time.perf_counter()
    csp = build_wine_csp(prefs, data)
    ctx = ScoringContext.from_data(data, distances=csp.distances)
    best = _BestOf(limit, Scorer(prefs, data.wines, data.venues, ctx))
    _, stats = solve_all(csp, solver_options or SolverOptions(), on_solution=best.add)
    ranked = best.ranked()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    hints = _relaxation_hints(prefs, data) if stats.solutions_found == 0 else []
    diagnostics = _diagnostics(
        stats, [c.id for c in csp.constraints], elapsed_ms, hints, csp.venue_exclusions()
    )
    return ranked, diagnostics, ctx


def solve_candidates(
    prefs: Preferences,
    data: AgentData,
    solver_options: SolverOptions | None = None,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> tuple[list[Candidate], Diagnostics]:
    """Return every solution of the wine CSP as a scored ``Candidate``, best first.

    The list is sorted by score descending, then ``wine_id``, ``venue_id``, ``dish_id``,
    and holds at most the best ``limit`` solutions; ``diagnostics.solutions_found`` is
    the full count. No per-venue cap is applied — that is :func:`recommend`'s job.
    The candidates do not depend on ``solver_options``; only the statistics do.

    Raises ``ValueError`` listing the problems if ``prefs`` (or ``limit``) is invalid.
    """
    ranked, diagnostics, ctx = _solve_ranked(prefs, data, solver_options, limit)
    candidates = [
        _candidate(wine_id, venue_id, dish_id, prefs, data, ctx)
        for _, wine_id, venue_id, dish_id in ranked
    ]
    return candidates, diagnostics


def _capped_indices(pairs: Iterable[tuple[str, str]], top_k: int, max_per_venue: int) -> list[int]:
    """Return the positions, in a ranked sequence of ``(wine_id, venue_id)``, to keep."""
    kept: list[int] = []
    per_venue: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for index, pair in enumerate(pairs):
        if len(kept) >= top_k:
            break
        venue_id = pair[1]
        if pair in seen or per_venue.get(venue_id, 0) >= max_per_venue:
            continue
        seen.add(pair)
        per_venue[venue_id] = per_venue.get(venue_id, 0) + 1
        kept.append(index)
    return kept


def cap_per_venue(
    candidates: Iterable[Candidate], top_k: int, max_per_venue: int = MAX_PER_VENUE
) -> list[Candidate]:
    """Walk a ranked list keeping the first ``top_k`` rows that respect the caps.

    At most ``max_per_venue`` rows per venue, and at most one row per
    ``(wine_id, venue_id)`` — the first met, i.e. the wine's best dish there.
    """
    rows = list(candidates)
    pairs = ((c.wine_id, c.venue_id) for c in rows)
    return [rows[i] for i in _capped_indices(pairs, top_k, max_per_venue)]


def recommend(
    prefs: Preferences,
    data: AgentData,
    solver_options: SolverOptions | None = None,
) -> RecommendationResult:
    """Recommend up to ``prefs.top_k`` (wine, venue, dish) picks, best first.

    Runs the same search as :func:`solve_candidates` and then keeps at most
    ``MAX_PER_VENUE`` rows per venue and one row per (wine, venue). The result equals
    ``cap_per_venue(solve_candidates(prefs, data)[0], prefs.top_k)``; only the kept rows
    are turned into full ``Candidate`` objects. When nothing satisfies the hard
    constraints, ``diagnostics.relaxation_hints`` says which single constraint to relax.

    Raises ``ValueError`` listing the problems if ``prefs`` is invalid.
    """
    ranked, diagnostics, ctx = _solve_ranked(prefs, data, solver_options, DEFAULT_CANDIDATE_LIMIT)
    pairs = ((wine_id, venue_id) for _, wine_id, venue_id, _ in ranked)
    candidates = [
        _candidate(*ranked[i][1:], prefs, data, ctx)
        for i in _capped_indices(pairs, prefs.top_k, MAX_PER_VENUE)
    ]
    return RecommendationResult(candidates=candidates, diagnostics=diagnostics)
