"""Adversarial search: minimax and alpha-beta, plus the "sommelier vs. you" game.

The generic half of this module knows nothing about wine. It is hand-written and
modelled on aima-python ``games.py`` (Russell & Norvig, *Artificial Intelligence:
A Modern Approach*, chapter 5 "Adversarial Search and Games"). Each docstring
names the AIMA function or textbook figure it mirrors.

Differences from aima-python, all deliberate:

* Utilities are always from MAX's point of view and ``to_move`` returns the
  literal ``"MAX"`` or ``"MIN"``, so the search needs no ``player`` argument.
* Both searches return a :class:`SearchResult` holding the value, the best
  action, the whole principal variation and the number of nodes visited
  (result monitoring), instead of only the move.
* Ties are broken by action order: the first action that reaches the best value
  wins. Games must therefore list their actions in a deterministic order.

The wine half (:class:`WineGame`, :func:`analyse`) plays a three-ply zero-sum
game over the solutions the CSP already found (DEVELOPMENT_PLAN.md section 14):

1. the **sommelier (MAX)** picks a venue among the top venues,
2. **you (MIN)** pick a dish that venue has a solution for,
3. the **sommelier (MAX)** picks a wine for that (venue, dish).

The terminal utility is the upsell value
``U = 0.5 * quality + 0.5 * min(1, price_at_venue / budget)``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from agent.models import Candidate

Player = Literal["MAX", "MIN"]

MAX: Player = "MAX"
MIN: Player = "MIN"

#: Default number of venues the sommelier may choose from.
DEFAULT_MAX_VENUES = 8


# --------------------------------------------------------------------------- #
# Generic game interface and search results
# --------------------------------------------------------------------------- #


class Game(ABC):
    """A two-player, zero-sum, perfect-information game.

    Mirrors aima-python ``games.Game`` (AIMA section 5.1.1: initial state,
    ``ACTIONS``, ``RESULT``, ``IS-TERMINAL``, ``UTILITY``, ``TO-MOVE``).
    Subclasses must provide an ``initial`` state. ``actions`` must return the
    legal actions in a deterministic order, because ties are broken by it.
    """

    #: The state the game starts in.
    initial: Any

    @abstractmethod
    def actions(self, state: Any) -> Sequence[Any]:
        """Return the legal actions in ``state``, in a deterministic order."""

    @abstractmethod
    def result(self, state: Any, action: Any) -> Any:
        """Return the state reached by playing ``action`` in ``state``."""

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Return ``True`` when the game is over in ``state``."""

    @abstractmethod
    def utility(self, state: Any) -> float:
        """Return the payoff of terminal ``state`` from MAX's point of view."""

    @abstractmethod
    def to_move(self, state: Any) -> Player:
        """Return whose turn it is in ``state``: ``"MAX"`` or ``"MIN"``."""


@dataclass(frozen=True)
class SearchResult:
    """Outcome of a game-tree search from one state."""

    value: float  # game-theoretic value of the searched state (MAX's view)
    best_action: Any  # first action of the optimal line; None at a terminal
    principal_variation: list[Any]  # actions along the optimal line of play
    nodes_visited: int  # states the search was called on, root included


class _Counter:
    """Mutable node counter shared by one search."""

    __slots__ = ("nodes",)

    def __init__(self) -> None:
        self.nodes = 0


def _result(value: float, line: tuple[Any, ...], counter: _Counter) -> SearchResult:
    return SearchResult(
        value=value,
        best_action=line[0] if line else None,
        principal_variation=list(line),
        nodes_visited=counter.nodes,
    )


def _check_has_actions(actions: Sequence[Any], state: Any) -> None:
    if not actions:
        raise ValueError(f"non-terminal state {state!r} has no actions")


# --------------------------------------------------------------------------- #
# Minimax
# --------------------------------------------------------------------------- #


def minimax(game: Game, state: Any = None) -> SearchResult:
    """Search the whole game tree and return the minimax decision.

    Mirrors aima-python ``games.minmax_decision`` / ``minimax_search``
    (AIMA figure 5.3, ``MINIMAX-SEARCH`` with ``MAX-VALUE`` and ``MIN-VALUE``).

    Args:
        game: the game to search.
        state: the state to search from; ``None`` means ``game.initial``.

    Returns:
        The value of ``state``, the best action for the player to move, the
        principal variation and the number of nodes visited.
    """
    counter = _Counter()
    start = game.initial if state is None else state
    value, line = _minimax_value(game, start, counter)
    return _result(value, line, counter)


def _minimax_value(game: Game, state: Any, counter: _Counter) -> tuple[float, tuple[Any, ...]]:
    """Return ``(value, optimal line)`` of ``state`` (AIMA figure 5.3)."""
    counter.nodes += 1
    if game.is_terminal(state):
        return game.utility(state), ()
    actions = game.actions(state)
    _check_has_actions(actions, state)
    maximising = game.to_move(state) == MAX
    best_value = -math.inf if maximising else math.inf
    best_line: tuple[Any, ...] | None = None
    for action in actions:
        value, line = _minimax_value(game, game.result(state, action), counter)
        better = value > best_value if maximising else value < best_value
        if best_line is None or better:
            best_value, best_line = value, (action, *line)
    assert best_line is not None
    return best_value, best_line


# --------------------------------------------------------------------------- #
# Alpha-beta
# --------------------------------------------------------------------------- #


def alphabeta(game: Game, state: Any = None) -> SearchResult:
    """Search the game tree with alpha-beta pruning.

    Mirrors aima-python ``games.alpha_beta_search`` (AIMA figure 5.7,
    ``ALPHA-BETA-SEARCH``). Returns the same value, best action and principal
    variation as :func:`minimax` while visiting no more nodes: a branch is cut
    as soon as it cannot change the decision (``v >= beta`` at a MAX node,
    ``v <= alpha`` at a MIN node).

    Args:
        game: the game to search.
        state: the state to search from; ``None`` means ``game.initial``.
    """
    counter = _Counter()
    start = game.initial if state is None else state
    value, line = _alphabeta_value(game, start, -math.inf, math.inf, counter)
    return _result(value, line, counter)


def _alphabeta_value(
    game: Game, state: Any, alpha: float, beta: float, counter: _Counter
) -> tuple[float, tuple[Any, ...]]:
    """Return ``(value, line)`` of ``state`` within the window (AIMA figure 5.7).

    ``alpha`` is the best value MAX can already guarantee on the path to
    ``state`` and ``beta`` the best MIN can. The value is exact when it lies
    strictly inside ``(alpha, beta)``; otherwise it is only a bound, which is
    all the caller needs.
    """
    counter.nodes += 1
    if game.is_terminal(state):
        return game.utility(state), ()
    actions = game.actions(state)
    _check_has_actions(actions, state)
    maximising = game.to_move(state) == MAX
    best_value = -math.inf if maximising else math.inf
    best_line: tuple[Any, ...] | None = None
    for action in actions:
        value, line = _alphabeta_value(game, game.result(state, action), alpha, beta, counter)
        better = value > best_value if maximising else value < best_value
        if best_line is None or better:
            best_value, best_line = value, (action, *line)
        if maximising:
            if best_value >= beta:
                break  # MIN above will never let play reach this node
            alpha = max(alpha, best_value)
        else:
            if best_value <= alpha:
                break  # MAX above already has something at least as good
            beta = min(beta, best_value)
    assert best_line is not None
    return best_value, best_line


# --------------------------------------------------------------------------- #
# The wine game
# --------------------------------------------------------------------------- #

#: A WineGame state is the tuple of choices made so far:
#: ``()`` -> ``(venue_id,)`` -> ``(venue_id, dish_id)`` -> ``(venue_id, dish_id, wine_id)``.
WineState = tuple[str, ...]


def upsell_utility(candidate: Candidate, budget_eur: float) -> float:
    """Return ``0.5 * quality + 0.5 * min(1, price_at_venue / budget_eur)``.

    ``quality`` is read from ``candidate.features`` (0 if absent). This is the
    terminal utility of :class:`WineGame` (AIMA section 5.1.1, ``UTILITY``).
    """
    quality = candidate.features.get("quality", 0.0)
    return 0.5 * quality + 0.5 * min(1.0, candidate.price_at_venue / budget_eur)


class WineGame(Game):
    """The three-ply "sommelier vs. you" game over CSP solutions.

    Plies: MAX (sommelier) picks a venue, MIN (you) picks a dish, MAX picks a
    wine. Only the top ``max_venues`` venues are playable; venues are ranked by
    the ``score`` of their best candidate, ties broken by ``venue_id``. Actions
    are ids in sorted order. If the same (wine, venue, dish) triple occurs more
    than once, the first occurrence is used.

    Args:
        candidates: CSP solutions, e.g. ``RecommendationResult.candidates``.
        budget_eur: the user's budget in euros; must be positive.
        max_venues: how many venues the sommelier may choose from (>= 1).

    Raises:
        ValueError: if ``budget_eur`` is not positive or ``max_venues`` < 1.
    """

    def __init__(
        self,
        candidates: Sequence[Candidate],
        budget_eur: float,
        max_venues: int = DEFAULT_MAX_VENUES,
    ) -> None:
        if not budget_eur > 0:
            raise ValueError(f"budget_eur must be positive, got {budget_eur!r}")
        if max_venues < 1:
            raise ValueError(f"max_venues must be at least 1, got {max_venues!r}")
        self.budget_eur = float(budget_eur)
        self.max_venues = max_venues
        self.initial: WineState = ()

        best_score: dict[str, float] = {}
        for cand in candidates:
            if cand.venue_id not in best_score or cand.score > best_score[cand.venue_id]:
                best_score[cand.venue_id] = cand.score
        ranked = sorted(best_score, key=lambda venue_id: (-best_score[venue_id], venue_id))
        #: Playable venues, sorted by id (the order MAX tries them in).
        self.venues: tuple[str, ...] = tuple(sorted(ranked[:max_venues]))

        playable = set(self.venues)
        self._tree: dict[str, dict[str, dict[str, Candidate]]] = {v: {} for v in self.venues}
        for cand in candidates:
            if cand.venue_id in playable:
                wines = self._tree[cand.venue_id].setdefault(cand.dish_id, {})
                wines.setdefault(cand.wine_id, cand)

    def actions(self, state: WineState) -> list[str]:
        """Return sorted venue ids, dish ids or wine ids, depending on the ply."""
        if len(state) == 0:
            return list(self.venues)
        if len(state) == 1:
            return sorted(self._tree[state[0]])
        if len(state) == 2:
            return sorted(self._tree[state[0]][state[1]])
        return []

    def result(self, state: WineState, action: str) -> WineState:
        """Return ``state`` extended with the chosen id."""
        return (*state, action)

    def is_terminal(self, state: WineState) -> bool:
        """Return ``True`` once venue, dish and wine are all chosen."""
        return len(state) == 3

    def utility(self, state: WineState) -> float:
        """Return the upsell value of the chosen candidate (see :func:`upsell_utility`)."""
        return upsell_utility(self.candidate(state), self.budget_eur)

    def to_move(self, state: WineState) -> Player:
        """Return ``"MIN"`` at the dish ply, ``"MAX"`` otherwise."""
        return MIN if len(state) == 1 else MAX

    def candidate(self, state: WineState) -> Candidate:
        """Return the :class:`Candidate` a terminal ``state`` stands for."""
        venue_id, dish_id, wine_id = state
        return self._tree[venue_id][dish_id][wine_id]


# --------------------------------------------------------------------------- #
# Analysis for the UI
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VenueLine:
    """Optimal play after the sommelier commits to one venue."""

    venue_id: str
    value: float  # minimax value of the venue: the worst-case upsell
    dish_id: str  # the dish MIN (you) chooses
    wine: Candidate  # the wine MAX (sommelier) then chooses
    price: float  # ``wine.price_at_venue``


@dataclass(frozen=True)
class SommelierAnalysis:
    """Everything the "sommelier vs. you" tab shows."""

    venues: list[VenueLine]  # one row per playable venue, lowest value first
    sommelier_proof: VenueLine  # lowest value; ties broken by venue_id
    sommelier_favourite: VenueLine  # highest value; ties broken by venue_id
    value: float  # minimax value of the whole game
    principal_variation: list[str]  # [venue_id, dish_id, wine_id]
    minimax_nodes: int  # nodes minimax visits over the whole game
    alphabeta_nodes: int  # nodes alpha-beta visits over the whole game
    percent_saved: float  # 100 * (minimax - alphabeta) / minimax


def analyse(
    candidates: Sequence[Candidate],
    budget_eur: float,
    max_venues: int = DEFAULT_MAX_VENUES,
) -> SommelierAnalysis | None:
    """Solve the wine game and summarise it for the UI.

    Args:
        candidates: CSP solutions, e.g. ``RecommendationResult.candidates``.
        budget_eur: the user's budget in euros; must be positive.
        max_venues: how many venues the sommelier may choose from.

    Returns:
        ``None`` when there are no candidates; otherwise per-venue optimal
        lines, the "sommelier-proof" venue, the sommelier's favourite, and the
        node counts of minimax and alpha-beta over the whole game.
    """
    if not candidates:
        return None
    game = WineGame(candidates, budget_eur, max_venues)

    lines: list[VenueLine] = []
    for venue_id in game.venues:
        sub = minimax(game, (venue_id,))
        dish_id, wine_id = sub.principal_variation
        wine = game.candidate((venue_id, dish_id, wine_id))
        lines.append(VenueLine(venue_id, sub.value, dish_id, wine, wine.price_at_venue))
    lines.sort(key=lambda line: (line.value, line.venue_id))

    full = minimax(game)
    pruned = alphabeta(game)
    saved = 100.0 * (full.nodes_visited - pruned.nodes_visited) / full.nodes_visited
    return SommelierAnalysis(
        venues=lines,
        sommelier_proof=lines[0],
        sommelier_favourite=min(lines, key=lambda line: (-line.value, line.venue_id)),
        value=full.value,
        principal_variation=full.principal_variation,
        minimax_nodes=full.nodes_visited,
        alphabeta_nodes=pruned.nodes_visited,
        percent_saved=saved,
    )
