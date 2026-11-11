"""Tests for agent/minimax.py (DEVELOPMENT_PLAN.md task 8.1)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import pytest

from agent.minimax import (
    Game,
    SommelierAnalysis,
    WineGame,
    alphabeta,
    analyse,
    minimax,
    upsell_utility,
)
from agent.models import Candidate

# --------------------------------------------------------------------------- #
# A generic tree game for the textbook and random-tree tests
# --------------------------------------------------------------------------- #


class TreeGame(Game):
    """A game given as nested lists; leaves are numbers; players alternate from MAX.

    A state is the tuple of child indices followed from the root.
    """

    def __init__(self, tree: Any) -> None:
        self.tree = tree
        self.initial: tuple[int, ...] = ()

    def _node(self, state: tuple[int, ...]) -> Any:
        node = self.tree
        for index in state:
            node = node[index]
        return node

    def actions(self, state: tuple[int, ...]) -> Sequence[int]:
        return range(len(self._node(state)))

    def result(self, state: tuple[int, ...], action: int) -> tuple[int, ...]:
        return (*state, action)

    def is_terminal(self, state: tuple[int, ...]) -> bool:
        return not isinstance(self._node(state), list)

    def utility(self, state: tuple[int, ...]) -> float:
        return float(self._node(state))

    def to_move(self, state: tuple[int, ...]) -> str:
        return "MAX" if len(state) % 2 == 0 else "MIN"


#: AIMA figure 5.2: MAX to move, three MIN nodes with three leaves each.
TEXTBOOK = [[3, 12, 8], [2, 4, 6], [14, 5, 2]]


def random_tree(rng: random.Random, depth: int) -> Any:
    """Return a random tree with uneven depth and branching 1-4."""
    if depth == 0 or rng.random() < 0.1:
        return rng.randint(-10, 10)  # small range, so ties are common
    return [random_tree(rng, depth - 1) for _ in range(rng.randint(1, 4))]


# --------------------------------------------------------------------------- #
# Generic search
# --------------------------------------------------------------------------- #


def test_textbook_tree_minimax() -> None:
    result = minimax(TreeGame(TEXTBOOK))
    assert result.value == 3
    assert result.best_action == 0
    assert result.principal_variation == [0, 0]
    assert result.nodes_visited == 13  # 1 root + 3 MIN nodes + 9 leaves


def test_textbook_tree_alphabeta() -> None:
    result = alphabeta(TreeGame(TEXTBOOK))
    assert result.value == 3
    assert result.best_action == 0
    assert result.principal_variation == [0, 0]
    # The textbook prunes the leaves 4 and 6 under the second MIN node.
    assert result.nodes_visited == 11


def test_search_from_a_given_state() -> None:
    game = TreeGame(TEXTBOOK)
    for search in (minimax, alphabeta):
        result = search(game, (2,))  # third MIN node: min(14, 5, 2)
        assert result.value == 2
        assert result.best_action == 2
        assert result.principal_variation == [2]


def test_terminal_state() -> None:
    game = TreeGame(7)
    for search in (minimax, alphabeta):
        result = search(game)
        assert result.value == 7
        assert result.best_action is None
        assert result.principal_variation == []
        assert result.nodes_visited == 1


def test_ties_go_to_the_first_action() -> None:
    game = TreeGame([[5, 9], [5, 7], [5]])
    for search in (minimax, alphabeta):
        result = search(game)
        assert result.value == 5
        assert result.principal_variation == [0, 0]


def test_non_terminal_state_without_actions_is_an_error() -> None:
    for search in (minimax, alphabeta):
        with pytest.raises(ValueError, match="no actions"):
            search(TreeGame([[1], []]))


@pytest.mark.parametrize("seed", range(50))
def test_alphabeta_agrees_with_minimax_on_random_trees(seed: int) -> None:
    rng = random.Random(seed)
    tree = random_tree(rng, depth=rng.randint(2, 5))
    game = TreeGame(tree)
    full = minimax(game)
    pruned = alphabeta(game)
    assert pruned.value == full.value
    assert pruned.best_action == full.best_action
    assert pruned.principal_variation == full.principal_variation
    assert pruned.nodes_visited <= full.nodes_visited


def test_alphabeta_prunes_something_across_the_random_trees() -> None:
    saved = 0
    for seed in range(50):
        rng = random.Random(seed)
        game = TreeGame(random_tree(rng, depth=rng.randint(2, 5)))
        saved += minimax(game).nodes_visited - alphabeta(game).nodes_visited
    assert saved > 0


# --------------------------------------------------------------------------- #
# The wine game
# --------------------------------------------------------------------------- #

BUDGET = 40.0


def cand(venue: str, dish: str, wine: str, quality: float, price: float, score: float) -> Candidate:
    return Candidate(
        wine_id=wine,
        venue_id=venue,
        dish_id=dish,
        price_at_venue=price,
        distance_m=100.0,
        features={"quality": quality},
        score=score,
        reasons=(),
    )


# Hand-worked game, budget 40 EUR, U = 0.5*quality + 0.5*min(1, price/40):
#
#   venue A  dish d1: w1 (q .8, 20 EUR) -> .650   w2 (q .6, 40 EUR) -> .800   MAX: .800
#            dish d2: w3 (q .5, 10 EUR) -> .375                               MAX: .375
#            MIN picks d2                                               value A = .375
#   venue B  dish d1: w1 (q .8, 30 EUR) -> .775                               MAX: .775
#            dish d2: w4 (q 1., 60 EUR) -> 1.00 (price capped)  w7 -> .625    MAX: 1.00
#            MIN picks d1                                               value B = .775
#   venue C  dish d3: w5 (q .4, 20 EUR) -> .450                               MAX: .450
#            dish d4: w6 (q .9, 40 EUR) -> .950                               MAX: .950
#            MIN picks d3                                               value C = .450
#
#   MAX picks B: game value .775, line B -> d1 -> w1.
#
# Minimax visits 1 root + 3 venues + 6 dishes + 8 wines = 18 nodes.
# Alpha-beta skips w7 (w4 already beats beta = .775) and all of C/d4 (C is already
# held to .450 <= alpha = .775): 18 - 1 - 2 = 15 nodes.
CANDIDATES = [
    cand("A", "d1", "w1", 0.8, 20.0, 0.90),
    cand("A", "d1", "w2", 0.6, 40.0, 0.50),
    cand("A", "d2", "w3", 0.5, 10.0, 0.40),
    cand("B", "d1", "w1", 0.8, 30.0, 0.80),
    cand("B", "d2", "w4", 1.0, 60.0, 0.30),
    cand("B", "d2", "w7", 0.5, 30.0, 0.20),
    cand("C", "d3", "w5", 0.4, 20.0, 0.70),
    cand("C", "d4", "w6", 0.9, 40.0, 0.60),
]


def test_upsell_utility() -> None:
    assert upsell_utility(CANDIDATES[0], BUDGET) == pytest.approx(0.65)
    assert upsell_utility(CANDIDATES[4], BUDGET) == pytest.approx(1.0)  # price capped


def test_wine_game_interface() -> None:
    game = WineGame(list(reversed(CANDIDATES)), BUDGET)  # input order must not matter
    assert game.initial == ()
    assert game.to_move(()) == "MAX"
    assert game.actions(()) == ["A", "B", "C"]
    assert game.to_move(("A",)) == "MIN"
    assert game.actions(("A",)) == ["d1", "d2"]
    assert game.to_move(("A", "d1")) == "MAX"
    assert game.actions(("A", "d1")) == ["w1", "w2"]
    assert game.result(("A", "d1"), "w2") == ("A", "d1", "w2")
    assert not game.is_terminal(("A", "d1"))
    assert game.is_terminal(("A", "d1", "w2"))
    assert game.utility(("A", "d1", "w2")) == pytest.approx(0.8)
    assert game.candidate(("B", "d1", "w1")).price_at_venue == 30.0


def test_wine_game_hand_worked_value() -> None:
    game = WineGame(CANDIDATES, BUDGET)
    full = minimax(game)
    pruned = alphabeta(game)
    for result in (full, pruned):
        assert result.value == pytest.approx(0.775)
        assert result.best_action == "B"
        assert result.principal_variation == ["B", "d1", "w1"]
    assert full.nodes_visited == 18
    assert pruned.nodes_visited == 15


def test_analyse_hand_worked() -> None:
    analysis = analyse(CANDIDATES, BUDGET)
    assert isinstance(analysis, SommelierAnalysis)
    # Fewer than 8 venues: all three are used. Rows are sorted by value.
    rows = [(v.venue_id, v.dish_id, v.wine.wine_id, v.price) for v in analysis.venues]
    assert rows == [("A", "d2", "w3", 10.0), ("C", "d3", "w5", 20.0), ("B", "d1", "w1", 30.0)]
    assert [v.value for v in analysis.venues] == pytest.approx([0.375, 0.45, 0.775])
    assert analysis.venues[2].wine is CANDIDATES[3]
    assert analysis.sommelier_proof.venue_id == "A"
    assert analysis.sommelier_favourite.venue_id == "B"
    assert analysis.value == pytest.approx(0.775)
    assert analysis.principal_variation == ["B", "d1", "w1"]
    assert analysis.minimax_nodes == 18
    assert analysis.alphabeta_nodes == 15
    assert analysis.percent_saved == pytest.approx(100 * 3 / 18)


def test_analyse_is_deterministic_under_input_order() -> None:
    assert analyse(list(reversed(CANDIDATES)), BUDGET) == analyse(CANDIDATES, BUDGET)


def test_max_venues_keeps_the_best_scored_venues() -> None:
    # Best scores: A .90, B .80, C .70 -> top two are A and B.
    analysis = analyse(CANDIDATES, BUDGET, max_venues=2)
    assert analysis is not None
    assert sorted(v.venue_id for v in analysis.venues) == ["A", "B"]
    # Top one is A alone: it is both the proof pick and the favourite.
    single = analyse(CANDIDATES, BUDGET, max_venues=1)
    assert single is not None
    assert single.sommelier_proof.venue_id == single.sommelier_favourite.venue_id == "A"
    assert single.value == pytest.approx(0.375)


def test_venue_ranking_ties_break_by_venue_id() -> None:
    tied = [cand(v, "d1", "w1", 0.5, 20.0, 0.5) for v in ("Z", "Y", "X")]
    assert WineGame(tied, BUDGET, max_venues=2).venues == ("X", "Y")
    analysis = analyse(tied, BUDGET)
    assert analysis is not None
    # Equal values: both picks fall back to the smallest venue_id.
    assert analysis.sommelier_proof.venue_id == "X"
    assert analysis.sommelier_favourite.venue_id == "X"


def test_default_limit_is_eight_venues() -> None:
    many = [cand(f"v{i:02d}", "d1", "w1", 0.5, 20.0, 1.0 - i / 100) for i in range(12)]
    analysis = analyse(many, BUDGET)
    assert analysis is not None
    assert sorted(v.venue_id for v in analysis.venues) == [f"v{i:02d}" for i in range(8)]


def test_missing_quality_feature_counts_as_zero() -> None:
    bare = Candidate("w1", "A", "d1", 20.0, 0.0, {}, 0.5, ())
    assert upsell_utility(bare, BUDGET) == pytest.approx(0.25)


def test_no_candidates_returns_none() -> None:
    assert analyse([], BUDGET) is None


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="budget_eur"):
        WineGame(CANDIDATES, 0.0)
    with pytest.raises(ValueError, match="max_venues"):
        WineGame(CANDIDATES, BUDGET, max_venues=0)
