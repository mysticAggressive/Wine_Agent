"""Tests for agent/scoring.py (DEVELOPMENT_PLAN.md task 5.2)."""

from __future__ import annotations

import dataclasses

import pytest

from agent.config import FEATURES
from agent.data_loader import AgentData
from agent.preferences import Preferences
from agent.scoring import Scorer, ScoringContext, evaluate, features, format_distance, reasons


@pytest.fixture
def ctx(agent_data: AgentData) -> ScoringContext:
    return ScoringContext.from_data(agent_data)


def feats(prefs, data, ctx, wine_id, venue_id, dish_id):
    return features(data.wines[wine_id], data.venues[venue_id], dish_id, prefs, ctx)


def test_feature_keys_and_range(prefs, agent_data, ctx):
    for venue_id in agent_data.venues:
        for wine_id in agent_data.inventory.wines_at(venue_id):
            for dish_id in agent_data.pairing_rules.dishes:
                f = feats(prefs, agent_data, ctx, wine_id, venue_id, dish_id)
                assert tuple(f) == FEATURES
                assert all(0.0 <= v <= 1.0 for v in f.values())


def test_taste_extremes(prefs, agent_data, ctx):
    # w0001 has body 5, w0003 body 2.
    p = dataclasses.replace(prefs, want_body=5)
    assert feats(p, agent_data, ctx, "w0001", "node/1", "grilled_red_meat")["taste"] == 1.0
    assert feats(p, agent_data, ctx, "w0003", "node/1", "shellfish")["taste"] == 0.0


def test_quality_extremes(prefs, agent_data, ctx):
    # Ratings run 4.3 (w0002, w0004) to 4.7 (w0005).
    assert feats(prefs, agent_data, ctx, "w0005", "node/1", "chocolate_dessert")["quality"] == 1.0
    assert feats(prefs, agent_data, ctx, "w0002", "node/1", "grilled_red_meat")["quality"] == 0.0


def test_price_fit_extremes(prefs, agent_data, ctx):
    p = dataclasses.replace(prefs, budget_eur=60)
    # w0001 costs exactly 60 at node/1; 150 at way/3 (over budget -> clipped to 0).
    assert feats(p, agent_data, ctx, "w0001", "node/1", "grilled_red_meat")["price_fit"] == 0.0
    assert feats(p, agent_data, ctx, "w0001", "way/3", "grilled_red_meat")["price_fit"] == 0.0
    assert feats(p, agent_data, ctx, "w0004", "node/2", "shellfish")["price_fit"] == (
        pytest.approx(1 - 8 / 60)
    )


def test_pairing_extremes(prefs, agent_data, ctx):
    # A white has no rule for red meat -> 0; PX with chocolate is a perfect match.
    assert feats(prefs, agent_data, ctx, "w0003", "node/1", "grilled_red_meat")["pairing"] == 0.0
    assert feats(prefs, agent_data, ctx, "w0005", "node/1", "chocolate_dessert")["pairing"] == 1.0


def test_distance_extremes(prefs, agent_data, ctx):
    v = agent_data.venues["node/1"]
    at_venue = dataclasses.replace(prefs, user_lat=v.lat, user_lon=v.lon)
    near = ScoringContext.from_data(agent_data)
    assert (
        feats(at_venue, agent_data, near, "w0002", "node/1", "grilled_red_meat")["distance"] == 1.0
    )
    tight = dataclasses.replace(prefs, max_distance_m=200)
    far = ScoringContext.from_data(agent_data)
    # node/2 is ~1.1 km from the default user position.
    assert feats(tight, agent_data, far, "w0002", "node/2", "grilled_red_meat")["distance"] == 0.0


def test_features_raise_for_unstocked_pair(prefs, agent_data, ctx):
    with pytest.raises(KeyError):
        feats(prefs, agent_data, ctx, "w0005", "node/2", "chocolate_dessert")


def test_evaluate_is_weighted_mean():
    f = {"taste": 1.0, "quality": 0.0, "price_fit": 0.5, "pairing": 0.0, "distance": 0.0}
    w = {"taste": 2, "quality": 0, "price_fit": 2, "pairing": 0, "distance": 0}
    assert evaluate(f, w) == pytest.approx(0.75)


def test_all_zero_weights_equal_all_equal_weights():
    f = {"taste": 0.9, "quality": 0.1, "price_fit": 0.4, "pairing": 0.7, "distance": 0.3}
    zeros = dict.fromkeys(FEATURES, 0.0)
    assert evaluate(f, zeros) == evaluate(f, dict.fromkeys(FEATURES, 1.0))
    for weight in (5.0, 7.0, 10.0):
        equal = dict.fromkeys(FEATURES, weight)
        assert evaluate(f, zeros) == pytest.approx(evaluate(f, equal), abs=1e-12)


def _triples(data: AgentData) -> list[tuple[str, str, str]]:
    return [
        (w, v, d)
        for v in sorted(data.venues)
        for w in sorted(data.inventory.wines_at(v))
        for d in sorted(data.pairing_rules.dishes)
    ]


@pytest.mark.parametrize("feature", FEATURES)
def test_raising_a_weight_never_lowers_the_best_on_that_feature(prefs, agent_data, feature):
    ctx = ScoringContext.from_data(agent_data)
    triples = _triples(agent_data)
    table = {t: feats(prefs, agent_data, ctx, *t) for t in triples}
    best = max(triples, key=lambda t: (table[t][feature], t))

    def rank(weights: dict[str, float]) -> int:
        order = sorted(triples, key=lambda t: (-evaluate(table[t], weights), t))
        return order.index(best)

    previous = None
    for weight in range(0, 11):
        weights = {**prefs.weights, feature: float(weight)}
        current = rank(weights)
        if previous is not None:
            assert current <= previous
        previous = current


def test_scorer_matches_evaluate_bit_for_bit(prefs: Preferences, agent_data):
    ctx = ScoringContext.from_data(agent_data)
    for weights in (prefs.weights, dict.fromkeys(FEATURES, 0.0), {**prefs.weights, "taste": 9}):
        p = dataclasses.replace(prefs, weights=weights)
        scorer = Scorer(p, agent_data.wines, agent_data.venues, ctx)
        for t in _triples(agent_data):
            assert scorer.score(*t) == evaluate(feats(p, agent_data, ctx, *t), weights)


def test_reasons_lines(prefs, agent_data, ctx):
    wine, venue = agent_data.wines["w0002"], agent_data.venues["node/2"]
    f = features(wine, venue, "grilled_red_meat", prefs, ctx)
    lines = reasons(wine, venue, "grilled_red_meat", f, prefs, ctx)
    assert 4 <= len(lines) <= 6
    assert lines[0] == "€12.00 — 76% under your €50 budget"
    assert lines[1] == "From Rioja (Rioja Red)"
    assert lines[2].endswith("min walk)")
    assert lines[4] == "Rated 4.3 by 2,500 reviewers"
    # node/2 has no opening hours.
    assert lines[-1] == "Hours unknown — check before you go"

    known = agent_data.venues["node/1"]
    f = features(wine, known, "grilled_red_meat", prefs, ctx)
    assert "Hours unknown — check before you go" not in reasons(
        wine, known, "grilled_red_meat", f, prefs, ctx
    )


def test_format_distance():
    assert format_distance(650) == "650 m"
    assert format_distance(999.4) == "999 m"
    assert format_distance(1234) == "1.2 km"
