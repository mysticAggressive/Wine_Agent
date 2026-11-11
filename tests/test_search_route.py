"""Tests for agent/search_route.py (DEVELOPMENT_PLAN.md task 6.1)."""

from __future__ import annotations

import random

import networkx as nx
import pytest
from shapely.geometry import LineString

from agent.config import DATA, WALK_SPEED_M_PER_MIN
from agent.search_route import (
    NoRouteError,
    astar,
    haversine_m,
    nearest_node,
    route,
    route_through,
    zero_heuristic,
)

# Five nodes ~111 m apart in latitude (0.001 deg). A-B-D-E is shorter than A-C-E.
#
#   E(40.003)
#   |  \
#   D   C
#   |  /
#   B  |
#   | /
#   A(40.000)
COORDS = {
    "A": (40.000, -3.700),
    "B": (40.001, -3.700),
    "C": (40.0015, -3.6985),
    "D": (40.002, -3.700),
    "E": (40.003, -3.700),
}
EDGES = [
    ("A", "B", 112.0),
    ("B", "D", 112.0),
    ("D", "E", 112.0),
    ("A", "C", 200.0),
    ("C", "E", 200.0),
]


def small_graph(bidirectional: bool = True) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for n, (lat, lon) in COORDS.items():
        g.add_node(n, y=lat, x=lon)
    for u, v, length in EDGES:
        g.add_edge(u, v, length=length)
        if bidirectional:
            g.add_edge(v, u, length=length)
    return g


def test_astar_finds_known_shortest_path():
    path, length, expanded = astar(small_graph(), "A", "E")
    assert path == ["A", "B", "D", "E"]
    assert length == pytest.approx(336.0)
    assert expanded >= 1


def test_parallel_edges_use_the_shortest():
    g = small_graph()
    g.add_edge("A", "E", length=500.0)
    g.add_edge("A", "E", length=300.0)
    path, length, _ = astar(g, "A", "E")
    assert path == ["A", "E"]
    assert length == pytest.approx(300.0)


def test_heuristic_expands_no_more_than_uniform_cost():
    g = small_graph()
    _, l1, with_h = astar(g, "A", "E")
    _, l2, without_h = astar(g, "A", "E", zero_heuristic)
    assert l1 == l2
    assert with_h <= without_h


def test_unreachable_goal_raises():
    g = small_graph(bidirectional=False)
    with pytest.raises(NoRouteError):
        astar(g, "E", "A")


def test_route_follows_geometry_and_times_walk():
    g = small_graph()
    key = next(iter(g["A"]["B"]))
    # Geometry stored in (lon, lat) order, reversed, with a bend: must be re-oriented.
    g.edges["A", "B", key]["geometry"] = LineString(
        [(-3.700, 40.001), (-3.7002, 40.0005), (-3.700, 40.000)]
    )
    r = route(g, COORDS["A"], COORDS["E"])
    assert r.path_latlon[0] == COORDS["A"]
    assert r.path_latlon[1] == (40.0005, -3.7002)
    assert r.path_latlon[-1] == COORDS["E"]
    assert r.walk_minutes == pytest.approx(r.length_m / WALK_SPEED_M_PER_MIN)


def test_same_node_gives_straight_two_point_route():
    g = small_graph()
    start, goal = (40.00001, -3.70001), (40.00002, -3.69999)
    r = route(g, start, goal)
    assert r.path_latlon == [start, goal]
    assert r.length_m == pytest.approx(haversine_m(*start, *goal))
    assert r.nodes_expanded == 0


def test_nearest_node():
    g = small_graph()
    assert nearest_node(g, 40.0016, -3.6986) == "C"
    assert nearest_node(g, 39.9, -3.7) == "A"


def test_route_through_visits_nearest_first():
    g = small_graph()
    result, order = route_through(g, COORDS["A"], [COORDS["E"], COORDS["B"]])
    assert order == [1, 0]
    assert result.length_m == pytest.approx(336.0)


# --- real graph -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_graph():
    if (
        not (DATA / "madrid_walk.graphml").exists()
        and not (DATA / "madrid_walk.graphml.gz").exists()
    ):
        pytest.skip("street graph not built")
    from agent.search_route import load_graph

    return load_graph()


def test_matches_networkx_on_real_graph(real_graph):
    rng = random.Random(42)
    nodes = sorted(real_graph.nodes)
    for _ in range(20):
        a, b = rng.choice(nodes), rng.choice(nodes)
        _, length, with_h = astar(real_graph, a, b)
        expected = nx.shortest_path_length(real_graph, a, b, weight="length")
        assert length == pytest.approx(expected, abs=1e-6)
        _, _, without_h = astar(real_graph, a, b, zero_heuristic)
        assert with_h <= without_h
