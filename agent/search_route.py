"""Walking routes on the street graph with A* search (DEVELOPMENT_PLAN.md task 6.1).

The search is hand-written and follows aima-python ``search.py``: ``astar_search`` is
``best_first_graph_search`` with ``f(n) = g(n) + h(n)`` (Russell & Norvig, *AIMA*,
section 3.5.2 "A* search"). NetworkX only *holds* the graph (and checks our answers
in the tests).

* Step cost: the edge ``length`` in metres; of parallel edges, the shortest.
* Heuristic: the haversine (great-circle) distance to the goal. A street path can
  never be shorter than the straight line, so ``h`` is admissible and consistent and
  the first time the goal is popped its path is optimal.
"""

from __future__ import annotations

import gzip
import heapq
import math
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from agent.config import DATA, WALK_SPEED_M_PER_MIN

EARTH_RADIUS_M = 6_371_008.8

Heuristic = Callable[[float, float, float, float], float]


class NoRouteError(Exception):
    """Raised when the goal cannot be reached from the start on the street graph."""


@dataclass(frozen=True)
class RouteResult:
    """A walking route, as the map and the results view need it."""

    path_latlon: list[tuple[float, float]]  # points along the streets, start to goal
    length_m: float
    nodes_expanded: int  # nodes A* took off the frontier and expanded
    walk_minutes: float


# --------------------------------------------------------------------------- #
# Loading and the search structure
# --------------------------------------------------------------------------- #


def load_graph(path: str | Path = DATA / "madrid_walk.graphml") -> nx.MultiDiGraph:
    """Load the street graph; also reads the ``.graphml.gz`` variant of task 1.3.

    If ``path`` does not exist but ``path + ".gz"`` does, the compressed file is used.
    """
    import osmnx as ox

    path = Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        path = path.with_name(path.name + ".gz")
    if path.suffix == ".gz":
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / path.stem
            with gzip.open(path, "rb") as src, open(plain, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return ox.load_graphml(plain)
    return ox.load_graphml(path)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def zero_heuristic(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """``h = 0``: turns A* into uniform-cost search (used to compare in tests/UI)."""
    return 0.0


class _SearchGraph:
    """Adjacency and coordinates extracted once from a NetworkX graph."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.adj: dict[Any, dict[Any, tuple[float, Any]]] = {}
        for u, v, key, data in graph.edges(keys=True, data=True):
            length = float(data["length"])
            best = self.adj.setdefault(u, {}).get(v)
            if best is None or length < best[0]:
                self.adj[u][v] = (length, key)
        self.coords: dict[Any, tuple[float, float]] = {
            n: (float(d["y"]), float(d["x"])) for n, d in graph.nodes(data=True)
        }
        self.node_ids = list(self.coords)
        self.lat = np.radians(np.array([self.coords[n][0] for n in self.node_ids]))
        self.lon = np.radians(np.array([self.coords[n][1] for n in self.node_ids]))


# id(graph) -> (graph, search structure). Holding the graph keeps the id unique.
_SEARCH_CACHE: dict[int, tuple[nx.MultiDiGraph, _SearchGraph]] = {}


def _search_graph(graph: nx.MultiDiGraph) -> _SearchGraph:
    cached = _SEARCH_CACHE.get(id(graph))
    if cached is not None and cached[0] is graph:
        return cached[1]
    sg = _SearchGraph(graph)
    _SEARCH_CACHE[id(graph)] = (graph, sg)
    return sg


def nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> Any:
    """Return the graph node closest (great-circle) to ``(lat, lon)``; ties by order.

    Vectorised haversine over every node; equivalent to
    ``ox.distance.nearest_nodes(G, X=lon, Y=lat)`` without needing scikit-learn.
    """
    sg = _search_graph(graph)
    if not sg.node_ids:
        raise NoRouteError("the street graph has no nodes")
    p, lam = math.radians(lat), math.radians(lon)
    a = (
        np.sin((sg.lat - p) / 2) ** 2
        + math.cos(p) * np.cos(sg.lat) * np.sin((sg.lon - lam) / 2) ** 2
    )
    return sg.node_ids[int(np.argmin(a))]


# --------------------------------------------------------------------------- #
# A*
# --------------------------------------------------------------------------- #


def astar(
    graph: nx.MultiDiGraph,
    start_node: Any,
    goal_node: Any,
    heuristic: Heuristic = haversine_m,
) -> tuple[list[Any], float, int]:
    """A* graph search from ``start_node`` to ``goal_node``.

    Mirrors aima-python ``astar_search`` / ``best_first_graph_search``: a priority
    queue ordered by ``f = g + h`` (``heapq`` with a counter so ties pop in insertion
    order), a ``best_g`` dict instead of an explored set (a node is re-opened only via
    a strictly cheaper path), and the goal test when a node is *popped*, which is what
    makes the answer optimal.

    Returns ``(path_nodes, length_m, nodes_expanded)``. Raises ``NoRouteError`` if
    the goal is unreachable and ``KeyError`` if either node is not in the graph.
    """
    sg = _search_graph(graph)
    coords, adj = sg.coords, sg.adj
    goal_lat, goal_lon = coords[goal_node]
    start_lat, start_lon = coords[start_node]

    tie = count()
    frontier: list[tuple[float, int, float, Any]] = [
        (heuristic(start_lat, start_lon, goal_lat, goal_lon), next(tie), 0.0, start_node)
    ]
    best_g: dict[Any, float] = {start_node: 0.0}
    parent: dict[Any, Any] = {start_node: None}
    expanded = 0

    while frontier:
        _, _, g, node = heapq.heappop(frontier)
        if g > best_g[node]:
            continue  # stale entry: a cheaper path to this node was found later
        if node == goal_node:
            path = [node]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.reverse()
            return path, g, expanded
        expanded += 1
        for child, (step, _) in adj.get(node, {}).items():
            new_g = g + step
            if new_g < best_g.get(child, math.inf):
                best_g[child] = new_g
                parent[child] = node
                lat, lon = coords[child]
                f = new_g + heuristic(lat, lon, goal_lat, goal_lon)
                heapq.heappush(frontier, (f, next(tie), new_g, child))

    raise NoRouteError(f"no walking route from node {start_node} to node {goal_node}")


def _edge_points(graph: nx.MultiDiGraph, sg: _SearchGraph, u: Any, v: Any) -> list:
    """``(lat, lon)`` points of the edge u->v used by the search, u first, v last."""
    _, key = sg.adj[u][v]
    geometry = graph.edges[u, v, key].get("geometry")
    if geometry is None:
        return [sg.coords[u], sg.coords[v]]
    points = [(float(y), float(x)) for x, y in geometry.coords]
    ulat, ulon = sg.coords[u]
    if haversine_m(ulat, ulon, *points[-1]) < haversine_m(ulat, ulon, *points[0]):
        points.reverse()
    return points


def route(
    graph: nx.MultiDiGraph,
    start: tuple[float, float],
    goal: tuple[float, float],
    heuristic: Heuristic = haversine_m,
) -> RouteResult:
    """Walking route between two ``(lat, lon)`` points.

    Both points snap to their nearest graph node; A* runs between those nodes and the
    path follows each street's ``geometry`` where the edge has one. If both points
    snap to the same node the route is the straight two-point line between them.
    """
    start_node = nearest_node(graph, *start)
    goal_node = nearest_node(graph, *goal)
    if start_node == goal_node:
        length = haversine_m(*start, *goal)
        return RouteResult([tuple(start), tuple(goal)], length, 0, length / WALK_SPEED_M_PER_MIN)

    nodes, length, expanded = astar(graph, start_node, goal_node, heuristic)
    sg = _search_graph(graph)
    points: list[tuple[float, float]] = [sg.coords[nodes[0]]]
    for u, v in zip(nodes, nodes[1:], strict=False):
        points.extend(_edge_points(graph, sg, u, v)[1:])
    return RouteResult(points, length, expanded, length / WALK_SPEED_M_PER_MIN)


def route_through(
    graph: nx.MultiDiGraph, start: tuple[float, float], stops: list[tuple[float, float]]
) -> tuple[RouteResult, list[int]]:
    """Visit every stop, always walking to the nearest unvisited one next (task 6.2).

    "Nearest" is straight-line distance; each leg is an A* route. Returns the joined
    route and the order in which the stops (by index) were visited.
    """
    remaining = list(range(len(stops)))
    order: list[int] = []
    here = tuple(start)
    points: list[tuple[float, float]] = [here]
    length = 0.0
    expanded = 0
    while remaining:
        nxt = min(remaining, key=lambda i: (haversine_m(*here, *stops[i]), i))
        leg = route(graph, here, stops[nxt])
        points.extend(leg.path_latlon[1:])
        length += leg.length_m
        expanded += leg.nodes_expanded
        here = tuple(stops[nxt])
        order.append(nxt)
        remaining.remove(nxt)
    return RouteResult(points, length, expanded, length / WALK_SPEED_M_PER_MIN), order
