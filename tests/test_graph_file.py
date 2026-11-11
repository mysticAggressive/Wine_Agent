"""Tests for the committed walking graph (task 1.3)."""

from __future__ import annotations

import gzip
import math

import pytest

from agent.config import DATA

PLAIN = DATA / "madrid_walk.graphml"
GZ = DATA / "madrid_walk.graphml.gz"


@pytest.fixture(scope="module")
def graph():
    ox = pytest.importorskip("osmnx")
    if PLAIN.exists():
        return ox.load_graphml(PLAIN)
    if GZ.exists():
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.graphml"
            path.write_bytes(gzip.decompress(GZ.read_bytes()))
            return ox.load_graphml(path)
    pytest.skip("data/madrid_walk.graphml has not been fetched yet")


def test_graph_size(graph):
    assert graph.number_of_nodes() > 5000


def test_edges_have_numeric_length(graph):
    for _, _, data in graph.edges(data=True):
        assert isinstance(data["length"], float) and math.isfinite(data["length"])


def test_nodes_have_coordinates(graph):
    for _, data in graph.nodes(data=True):
        assert isinstance(data["x"], float) and isinstance(data["y"], float)
