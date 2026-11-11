"""Fetch the walking street graph around Puerta del Sol with OSMnx (task 1.3).

Run by hand, once:  ``python data/fetch_graph.py``

Downloads the ``walk`` network within ``RADIUS_M + 300`` m of ``CENTER`` (the extra
300 m keeps venues at the edge of the circle reachable), keeps the largest strongly
connected component and writes ``data/madrid_walk.graphml``. If that file is larger
than 50 MB the radius drops to ``RADIUS_M``; if it is still too large the file is
written gzip-compressed as ``madrid_walk.graphml.gz`` instead.

OSMnx's HTTP cache goes to ``./cache`` (git-ignored), so a failed run resumes from the
sub-queries that already succeeded.

Data © OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

import osmnx as ox

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import CENTER, DATA, RADIUS_M  # noqa: E402

OUT_PATH = DATA / "madrid_walk.graphml"
GZ_PATH = DATA / "madrid_walk.graphml.gz"
MAX_PLAIN_BYTES = 50 * 1024 * 1024
MAX_COMMIT_BYTES = 95 * 1024 * 1024
EXTRA_M = 300
DEFAULT_OVERPASS = "https://overpass-api.de/api"
# OSMnx splits the query polygon into pieces of at most this area (m²). Its default
# (2,500 km²) sends Madrid as one big query, which busy public servers time out on.
SUBQUERY_AREA_M2 = 4_000_000


def configure(overpass_url: str, subquery_area_m2: float) -> None:
    """Point OSMnx at an Overpass endpoint and set caching, timeouts and splitting."""
    ox.settings.overpass_url = overpass_url.rstrip("/")
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(ROOT / "cache")
    ox.settings.requests_timeout = 300
    ox.settings.max_query_area_size = subquery_area_m2
    ox.settings.log_console = True
    # Some mirrors have no /status endpoint; the pause OSMnx computes from it is optional.
    ox.settings.overpass_rate_limit = False
    ox.settings.http_user_agent = "madrid-wine-agent/0.1 (course project)"


def use_extract(pbf: Path) -> None:
    """Make OSMnx read its network data from a ``.osm.pbf`` extract, not Overpass.

    Only OSMnx's download step is replaced: for each polygon it would query, it gets
    the elements ``way<walk filter>(poly); >; out;`` would return (see
    ``data/osm_extract.network_payload``). Buffering, truncation, simplification and
    the walk filter itself are OSMnx's own, so the graph is the one an online fetch
    of the same OSM data gives.
    """
    from osmnx import _overpass

    from data.osm_extract import network_payload

    def download_from_extract(polygon, network_type, custom_filter):
        if custom_filter is None:
            filters = [_overpass._get_network_filter(network_type)]
        elif isinstance(custom_filter, str):
            filters = [custom_filter]
        else:
            filters = list(custom_filter)
        west, south, east, north = polygon.bounds
        for way_filter in filters:
            yield network_payload(pbf, (south, west, north, east), way_filter)

    _overpass._download_overpass_network = download_from_extract


def fetch(dist_m: float):
    """Download, simplify and keep the largest strongly connected component."""
    graph = ox.graph_from_point(CENTER, dist=dist_m, network_type="walk", simplify=True)
    return ox.truncate.largest_component(graph, strongly=True)


def main(argv: list[str] | None = None) -> int:
    """Fetch the graph, write it (shrinking or compressing it if needed), print a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--overpass-url",
        default=DEFAULT_OVERPASS,
        help=f"Overpass API base URL, without /interpreter (default {DEFAULT_OVERPASS})",
    )
    parser.add_argument(
        "--subquery-area-km2",
        type=float,
        default=SUBQUERY_AREA_M2 / 1e6,
        help="largest area OSMnx sends in one Overpass query",
    )
    parser.add_argument(
        "--from-pbf",
        type=Path,
        metavar="PATH",
        help="read a regional .osm.pbf extract (see data/osm_extract.py) instead of Overpass",
    )
    args = parser.parse_args(argv)
    configure(args.overpass_url, args.subquery_area_km2 * 1e6)
    if args.from_pbf:
        use_extract(args.from_pbf)

    dist = RADIUS_M + EXTRA_M
    graph = fetch(dist)
    ox.save_graphml(graph, OUT_PATH)
    size = OUT_PATH.stat().st_size
    if size > MAX_PLAIN_BYTES:
        print(f"{size / 1e6:.1f} MB is over 50 MB; refetching with dist={RADIUS_M}")
        dist = RADIUS_M
        graph = fetch(dist)
        ox.save_graphml(graph, OUT_PATH)
        size = OUT_PATH.stat().st_size

    written = OUT_PATH
    if size > MAX_PLAIN_BYTES:
        with open(OUT_PATH, "rb") as src, gzip.GzipFile(GZ_PATH, "wb", mtime=0) as dst:
            shutil.copyfileobj(src, dst)
        OUT_PATH.unlink()
        written = GZ_PATH
        size = GZ_PATH.stat().st_size
        print(f"Still over 50 MB; wrote gzip-compressed {GZ_PATH.name}")
    if size > MAX_COMMIT_BYTES:
        print(f"ERROR: {written.name} is {size / 1e6:.1f} MB, too large to commit")
        return 1

    print(f"Radius: {dist} m")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"File: {written} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
