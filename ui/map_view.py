"""Folium map: start point, distance circle, ranked venues, ruled-out venues, route.

DEVELOPMENT_PLAN.md task 7.3. All OSM-derived text is escaped before it goes into
popup HTML.
"""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

import folium
from streamlit_folium import st_folium

from agent.data_loader import AgentData
from agent.models import Candidate, OpenState, RecommendationResult, VenueKind
from agent.opening import open_state
from agent.preferences import Preferences, haversine_m
from agent.search_route import RouteResult

KIND_COLOURS = {
    VenueKind.SHOP: "#6a3d9a",  # purple
    VenueKind.RESTAURANT: "#b2182b",  # wine red
    VenueKind.BAR: "#e08214",  # amber
}
KIND_LABELS = {
    VenueKind.SHOP: "Wine shop",
    VenueKind.RESTAURANT: "Restaurant",
    VenueKind.BAR: "Bar",
}
OPEN_LABELS = {
    OpenState.OPEN: "Open at the chosen time",
    OpenState.CLOSED: "Closed at the chosen time",
    OpenState.UNKNOWN: "Hours unknown",
}
MAX_RULED_OUT_MARKERS = 300

#: Where Streamlit serves ``static/`` (``server.enableStaticServing`` in .streamlit/config.toml).
STATIC_URL = "/app/static/vendor"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static" / "vendor"
#: Folium's CDN assets, replaced by the vendored copies (same names Folium uses).
LOCAL_JS = [
    ("leaflet", "leaflet-1.9.3/leaflet.js"),
    ("jquery", "jquery-3.7.1/jquery.min.js"),
    ("bootstrap", "bootstrap-5.2.2/js/bootstrap.bundle.min.js"),
    ("awesome_markers", "leaflet.awesome-markers-2.0.5/leaflet.awesome-markers.js"),
]
LOCAL_CSS = [
    ("leaflet_css", "leaflet-1.9.3/leaflet.css"),
    ("bootstrap_css", "bootstrap-5.2.2/css/bootstrap.min.css"),
    ("awesome_markers_font_css", "fontawesome-free-6.2.0/css/all.min.css"),
    ("awesome_markers_css", "leaflet.awesome-markers-2.0.5/leaflet.awesome-markers.css"),
]


def use_local_assets(fmap: folium.Map) -> None:
    """Load Leaflet and friends from this app instead of public CDNs.

    So the map works offline (only the tiles need the network). Skipped if the
    vendored files are missing, in which case Folium's CDN links stay.
    """
    if not all((STATIC_DIR / rel).exists() for _, rel in LOCAL_JS + LOCAL_CSS):
        return
    fmap.default_js = [(name, f"{STATIC_URL}/{rel}") for name, rel in LOCAL_JS]
    fmap.default_css = [(name, f"{STATIC_URL}/{rel}") for name, rel in LOCAL_CSS]


ROUTE_COLOUR = "#2166ac"


def _rank_icon(rank: int, colour: str, selected: bool) -> folium.DivIcon:
    size = 30 if selected else 24
    ring = "3px solid #111" if selected else "2px solid #fff"
    html = (
        f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:{colour};'
        f"border:{ring};color:#fff;font:bold 12px/1 sans-serif;display:flex;"
        f'align-items:center;justify-content:center;box-shadow:0 1px 4px rgba(0,0,0,.4)">'
        f"{rank}</div>"
    )
    return folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(size // 2, size // 2))


def _popup_html(
    venue_id: str,
    picks: list[tuple[int, Candidate]],
    data: AgentData,
    prefs: Preferences,
) -> str:
    venue = data.venues[venue_id]
    state = open_state(venue.opening_hours, prefs.at_time)
    parts = [
        f"<b>{escape(venue.name)}</b><br>",
        f"{escape(KIND_LABELS[venue.kind])}",
        f" · {escape(venue.address)}" if venue.address else "",
        f"<br><i>{escape(OPEN_LABELS[state])}</i>",
    ]
    if venue.opening_hours:
        parts.append(f" <small>({escape(venue.opening_hours)})</small>")
    for rank, cand in picks:
        wine = data.wines[cand.wine_id]
        year = wine.year if wine.year is not None else "N.V."
        dish = data.pairing_rules.dishes[cand.dish_id].label
        parts.append(
            "<hr style='margin:6px 0'>"
            f"<b>#{rank} {escape(wine.name)}</b> ({escape(str(year))})<br>"
            f"{escape(wine.winery)} · {escape(wine.region)}<br>"
            f"€{cand.price_at_venue:.2f} · with {escape(dish.lower())} · "
            f"score <b>{cand.score:.2f}</b>"
            "<ul style='margin:4px 0;padding-left:16px'>"
            + "".join(f"<li>{escape(line)}</li>" for line in cand.reasons)
            + "</ul>"
        )
    parts.append("<small><i>Wine list simulated — not the venue's real list.</i></small>")
    return "<div style='width:280px;font-size:12px'>" + "".join(parts) + "</div>"


MAP_HEIGHT_PX = 520
MAP_WIDTH_PX = 900  # a conservative guess at the wide-layout map width


def view_for(
    user: tuple[float, float], route: RouteResult | None
) -> tuple[tuple[float, float], int]:
    """Centre and zoom that show the start point and the whole route.

    Computed here (Web Mercator maths) rather than with ``fit_bounds``, which
    streamlit-folium does not apply reliably. Zoom is clamped to 13–16 so
    neighbouring picks stay in view.
    """
    points = [user, *(route.path_latlon if route is not None else [])]
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    south, north, west, east = min(lats), max(lats), min(lons), max(lons)

    def merc_y(lat: float) -> float:
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    lon_frac = max((east - west) / 360.0, 1e-9)
    lat_frac = max((merc_y(north) - merc_y(south)) / (2 * math.pi), 1e-9)
    usable = 0.8  # leave a margin around the route
    zoom_x = math.log2(usable * MAP_WIDTH_PX / 256 / lon_frac)
    zoom_y = math.log2(usable * MAP_HEIGHT_PX / 256 / lat_frac)
    zoom = int(max(13, min(16, math.floor(min(zoom_x, zoom_y)))))
    return ((south + north) / 2, (west + east) / 2), zoom


def build_map(
    result: RecommendationResult,
    data: AgentData,
    prefs: Preferences,
    route: RouteResult | None,
    selected: int,
    show_ruled_out: bool,
) -> folium.Map:
    """Build the Folium map (separate from rendering, so it can be tested)."""
    user = (prefs.user_lat, prefs.user_lon)
    fmap = folium.Map(location=user, zoom_start=15, tiles="OpenStreetMap", control_scale=True)
    use_local_assets(fmap)

    folium.Circle(
        user,
        radius=prefs.max_distance_m,
        color=ROUTE_COLOUR,
        weight=1,
        fill=True,
        fill_opacity=0.06,
        tooltip=f"Max walking distance: {prefs.max_distance_m:.0f} m (straight line)",
    ).add_to(fmap)

    if show_ruled_out:
        excluded = result.diagnostics.venue_exclusions
        nearest = sorted(
            excluded,
            key=lambda vid: (
                haversine_m(*user, data.venues[vid].lat, data.venues[vid].lon),
                vid,
            ),
        )[:MAX_RULED_OUT_MARKERS]
        for venue_id in nearest:
            venue = data.venues[venue_id]
            folium.CircleMarker(
                (venue.lat, venue.lon),
                radius=4,
                color="#777",
                weight=1,
                fill=True,
                fill_color="#999",
                fill_opacity=0.6,
                tooltip=f"{escape(venue.name)} — ruled out by: {escape(excluded[venue_id])}",
            ).add_to(fmap)

    by_venue: dict[str, list[tuple[int, Candidate]]] = {}
    for rank, cand in enumerate(result.candidates, 1):
        by_venue.setdefault(cand.venue_id, []).append((rank, cand))
    selected_venue = (
        result.candidates[selected].venue_id if 0 <= selected < len(result.candidates) else None
    )
    # Draw the selected venue last so its marker sits on top.
    for venue_id, picks in sorted(by_venue.items(), key=lambda kv: kv[0] == selected_venue):
        venue = data.venues[venue_id]
        is_selected = venue_id == selected_venue
        folium.Marker(
            (venue.lat, venue.lon),
            icon=_rank_icon(picks[0][0], KIND_COLOURS[venue.kind], is_selected),
            tooltip=f"#{picks[0][0]} {escape(venue.name)}",
            popup=folium.Popup(_popup_html(venue_id, picks, data, prefs), max_width=320),
        ).add_to(fmap)

    if route is not None and len(route.path_latlon) >= 2:
        folium.PolyLine(
            route.path_latlon,
            color=ROUTE_COLOUR,
            weight=5,
            opacity=0.85,
            tooltip=f"Walking route: {route.length_m:.0f} m",
        ).add_to(fmap)

    folium.Marker(
        user,
        tooltip="You start here",
        icon=folium.Icon(color="blue", icon="user", prefix="fa"),
    ).add_to(fmap)

    center, zoom = view_for(user, route)
    fmap.location = list(center)
    fmap.options["zoom"] = zoom
    return fmap


def render_map(
    result: RecommendationResult,
    data: AgentData,
    prefs: Preferences,
    route: RouteResult | None,
    selected: int = 0,
    show_ruled_out: bool = False,
) -> dict:
    """Draw the map and return the ``st_folium`` output (last click, last object)."""
    fmap = build_map(result, data, prefs, route, selected, show_ruled_out)
    center, zoom = view_for((prefs.user_lat, prefs.user_lon), route)
    return st_folium(
        fmap,
        center=center,
        zoom=zoom,
        height=MAP_HEIGHT_PX,
        use_container_width=True,
        returned_objects=["last_clicked"],  # marker clicks do not re-run the app
        key="map",
    )
