"""Madrid Wine Agent — Streamlit entry point. Run with: ``streamlit run app.py``

Layout (DEVELOPMENT_PLAN.md phase 7): a bar of filter chips holds the preferences; below
it come the selected pick, the map beside the ranked list, then tabs for the score
breakdown, the "sommelier vs. you" game and the solver diagnostics.
Every widget change re-runs the CSP.
"""

from __future__ import annotations

import dataclasses

import streamlit as st

from agent.csp_solver import SolverOptions
from agent.data_loader import AgentData, load_all
from agent.minimax import SommelierAnalysis, analyse
from agent.models import RecommendationResult, Style, VenueKind
from agent.preferences import Preferences
from agent.recommend import recommend, sommelier_candidates
from agent.search_route import NoRouteError, RouteResult, load_graph, route
from ui.map_view import render_map
from ui.results_view import (
    render_diagnostics_tab,
    render_pick_card,
    render_results_tab,
    render_why_tab,
    selected_index,
    table_key,
)
from ui.sidebar import CLICKED_POINT_KEY, PENDING_CLICK_KEY, render_sidebar
from ui.sommelier_view import render_sommelier_tab
from ui.styles import brand_header, glass_logo, inject_css

ATTRIBUTION = (
    "Venues and streets © [OpenStreetMap](https://www.openstreetmap.org/copyright) "
    "contributors (ODbL). Wines: fedesoriano, “Spanish Wine Quality Dataset” (Kaggle). "
    "**Venue wine lists are simulated** (a deterministic sample of the dataset per venue), "
    "and restaurant/bar prices apply a fixed markup to retail."
)

# --------------------------------------------------------------------------- #
# Cached loading and solving (task 7.1)
# --------------------------------------------------------------------------- #

PrefsKey = tuple


@st.cache_resource(show_spinner="Loading wines, venues and inventory…")
def get_data() -> AgentData:
    """All files under ``data/``, loaded once per server process."""
    return load_all()


@st.cache_resource(show_spinner="Loading the Madrid street graph (first run only)…")
def get_graph():
    """The walking street graph, loaded once per server process."""
    return load_graph()


def prefs_key(prefs: Preferences) -> PrefsKey:
    """A hashable cache key for ``prefs``: weights as a sorted tuple, time to the minute."""
    fields = dataclasses.asdict(prefs)
    fields["styles"] = tuple(sorted(s.value for s in prefs.styles))
    fields["venue_kinds"] = tuple(sorted(k.value for k in prefs.venue_kinds))
    fields["weights"] = tuple(sorted(prefs.weights.items()))
    fields["at_time"] = prefs.at_time.replace(second=0, microsecond=0)
    return tuple(sorted(fields.items()))


def prefs_from_key(key: PrefsKey) -> Preferences:
    """Inverse of :func:`prefs_key`."""
    fields = dict(key)
    fields["styles"] = frozenset(Style(s) for s in fields["styles"])
    fields["venue_kinds"] = frozenset(VenueKind(k) for k in fields["venue_kinds"])
    fields["weights"] = dict(fields["weights"])
    return Preferences(**fields)


@st.cache_data(max_entries=64, show_spinner=False)
def cached_recommend(key: PrefsKey, options: tuple[bool, ...]) -> RecommendationResult:
    """``recommend`` keyed on the preferences and the four solver switches."""
    return recommend(prefs_from_key(key), get_data(), SolverOptions(*options))


@st.cache_data(max_entries=32, show_spinner=False)
def cached_sommelier(key: PrefsKey) -> SommelierAnalysis | None:
    """The minimax game over every CSP solution at the best venues."""
    prefs = prefs_from_key(key)
    return analyse(sommelier_candidates(prefs, get_data()), prefs.budget_eur)


@st.cache_data(max_entries=256, show_spinner=False)
def cached_route(start: tuple[float, float], goal: tuple[float, float]) -> RouteResult | None:
    """A* walking route, or ``None`` if the goal is unreachable."""
    try:
        return route(get_graph(), start, goal)
    except NoRouteError:
        return None


def options_tuple(options: SolverOptions) -> tuple[bool, ...]:
    """The solver switches as a hashable tuple, in ``SolverOptions`` field order."""
    return tuple(getattr(options, f.name) for f in dataclasses.fields(options))


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


def handle_map_click(output: dict | None) -> None:
    """A new click on empty map moves the starting point there (via a rerun)."""
    if not output:
        return
    click = output.get("last_clicked")
    if not click:
        return
    # Leaflet does not report marker clicks as map clicks, so this is empty map.
    point = (round(float(click["lat"]), 6), round(float(click["lng"]), 6))
    if st.session_state.get("_last_seen_click") == point:
        return
    st.session_state["_last_seen_click"] = point
    st.session_state[CLICKED_POINT_KEY] = point
    st.session_state[PENDING_CLICK_KEY] = True
    st.rerun()


def main() -> None:
    """Render the whole app."""
    st.set_page_config(page_title="Madrid Wine Agent", page_icon="🍷", layout="wide")
    inject_css()
    logo = brand_header()

    data = get_data()
    prefs, options = render_sidebar(data)
    key = prefs_key(prefs)
    result = cached_recommend(key, options_tuple(options))

    glass_logo(logo, table_key(result))  # new picks reload it, so the wine sloshes
    selected = selected_index(result)
    route_result: RouteResult | None = None
    target = None
    if result.candidates:
        target = data.venues[result.candidates[selected].venue_id]
        route_result = cached_route((prefs.user_lat, prefs.user_lon), (target.lat, target.lon))

    render_pick_card(result, data, selected, route_result)

    map_col, list_col = st.columns([3, 2], gap="medium")
    with map_col.container(key="map_panel"):
        show_ruled_out = st.toggle(
            "Show ruled-out venues", help="Grey dots, up to the 300 nearest. Hover for the reason."
        )
        output = render_map(result, data, prefs, route_result, selected, show_ruled_out)
        if target is not None and route_result is not None:
            st.caption(
                f"Walk to {target.name}: {route_result.length_m:,.0f} m — A* expanded "
                f"{route_result.nodes_expanded:,} nodes"
            )
        elif target is not None:
            st.caption(f"No walking route found to {target.name}.")
    with list_col.container(key="list_panel"):
        render_results_tab(result, data)
    handle_map_click(output)

    tabs = st.tabs(["Why this pick", "Sommelier vs. you", "Solver diagnostics"])
    with tabs[0]:
        render_why_tab(result, data, prefs, selected)
    with tabs[1]:
        render_sommelier_tab(cached_sommelier(key), result, data)
    with tabs[2]:
        render_diagnostics_tab(result, lambda opts: cached_recommend(key, options_tuple(opts)))

    with st.expander("How it works", icon=":material/info:"):
        st.markdown(
            "The agent solves a constraint satisfaction problem over wines, venues and "
            "dishes, ranks the answers with an evaluation function, and plans the walk "
            "with A* on the street graph. The tabs above show each step."
        )
    st.caption(ATTRIBUTION)


main()
