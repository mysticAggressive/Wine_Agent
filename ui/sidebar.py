"""Preference panel (DEVELOPMENT_PLAN.md task 7.2).

There is no "Submit" button: every widget change re-runs the script, so the solver
re-runs and the results redraw at once (the "Dynamic" rubric item).
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import streamlit as st

from agent.config import CENTER, FEATURES, LANDMARKS, TIMEZONE
from agent.csp_solver import SolverOptions
from agent.data_loader import AgentData
from agent.models import Style, VenueKind
from agent.preferences import MAX_USER_OFFSET_M, Preferences, haversine_m

CLICKED = "Clicked on map"
#: Session-state key holding the last point clicked on the map, ``(lat, lon)``.
CLICKED_POINT_KEY = "clicked_point"
#: Set by the map on a new click; applied before the start widget is drawn.
PENDING_CLICK_KEY = "pending_click"
START_KEY = "start_choice"

STYLE_LABELS = {
    "Red": Style.RED,
    "White": Style.WHITE,
    "Sparkling": Style.SPARKLING,
    "Fortified": Style.FORTIFIED,
}
WHERE = {
    "Anywhere": frozenset(),
    "Shop": frozenset({VenueKind.SHOP}),
    "Restaurant or bar": frozenset({VenueKind.RESTAURANT, VenueKind.BAR}),
}
BODY = {"Light": 2, "Medium": 3, "Full": 4, "Very full": 5}
SURPRISE = "Surprise me"
DEFAULT_WEIGHT = 5
FEATURE_LABELS = {
    "taste": "Taste (body match)",
    "quality": "Quality (rating)",
    "price_fit": "Price (cheaper is better)",
    "pairing": "Food pairing",
    "distance": "Closeness",
}
TECHNIQUES = {
    "use_mrv": "MRV (minimum remaining values)",
    "use_lcv": "LCV (least constraining value)",
    "use_forward_checking": "Forward checking",
    "use_ac3": "AC-3 arc consistency",
}


def now_madrid() -> datetime:
    """The current time in Madrid, rounded down to the minute (so caches can hit)."""
    return datetime.now(ZoneInfo(TIMEZONE)).replace(second=0, microsecond=0)


def start_point(choice: str) -> tuple[float, float]:
    """Coordinates of the chosen starting point (a landmark or the clicked point)."""
    if choice == CLICKED:
        return st.session_state.get(CLICKED_POINT_KEY, CENTER)
    return LANDMARKS[choice]


def filter_summary(prefs: Preferences, start: str, food: str) -> str:
    """One line describing the active filters, e.g. "Red · ≤ €50 · shop · 2 km from Sol"."""
    parts = [
        " / ".join(sorted(s.value.capitalize() for s in prefs.styles)) or "Any style",
        f"≤ €{prefs.budget_eur:.0f}",
    ]
    where = next(k for k, v in WHERE.items() if v == prefs.venue_kinds)
    if where != "Anywhere":
        parts.append(where.lower())
    if food != SURPRISE:
        parts.append(f"with {food.lower()}")
    parts.append(f"{prefs.max_distance_m / 1000:g} km from {start}")
    if prefs.open_only:
        parts.append(f"open {prefs.at_time:%a %H:%M}")
    return " · ".join(parts)


def reset_weights() -> None:
    """Button callback: every ranking weight back to the default."""
    for name in FEATURES:
        st.session_state[f"w_{name}"] = DEFAULT_WEIGHT


def render_sidebar(data: AgentData) -> tuple[Preferences, SolverOptions]:
    """Draw the preference panel and return what it describes."""
    sb = st.sidebar
    sb.header("Your preferences")
    summary = sb.container()  # filled last, once every widget has a value

    with sb.expander("What", expanded=True, icon=":material/wine_bar:"):
        style_names = st.multiselect(
            "Wine style", list(STYLE_LABELS), help="Leave empty to accept any style."
        )
        dish_labels = {rule.label: dish_id for dish_id, rule in data.pairing_rules.dishes.items()}
        food = st.selectbox("Food", [SURPRISE, *dish_labels])
        body = st.select_slider("Body", options=list(BODY), value="Full")
        budget = st.slider("Budget per bottle (€)", 10, 300, 50, step=5)
        where = st.radio("Venue type", list(WHERE), horizontal=True)

    with sb.expander("Where", icon=":material/location_on:"):
        clicked = CLICKED_POINT_KEY in st.session_state
        options = [*LANDMARKS, CLICKED]
        if st.session_state.pop(PENDING_CLICK_KEY, False):
            st.session_state[START_KEY] = CLICKED  # a map click moves the start point
        start = st.selectbox(
            "Starting point",
            options,
            key=START_KEY,
            help="Or click anywhere on the map to start from there.",
        )
        if start == CLICKED and not clicked:
            st.caption("No point clicked yet — using Puerta del Sol. Click on the map.")
        user_lat, user_lon = start_point(start)
        if haversine_m(user_lat, user_lon, *CENTER) > MAX_USER_OFFSET_M:
            st.warning("That point is outside the mapped area; using Puerta del Sol instead.")
            user_lat, user_lon = CENTER
        max_distance = st.slider("Max walking distance (m)", 200, 4000, 2000, step=100)

    with sb.expander("When", icon=":material/schedule:"):
        open_only = st.checkbox("Open venues only", value=True)
        when = st.radio("Time", ["Now", "Pick a time"], horizontal=True, disabled=not open_only)
        at_time = now_madrid()
        if when == "Pick a time" and open_only:
            # The key keeps the user's pick: without it the moving default resets the widget.
            day = st.date_input("Date", value=at_time.date(), key="pick_date")
            clock = st.time_input(
                "Time of day", value=time(at_time.hour, at_time.minute), step=900, key="pick_time"
            )
            at_time = datetime.combine(day, clock, tzinfo=ZoneInfo(TIMEZONE))
        exclude_unknown = st.checkbox(
            "Exclude venues with unknown hours", value=False, disabled=not open_only
        )

    with sb.expander("Fine-tune ranking", icon=":material/tune:"):
        st.caption("How much each factor counts towards the score.")
        for name in FEATURES:
            # Defaults live in session state (not value=) so the reset callback can set them.
            st.session_state.setdefault(f"w_{name}", DEFAULT_WEIGHT)
        weights = {
            name: float(st.slider(FEATURE_LABELS[name], 0, 10, key=f"w_{name}"))
            for name in FEATURES
        }
        st.button("Reset weights", on_click=reset_weights)

    with sb.expander("Advanced: solver techniques", icon=":material/science:"):
        st.caption(
            "Switching these off changes the search statistics, never the answer. "
            "Without forward checking the search tries millions more values and takes "
            "about half a minute on the full data."
        )
        flags = {
            name: st.checkbox(label, value=True, key=name) for name, label in TECHNIQUES.items()
        }

    prefs = Preferences(
        styles=frozenset(STYLE_LABELS[name] for name in style_names),
        budget_eur=float(budget),
        venue_kinds=WHERE[where],
        dish_id=None if food == SURPRISE else dish_labels[food],
        want_body=BODY[body],
        user_lat=float(user_lat),
        user_lon=float(user_lon),
        max_distance_m=float(max_distance),
        at_time=at_time,
        open_only=open_only,
        exclude_unknown_hours=exclude_unknown and open_only,
        weights=weights,
    )
    summary.caption(filter_summary(prefs, start, food))
    return prefs, SolverOptions(**flags)
