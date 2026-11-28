"""Preference panel (DEVELOPMENT_PLAN.md task 7.2), drawn as a bar of filter chips.

Each chip is a popover holding one group of widgets; its label summarises that group's
current values. There is no "Submit" button: every widget change re-runs the script, so
the solver re-runs and the results redraw at once (the "Dynamic" rubric item).

Popover labels are fixed when the popover is created, before its widgets run, so every
widget is keyed and the labels are built from session state (which already holds the
new value when a widget change triggers the rerun).
"""

from __future__ import annotations

from collections.abc import Mapping
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
NOW, PICK = "Now", "Pick a time"

#: Every always-drawn widget's key and default (set in session state, not ``value=``,
#: so the chip labels can read them before the widgets run).
DEFAULTS = {
    "styles": [],
    "food": SURPRISE,
    "body": "Full",
    "budget": 50,
    "where": "Anywhere",
    START_KEY: next(iter(LANDMARKS)),
    "max_distance": 2000,
    "open_only": True,
    "when": NOW,
    "exclude_unknown": False,
    **{f"w_{name}": DEFAULT_WEIGHT for name in FEATURES},
    **{name: True for name in TECHNIQUES},
}


def now_madrid() -> datetime:
    """The current time in Madrid, rounded down to the minute (so caches can hit)."""
    return datetime.now(ZoneInfo(TIMEZONE)).replace(second=0, microsecond=0)


def start_point(choice: str) -> tuple[float, float]:
    """Coordinates of the chosen starting point (a landmark or the clicked point)."""
    if choice == CLICKED:
        return st.session_state.get(CLICKED_POINT_KEY, CENTER)
    return LANDMARKS[choice]


def chip_labels(state: Mapping) -> dict[str, str]:
    """The label of each filter chip, from the widget values in ``state``."""
    what = [" / ".join(state["styles"]) or "Any wine", state["body"].lower()]
    what.append(f"≤ €{state['budget']}")
    if state["where"] != "Anywhere":
        what.append(state["where"].lower())
    if state["food"] != SURPRISE:
        what.append(f"with {state['food'].lower()}")
    start = "clicked point" if state[START_KEY] == CLICKED else state[START_KEY]
    if not state["open_only"]:
        when = "Any time"
    elif state["when"] == NOW:
        when = "Open now"
    elif "pick_date" not in state:  # the date widget has not run yet
        when = "Open at a picked time"
    else:
        when = f"Open {datetime.combine(state['pick_date'], state['pick_time']):%a %d %b, %H:%M}"
    custom = any(state[f"w_{n}"] != DEFAULT_WEIGHT for n in FEATURES)
    off = sum(not state[n] for n in TECHNIQUES)
    return {
        "what": " · ".join(what),
        "where": f"{state['max_distance'] / 1000:g} km from {start}",
        "when": when,
        "ranking": "Ranking · custom" if custom else "Ranking",
        "solver": f"Solver · {off} off" if off else "Solver",
    }


def reset_weights() -> None:
    """Button callback: every ranking weight back to the default."""
    for name in FEATURES:
        st.session_state[f"w_{name}"] = DEFAULT_WEIGHT


def render_sidebar(data: AgentData) -> tuple[Preferences, SolverOptions]:
    """Draw the filter chip bar and return the preferences it describes."""
    state = st.session_state
    for key, value in DEFAULTS.items():
        state.setdefault(key, value)
    if state.pop(PENDING_CLICK_KEY, False):
        state[START_KEY] = CLICKED  # a map click moves the start point
    labels = chip_labels(state)
    dish_labels = {rule.label: dish_id for dish_id, rule in data.pairing_rules.dishes.items()}

    bar = st.container(horizontal=True, horizontal_alignment="center", key="filters", gap="small")
    with bar.popover(labels["what"], icon=":material/wine_bar:"):
        style_names = st.pills(
            "Wine style",
            list(STYLE_LABELS),
            selection_mode="multi",
            key="styles",
            help="Pick none to accept any style.",
        )
        food = st.selectbox("Food", [SURPRISE, *dish_labels], key="food")
        body = st.select_slider("Body", options=list(BODY), key="body")
        budget = st.slider("Budget per bottle (€)", 10, 300, step=5, key="budget")
        where = st.segmented_control("Venue type", list(WHERE), required=True, key="where")

    with bar.popover(labels["where"], icon=":material/location_on:"):
        start = st.selectbox(
            "Starting point",
            [*LANDMARKS, CLICKED],
            key=START_KEY,
            help="Or click anywhere on the map to start from there.",
        )
        if start == CLICKED and CLICKED_POINT_KEY not in state:
            st.caption("No point clicked yet — using Puerta del Sol. Click on the map.")
        max_distance = st.slider(
            "Max walking distance (m)", 200, 4000, step=100, key="max_distance"
        )

    with bar.popover(labels["when"], icon=":material/schedule:"):
        open_only = st.toggle("Open venues only", key="open_only")
        when = st.segmented_control(
            "Time", [NOW, PICK], required=True, disabled=not open_only, key="when"
        )
        at_time = now_madrid()
        if when == PICK and open_only:
            # The key keeps the user's pick: without it the moving default resets the widget.
            day = st.date_input("Date", value=at_time.date(), key="pick_date")
            clock = st.time_input(
                "Time of day", value=time(at_time.hour, at_time.minute), step=900, key="pick_time"
            )
            at_time = datetime.combine(day, clock, tzinfo=ZoneInfo(TIMEZONE))
        exclude_unknown = st.toggle(
            "Exclude venues with unknown hours", disabled=not open_only, key="exclude_unknown"
        )

    with bar.popover(labels["ranking"], icon=":material/tune:"):
        st.caption("How much each factor counts towards the score.")
        weights = {
            name: float(st.slider(FEATURE_LABELS[name], 0, 10, key=f"w_{name}"))
            for name in FEATURES
        }
        st.button("Reset weights", on_click=reset_weights)

    with bar.popover(labels["solver"], icon=":material/science:"):
        st.caption(
            "Switching these off changes the search statistics, never the answer. "
            "Without forward checking the search tries millions more values and takes "
            "about half a minute on the full data."
        )
        flags = {name: st.toggle(label, key=name) for name, label in TECHNIQUES.items()}

    # Outside the popovers, so it shows without opening one.
    user_lat, user_lon = start_point(start)
    if haversine_m(user_lat, user_lon, *CENTER) > MAX_USER_OFFSET_M:
        st.warning("That point is outside the mapped area; using Puerta del Sol instead.")
        user_lat, user_lon = CENTER

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
    return prefs, SolverOptions(**flags)
