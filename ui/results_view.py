"""Result monitoring: ranked table, score breakdown, solver diagnostics (task 7.4)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from agent.config import FEATURES
from agent.csp_solver import SolverOptions
from agent.data_loader import AgentData
from agent.models import Candidate, RecommendationResult
from agent.preferences import Preferences
from ui.map_view import KIND_LABELS
from ui.sidebar import FEATURE_LABELS, TECHNIQUES

#: Widget key of the results table; its selection says where the route goes.
TABLE_KEY = "results_table"
VARIABLES = ("wine", "venue", "dish")


def selected_index(n_candidates: int) -> int:
    """Row selected in the results table (from session state), default 0."""
    state = st.session_state.get(TABLE_KEY)
    rows = []
    if state is not None:
        selection = state.get("selection", {}) if isinstance(state, dict) else state.selection
        rows = selection.get("rows", []) if isinstance(selection, dict) else selection.rows
    if rows and 0 <= rows[0] < n_candidates:
        return int(rows[0])
    return 0


def results_frame(result: RecommendationResult, data: AgentData) -> pd.DataFrame:
    """One row per recommended candidate, in rank order."""
    rows = []
    for rank, c in enumerate(result.candidates, 1):
        wine, venue = data.wines[c.wine_id], data.venues[c.venue_id]
        rows.append(
            {
                "Rank": rank,
                "Wine": wine.name,
                "Winery": wine.winery,
                "Region": wine.region,
                "Year": str(wine.year) if wine.year is not None else "N.V.",
                "Style": wine.style.value.capitalize(),
                "Venue": venue.name,
                "Kind": KIND_LABELS[venue.kind],
                "Dish": data.pairing_rules.dishes[c.dish_id].label,
                "Price (€)": c.price_at_venue,
                "Distance (m)": round(c.distance_m),
                "Score": c.score,
            }
        )
    return pd.DataFrame(rows)


def render_hints(result: RecommendationResult) -> None:
    """Warning plus relaxation hints, for when nothing satisfies the constraints."""
    st.warning("No wine, venue and dish satisfy all your constraints.")
    hints = result.diagnostics.relaxation_hints
    if hints:
        st.markdown("**What would help:**\n" + "\n".join(f"- {h}" for h in hints))


def render_results_tab(result: RecommendationResult, data: AgentData) -> None:
    """The ranked table; selecting a row moves the route to that venue."""
    if not result.candidates:
        render_hints(result)
        return
    st.caption(
        f"Top {len(result.candidates)} of {result.diagnostics.solutions_found:,} "
        "combinations that pass every hard constraint (at most 2 per venue). "
        "Select a row to route to it."
    )
    st.dataframe(
        results_frame(result, data),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=TABLE_KEY,
        column_config={
            "Price (€)": st.column_config.NumberColumn(format="€%.2f"),
            "Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )


def score_breakdown(cand: Candidate, weights: dict[str, float]) -> tuple[pd.DataFrame, str]:
    """Weighted contribution per feature and the formula with the actual numbers."""
    used = {n: float(weights.get(n, 0.0)) for n in FEATURES}
    if not any(used.values()):
        used = dict.fromkeys(FEATURES, 1.0)
    total_w = sum(used.values())
    frame = pd.DataFrame(
        {
            "feature": [FEATURE_LABELS[n] for n in FEATURES],
            "value": [cand.features[n] for n in FEATURES],
            "weight": [used[n] for n in FEATURES],
            "weighted": [used[n] * cand.features[n] / total_w for n in FEATURES],
        }
    )
    terms = " + ".join(f"{used[n]:g}×{cand.features[n]:.2f}" for n in FEATURES)
    formula = f"Eval = ({terms}) / {total_w:g} = {cand.score:.2f}"
    return frame, formula


def render_why_tab(
    result: RecommendationResult, data: AgentData, prefs: Preferences, selected: int
) -> None:
    """Feature values times weights for the selected pick, the formula, the reasons."""
    if not result.candidates:
        render_hints(result)
        return
    cand = result.candidates[selected]
    wine, venue = data.wines[cand.wine_id], data.venues[cand.venue_id]
    st.subheader(f"#{selected + 1} {wine.name} at {venue.name}")
    frame, formula = score_breakdown(cand, prefs.weights)
    st.markdown("**Contribution of each feature to the score** (weight × value ÷ total weight)")
    st.bar_chart(frame.set_index("feature")["weighted"], horizontal=True, height=240)
    st.code(formula, language=None)
    st.dataframe(
        frame.rename(columns={"weighted": "contribution"}),
        hide_index=True,
        width="stretch",
        column_config={
            "value": st.column_config.NumberColumn(format="%.2f"),
            "contribution": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.markdown("**Why this pick**\n" + "\n".join(f"- {line}" for line in cand.reasons))


def _stats_row(label: str, result: RecommendationResult) -> dict:
    d = result.diagnostics
    return {
        "Techniques": label,
        "Solutions": d.solutions_found,
        "Assignments": d.assignments,
        "Backtracks": d.backtracks,
        "Forward-check prunes": d.forward_check_prunes,
        "AC-3 revisions": d.ac3_revisions,
        "Time (ms)": round(d.elapsed_ms, 1),
        "Same top picks": None,
    }


def compare_techniques(
    run: Callable[[SolverOptions], RecommendationResult],
) -> pd.DataFrame:
    """Run with all techniques on, then with each one off in turn; one row each."""
    baseline = run(SolverOptions())
    reference = [(c.wine_id, c.venue_id, c.dish_id) for c in baseline.candidates]
    rows = [_stats_row("All on", baseline)]
    rows[0]["Same top picks"] = True
    for name, label in TECHNIQUES.items():
        res = run(SolverOptions(**{name: False}))
        row = _stats_row(f"Without {label.split(' (')[0]}", res)
        row["Same top picks"] = [(c.wine_id, c.venue_id, c.dish_id) for c in res.candidates] == (
            reference
        )
        rows.append(row)
    return pd.DataFrame(rows)


def render_diagnostics_tab(
    result: RecommendationResult,
    run: Callable[[SolverOptions], RecommendationResult],
) -> None:
    """Domain sizes, prunes per constraint, search counters, MRV order, comparison."""
    d = result.diagnostics
    cols = st.columns(6)
    cols[0].metric("Solutions found", f"{d.solutions_found:,}")
    cols[1].metric("Assignments", f"{d.assignments:,}")
    cols[2].metric("Backtracks", f"{d.backtracks:,}")
    cols[3].metric("Forward-check prunes", f"{d.forward_check_prunes:,}")
    cols[4].metric("AC-3 revisions", f"{d.ac3_revisions:,}")
    cols[5].metric("Solve time", f"{d.elapsed_ms:.0f} ms")

    left, right = st.columns(2)
    with left:
        st.markdown("**Domain sizes**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Variable": list(VARIABLES),
                    "Start": [d.initial_domain_sizes.get(v, 0) for v in VARIABLES],
                    "After node consistency": [
                        d.domain_sizes_after_node_consistency.get(v, 0) for v in VARIABLES
                    ],
                    "After AC-3": [d.domain_sizes_after_ac3.get(v, 0) for v in VARIABLES],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        if d.variable_order:
            st.markdown(
                f"**Variable order chosen:** {' → '.join(d.variable_order)}  \n"
                "With MRV on, the search always branches next on the variable with the "
                "fewest values left, so dead ends are found as early as possible."
            )
    with right:
        st.markdown("**Values pruned per constraint** (node consistency + AC-3)")
        pruned = pd.Series(d.pruned_by_constraint, name="values removed")
        st.bar_chart(pruned, horizontal=True, height=260)

    st.markdown("**Compare techniques**")
    st.caption(
        "Runs the solver with every technique on, then with each one switched off in "
        "turn. The answer never changes; the amount of work does. (About half a minute: the "
        "run without forward checking is the slow one.)"
    )
    if st.button("Compare techniques"):
        with st.spinner("Solving five times (the run without forward checking takes ~30 s)…"):
            st.dataframe(compare_techniques(run), hide_index=True, width="stretch")
