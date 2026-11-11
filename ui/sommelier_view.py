"""The "Sommelier vs. you" tab: minimax and alpha-beta over the CSP solutions (task 8.2)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from agent.data_loader import AgentData
from agent.minimax import SommelierAnalysis
from agent.models import RecommendationResult
from ui.results_view import render_hints


def sommelier_frame(analysis: SommelierAnalysis, data: AgentData) -> pd.DataFrame:
    """One row per playable venue, lowest minimax value (most sommelier-proof) first."""
    rows = []
    for line in analysis.venues:
        tag = ""
        if line.venue_id == analysis.sommelier_proof.venue_id:
            tag = "🛡️ sommelier-proof"
        elif line.venue_id == analysis.sommelier_favourite.venue_id:
            tag = "💰 sommelier's favourite"
        wine = data.wines[line.wine.wine_id]
        rows.append(
            {
                "Venue": data.venues[line.venue_id].name,
                "Minimax value": line.value,
                "You choose": data.pairing_rules.dishes[line.dish_id].label,
                "Sommelier then pours": f"{wine.name} ({wine.winery})",
                "Price (€)": line.price,
                "": tag,
            }
        )
    return pd.DataFrame(rows)


def render_sommelier_tab(
    analysis: SommelierAnalysis | None, result: RecommendationResult, data: AgentData
) -> None:
    """Explain the game, show each venue's optimal line, and the search node counts."""
    st.markdown(
        "The sommelier wants to upsell you (score = ½ quality + ½ price ÷ budget); "
        "you want to keep that down. The sommelier picks one of the top venues, you "
        "pick a dish it serves, then the sommelier picks the wine — so each venue's "
        "**minimax value** is the upsell you can hold it to with best play."
    )
    if analysis is None:
        render_hints(result)
        return

    proof, fav = analysis.sommelier_proof, analysis.sommelier_favourite
    c1, c2 = st.columns(2)
    c1.success(
        f"🛡️ **Sommelier-proof:** {data.venues[proof.venue_id].name} "
        f"(value {proof.value:.2f}) — order "
        f"{data.pairing_rules.dishes[proof.dish_id].label.lower()}."
    )
    c2.error(
        f"💰 **Sommelier's favourite:** {data.venues[fav.venue_id].name} "
        f"(value {fav.value:.2f}). Optimal play over the whole game: "
        + " → ".join(
            [
                data.venues[analysis.principal_variation[0]].name,
                data.pairing_rules.dishes[analysis.principal_variation[1]].label,
                data.wines[analysis.principal_variation[2]].name,
            ]
        )
    )
    st.dataframe(
        sommelier_frame(analysis, data),
        hide_index=True,
        width="stretch",
        column_config={
            "Minimax value": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "Price (€)": st.column_config.NumberColumn(format="€%.2f"),
        },
    )
    m1, m2 = st.columns(2)
    m1.metric("Minimax visited", f"{analysis.minimax_nodes:,} nodes")
    m2.metric(
        "Alpha-beta visited",
        f"{analysis.alphabeta_nodes:,} nodes",
        delta=f"{analysis.percent_saved:.0f}% fewer",
        delta_color="normal",
    )
