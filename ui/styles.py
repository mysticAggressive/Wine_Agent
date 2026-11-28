"""Page CSS for the "Cellar Night" look (colours and fonts live in .streamlit/config.toml).

Rules target ``st-key-<key>`` classes, which Streamlit adds to keyed containers, so they
do not depend on Streamlit's internal markup.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from streamlit.delta_generator import DeltaGenerator

GOLD = "#e3b35b"
WINE = "#c2375a"

CSS = f"""
<style>
/* Clear Streamlit's fixed top bar (Deploy menu), which otherwise covers the title. */
.block-container {{ padding-top: 4.5rem; max-width: 1280px; margin: 0 auto; }}

/* Brand header: logo slot, title, tagline */
.st-key-brand iframe {{ display: block; }}
.wa-brand h1 {{
  margin: 0; padding: 0; font-size: 2.6rem; font-weight: 700; line-height: 1.1;
  letter-spacing: .01em;
  /* Wine-label display serifs; all system fonts, so nothing is fetched at run time. */
  font-family: Didot, "Bodoni 72", "Bodoni MT", "Playfair Display", Georgia, serif;
}}
.wa-brand h1 span {{ color: {GOLD}; font-style: italic; }}
.wa-tagline {{ margin: 0; opacity: .65; text-align: center; }}

/* Filter chips: popover buttons as pills */
.st-key-filters button {{
  border-radius: 999px !important;
  border: 1px solid #3a2832 !important;
  background: #22161e !important;
  transition: border-color .15s, background .15s, transform .15s;
}}
.st-key-filters button:hover {{
  border-color: {WINE} !important; background: #2c1c26 !important; transform: translateY(-1px);
}}

/* Pick card */
.st-key-pick {{
  background: radial-gradient(120% 140% at 0% 0%, #3b1426 0%, #22161e 55%, #1a1117 100%);
  border: 1px solid #4a2a3a; border-radius: 1.1rem; padding: 1rem 1.25rem;
  box-shadow: 0 10px 30px -12px rgba(194, 55, 90, .35);
  animation: wa-rise .45s ease-out;
}}
.st-key-pick [data-testid="stMetricValue"] {{ color: {GOLD}; }}
.wa-pick-title {{ font-family: var(--st-heading-font, serif); font-size: 1.35rem;
  line-height: 1.25; }}
.wa-pick-sub {{ opacity: .75; margin-top: .2rem; }}
.wa-rank {{ color: {GOLD}; font-weight: 600; letter-spacing: .04em; font-size: .8rem;
  text-transform: uppercase; }}

/* Score ring: conic gradient filled to the score */
.wa-ring {{
  --p: 0; width: 78px; height: 78px; border-radius: 50%;
  background: conic-gradient({WINE} calc(var(--p) * 1%), #3a2832 0);
  display: grid; place-items: center; animation: wa-fade .6s ease-out;
}}
.wa-ring div {{
  width: 62px; height: 62px; border-radius: 50%; background: #1d1218;
  display: grid; place-items: center; font-weight: 700; font-size: 1.15rem;
}}
.wa-ring small {{ display: block; font-size: .6rem; font-weight: 500; opacity: .6;
  text-transform: uppercase; letter-spacing: .08em; margin-top: -.35rem; }}

/* Map and list panels */
.st-key-map_panel, .st-key-list_panel {{ animation: wa-fade .4s ease-out; }}
.st-key-map_panel iframe {{ border-radius: .9rem; }}

/* Tabs: gold underline */
.stTabs [data-baseweb="tab-highlight"] {{ background: {GOLD}; }}
.stTabs [aria-selected="true"] p {{ color: {GOLD}; }}

@keyframes wa-rise {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; }} }}
@keyframes wa-fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@media (prefers-reduced-motion: reduce) {{
  * {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


def inject_css() -> None:
    """Add the page stylesheet (once per run, before anything it styles)."""
    st.html(CSS)


def brand_header() -> DeltaGenerator:
    """Centred logo slot, title and tagline; returns the slot for :func:`glass_logo`."""
    row = st.container(
        key="brand", horizontal=True, horizontal_alignment="center", vertical_alignment="center"
    )
    slot = row.container(width=28)
    row.html('<div class="wa-brand"><h1>Madrid <span>Wine</span> Agent</h1></div>', width="content")
    st.html(
        '<p class="wa-tagline">The best wine, venue and dish near you, and how to walk there.</p>'
    )
    return slot


GLASS_ICON = (Path(__file__).parent / "glass_icon.html").read_text(encoding="utf-8")


def glass_logo(slot: DeltaGenerator, signature: str) -> None:
    """Draw the animated glass in ``slot``; a new ``signature`` reloads it with a slosh."""
    with slot:
        components.html(f"{GLASS_ICON}<!-- {signature} -->", width=28, height=52)


def score_ring(score: float) -> str:
    """HTML for a ring filled to ``score`` (0–1)."""
    pct = max(0.0, min(1.0, score)) * 100
    return (
        f'<div class="wa-ring" style="--p:{pct:.0f}" title="Evaluation score">'
        f"<div><span>{score:.2f}<small>score</small></span></div></div>"
    )
