"""Tests for ui/map_view.py on the miniature fixture world (task 7.3)."""

from __future__ import annotations

import dataclasses

from agent.data_loader import AgentData
from agent.models import Venue
from agent.recommend import recommend
from agent.search_route import RouteResult
from ui.map_view import MAX_RULED_OUT_MARKERS, build_map, view_for
from ui.results_view import results_frame, score_breakdown


def _html(fmap) -> str:
    return fmap.get_root().render()


def test_map_escapes_osm_text_and_uses_local_assets(prefs, agent_data: AgentData):
    venues = dict(agent_data.venues)
    venues["node/1"] = dataclasses.replace(venues["node/1"], name="<script>alert(1)</script>")
    data = dataclasses.replace(agent_data, venues=venues)
    result = recommend(prefs, data)
    assert any(c.venue_id == "node/1" for c in result.candidates)
    html = _html(build_map(result, data, prefs, None, 0, show_ruled_out=False))
    assert "<script>alert(1)</script>" not in html
    assert "cdn.jsdelivr.net" not in html and "/app/static/vendor/leaflet-1.9.3/leaflet.js" in html
    assert "Wine list simulated" in html


def test_ruled_out_markers_are_capped(prefs, agent_data: AgentData):
    venues = dict(agent_data.venues)
    template = venues["node/4"]  # a bar closed at noon -> ruled out by open_now
    for i in range(MAX_RULED_OUT_MARKERS + 50):
        vid = f"node/{1000 + i}"
        venues[vid] = Venue(**{**dataclasses.asdict(template), "venue_id": vid, "name": f"Bar {i}"})
    data = dataclasses.replace(agent_data, venues=venues)
    result = recommend(prefs, data)
    assert len(result.diagnostics.venue_exclusions) > MAX_RULED_OUT_MARKERS
    html = _html(build_map(result, data, prefs, None, 0, show_ruled_out=True))
    assert html.count("ruled out by: open_now") == MAX_RULED_OUT_MARKERS


def test_view_fits_route():
    user = (40.4168, -3.7038)
    short = RouteResult([user, (40.418, -3.703)], 150.0, 3, 2.0)
    long = RouteResult([user, (40.40, -3.72)], 2500.0, 3, 30.0)
    (lat, lon), zoom_short = view_for(user, short)
    assert min(user[0], 40.418) <= lat <= max(user[0], 40.418)
    assert view_for(user, long)[1] < zoom_short <= 16
    assert view_for(user, None) == (user, 16)


def test_results_frame_and_breakdown(prefs, agent_data):
    result = recommend(prefs, agent_data)
    frame = results_frame(result, agent_data)
    assert list(frame["Rank"]) == list(range(1, len(result.candidates) + 1))
    best = result.candidates[0]
    breakdown, formula = score_breakdown(best, prefs.weights)
    assert abs(breakdown["weighted"].sum() - best.score) < 1e-9
    assert formula.endswith(f"= {best.score:.2f}")
