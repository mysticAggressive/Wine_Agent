"""Front-end smoke tests with Streamlit's AppTest (tasks 7.5 and 8.2)."""

from __future__ import annotations

import pytest

from agent.config import DATA, ROOT

REAL_FILES = (
    "wines.csv",
    "venues_madrid.json",
    "inventory.csv",
    "pairing_rules.json",
    "madrid_walk.graphml",
)
pytestmark = pytest.mark.skipif(
    not all((DATA / name).exists() for name in REAL_FILES), reason="real data not built"
)
APP = str(ROOT / "app.py")
TIMEOUT = 240


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_defaults_render_results(app):
    assert len(app.tabs) == 4
    assert len(app.dataframe) >= 1
    results = app.dataframe[0].value
    assert 1 <= len(results) <= 10
    assert list(results["Rank"]) == list(range(1, len(results) + 1))
    assert not app.warning
    assert any(m.value.startswith("**Route to ") for m in app.markdown)
    assert any("OpenStreetMap" in c.value for c in app.caption)
    assert any("simulated" in c.value for c in app.caption)


def test_diagnostics_and_sommelier_tab_render(app):
    labels = {m.label for m in app.metric}
    assert {"Solutions found", "Assignments", "AC-3 revisions", "Solve time"} <= labels
    assert {"Minimax visited", "Alpha-beta visited"} <= labels


def test_compare_techniques_button(app):
    at = app.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    table = next(d.value for d in at.dataframe if "Techniques" in d.value.columns)
    assert len(table) == 5
    assert table["Same top picks"].all()
    assert table["Solutions"].nunique() == 1


def test_impossible_preferences_show_hints_not_a_crash():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.sidebar.slider[0].set_value(10)  # budget
    at.sidebar.multiselect[0].set_value(["Fortified"])
    at.sidebar.radio[0].set_value("Restaurant or bar")
    at.sidebar.slider[1].set_value(200)  # max walking distance
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.warning
    assert "No wine, venue and dish" in at.warning[0].value
    assert any("What would help" in m.value for m in at.markdown)
