"""Tests for agent/opening.py (task 2.4)."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from agent.models import OpenState
from agent.opening import _state_at, open_state

MADRID = ZoneInfo("Europe/Madrid")


def madrid(day: int, hour: int, minute: int = 0) -> datetime:
    """A November 2026 datetime in Madrid. The 9th is a Monday, the 15th a Sunday."""
    return datetime(2026, 11, day, hour, minute, tzinfo=MADRID)


def test_weekday_range():
    assert open_state("Mo-Fr 10:00-20:00", madrid(11, 12)) is OpenState.OPEN  # Wednesday
    assert open_state("Mo-Fr 10:00-20:00", madrid(15, 12)) is OpenState.CLOSED  # Sunday
    assert open_state("Mo-Fr 10:00-20:00", madrid(11, 9, 59)) is OpenState.CLOSED
    assert open_state("Mo-Fr 10:00-20:00", madrid(11, 20, 0)) is OpenState.CLOSED


def test_always_open():
    assert open_state("24/7", madrid(15, 3)) is OpenState.OPEN


@pytest.mark.parametrize("tag", ["", "   ", "garbage string", "by appointment", "xx 99:99"])
def test_empty_or_garbage_is_unknown(tag):
    assert open_state(tag, madrid(11, 12)) is OpenState.UNKNOWN


def test_range_crossing_midnight():
    assert open_state("Fr 20:00-02:00", madrid(14, 1)) is OpenState.OPEN  # Saturday 01:00
    assert open_state("Fr 20:00-02:00", madrid(14, 3)) is OpenState.CLOSED
    assert open_state("Fr 20:00-02:00", madrid(13, 1)) is OpenState.CLOSED  # Friday 01:00


def test_split_hours_and_closed_day():
    tag = "Tu-Sa 10:00-14:00,17:00-20:30; Su 11:00-15:00"
    assert open_state(tag, madrid(9, 12)) is OpenState.CLOSED  # Monday
    assert open_state(tag, madrid(10, 12)) is OpenState.OPEN
    assert open_state(tag, madrid(10, 15)) is OpenState.CLOSED  # siesta
    assert open_state(tag, madrid(15, 12)) is OpenState.OPEN


def test_explicit_unknown_modifier():
    assert open_state("Mo-Fr 10:00-20:00 unknown", madrid(11, 12)) is OpenState.UNKNOWN


def test_other_timezones_are_converted_to_madrid():
    # 19:30 UTC in November is 20:30 in Madrid: closed.
    assert open_state("Mo-Fr 10:00-20:00", datetime(2026, 11, 11, 19, 30, tzinfo=UTC)) is (
        OpenState.CLOSED
    )
    # 09:30 UTC is 10:30 in Madrid: open.
    assert open_state("Mo-Fr 10:00-20:00", datetime(2026, 11, 11, 9, 30, tzinfo=UTC)) is (
        OpenState.OPEN
    )
    # summer time: 08:30 UTC in July is 10:30 in Madrid.
    assert open_state("Mo-Fr 10:00-20:00", datetime(2026, 7, 8, 8, 30, tzinfo=UTC)) is (
        OpenState.OPEN
    )


def test_naive_datetime_is_madrid_local():
    assert open_state("Mo-Fr 10:00-20:00", datetime(2026, 11, 11, 12, 0)) is OpenState.OPEN


def test_results_are_cached_per_minute():
    _state_at.cache_clear()
    tag = "Mo-Fr 10:00-20:00"
    open_state(tag, madrid(11, 12).replace(second=5))
    open_state(tag, madrid(11, 12).replace(second=40, microsecond=7))
    info = _state_at.cache_info()
    assert (info.misses, info.hits) == (1, 1)
    open_state(tag, madrid(11, 12, 1))
    assert _state_at.cache_info().misses == 2
