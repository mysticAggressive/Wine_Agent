"""Opening-hours evaluation. See DEVELOPMENT_PLAN.md section 3.6 and task 2.4."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from opening_hours import OpeningHours

from agent.config import TIMEZONE
from agent.models import OpenState

_TZ = ZoneInfo(TIMEZONE)


@lru_cache(maxsize=8192)
def _parse(tag: str) -> OpeningHours | None:
    """Parse an OSM ``opening_hours`` tag; None if it cannot be parsed."""
    try:
        return OpeningHours(tag)
    except Exception:  # noqa: BLE001 - any parser failure means "unknown"
        return None


@lru_cache(maxsize=65536)
def _state_at(tag: str, local_minute: datetime) -> OpenState:
    """State of ``tag`` at a naive Europe/Madrid datetime rounded to the minute."""
    parsed = _parse(tag)
    if parsed is None:
        return OpenState.UNKNOWN
    try:
        if parsed.is_unknown(local_minute):
            return OpenState.UNKNOWN
        return OpenState.OPEN if parsed.is_open(local_minute) else OpenState.CLOSED
    except Exception:  # noqa: BLE001 - evaluation failure means "unknown"
        return OpenState.UNKNOWN


def open_state(opening_hours: str, at: datetime) -> OpenState:
    """Say whether a venue with the OSM tag ``opening_hours`` is open at ``at``.

    ``at`` is converted to Europe/Madrid local time (a naive datetime is taken to be
    Madrid local time already). An empty or unparseable tag gives ``UNKNOWN``. Public
    holidays are not taken into account. Results are cached per (tag, minute).
    """
    tag = (opening_hours or "").strip()
    if not tag:
        return OpenState.UNKNOWN
    try:
        if at.tzinfo is not None:
            at = at.astimezone(_TZ).replace(tzinfo=None)
        local_minute = at.replace(second=0, microsecond=0)
    except Exception:  # noqa: BLE001
        return OpenState.UNKNOWN
    return _state_at(tag, local_minute)
