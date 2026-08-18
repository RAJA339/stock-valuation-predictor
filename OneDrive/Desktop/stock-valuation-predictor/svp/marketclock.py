"""
US equity market session clock.
===============================

One question, answered honestly: is the regular session open right now?
Weekday and clock-time only — exchange holidays and half-days are *not*
modelled, and the note says so rather than pretending. A wrong "Open" on
Thanksgiving would be exactly the kind of silent lie this app refuses
elsewhere, so the label carries its own caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

_PRE_START = time(4, 0)
_OPEN = time(9, 30)
_CLOSE = time(16, 0)
_POST_END = time(20, 0)


@dataclass
class MarketStatus:
    label: str        # Open | Pre-market | After-hours | Closed
    note: str
    is_open: bool     # regular session only


def status(now: datetime | None = None) -> MarketStatus:
    """Session status for US equities at ``now`` (any tz; default: current)."""
    now = (now.astimezone(ET) if now is not None and now.tzinfo
           else now.replace(tzinfo=ET) if now is not None
           else datetime.now(ET))
    caveat = "Holidays not modelled — a market holiday reads as a normal day."
    stamp = now.strftime("%a %b %d, %I:%M %p ET")

    if now.weekday() >= 5:
        return MarketStatus("Closed", f"Weekend · {stamp}", False)
    t = now.time()
    if _OPEN <= t < _CLOSE:
        return MarketStatus("Open", f"Regular session · {stamp} · {caveat}", True)
    if _PRE_START <= t < _OPEN:
        return MarketStatus("Pre-market", f"Regular open 9:30 ET · {stamp} · {caveat}",
                            False)
    if _CLOSE <= t < _POST_END:
        return MarketStatus("After-hours", f"Regular close was 4:00 ET · {stamp} · "
                            f"{caveat}", False)
    return MarketStatus("Closed", f"Next regular session 9:30–16:00 ET · {stamp}",
                        False)
