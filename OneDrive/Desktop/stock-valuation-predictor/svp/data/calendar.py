"""
Event calendar — when the next thing that moves the price actually happens.
===========================================================================

Two kinds of date, from two sources that fail differently:

**Earnings** come from yfinance and are a genuine forward-looking calendar
entry. They are also the flakiest field in that API — frequently absent, and
rate-limited when it is not — so every path here returns ``None`` rather than
raising, and the caller shows what it has.

**Filings** come from EDGAR, which only publishes the past. The next 10-Q is
therefore *projected* from the company's own cadence: the median gap between
its recent filings of that form, applied to the latest one. That is an estimate
and is labelled as one — a projected date presented as a fact would be worse
than no date, because a reader would plan around it.

This is what makes a return visit worth something. An event calendar is an
external trigger in the honest sense: it is time-relevant information the user
would want anyway, not a manufactured reason to open the app.
"""

from __future__ import annotations

import datetime as dt
import statistics
from dataclasses import dataclass
from typing import Optional

from . import storage

_EARNINGS_TTL = 6 * 3600
_FILING_TTL = 6 * 3600

# Forms whose cadence is regular enough to project forward. An 8-K is filed
# when something happens, so projecting one would be meaningless.
PERIODIC_FORMS = ("10-Q", "10-K")


@dataclass(frozen=True)
class Event:
    ticker: str
    kind: str                 # "Earnings" | "10-Q" | "10-K"
    date: dt.date
    projected: bool           # True when derived from cadence, not announced
    note: str = ""

    @property
    def days_away(self) -> int:
        return (self.date - dt.date.today()).days

    @property
    def is_past(self) -> bool:
        return self.days_away < 0


def _as_date(value) -> Optional[dt.date]:
    """Coerce the several shapes yfinance and EDGAR use into a date."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except Exception:
        return None


# ── Earnings ─────────────────────────────────────────────────────────────────
def next_earnings(ticker: str) -> Optional[Event]:
    """
    The next scheduled earnings date, or ``None`` if the feed does not have one.

    yfinance exposes this in two places that disagree in shape and availability
    across versions, so both are tried. A company between reporting cycles
    genuinely has no next date, and that is not an error.
    """
    tkr = ticker.upper().strip()
    key = f"earnings_date:{tkr}"
    cached = storage.cache_get(key)
    if cached is not None:
        d = _as_date(cached.get("date"))
        return Event(tkr, "Earnings", d, projected=False) if d else None

    found: Optional[dt.date] = None
    try:
        import yfinance as yf

        t = yf.Ticker(tkr)

        try:
            cal = t.calendar
            if isinstance(cal, dict):
                vals = cal.get("Earnings Date") or []
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                for v in vals:
                    found = _as_date(v)
                    if found:
                        break
            elif cal is not None and getattr(cal, "empty", True) is False:
                # Older yfinance returned a DataFrame with dates in the columns.
                for v in list(cal.columns) + list(cal.values.ravel()):
                    found = _as_date(v)
                    if found:
                        break
        except Exception:
            pass

        if found is None:
            try:
                df = t.get_earnings_dates(limit=8)
                if df is not None and not df.empty:
                    today = dt.date.today()
                    upcoming = [d for d in (_as_date(i) for i in df.index)
                                if d and d >= today]
                    found = min(upcoming) if upcoming else None
            except Exception:
                pass
    except Exception:
        return None

    if found is None:
        return None
    storage.cache_set(key, {"date": found.isoformat()}, ttl=_EARNINGS_TTL)
    return Event(tkr, "Earnings", found, projected=False)


# ── Filings ──────────────────────────────────────────────────────────────────
def recent_filings(ticker: str, limit: int = 8) -> list[Event]:
    """Filings already made, newest first. Past events, plainly labelled."""
    from ..rag import filings as filings_mod

    out: list[Event] = []
    try:
        for f in filings_mod.list_filings(ticker, forms=PERIODIC_FORMS, limit=limit):
            d = _as_date(f.filed)
            if d:
                out.append(Event(ticker.upper(), f.form, d, projected=False,
                                 note="filed"))
    except Exception:
        return []
    return out


def next_filing(ticker: str) -> Optional[Event]:
    """
    Project the next periodic filing from the company's own cadence.

    EDGAR publishes only what has been filed, so this is arithmetic on history:
    the median gap between recent filings of the same form, added to the most
    recent one. Marked ``projected`` so the UI can say so — a guess dressed as a
    fact is worse than no date, because the reader plans around it.
    """
    tkr = ticker.upper().strip()
    key = f"next_filing:{tkr}"
    cached = storage.cache_get(key)
    if cached is not None:
        d = _as_date(cached.get("date"))
        return (Event(tkr, cached.get("form", "10-Q"), d, projected=True,
                      note="projected from filing cadence") if d else None)

    filed = recent_filings(tkr, limit=8)
    if len(filed) < 3:
        return None

    # Quarterlies set the rhythm; a 10-K substitutes for one 10-Q each year, so
    # the combined series is the right one to measure.
    dates = sorted({e.date for e in filed}, reverse=True)
    if len(dates) < 3:
        return None
    gaps = [(dates[i] - dates[i + 1]).days for i in range(len(dates) - 1)]
    gaps = [g for g in gaps if 30 <= g <= 200]
    if not gaps:
        return None

    median_gap = int(statistics.median(gaps))
    nxt = dates[0] + dt.timedelta(days=median_gap)

    # A projection already in the past means the company is late or the cadence
    # changed. Roll forward rather than showing a date that has been and gone.
    today = dt.date.today()
    while nxt < today:
        nxt += dt.timedelta(days=median_gap)

    # Which form falls next: three of every four periodic filings are 10-Qs, and
    # the annual takes the slot roughly a year after the last one. If no 10-K is
    # in the visible history, say 10-Q rather than guessing at an annual.
    last_annual = max((e.date for e in filed if e.kind == "10-K"), default=None)
    form = "10-K" if last_annual and (nxt - last_annual).days >= 350 else "10-Q"

    storage.cache_set(key, {"date": nxt.isoformat(), "form": form}, ttl=_FILING_TTL)
    return Event(tkr, form, nxt, projected=True, note="projected from filing cadence")


# ── Aggregate ────────────────────────────────────────────────────────────────
def upcoming(tickers, within_days: int = 120) -> list[Event]:
    """
    Every known forward-looking event across a list of tickers, soonest first.

    Failures are per-ticker: one symbol with no EDGAR history or a rate-limited
    quote must not empty the whole calendar.
    """
    out: list[Event] = []
    for t in tickers:
        for fn in (next_earnings, next_filing):
            try:
                e = fn(t)
            except Exception:
                e = None
            if e and not e.is_past and e.days_away <= within_days:
                out.append(e)
    return sorted(out, key=lambda e: e.date)
