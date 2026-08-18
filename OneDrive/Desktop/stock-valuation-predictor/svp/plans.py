"""
Plan limits — one table of numbers, enforced where the rows are written.
========================================================================

The app stays anonymous-first: an anonymous visitor saves work exactly like a
signed-in one, because the ledger-key identity *is* the product's identity
model and a save-wall would gut it. What plans gate is **capacity**, not
capability — how many scenarios, journal entries, holdings and active shared
links one key can hold. Free limits are set high enough that a genuine
personal workflow never hits them; premium exists as headroom and is granted
by the ``plan`` column on the profile row (there is no billing flow — an
operator sets it in the database, and the UI never pretends otherwise).

Enforcement happens in the data modules' save paths via the ``cap`` argument
each accepts — UI checks alone would be decoration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    scenarios: int          # saved DCF scenarios per key (across tickers)
    journal_entries: int
    holdings: int
    active_shares: int      # live shared-report links
    watchlist: int


_PLANS = {
    "free": Limits(scenarios=40, journal_entries=60, holdings=30,
                   active_shares=10, watchlist=50),
    "premium": Limits(scenarios=200, journal_entries=500, holdings=100,
                      active_shares=50, watchlist=200),
    "admin": Limits(scenarios=1000, journal_entries=2000, holdings=500,
                    active_shares=200, watchlist=500),
}


def limits(plan: str | None) -> Limits:
    """The limits for a plan name; unknown or absent plans read as free."""
    return _PLANS.get((plan or "free").strip().lower(), _PLANS["free"])
