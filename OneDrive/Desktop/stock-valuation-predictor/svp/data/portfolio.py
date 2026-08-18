"""
Portfolio — manually entered holdings, never brokerage credentials.
===================================================================

A row is what the user typed: ticker, share count, average cost, a note.
Everything derived (market value, unrealized P/L, weights) is computed at
render time from live prices, so the storage never goes stale and never
pretends to know a price. No brokerage linking exists or will: the trust
copy says "we never access your brokerage account", and the schema is how
that stays true.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from . import _userdb
from .watchlist import normalise

DEFAULT_CAP = 30
MAX_NOTE = 300

_DDL_PG = """
CREATE TABLE IF NOT EXISTS svp_holdings (
    ledger_key TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    shares     DOUBLE PRECISION NOT NULL,
    avg_cost   DOUBLE PRECISION NOT NULL,
    note       TEXT,
    updated_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ledger_key, ticker)
)
"""

_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS svp_holdings (
    ledger_key TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    shares     REAL NOT NULL,
    avg_cost   REAL NOT NULL,
    note       TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (ledger_key, ticker)
)
"""

_PG_DDL = (_DDL_PG,)
_SQ_DDL = (_DDL_SQLITE,)


@dataclass
class Holding:
    ticker: str
    shares: float
    avg_cost: float
    note: str
    updated_at: float

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost


def upsert(ledger_key: str, ticker: str, shares: float, avg_cost: float,
           note: str = "", cap: int = DEFAULT_CAP) -> bool:
    """
    Add or replace one holding. Returns True when stored.

    Re-entering a ticker replaces the row — "make my AAL position be this" —
    so the cap blocks only genuinely new tickers. Zero/negative shares or
    cost are refused: a closed position is deleted, not stored as zero.
    """
    tkr = normalise(ticker)
    if not _userdb.valid_key(ledger_key) or not tkr:
        return False
    if shares is None or avg_cost is None or shares <= 0 or avg_cost <= 0:
        return False
    key = ledger_key.strip().lower()
    note = (note or "").strip()[:MAX_NOTE]

    existing = {h.ticker for h in list_for(key)}
    if tkr not in existing and len(existing) >= cap:
        return False

    now = time.time()
    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO svp_holdings "
                        "(ledger_key, ticker, shares, avg_cost, note, updated_at) "
                        "VALUES (%s,%s,%s,%s,%s,%s) "
                        "ON CONFLICT (ledger_key, ticker) DO UPDATE SET "
                        "shares=EXCLUDED.shares, avg_cost=EXCLUDED.avg_cost, "
                        "note=EXCLUDED.note, updated_at=EXCLUDED.updated_at",
                        (key, tkr, float(shares), float(avg_cost), note, now),
                    )
                return True
            except Exception:
                pass
        try:
            c = _userdb.sq(_SQ_DDL)
            c.execute(
                "INSERT OR REPLACE INTO svp_holdings "
                "(ledger_key, ticker, shares, avg_cost, note, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (key, tkr, float(shares), float(avg_cost), note, now),
            )
            c.commit()
            return True
        except Exception:
            return False


def remove(ledger_key: str, ticker: str) -> bool:
    tkr = normalise(ticker)
    if not _userdb.valid_key(ledger_key) or not tkr:
        return False
    key = ledger_key.strip().lower()
    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute("DELETE FROM svp_holdings WHERE ledger_key=%s "
                                "AND ticker=%s", (key, tkr))
                return True
            except Exception:
                pass
        try:
            c = _userdb.sq(_SQ_DDL)
            c.execute("DELETE FROM svp_holdings WHERE ledger_key=? AND ticker=?",
                      (key, tkr))
            c.commit()
            return True
        except Exception:
            return False


def list_for(ledger_key: str) -> list[Holding]:
    """This key's holdings, largest cost basis first."""
    if not _userdb.valid_key(ledger_key):
        return []
    key = ledger_key.strip().lower()
    q = ("SELECT ticker, shares, avg_cost, note, updated_at FROM svp_holdings "
         "WHERE ledger_key={}")

    rows = None
    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(q.format("%s"), (key,))
                    rows = cur.fetchall()
            except Exception:
                rows = None
        if rows is None:
            try:
                rows = _userdb.sq(_SQ_DDL).execute(q.format("?"), (key,)).fetchall()
            except Exception:
                rows = []

    out = [Holding(ticker=r[0], shares=float(r[1]), avg_cost=float(r[2]),
                   note=r[3] or "", updated_at=float(r[4])) for r in rows]
    return sorted(out, key=lambda h: -h.cost_basis)


@dataclass
class PortfolioView:
    """Derived totals over holdings priced by the caller."""
    market_value: float
    cost_basis: float
    unrealized: float
    unrealized_pct: Optional[float]
    priced: int                 # holdings a live price was found for
    unpriced: list[str]         # tickers with no price — shown, not hidden
    top_weight: Optional[tuple[str, float]] = None   # (ticker, share of MV)


def value(holdings: list[Holding], price_of) -> PortfolioView:
    """
    Price the book with ``price_of(ticker) -> Optional[float]``.

    A ticker the pricer cannot answer for is *excluded from the totals and
    named in* ``unpriced`` — folding a stale cost basis into market value
    would make the total a mixture of two different kinds of number.
    """
    mv = cb = 0.0
    unpriced: list[str] = []
    weights: list[tuple[str, float]] = []
    for h in holdings:
        try:
            px = price_of(h.ticker)
        except Exception:
            px = None
        if px is None or px <= 0:
            unpriced.append(h.ticker)
            continue
        v = h.shares * float(px)
        mv += v
        cb += h.cost_basis
        weights.append((h.ticker, v))
    unreal = mv - cb
    top = None
    if weights and mv > 0:
        t, v = max(weights, key=lambda x: x[1])
        top = (t, v / mv)
    return PortfolioView(
        market_value=mv, cost_basis=cb, unrealized=unreal,
        unrealized_pct=(unreal / cb if cb > 0 else None),
        priced=len(weights), unpriced=unpriced, top_weight=top,
    )
