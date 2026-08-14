"""
Watchlist — the tickers one person is actually following.
=========================================================

Small on purpose. A watchlist is a set of strings against an anonymous key; the
value is not in the storage, it is in what the rest of the app can then do —
score a whole list at once, tell the user which names report this week, and
give them a page that is about *their* companies rather than whichever ticker
they last typed.

Ordering is by insertion, newest first, so the list reads as a history of what
someone has been looking at rather than an alphabetical directory.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from . import _userdb

MAX_ENTRIES = 50

# Tickers on US venues: letters, with an optional class suffix after a dash or
# dot. Also allows a leading caret so index symbols can be followed even though
# the valuation model cannot value them.
_TICKER_RE = re.compile(r"^\^?[A-Z]{1,6}([.\-][A-Z]{1,3})?$")

_DDL_PG = """
CREATE TABLE IF NOT EXISTS svp_watchlist (
    ledger_key TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    added_at   DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ledger_key, ticker)
)
"""

_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS svp_watchlist (
    ledger_key TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    added_at   REAL NOT NULL,
    PRIMARY KEY (ledger_key, ticker)
)
"""

_IDX = ("CREATE INDEX IF NOT EXISTS svp_watch_key ON svp_watchlist (ledger_key)",)
_PG_DDL = (_DDL_PG, *_IDX)
_SQ_DDL = (_DDL_SQLITE, *_IDX)


def normalise(ticker: str) -> Optional[str]:
    """Upper-case and validate a symbol, or ``None`` if it is not one."""
    t = (ticker or "").strip().upper()
    return t if _TICKER_RE.match(t) else None


def add(ledger_key: str, ticker: str) -> bool:
    """
    Add a ticker. Returns True if it is on the list afterwards.

    Idempotent: adding a symbol already present succeeds silently rather than
    erroring, because from the user's point of view the request was "make sure
    this is on my list" and it is.
    """
    tkr = normalise(ticker)
    if not _userdb.valid_key(ledger_key) or not tkr:
        return False
    key = ledger_key.strip().lower()

    if tkr not in get(key) and len(get(key)) >= MAX_ENTRIES:
        return False

    now = time.time()
    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO svp_watchlist (ledger_key, ticker, added_at) "
                        "VALUES (%s,%s,%s) ON CONFLICT DO NOTHING", (key, tkr, now),
                    )
                return True
            except Exception:
                pass
        try:
            c = _userdb.sq(_SQ_DDL)
            c.execute(
                "INSERT OR IGNORE INTO svp_watchlist (ledger_key, ticker, added_at) "
                "VALUES (?,?,?)", (key, tkr, now),
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
                    cur.execute(
                        "DELETE FROM svp_watchlist WHERE ledger_key=%s AND ticker=%s",
                        (key, tkr),
                    )
                return True
            except Exception:
                pass
        try:
            c = _userdb.sq(_SQ_DDL)
            c.execute("DELETE FROM svp_watchlist WHERE ledger_key=? AND ticker=?",
                      (key, tkr))
            c.commit()
            return True
        except Exception:
            return False


def get(ledger_key: str) -> list[str]:
    """The watchlist, most recently added first."""
    if not _userdb.valid_key(ledger_key):
        return []
    key = ledger_key.strip().lower()

    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "SELECT ticker FROM svp_watchlist WHERE ledger_key=%s "
                        "ORDER BY added_at DESC LIMIT %s", (key, MAX_ENTRIES),
                    )
                    return [r[0] for r in cur.fetchall()]
            except Exception:
                pass
        try:
            cur = _userdb.sq(_SQ_DDL).execute(
                "SELECT ticker FROM svp_watchlist WHERE ledger_key=? "
                "ORDER BY added_at DESC LIMIT ?", (key, MAX_ENTRIES),
            )
            return [r[0] for r in cur.fetchall()]
        except Exception:
            return []


def contains(ledger_key: str, ticker: str) -> bool:
    tkr = normalise(ticker)
    return bool(tkr) and tkr in get(ledger_key)


def toggle(ledger_key: str, ticker: str) -> bool:
    """Add if absent, remove if present. Returns True if it is now on the list."""
    if contains(ledger_key, ticker):
        remove(ledger_key, ticker)
        return False
    add(ledger_key, ticker)
    return contains(ledger_key, ticker)
