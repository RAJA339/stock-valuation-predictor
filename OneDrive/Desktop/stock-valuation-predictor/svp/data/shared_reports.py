"""
Shared reports — a frozen snapshot behind a revocable slug.
===========================================================

A ``?t=`` link re-runs the live analysis; a shared report is the opposite: a
snapshot of what the app said at one moment, addressed by a random slug, so
the sender controls exactly what the recipient sees and can take it back.
Three properties carry the feature:

- **Frozen.** The payload is stored JSON; the recipient sees what was shared,
  stamped with when, and is pointed at the live analysis for today's view.
- **Revocable.** Deleting the row kills the link. Expiry does the same on a
  clock. Both are the *sender's* controls.
- **Nothing private.** The payload is composed by the sharing UI from the
  analysis surface only — journal entries, holdings, watchlists and the
  ledger key never enter it, and :func:`create` strips any key that looks
  like it is trying to (defense against a future caller's mistake).
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from typing import Optional

from . import _userdb
from .watchlist import normalise

DEFAULT_CAP = 10
_FORBIDDEN_KEYS = {"ledger_key", "journal", "holdings", "watchlist", "email",
                   "profile", "sub", "id"}

_DDL_PG = """
CREATE TABLE IF NOT EXISTS svp_shared_reports (
    slug       TEXT PRIMARY KEY,
    ledger_key TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION
)
"""

_DDL_SQLITE = _DDL_PG.replace("DOUBLE PRECISION", "REAL")

_IDX = ("CREATE INDEX IF NOT EXISTS svp_shared_key ON svp_shared_reports (ledger_key)",)
_PG_DDL = (_DDL_PG, *_IDX)
_SQ_DDL = (_DDL_SQLITE, *_IDX)


@dataclass
class SharedReport:
    slug: str
    ticker: str
    payload: dict
    created_at: float
    expires_at: Optional[float]

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at


def create(ledger_key: str, ticker: str, payload: dict,
           expires_days: Optional[int] = None,
           cap: int = DEFAULT_CAP) -> Optional[str]:
    """Store a snapshot; returns the slug, or ``None`` (cap, bad input)."""
    tkr = normalise(ticker)
    if not _userdb.valid_key(ledger_key) or not tkr or not isinstance(payload, dict):
        return None
    key = ledger_key.strip().lower()
    if len(list_for(key)) >= cap:
        return None

    clean = {k: v for k, v in payload.items()
             if k.lower() not in _FORBIDDEN_KEYS}
    try:
        blob = json.dumps(clean, default=str)
    except Exception:
        return None

    slug = secrets.token_urlsafe(8)
    now = time.time()
    exp = now + expires_days * 86400 if expires_days else None

    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO svp_shared_reports "
                        "(slug, ledger_key, ticker, payload, created_at, expires_at) "
                        "VALUES (%s,%s,%s,%s,%s,%s)",
                        (slug, key, tkr, blob, now, exp),
                    )
                return slug
            except Exception:
                pass
        try:
            c = _userdb.sq(_SQ_DDL)
            c.execute(
                "INSERT INTO svp_shared_reports "
                "(slug, ledger_key, ticker, payload, created_at, expires_at) "
                "VALUES (?,?,?,?,?,?)", (slug, key, tkr, blob, now, exp))
            c.commit()
            return slug
        except Exception:
            return None


def get(slug: str) -> Optional[SharedReport]:
    """The snapshot behind a slug — ``None`` if unknown, revoked or expired."""
    slug = (slug or "").strip()
    if not slug or len(slug) > 64:
        return None
    q = ("SELECT slug, ticker, payload, created_at, expires_at "
         "FROM svp_shared_reports WHERE slug={}")

    row = None
    answered = False
    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(q.format("%s"), (slug,))
                    row = cur.fetchone()
                answered = True
            except Exception:
                answered = False
        if not answered:
            try:
                row = _userdb.sq(_SQ_DDL).execute(q.format("?"), (slug,)).fetchone()
            except Exception:
                row = None
    if not row:
        return None
    try:
        rep = SharedReport(slug=row[0], ticker=row[1],
                           payload=json.loads(row[2]),
                           created_at=float(row[3]),
                           expires_at=float(row[4]) if row[4] is not None else None)
    except Exception:
        return None
    return None if rep.expired else rep


def list_for(ledger_key: str) -> list[SharedReport]:
    """The sender's own links (expired ones included, so they can be culled)."""
    if not _userdb.valid_key(ledger_key):
        return []
    key = ledger_key.strip().lower()
    q = ("SELECT slug, ticker, payload, created_at, expires_at "
         "FROM svp_shared_reports WHERE ledger_key={} ORDER BY created_at DESC")

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
    out = []
    for r in rows:
        try:
            out.append(SharedReport(slug=r[0], ticker=r[1],
                                    payload=json.loads(r[2]),
                                    created_at=float(r[3]),
                                    expires_at=float(r[4]) if r[4] is not None
                                    else None))
        except Exception:
            continue
    return out


def revoke(ledger_key: str, slug: str) -> bool:
    """Kill one link. Only the owning key can."""
    if not _userdb.valid_key(ledger_key) or not slug:
        return False
    key = ledger_key.strip().lower()
    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute("DELETE FROM svp_shared_reports WHERE "
                                "ledger_key=%s AND slug=%s", (key, slug))
                return True
            except Exception:
                pass
        try:
            c = _userdb.sq(_SQ_DDL)
            c.execute("DELETE FROM svp_shared_reports WHERE ledger_key=? AND slug=?",
                      (key, slug))
            c.commit()
            return True
        except Exception:
            return False
