"""
Investment journal — the thesis, written down before the outcome.
=================================================================

The ledger records what the *model* said; the journal records what the
*person* believed and, crucially, what would prove them wrong. The
``invalidation`` field is the one that pays for the feature: a thesis with a
falsifier written in advance can be honestly closed, while one without can
always be re-narrated. Entries are keyed by id so several theses about the
same ticker can coexist (pre-earnings, post-earnings).

Nothing here is advice, and nothing here is shared — journal entries never
appear in shared reports.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from . import _userdb
from .watchlist import normalise

DEFAULT_CAP = 60
MAX_FIELD = 2000

#: The free-text sections, in the order the UI shows them.
SECTIONS = ("thesis", "bull_case", "bear_case", "catalysts", "risks",
            "invalidation", "notes")

_COLS = ("entry_id", "ledger_key", "ticker", "thesis", "bull_case",
         "bear_case", "catalysts", "risks", "invalidation", "entry_price",
         "target_price", "horizon", "notes", "created_at", "updated_at")

_DDL_PG = """
CREATE TABLE IF NOT EXISTS svp_journal (
    entry_id     TEXT PRIMARY KEY,
    ledger_key   TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    thesis       TEXT,
    bull_case    TEXT,
    bear_case    TEXT,
    catalysts    TEXT,
    risks        TEXT,
    invalidation TEXT,
    entry_price  DOUBLE PRECISION,
    target_price DOUBLE PRECISION,
    horizon      TEXT,
    notes        TEXT,
    created_at   DOUBLE PRECISION NOT NULL,
    updated_at   DOUBLE PRECISION NOT NULL
)
"""

_DDL_SQLITE = _DDL_PG.replace("DOUBLE PRECISION", "REAL")

_IDX = ("CREATE INDEX IF NOT EXISTS svp_journal_key ON svp_journal (ledger_key)",)
_PG_DDL = (_DDL_PG, *_IDX)
_SQ_DDL = (_DDL_SQLITE, *_IDX)


@dataclass
class Entry:
    entry_id: str
    ticker: str
    thesis: str = ""
    bull_case: str = ""
    bear_case: str = ""
    catalysts: str = ""
    risks: str = ""
    invalidation: str = ""
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    horizon: str = ""
    notes: str = ""
    created_at: float = field(default=0.0)
    updated_at: float = field(default=0.0)


def _clip(s: Optional[str]) -> str:
    return (s or "").strip()[:MAX_FIELD]


def save(ledger_key: str, ticker: str, *, entry_id: Optional[str] = None,
         thesis: str = "", bull_case: str = "", bear_case: str = "",
         catalysts: str = "", risks: str = "", invalidation: str = "",
         entry_price: Optional[float] = None,
         target_price: Optional[float] = None, horizon: str = "",
         notes: str = "", cap: int = DEFAULT_CAP) -> Optional[str]:
    """
    Create (``entry_id=None``) or update one entry. Returns the entry id.

    A new entry needs at least a thesis — an empty journal row is noise. The
    cap counts this key's entries and blocks only creation, never updates.
    """
    tkr = normalise(ticker)
    if not _userdb.valid_key(ledger_key) or not tkr:
        return None
    key = ledger_key.strip().lower()
    creating = entry_id is None
    if creating and not _clip(thesis):
        return None
    if creating and len(list_for(key)) >= cap:
        return None
    eid = entry_id or uuid.uuid4().hex[:16]
    now = time.time()

    ep = float(entry_price) if entry_price else None
    tp = float(target_price) if target_price else None
    vals = (eid, key, tkr, _clip(thesis), _clip(bull_case), _clip(bear_case),
            _clip(catalysts), _clip(risks), _clip(invalidation), ep, tp,
            _clip(horizon)[:100], _clip(notes), now, now)
    upd_cols = ("ticker", "thesis", "bull_case", "bear_case", "catalysts",
                "risks", "invalidation", "entry_price", "target_price",
                "horizon", "notes", "updated_at")

    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                sets = ", ".join(f"{c}=EXCLUDED.{c}" for c in upd_cols)
                with conn, conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO svp_journal ({', '.join(_COLS)}) "
                        f"VALUES ({', '.join(['%s'] * len(_COLS))}) "
                        f"ON CONFLICT (entry_id) DO UPDATE SET {sets}",
                        vals,
                    )
                return eid
            except Exception:
                pass
        try:
            c = _userdb.sq(_SQ_DDL)
            # Preserve created_at on update: REPLACE would reset it, so fetch.
            old = c.execute("SELECT created_at FROM svp_journal WHERE entry_id=? "
                            "AND ledger_key=?", (eid, key)).fetchone()
            created = old[0] if old else now
            vals_sq = vals[:-2] + (created, now)
            c.execute(
                f"INSERT OR REPLACE INTO svp_journal ({', '.join(_COLS)}) "
                f"VALUES ({', '.join(['?'] * len(_COLS))})", vals_sq)
            c.commit()
            return eid
        except Exception:
            return None


def list_for(ledger_key: str, ticker: Optional[str] = None) -> list[Entry]:
    """This key's entries, newest first; optionally one ticker's."""
    if not _userdb.valid_key(ledger_key):
        return []
    key = ledger_key.strip().lower()
    tkr = normalise(ticker) if ticker else None

    where = "ledger_key={}" + (" AND ticker={}" if tkr else "")
    args = (key, tkr) if tkr else (key,)
    q = (f"SELECT {', '.join(_COLS)} FROM svp_journal WHERE {where} "
         "ORDER BY updated_at DESC")

    rows = None
    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(q.format(*(["%s"] * len(args))), args)
                    rows = cur.fetchall()
            except Exception:
                rows = None
        if rows is None:
            try:
                rows = _userdb.sq(_SQ_DDL).execute(
                    q.format(*(["?"] * len(args))), args).fetchall()
            except Exception:
                rows = []

    out = []
    for r in rows:
        d = dict(zip(_COLS, r))
        d.pop("ledger_key")
        out.append(Entry(
            entry_id=d.pop("entry_id"), ticker=d.pop("ticker"),
            entry_price=d.pop("entry_price"), target_price=d.pop("target_price"),
            horizon=d.pop("horizon") or "", created_at=float(d.pop("created_at")),
            updated_at=float(d.pop("updated_at")),
            **{k: (v or "") for k, v in d.items()},
        ))
    return out


def delete(ledger_key: str, entry_id: str) -> bool:
    if not _userdb.valid_key(ledger_key) or not entry_id:
        return False
    key = ledger_key.strip().lower()
    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute("DELETE FROM svp_journal WHERE ledger_key=%s "
                                "AND entry_id=%s", (key, entry_id))
                return True
            except Exception:
                pass
        try:
            c = _userdb.sq(_SQ_DDL)
            c.execute("DELETE FROM svp_journal WHERE ledger_key=? AND entry_id=?",
                      (key, entry_id))
            c.commit()
            return True
        except Exception:
            return False
