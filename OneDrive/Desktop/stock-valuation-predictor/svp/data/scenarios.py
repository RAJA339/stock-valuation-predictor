"""
Saved DCF scenarios — named assumption sets a person can come back to.
======================================================================

The Scenario Lab's sliders are ephemeral; this table is what makes an
assumption set a durable artefact: "my bear case for AAL, October" survives
the session, follows the ledger key (and therefore the signed-in account),
and can be reloaded, renamed by re-saving, or deleted.

Same shape as the watchlist store: keyed by ``(ledger_key, ticker, name)``,
Postgres first with SQLite fallback, every query filtered by the owning key.
Parameters travel as JSON so the schema does not chase the slider set.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

from . import _userdb

MAX_PER_KEY = 40
MAX_NAME = 60

_DDL_PG = """
CREATE TABLE IF NOT EXISTS svp_scenarios (
    ledger_key TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    name       TEXT NOT NULL,
    params     TEXT NOT NULL,
    fair_value DOUBLE PRECISION,
    updated_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ledger_key, ticker, name)
)
"""

_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS svp_scenarios (
    ledger_key TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    name       TEXT NOT NULL,
    params     TEXT NOT NULL,
    fair_value REAL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (ledger_key, ticker, name)
)
"""

_PG_DDL = (_DDL_PG,)
_SQ_DDL = (_DDL_SQLITE,)


@dataclass
class Scenario:
    ticker: str
    name: str
    params: dict
    fair_value: Optional[float]
    updated_at: float


def _count(key: str) -> int:
    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM svp_scenarios WHERE ledger_key=%s",
                                (key,))
                    return int(cur.fetchone()[0])
            except Exception:
                pass
        try:
            row = _userdb.sq(_SQ_DDL).execute(
                "SELECT COUNT(*) FROM svp_scenarios WHERE ledger_key=?", (key,)
            ).fetchone()
            return int(row[0])
        except Exception:
            return 0


def save(ledger_key: str, ticker: str, name: str, params: dict,
         fair_value: Optional[float] = None) -> bool:
    """
    Upsert one named scenario. Returns True when stored.

    Saving an existing name overwrites it — from the user's point of view the
    request was "make 'bear case' be *this* now". The per-key cap only blocks
    genuinely new rows, never updates.
    """
    name = (name or "").strip()[:MAX_NAME]
    ticker = (ticker or "").strip().upper()
    if not _userdb.valid_key(ledger_key) or not name or not ticker:
        return False
    key = ledger_key.strip().lower()

    existing = {s.name for s in list_for(key, ticker)}
    if name not in existing and _count(key) >= MAX_PER_KEY:
        return False

    try:
        blob = json.dumps(params or {}, default=float)
    except Exception:
        return False
    now = time.time()
    fv = float(fair_value) if fair_value is not None else None

    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO svp_scenarios "
                        "(ledger_key, ticker, name, params, fair_value, updated_at) "
                        "VALUES (%s,%s,%s,%s,%s,%s) "
                        "ON CONFLICT (ledger_key, ticker, name) DO UPDATE SET "
                        "params=EXCLUDED.params, fair_value=EXCLUDED.fair_value, "
                        "updated_at=EXCLUDED.updated_at",
                        (key, ticker, name, blob, fv, now),
                    )
                return True
            except Exception:
                pass
        try:
            c = _userdb.sq(_SQ_DDL)
            c.execute(
                "INSERT OR REPLACE INTO svp_scenarios "
                "(ledger_key, ticker, name, params, fair_value, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (key, ticker, name, blob, fv, now),
            )
            c.commit()
            return True
        except Exception:
            return False


def list_for(ledger_key: str, ticker: str) -> list[Scenario]:
    """This key's scenarios for one ticker, most recently updated first."""
    ticker = (ticker or "").strip().upper()
    if not _userdb.valid_key(ledger_key) or not ticker:
        return []
    key = ledger_key.strip().lower()
    q = ("SELECT ticker, name, params, fair_value, updated_at FROM svp_scenarios "
         "WHERE ledger_key={} AND ticker={} ORDER BY updated_at DESC")

    # rows is None until a backend has *answered*; an empty Postgres answer is
    # authoritative and must not fall through to a second source of truth.
    rows = None
    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(q.format("%s", "%s"), (key, ticker))
                    rows = cur.fetchall()
            except Exception:
                rows = None
        if rows is None:
            try:
                rows = _userdb.sq(_SQ_DDL).execute(
                    q.format("?", "?"), (key, ticker)).fetchall()
            except Exception:
                rows = []

    out = []
    for r in rows:
        try:
            out.append(Scenario(ticker=r[0], name=r[1], params=json.loads(r[2]),
                                fair_value=r[3], updated_at=float(r[4])))
        except Exception:
            continue
    return out


def delete(ledger_key: str, ticker: str, name: str) -> bool:
    ticker = (ticker or "").strip().upper()
    name = (name or "").strip()
    if not _userdb.valid_key(ledger_key) or not ticker or not name:
        return False
    key = ledger_key.strip().lower()

    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM svp_scenarios WHERE ledger_key=%s AND "
                        "ticker=%s AND name=%s", (key, ticker, name))
                return True
            except Exception:
                pass
        try:
            c = _userdb.sq(_SQ_DDL)
            c.execute("DELETE FROM svp_scenarios WHERE ledger_key=? AND "
                      "ticker=? AND name=?", (key, ticker, name))
            c.commit()
            return True
        except Exception:
            return False
