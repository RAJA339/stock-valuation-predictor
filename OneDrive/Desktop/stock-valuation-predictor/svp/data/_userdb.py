"""
Shared persistence plumbing for user data.
==========================================

:mod:`svp.data.storage` is a cache: every row has a TTL and losing it costs a
round-trip. The ledger and the watchlist are the opposite — rows are a user's
own work and must not expire — so they live in their own tables, and both need
the same Postgres-or-SQLite fallback. This module holds that once.

Postgres is used when ``SVP_DATABASE_URL`` (or ``DATABASE_URL``) is set and
reachable; otherwise a local SQLite file. On Streamlit Cloud the SQLite file
does not survive a redeploy, which is exactly why the Postgres path exists —
a track record that vanishes on deploy is worse than none, because the user
believed it was being kept.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import uuid
from typing import Optional, Sequence

_DB_FILE = os.environ.get("SVP_SQLITE_PATH", "svp_cache.db")
LOCK = threading.Lock()

_KEY_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


# ── Identity ─────────────────────────────────────────────────────────────────
def new_key() -> str:
    """
    Mint an anonymous key identifying one person's saved work.

    Deliberately not a login. A sign-up wall in front of a tool nobody has used
    yet costs more traffic than it captures, and this app has no reason to hold
    an email address.

    The key is also kept out of the shareable ``?t=`` link: putting it there
    would mean a user who shares a result hands over their whole saved history
    to whoever opens it.
    """
    return str(uuid.uuid4())


def valid_key(key: str) -> bool:
    """True for a well-formed key. Rejects junk before it reaches SQL."""
    return bool(key) and bool(_KEY_RE.match(key.strip().lower()))


# ── Postgres (optional) ──────────────────────────────────────────────────────
def _dsn() -> Optional[str]:
    return os.environ.get("SVP_DATABASE_URL") or os.environ.get("DATABASE_URL")


_pg_conn = None
_pg_checked = False
_pg_ddl_done: set[str] = set()


def pg(ddl: Sequence[str] = ()):
    """
    The Postgres connection, or ``None``.

    ``ddl`` is applied once per distinct statement, so each caller can declare
    the tables it needs without coordinating with the others.
    """
    global _pg_conn, _pg_checked
    if not _pg_checked:
        _pg_checked = True
        dsn = _dsn()
        if dsn:
            try:
                import psycopg2  # type: ignore

                _pg_conn = psycopg2.connect(dsn)
            except Exception:
                _pg_conn = None

    if _pg_conn is not None and ddl:
        todo = [s for s in ddl if s not in _pg_ddl_done]
        if todo:
            try:
                with _pg_conn, _pg_conn.cursor() as cur:
                    for stmt in todo:
                        cur.execute(stmt)
                _pg_ddl_done.update(todo)
            except Exception:
                return None
    return _pg_conn


# ── SQLite (always available) ────────────────────────────────────────────────
_sqlite_conn: Optional[sqlite3.Connection] = None
_sqlite_ddl_done: set[str] = set()


def sq(ddl: Sequence[str] = ()) -> sqlite3.Connection:
    """The SQLite connection, applying each DDL statement once."""
    global _sqlite_conn
    if _sqlite_conn is None:
        _sqlite_conn = sqlite3.connect(_DB_FILE, check_same_thread=False)

    todo = [s for s in ddl if s not in _sqlite_ddl_done]
    if todo:
        for stmt in todo:
            _sqlite_conn.execute(stmt)
        _sqlite_conn.commit()
        _sqlite_ddl_done.update(todo)
    return _sqlite_conn


def backend_name() -> str:
    return "PostgreSQL" if pg() is not None else "SQLite"


def reset_for_tests() -> None:
    """Drop cached connections so a test can point SVP_SQLITE_PATH elsewhere."""
    global _sqlite_conn, _pg_conn, _pg_checked
    if _sqlite_conn is not None:
        try:
            _sqlite_conn.close()
        except Exception:
            pass
    _sqlite_conn = None
    _pg_conn = None
    _pg_checked = False
    _sqlite_ddl_done.clear()
    _pg_ddl_done.clear()
