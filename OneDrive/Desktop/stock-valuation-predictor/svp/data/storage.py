"""
Data caching / storage backend.
===============================

Provides a small key/value cache backed by a database so parsed SEC filings,
macro pulls and other slow API calls can be reused across sessions and speed up
response times.

Backend selection (in priority order):
  1. PostgreSQL  — if ``DATABASE_URL`` / ``SVP_DATABASE_URL`` is set and
     ``psycopg2`` is importable (drop-in for the Snowflake/Postgres requirement).
  2. SQLite      — a local file (``svp_cache.db``), always available.

The public API is deliberately tiny and JSON-oriented so callers never need to
know which backend is active::

    storage.cache_set("sec:AAPL", payload_dict, ttl=86400)
    payload = storage.cache_get("sec:AAPL")        # None if missing / expired

This pairs with Streamlit's ``@st.cache_data`` (in-memory, per-session) to give
a two-tier cache: memory first, then the persistent DB.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
from typing import Any, Optional

from ._userdb import _dsn as _shared_dsn

_DEFAULT_TTL = 24 * 3600  # 1 day


def _default_sqlite_path() -> str:
    """
    A writable path for the SQLite fallback.

    A bare ``svp_cache.db`` resolves against the working directory, which on
    Streamlit Cloud is the *read-only* ``/mount/src`` checkout — so the first
    cache write raised ``sqlite3.OperationalError`` and took the whole app down.
    The system temp directory is always writable; the cache is TTL'd and
    disposable, so putting it there loses nothing. ``SVP_SQLITE_PATH`` still
    overrides for anyone who wants a specific location.
    """
    override = os.environ.get("SVP_SQLITE_PATH")
    if override:
        return override
    return os.path.join(tempfile.gettempdir(), "svp_cache.db")


_DB_FILE = _default_sqlite_path()
_lock = threading.Lock()


# ── Postgres (optional) ──────────────────────────────────────────────────────
def _postgres_dsn() -> Optional[str]:
    """
    The Postgres connection string.

    Delegates to :func:`svp.data._userdb._dsn`, which reads ``st.secrets`` as
    well as ``os.environ``. Reading only the environment here was a real bug:
    on Streamlit Cloud the DSN entered in the Secrets UI lands in ``st.secrets``
    and never appears as an environment variable, so this cache silently fell
    back to SQLite while the ledger (which already used the shared lookup)
    correctly reached Postgres.
    """
    return _shared_dsn()


def _try_postgres():
    dsn = _postgres_dsn()
    if not dsn:
        return None
    try:
        import psycopg2  # type: ignore

        conn = psycopg2.connect(dsn)
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS svp_cache (
                    key       TEXT PRIMARY KEY,
                    value     TEXT NOT NULL,
                    expires   DOUBLE PRECISION NOT NULL
                )
                """
            )
        return conn
    except Exception:
        return None


_pg_conn = None
_pg_checked = False


def _pg():
    global _pg_conn, _pg_checked
    if not _pg_checked:
        _pg_conn = _try_postgres()
        _pg_checked = True
    return _pg_conn


# ── SQLite (always available fallback) ───────────────────────────────────────
def _sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_FILE, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS svp_cache (
            key     TEXT PRIMARY KEY,
            value   TEXT NOT NULL,
            expires REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


_sqlite_conn: Optional[sqlite3.Connection] = None


def _sq() -> sqlite3.Connection:
    global _sqlite_conn
    if _sqlite_conn is None:
        _sqlite_conn = _sqlite()
    return _sqlite_conn


def backend_name() -> str:
    """Return a human-readable label for the active persistent backend."""
    return "PostgreSQL" if _pg() is not None else "SQLite"


# ── Public API ───────────────────────────────────────────────────────────────
def cache_set(key: str, value: Any, ttl: int = _DEFAULT_TTL) -> None:
    """Store a JSON-serialisable ``value`` under ``key`` for ``ttl`` seconds."""
    expires = time.time() + ttl
    payload = json.dumps(value, default=str)
    with _lock:
        pg = _pg()
        if pg is not None:
            try:
                with pg, pg.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO svp_cache (key, value, expires)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (key) DO UPDATE
                        SET value = EXCLUDED.value, expires = EXCLUDED.expires
                        """,
                        (key, payload, expires),
                    )
                return
            except Exception:
                pass  # fall through to sqlite
        try:
            conn = _sq()
            conn.execute(
                "INSERT OR REPLACE INTO svp_cache (key, value, expires) VALUES (?, ?, ?)",
                (key, payload, expires),
            )
            conn.commit()
        except sqlite3.Error:
            # A cache write is best-effort. If the SQLite file is unwritable
            # (e.g. a read-only host filesystem with no Postgres configured),
            # skip caching rather than crash the request that triggered it.
            pass


def cache_get(key: str) -> Optional[Any]:
    """Return the stored value for ``key`` or ``None`` if missing/expired."""
    now = time.time()
    with _lock:
        pg = _pg()
        if pg is not None:
            try:
                with pg, pg.cursor() as cur:
                    cur.execute("SELECT value, expires FROM svp_cache WHERE key = %s", (key,))
                    row = cur.fetchone()
                if row and row[1] > now:
                    return json.loads(row[0])
                return None
            except Exception:
                pass
        try:
            conn = _sq()
            cur = conn.execute("SELECT value, expires FROM svp_cache WHERE key = ?", (key,))
            row = cur.fetchone()
        except sqlite3.Error:
            return None
        if row and row[1] > now:
            try:
                return json.loads(row[0])
            except Exception:
                return None
        return None


def cache_clear() -> int:
    """Delete all cached rows. Returns the number of rows removed (best effort)."""
    with _lock:
        n = 0
        try:
            conn = _sq()
            cur = conn.execute("SELECT COUNT(*) FROM svp_cache")
            n = cur.fetchone()[0]
            conn.execute("DELETE FROM svp_cache")
            conn.commit()
        except sqlite3.Error:
            n = 0
        pg = _pg()
        if pg is not None:
            try:
                with pg, pg.cursor() as c:
                    c.execute("DELETE FROM svp_cache")
            except Exception:
                pass
        return int(n)


def stats() -> dict:
    """Return simple cache statistics for display in the UI."""
    now = time.time()
    try:
        conn = _sq()
        total = conn.execute("SELECT COUNT(*) FROM svp_cache").fetchone()[0]
        live = conn.execute(
            "SELECT COUNT(*) FROM svp_cache WHERE expires > ?", (now,)
        ).fetchone()[0]
    except sqlite3.Error:
        total = live = 0
    return {"backend": backend_name(), "rows": int(total), "fresh": int(live)}
