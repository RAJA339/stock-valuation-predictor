"""
Profiles — the binding between a signed-in account and a ledger key.
====================================================================

The app's identity model stays anonymous-first: every visitor gets a ledger
key, and everything they save hangs off it. Signing in adds exactly one
thing — a durable pointer from an OIDC account (``sub``) to one ledger key —
so the same record follows the person across devices without the URL.

The adoption rule matters most on the *first* sign-in: whatever anonymous
work the visitor did before creating the account is theirs, so the new
profile binds to the ledger key already in their session rather than minting
a fresh one and orphaning their history. On every later sign-in the stored
key wins, because that is the whole point of having signed in.

``plan`` exists now (default ``free``) so later feature gates are a column
read, not a migration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from . import _userdb

PLANS = ("free", "premium", "admin")

_DDL_PG = """
CREATE TABLE IF NOT EXISTS svp_profiles (
    sub          TEXT PRIMARY KEY,
    email        TEXT,
    display_name TEXT,
    ledger_key   TEXT NOT NULL,
    plan         TEXT NOT NULL DEFAULT 'free',
    created_at   DOUBLE PRECISION NOT NULL,
    last_login   DOUBLE PRECISION NOT NULL
)
"""

_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS svp_profiles (
    sub          TEXT PRIMARY KEY,
    email        TEXT,
    display_name TEXT,
    ledger_key   TEXT NOT NULL,
    plan         TEXT NOT NULL DEFAULT 'free',
    created_at   REAL NOT NULL,
    last_login   REAL NOT NULL
)
"""

_PG_DDL = (_DDL_PG,)
_SQ_DDL = (_DDL_SQLITE,)


@dataclass
class Profile:
    sub: str
    email: Optional[str]
    display_name: Optional[str]
    ledger_key: str
    plan: str
    created: bool = False       # True only on the call that created the row


def get(sub: str) -> Optional[Profile]:
    """The stored profile for an OIDC subject, or ``None``."""
    sub = (sub or "").strip()
    if not sub:
        return None
    q = "SELECT sub, email, display_name, ledger_key, plan FROM svp_profiles WHERE sub="
    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(q + "%s", (sub,))
                    row = cur.fetchone()
                if row:
                    return Profile(row[0], row[1], row[2], row[3], row[4])
                return None
            except Exception:
                pass
        try:
            row = _userdb.sq(_SQ_DDL).execute(q + "?", (sub,)).fetchone()
        except Exception:
            return None
    return Profile(row[0], row[1], row[2], row[3], row[4]) if row else None


def get_or_create(sub: str, email: Optional[str], display_name: Optional[str],
                  session_key: str) -> Optional[Profile]:
    """
    The profile for ``sub``, creating it on first sign-in.

    On creation the profile adopts ``session_key`` — the anonymous key already
    active in the visitor's session — so pre-sign-in work is kept, not
    orphaned. If that key is malformed a fresh one is minted instead. On an
    existing profile the stored key is authoritative, and email/name/login
    time are refreshed.
    """
    sub = (sub or "").strip()
    if not sub:
        return None

    existing = get(sub)
    now = time.time()
    if existing is not None:
        _touch(sub, email, display_name, now)
        existing.email = email or existing.email
        existing.display_name = display_name or existing.display_name
        return existing

    key = (session_key or "").strip().lower()
    if not _userdb.valid_key(key):
        key = _userdb.new_key()

    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO svp_profiles "
                        "(sub, email, display_name, ledger_key, plan, created_at, last_login) "
                        "VALUES (%s,%s,%s,%s,'free',%s,%s) ON CONFLICT (sub) DO NOTHING",
                        (sub, email, display_name, key, now, now),
                    )
                prof = get(sub)
                return _mark_created(prof) if prof else None
            except Exception:
                pass
        try:
            c = _userdb.sq(_SQ_DDL)
            c.execute(
                "INSERT OR IGNORE INTO svp_profiles "
                "(sub, email, display_name, ledger_key, plan, created_at, last_login) "
                "VALUES (?,?,?,?,'free',?,?)",
                (sub, email, display_name, key, now, now),
            )
            c.commit()
        except Exception:
            return None
    prof = get(sub)
    return _mark_created(prof) if prof else None


def _mark_created(prof: Profile) -> Profile:
    prof.created = True
    return prof


def _touch(sub: str, email: Optional[str], display_name: Optional[str],
           now: float) -> None:
    """Refresh mutable claims and the last-login stamp; never the ledger key."""
    with _userdb.LOCK:
        conn = _userdb.pg(_PG_DDL)
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE svp_profiles SET last_login=%s, "
                        "email=COALESCE(%s, email), "
                        "display_name=COALESCE(%s, display_name) WHERE sub=%s",
                        (now, email, display_name, sub),
                    )
                return
            except Exception:
                pass
        try:
            c = _userdb.sq(_SQ_DDL)
            c.execute(
                "UPDATE svp_profiles SET last_login=?, "
                "email=COALESCE(?, email), "
                "display_name=COALESCE(?, display_name) WHERE sub=?",
                (now, email, display_name, sub),
            )
            c.commit()
        except Exception:
            pass
