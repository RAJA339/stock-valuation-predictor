"""
Pytest suite for the durable-storage layer.

The PostgreSQL branch is the one that has never actually run in this project —
development and CI both fall back to SQLite, so every ``%s``-parameterised
statement in the ledger and the watchlist is unexercised code. A wrong
placeholder style or column order there would not surface until the first
deploy that sets SVP_DATABASE_URL, which is exactly the moment it matters.

There is no Postgres server available in CI, so the connection is stubbed. That
does not prove the SQL is valid Postgres, and this file does not claim it does.
What it does prove: the Postgres branch is taken when a DSN is present, the
statements are built with Postgres placeholders rather than SQLite ones, and a
refused connection falls back to SQLite rather than losing the write.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SVP_SQLITE_PATH"] = os.path.join(tempfile.mkdtemp(), "userdb_test.db")

from svp.data import _userdb, predictions as P, watchlist as W   # noqa: E402


class FakeCursor:
    def __init__(self, log, rows):
        self.log, self._rows = log, rows

    def execute(self, sql, args=None):
        self.log.append((" ".join(sql.split()), args))

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    """Enough of a psycopg2 connection for the call sites in this package."""

    def __init__(self, rows=()):
        self.log: list = []
        self._rows = list(rows)

    def cursor(self):
        return FakeCursor(self.log, self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def pg(monkeypatch):
    """Force the Postgres branch with a stub connection."""
    conn = FakeConn()
    _userdb.reset_for_tests()
    monkeypatch.setattr(_userdb, "_dsn", lambda: "postgresql://stub/db")
    monkeypatch.setattr(_userdb, "_pg_conn", conn, raising=False)
    monkeypatch.setattr(_userdb, "_pg_checked", True, raising=False)
    yield conn
    _userdb.reset_for_tests()


@pytest.fixture(autouse=True)
def _clean():
    yield
    _userdb.reset_for_tests()


# ── Backend selection ────────────────────────────────────────────────────────
class TestBackendSelection:
    def test_sqlite_when_no_dsn(self, monkeypatch):
        _userdb.reset_for_tests()
        monkeypatch.delenv("SVP_DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert _userdb.pg() is None
        assert _userdb.backend_name() == "SQLite"

    def test_postgres_when_connectable(self, pg):
        assert _userdb.pg() is pg
        assert _userdb.backend_name() == "PostgreSQL"

    def test_either_env_var_is_read(self, monkeypatch):
        for var in ("SVP_DATABASE_URL", "DATABASE_URL"):
            _userdb.reset_for_tests()
            monkeypatch.delenv("SVP_DATABASE_URL", raising=False)
            monkeypatch.delenv("DATABASE_URL", raising=False)
            monkeypatch.setenv(var, "postgresql://x/y")
            assert _userdb._dsn() == "postgresql://x/y"

    def test_unreachable_database_falls_back_rather_than_raising(self, monkeypatch):
        """
        A database misconfiguration must not take down the valuation model. The
        app is useful without a ledger; it is useless if it will not boot.
        """
        _userdb.reset_for_tests()
        monkeypatch.setenv("SVP_DATABASE_URL", "postgresql://nowhere:1/none")
        assert _userdb.pg() is None
        assert _userdb.backend_name() == "SQLite"


# ── Durability reporting ─────────────────────────────────────────────────────
class TestDurability:
    def test_sqlite_is_reported_as_not_durable(self, monkeypatch):
        _userdb.reset_for_tests()
        monkeypatch.delenv("SVP_DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert _userdb.is_durable() is False
        assert "can be lost" in _userdb.durability_note()

    def test_postgres_is_reported_as_durable(self, pg):
        assert _userdb.is_durable() is True
        assert "persists" in _userdb.durability_note()

    def test_the_note_names_the_variable_to_set(self, monkeypatch):
        _userdb.reset_for_tests()
        monkeypatch.delenv("SVP_DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert "SVP_DATABASE_URL" in _userdb.durability_note()


# ── The unexercised SQL ──────────────────────────────────────────────────────
class TestPostgresStatements:
    def test_ledger_insert_uses_postgres_placeholders(self, pg):
        P.record(_userdb.new_key(), "NVDA", 100.0, 90.0, 110.0, 130.0,
                 dedupe_hours=0)
        inserts = [sql for sql, _ in pg.log if sql.startswith("INSERT INTO svp_predictions")]
        assert inserts, f"no insert reached Postgres: {pg.log}"
        assert "%s" in inserts[0] and "?" not in inserts[0]

    def test_ledger_insert_arity_matches_the_column_list(self, pg):
        P.record(_userdb.new_key(), "NVDA", 100.0, 90.0, 110.0, 130.0,
                 dedupe_hours=0)
        sql, args = next((s, a) for s, a in pg.log
                         if s.startswith("INSERT INTO svp_predictions"))
        assert sql.count("%s") == len(args)

    def test_ledger_select_is_parameterised(self, pg):
        P.for_key(_userdb.new_key())
        sql, args = next((s, a) for s, a in pg.log if s.startswith("SELECT"))
        assert "%s" in sql and "?" not in sql
        assert sql.count("%s") == len(args)

    def test_watchlist_insert_uses_on_conflict_not_or_ignore(self, pg):
        """INSERT OR IGNORE is SQLite syntax and is a syntax error in Postgres."""
        W.add(_userdb.new_key(), "NVDA")
        inserts = [s for s, _ in pg.log if s.startswith("INSERT INTO svp_watchlist")]
        assert inserts
        assert "ON CONFLICT" in inserts[0]
        assert "OR IGNORE" not in inserts[0]

    def test_watchlist_delete_is_parameterised(self, pg):
        W.remove(_userdb.new_key(), "NVDA")
        sql, args = next((s, a) for s, a in pg.log if s.startswith("DELETE"))
        assert sql.count("%s") == len(args)

    def test_ddl_runs_before_the_first_statement(self, pg):
        P.for_key(_userdb.new_key())
        assert any("CREATE TABLE IF NOT EXISTS svp_predictions" in s
                   for s, _ in pg.log)

    def test_ddl_is_applied_once_not_per_call(self, pg):
        for _ in range(5):
            P.for_key(_userdb.new_key())
        creates = [s for s, _ in pg.log if s.startswith("CREATE TABLE")]
        assert len(creates) == 1

    def test_each_table_declares_its_own_ddl(self, pg):
        P.for_key(_userdb.new_key())
        W.get(_userdb.new_key())
        tables = {s.split()[5] for s, _ in pg.log if s.startswith("CREATE TABLE")}
        assert tables == {"svp_predictions", "svp_watchlist"}


# ── Fallback behaviour ───────────────────────────────────────────────────────
class TestWriteFallback:
    def test_a_failing_postgres_write_still_lands_in_sqlite(self, monkeypatch):
        """The write must survive a database that accepts a connection then fails."""
        class Broken(FakeConn):
            def cursor(self):
                raise RuntimeError("connection reset")

        _userdb.reset_for_tests()
        monkeypatch.setattr(_userdb, "_dsn", lambda: "postgresql://stub/db")
        monkeypatch.setattr(_userdb, "_pg_conn", Broken(), raising=False)
        monkeypatch.setattr(_userdb, "_pg_checked", True, raising=False)

        key = _userdb.new_key()
        assert P.record(key, "NVDA", 100.0, 90.0, 110.0, 130.0,
                        dedupe_hours=0) is not None

        _userdb.reset_for_tests()
        monkeypatch.setattr(_userdb, "_dsn", lambda: None)
        assert [p.ticker for p in P.for_key(key)] == ["NVDA"]


# ── DSN resolution ───────────────────────────────────────────────────────────
class TestDsnResolution:
    """
    Where the connection string is read from. The bug this guards: reading only
    os.environ meant a URL set in Streamlit Cloud's Secrets UI — which lands in
    st.secrets, not the environment — was silently ignored, leaving the app on
    SQLite while looking fully configured.
    """

    def _clear_env(self, monkeypatch):
        monkeypatch.delenv("SVP_DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)

    def _fake_streamlit(self, monkeypatch, secrets):
        import types
        mod = types.ModuleType("streamlit")
        mod.secrets = secrets
        monkeypatch.setitem(sys.modules, "streamlit", mod)

    def test_none_when_nothing_is_set(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setitem(sys.modules, "streamlit", None)
        assert _userdb._dsn() is None

    def test_reads_svp_env_var(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("SVP_DATABASE_URL", "postgresql://env/db")
        assert _userdb._dsn() == "postgresql://env/db"

    def test_reads_database_url_alias(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("DATABASE_URL", "postgresql://alias/db")
        assert _userdb._dsn() == "postgresql://alias/db"

    def test_reads_streamlit_secret_when_env_absent(self, monkeypatch):
        self._clear_env(monkeypatch)
        self._fake_streamlit(monkeypatch, {"SVP_DATABASE_URL": "postgresql://secret/db"})
        assert _userdb._dsn() == "postgresql://secret/db"

    def test_environment_beats_secret(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("SVP_DATABASE_URL", "postgresql://env/wins")
        self._fake_streamlit(monkeypatch, {"SVP_DATABASE_URL": "postgresql://secret/loses"})
        assert _userdb._dsn() == "postgresql://env/wins"

    def test_a_broken_secrets_object_does_not_raise(self, monkeypatch):
        """No secrets file makes st.secrets raise on access; that must fall to None."""
        self._clear_env(monkeypatch)

        class Boom:
            def get(self, *a):
                raise FileNotFoundError("no secrets.toml")

        import types
        mod = types.ModuleType("streamlit")
        mod.secrets = Boom()
        monkeypatch.setitem(sys.modules, "streamlit", mod)
        assert _userdb._dsn() is None
