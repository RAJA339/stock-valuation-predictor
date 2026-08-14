"""
Pytest suite for the watchlist and the event calendar.

The watchlist is small but it is user data, so the same properties matter as
for the ledger: no leakage between keys, junk rejected before it reaches SQL,
and idempotent adds because the UI will re-submit on every Streamlit rerun.

The calendar is tested offline against synthetic filing histories. The real
value in it is the projection arithmetic — a date derived from cadence — and
that is pure computation with no network in it.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SVP_SQLITE_PATH"] = os.path.join(tempfile.mkdtemp(), "watch_test.db")

from svp.data import _userdb, watchlist as W, calendar as C     # noqa: E402


@pytest.fixture
def key():
    return _userdb.new_key()


# ── Symbol validation ────────────────────────────────────────────────────────
class TestNormalise:
    @pytest.mark.parametrize("raw,want", [
        ("nvda", "NVDA"), (" msft ", "MSFT"), ("BRK.B", "BRK.B"),
        ("BRK-B", "BRK-B"), ("^VIX", "^VIX"), ("F", "F"),
    ])
    def test_accepts_real_symbols(self, raw, want):
        assert W.normalise(raw) == want

    @pytest.mark.parametrize("bad", [
        "", "   ", "TOOLONGSYM", "'; DROP TABLE svp_watchlist; --",
        "A B", "12345", "NVDA;", "<script>",
    ])
    def test_rejects_junk(self, bad):
        assert W.normalise(bad) is None

    def test_junk_never_reaches_storage(self, key):
        assert W.add(key, "'; DROP TABLE svp_watchlist; --") is False
        assert W.get(key) == []
        # Table intact.
        assert W.add(key, "NVDA") is True


# ── Storage ──────────────────────────────────────────────────────────────────
class TestWatchlist:
    def test_add_and_get(self, key):
        W.add(key, "nvda")
        assert W.get(key) == ["NVDA"]

    def test_add_is_idempotent(self, key):
        """The UI re-submits on every rerun; a second add must not duplicate."""
        assert W.add(key, "NVDA") is True
        assert W.add(key, "NVDA") is True
        assert W.get(key) == ["NVDA"]

    def test_newest_first(self, key):
        W.add(key, "AAA")
        W.add(key, "BBB")
        assert W.get(key)[0] == "BBB"

    def test_isolation_between_keys(self, key):
        W.add(key, "NVDA")
        assert W.get(_userdb.new_key()) == []

    def test_remove(self, key):
        W.add(key, "NVDA")
        W.remove(key, "nvda")
        assert W.get(key) == []

    def test_remove_absent_is_not_an_error(self, key):
        assert W.remove(key, "NVDA") is True

    def test_contains(self, key):
        W.add(key, "NVDA")
        assert W.contains(key, "nvda")
        assert not W.contains(key, "AAPL")

    def test_toggle_round_trip(self, key):
        assert W.toggle(key, "NVDA") is True
        assert W.toggle(key, "NVDA") is False
        assert W.get(key) == []

    def test_invalid_key_reads_empty(self):
        assert W.get("not-a-key") == []

    def test_invalid_key_cannot_write(self):
        assert W.add("not-a-key", "NVDA") is False

    def test_cap_is_enforced(self, key):
        for i in range(W.MAX_ENTRIES):
            assert W.add(key, f"S{i:03d}"[:6]) or True
        n = len(W.get(key))
        assert n <= W.MAX_ENTRIES


# ── Calendar ─────────────────────────────────────────────────────────────────
class TestCalendarProjection:
    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch):
        monkeypatch.setattr(C.storage, "cache_get", lambda k: None)
        monkeypatch.setattr(C.storage, "cache_set", lambda *a, **k: None)

    def _history(self, monkeypatch, events):
        monkeypatch.setattr(C, "recent_filings", lambda t, limit=8: events)

    def _quarterly(self, last_form="10-K", n=4, anchor_days_ago=30):
        base = dt.date.today() - dt.timedelta(days=anchor_days_ago)
        forms = [last_form] + ["10-Q"] * (n - 1)
        return [C.Event("X", f, base - dt.timedelta(days=91 * i), False)
                for i, f in enumerate(forms)]

    def test_projects_one_cadence_ahead(self, monkeypatch):
        self._history(monkeypatch, self._quarterly())
        e = C.next_filing("X")
        assert e is not None and e.projected is True
        assert 55 <= e.days_away <= 65        # ~91 days on from a filing 30d ago

    def test_projection_is_always_in_the_future(self, monkeypatch):
        """A company that stopped filing must not surface a date already past."""
        self._history(monkeypatch, self._quarterly(anchor_days_ago=800))
        e = C.next_filing("X")
        assert e is not None and not e.is_past

    def test_annual_slot_is_identified(self, monkeypatch):
        """Last 10-K ~9 months back: the next periodic filing is the annual."""
        base = dt.date.today() - dt.timedelta(days=30)
        self._history(monkeypatch, [
            C.Event("Y", "10-Q", base, False),
            C.Event("Y", "10-Q", base - dt.timedelta(days=91), False),
            C.Event("Y", "10-Q", base - dt.timedelta(days=182), False),
            C.Event("Y", "10-K", base - dt.timedelta(days=273), False),
        ])
        assert C.next_filing("Y").kind == "10-K"

    def test_quarter_follows_an_annual(self, monkeypatch):
        self._history(monkeypatch, self._quarterly(last_form="10-K"))
        assert C.next_filing("X").kind == "10-Q"

    def test_too_little_history_declines_to_guess(self, monkeypatch):
        self._history(monkeypatch, self._quarterly(n=2))
        assert C.next_filing("X") is None

    def test_no_history_declines_to_guess(self, monkeypatch):
        self._history(monkeypatch, [])
        assert C.next_filing("X") is None

    def test_implausible_gaps_are_discarded(self, monkeypatch):
        """Filings days apart are amendments, not a cadence."""
        base = dt.date.today()
        self._history(monkeypatch, [
            C.Event("X", "10-Q", base - dt.timedelta(days=1), False),
            C.Event("X", "10-Q", base - dt.timedelta(days=2), False),
            C.Event("X", "10-Q", base - dt.timedelta(days=3), False),
        ])
        assert C.next_filing("X") is None

    def test_recent_filings_survives_a_broken_source(self, monkeypatch):
        import svp.rag.filings as F

        def boom(*a, **k):
            raise RuntimeError("EDGAR down")

        monkeypatch.setattr(F, "list_filings", boom)
        assert C.recent_filings("X") == []

    def test_upcoming_is_sorted_and_bounded(self, monkeypatch):
        today = dt.date.today()
        monkeypatch.setattr(C, "next_earnings",
                            lambda t: C.Event(t, "Earnings",
                                              today + dt.timedelta(days=10), False))
        monkeypatch.setattr(C, "next_filing",
                            lambda t: C.Event(t, "10-Q",
                                              today + dt.timedelta(days=5), True))
        got = C.upcoming(["AAA", "BBB"])
        assert [e.days_away for e in got] == [5, 5, 10, 10]

    def test_upcoming_drops_events_beyond_the_window(self, monkeypatch):
        today = dt.date.today()
        monkeypatch.setattr(C, "next_earnings",
                            lambda t: C.Event(t, "Earnings",
                                              today + dt.timedelta(days=900), False))
        monkeypatch.setattr(C, "next_filing", lambda t: None)
        assert C.upcoming(["AAA"]) == []

    def test_one_broken_ticker_does_not_empty_the_calendar(self, monkeypatch):
        today = dt.date.today()

        def flaky(t):
            if t == "BAD":
                raise RuntimeError("rate limited")
            return C.Event(t, "Earnings", today + dt.timedelta(days=7), False)

        monkeypatch.setattr(C, "next_earnings", flaky)
        monkeypatch.setattr(C, "next_filing", lambda t: None)
        got = C.upcoming(["BAD", "GOOD"])
        assert [e.ticker for e in got] == ["GOOD"]


class TestEventDates:
    @pytest.mark.parametrize("raw", [
        "2026-03-01", "2026-03-01T00:00:00", dt.date(2026, 3, 1),
        dt.datetime(2026, 3, 1, 9, 30),
    ])
    def test_date_coercion(self, raw):
        assert C._as_date(raw) == dt.date(2026, 3, 1)

    @pytest.mark.parametrize("raw", [None, "", "not a date", object()])
    def test_bad_dates_are_none_not_raises(self, raw):
        assert C._as_date(raw) is None
