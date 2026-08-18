"""
Pytest suite for Phase 4: plans, portfolio, journal, shared reports.

The properties that matter: caps block creation but never updates (a limit
that eats edits would corrupt data to enforce a quota); portfolio totals
exclude unpriceable tickers and *name* them rather than folding stale
numbers into the total; journal creation requires a thesis and updates
preserve creation time; shared slugs are revocable, expire on the clock, and
strip anything that looks private no matter what a future caller passes in.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SVP_SQLITE_PATH",
                      os.path.join(tempfile.mkdtemp(), "phase4_test.db"))
os.environ.pop("SVP_DATABASE_URL", None)
os.environ.pop("DATABASE_URL", None)

from svp import plans                                     # noqa: E402
from svp.data import (                                    # noqa: E402
    _userdb, journal as J, portfolio as PF, shared_reports as SR,
)


class _Base:
    def setup_method(self):
        _userdb.reset_for_tests()
        self.key = _userdb.new_key()


# ── Plans ────────────────────────────────────────────────────────────────────
class TestPlans:
    def test_unknown_or_missing_plan_reads_as_free(self):
        assert plans.limits(None) == plans.limits("free")
        assert plans.limits("enterprise") == plans.limits("free")
        assert plans.limits(" Premium ") == plans.limits("premium")

    def test_premium_is_a_superset_of_free(self):
        f, p = plans.limits("free"), plans.limits("premium")
        for attr in ("scenarios", "journal_entries", "holdings",
                     "active_shares", "watchlist"):
            assert getattr(p, attr) > getattr(f, attr)


# ── Portfolio ────────────────────────────────────────────────────────────────
class TestPortfolio(_Base):
    def test_round_trip(self):
        assert PF.upsert(self.key, "AAL", 100, 12.50, "starter")
        h = PF.list_for(self.key)
        assert len(h) == 1 and h[0].ticker == "AAL"
        assert h[0].cost_basis == pytest.approx(1250.0)

    def test_reentering_a_ticker_replaces_it(self):
        PF.upsert(self.key, "AAL", 100, 12.50)
        PF.upsert(self.key, "AAL", 150, 13.00)
        h = PF.list_for(self.key)
        assert len(h) == 1 and h[0].shares == 150

    def test_zero_or_negative_positions_refused(self):
        assert not PF.upsert(self.key, "AAL", 0, 12.0)
        assert not PF.upsert(self.key, "AAL", -5, 12.0)
        assert not PF.upsert(self.key, "AAL", 5, 0.0)

    def test_cap_blocks_new_tickers_not_updates(self):
        for i in range(5):
            assert PF.upsert(self.key, f"T{chr(65 + i)}", 1, 1.0, cap=5)
        assert not PF.upsert(self.key, "NEWT", 1, 1.0, cap=5)
        assert PF.upsert(self.key, "TA", 9, 9.0, cap=5)

    def test_scoped_to_key(self):
        PF.upsert(self.key, "AAL", 1, 1.0)
        assert PF.list_for(_userdb.new_key()) == []

    def test_valuation_excludes_and_names_unpriced(self):
        PF.upsert(self.key, "AAL", 10, 10.0)     # cost 100
        PF.upsert(self.key, "TSLA", 2, 200.0)    # cost 400
        prices = {"AAL": 15.0, "TSLA": None}
        v = PF.value(PF.list_for(self.key), lambda t: prices.get(t))
        assert v.priced == 1
        assert v.unpriced == ["TSLA"]
        assert v.market_value == pytest.approx(150.0)
        assert v.cost_basis == pytest.approx(100.0)     # TSLA excluded from BOTH
        assert v.unrealized == pytest.approx(50.0)

    def test_top_weight(self):
        PF.upsert(self.key, "AAL", 10, 10.0)
        PF.upsert(self.key, "KO", 1, 10.0)
        v = PF.value(PF.list_for(self.key), lambda t: 10.0)
        assert v.top_weight[0] == "AAL"
        assert v.top_weight[1] == pytest.approx(100 / 110)

    def test_remove(self):
        PF.upsert(self.key, "AAL", 1, 1.0)
        assert PF.remove(self.key, "AAL")
        assert PF.list_for(self.key) == []


# ── Journal ──────────────────────────────────────────────────────────────────
class TestJournal(_Base):
    def test_create_and_list(self):
        eid = J.save(self.key, "AAL", thesis="Undervalued on refi",
                     invalidation="Loses hub slots", entry_price=12.0,
                     target_price=18.0, horizon="12 months")
        assert eid
        e = J.list_for(self.key)[0]
        assert e.ticker == "AAL" and e.thesis == "Undervalued on refi"
        assert e.invalidation == "Loses hub slots"
        assert e.entry_price == pytest.approx(12.0)

    def test_creation_requires_a_thesis(self):
        assert J.save(self.key, "AAL", thesis="   ") is None

    def test_update_keeps_id_and_created_at(self):
        eid = J.save(self.key, "AAL", thesis="v1")
        first = J.list_for(self.key)[0]
        time.sleep(0.02)
        eid2 = J.save(self.key, "AAL", entry_id=eid, thesis="v2",
                      bear_case="fuel spike")
        assert eid2 == eid
        e = J.list_for(self.key)[0]
        assert e.thesis == "v2" and e.bear_case == "fuel spike"
        assert e.created_at == pytest.approx(first.created_at, abs=1e-3)
        assert e.updated_at >= first.updated_at

    def test_two_theses_same_ticker_coexist(self):
        J.save(self.key, "AAL", thesis="pre-earnings")
        J.save(self.key, "AAL", thesis="post-earnings")
        assert len(J.list_for(self.key, "AAL")) == 2

    def test_cap_blocks_creation_not_updates(self):
        eids = [J.save(self.key, "AAL", thesis=f"t{i}", cap=3) for i in range(3)]
        assert all(eids)
        assert J.save(self.key, "AAL", thesis="one more", cap=3) is None
        assert J.save(self.key, "AAL", entry_id=eids[0], thesis="edited",
                      cap=3) == eids[0]

    def test_delete_scoped_to_key(self):
        eid = J.save(self.key, "AAL", thesis="mine")
        assert not J.delete(_userdb.new_key(), eid) or J.list_for(self.key)
        assert J.delete(self.key, eid)
        assert J.list_for(self.key) == []

    def test_oversized_fields_are_clipped_not_refused(self):
        eid = J.save(self.key, "AAL", thesis="x" * 10_000)
        assert len(J.list_for(self.key)[0].thesis) == J.MAX_FIELD
        assert eid


# ── Shared reports ───────────────────────────────────────────────────────────
class TestSharedReports(_Base):
    PAYLOAD = {"price": 12.0, "point": 15.0, "low": 11.0, "high": 19.0,
               "signal": "Undervalued"}

    def test_create_get_round_trip(self):
        slug = SR.create(self.key, "AAL", self.PAYLOAD)
        assert slug
        rep = SR.get(slug)
        assert rep is not None and rep.ticker == "AAL"
        assert rep.payload["point"] == 15.0

    def test_unknown_slug_is_none(self):
        assert SR.get("nope") is None
        assert SR.get("") is None

    def test_revoke_kills_the_link(self):
        slug = SR.create(self.key, "AAL", self.PAYLOAD)
        assert SR.revoke(self.key, slug)
        assert SR.get(slug) is None

    def test_only_the_owner_can_revoke(self):
        slug = SR.create(self.key, "AAL", self.PAYLOAD)
        SR.revoke(_userdb.new_key(), slug)
        assert SR.get(slug) is not None

    def test_expiry_is_enforced_on_read(self):
        slug = SR.create(self.key, "AAL", self.PAYLOAD, expires_days=1)
        assert SR.get(slug) is not None
        # Rewind the stored expiry to the past.
        c = _userdb.sq()
        c.execute("UPDATE svp_shared_reports SET expires_at=? WHERE slug=?",
                  (time.time() - 10, slug))
        c.commit()
        assert SR.get(slug) is None
        # …but the owner still sees it in their list, marked expired, to cull.
        mine = SR.list_for(self.key)
        assert len(mine) == 1 and mine[0].expired

    def test_private_looking_keys_are_stripped(self):
        slug = SR.create(self.key, "AAL",
                         {**self.PAYLOAD, "ledger_key": "leak",
                          "journal": ["x"], "email": "a@b.com"})
        rep = SR.get(slug)
        assert "ledger_key" not in rep.payload
        assert "journal" not in rep.payload
        assert "email" not in rep.payload
        assert rep.payload["point"] == 15.0

    def test_cap(self):
        for _ in range(3):
            assert SR.create(self.key, "AAL", self.PAYLOAD, cap=3)
        assert SR.create(self.key, "AAL", self.PAYLOAD, cap=3) is None

    def test_slugs_are_not_guessable_length(self):
        slug = SR.create(self.key, "AAL", self.PAYLOAD)
        assert len(slug) >= 10
