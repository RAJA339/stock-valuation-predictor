"""
Pytest suite for Deribit crypto options — skew and dealer GEX.

The network is stubbed; the value is in the parsing and the two analytics, which
are pure. The properties that must hold: instrument names parse (and malformed
ones are dropped, not guessed), IV is converted from Deribit's percent, a
put-heavy smile reads as put skew, and the dealer-gamma sign follows its stated
convention — all calls net long gamma, all puts net short.
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.data import crypto_options as CO          # noqa: E402


# A fixed "now" so expiry-day arithmetic is deterministic.
_NOW = dt.datetime(2024, 12, 1, 8, 0, tzinfo=dt.timezone.utc).timestamp()


def _row(name, iv_pct, oi=100.0, spot=100000.0):
    return {"instrument_name": name, "mark_iv": iv_pct,
            "open_interest": oi, "underlying_price": spot}


def _chain(spot=100000.0, expiry="27DEC24", put_extra=0.0, oi_call=100.0, oi_put=100.0):
    """A symmetric smile, with `put_extra` added to every put IV to create skew."""
    rows = []
    for k in (80000, 90000, 95000, 100000, 105000, 110000, 120000):
        wing = 0.0015 * abs(k - spot) / 1000
        rows.append(_row(f"BTC-{expiry}-{k}-C", (0.60 + wing) * 100, oi_call, spot))
        rows.append(_row(f"BTC-{expiry}-{k}-P", (0.60 + wing + put_extra) * 100, oi_put, spot))
    return CO.parse_summary(rows, now=_NOW)


# ── Parsing ──────────────────────────────────────────────────────────────────
class TestParse:
    def test_parses_a_valid_instrument(self):
        q = CO.parse_summary([_row("BTC-27DEC24-100000-C", 65.0)], now=_NOW)
        assert len(q) == 1
        assert q[0].strike == 100000.0 and q[0].kind == "C"

    def test_iv_is_converted_from_percent(self):
        q = CO.parse_summary([_row("BTC-27DEC24-100000-C", 65.0)], now=_NOW)
        assert q[0].iv == pytest.approx(0.65)

    def test_expiry_days_are_computed(self):
        q = CO.parse_summary([_row("BTC-27DEC24-100000-C", 65.0)], now=_NOW)
        assert q[0].expiry_days == pytest.approx(26.0, abs=0.1)   # Dec 1 → Dec 27

    @pytest.mark.parametrize("name", [
        "GARBAGE", "BTC-100000-C", "BTC-27DEC24-100000", "BTC-99XXX24-100000-C",
        "BTC-27DEC24-100000-X",
    ])
    def test_malformed_names_are_dropped(self, name):
        assert CO.parse_summary([_row(name, 65.0)], now=_NOW) == []

    def test_expired_contracts_are_dropped(self):
        # 27DEC20 is in the past relative to _NOW.
        assert CO.parse_summary([_row("BTC-27DEC20-100000-C", 65.0)], now=_NOW) == []

    def test_missing_iv_is_dropped(self):
        assert CO.parse_summary([_row("BTC-27DEC24-100000-C", None)], now=_NOW) == []

    def test_empty_input(self):
        assert CO.parse_summary([]) == []
        assert CO.parse_summary(None) == []


# ── Skew ─────────────────────────────────────────────────────────────────────
class TestSkew:
    def test_put_heavy_smile_reads_as_put_skew(self):
        sk = CO.nearest_expiry_skew(_chain(put_extra=0.08))
        assert sk is not None
        assert sk.rr_25 > 0.02
        assert "Put skew" in sk.reading

    def test_call_heavy_smile_reads_as_call_skew(self):
        sk = CO.nearest_expiry_skew(_chain(put_extra=-0.08))
        assert sk.rr_25 < -0.02
        assert "Call skew" in sk.reading

    def test_symmetric_smile_is_neutral(self):
        sk = CO.nearest_expiry_skew(_chain(put_extra=0.0))
        assert abs(sk.rr_25) <= 0.02
        assert "symmetric" in sk.reading

    def test_atm_iv_is_reasonable(self):
        sk = CO.nearest_expiry_skew(_chain())
        assert 0.4 < sk.atm_iv < 0.9

    def test_empty_chain_is_none(self):
        assert CO.nearest_expiry_skew([]) is None

    def test_too_few_strikes_is_none(self):
        thin = CO.parse_summary([_row("BTC-27DEC24-100000-C", 60.0),
                                 _row("BTC-27DEC24-100000-P", 60.0)], now=_NOW)
        assert CO.nearest_expiry_skew(thin) is None


# ── Dealer GEX ───────────────────────────────────────────────────────────────
class TestGEX:
    def test_all_calls_net_positive_gamma(self):
        calls = [q for q in _chain() if q.kind == "C"]
        g = CO.dealer_gex(calls, spot=100000.0)
        assert g.total > 0
        assert g.regime == "Positive"

    def test_all_puts_net_negative_gamma(self):
        puts = [q for q in _chain() if q.kind == "P"]
        g = CO.dealer_gex(puts, spot=100000.0)
        assert g.total < 0
        assert g.regime == "Negative"

    def test_balanced_book_can_flip(self):
        g = CO.dealer_gex(_chain(), spot=100000.0)
        assert g is not None
        assert g.flip_strike is None or 80000 <= g.flip_strike <= 120000

    def test_gamma_is_largest_near_the_money(self):
        """Per-strike |gamma$| should peak near spot, where BS gamma is highest."""
        calls = [q for q in _chain() if q.kind == "C"]
        g = CO.dealer_gex(calls, spot=100000.0)
        peak = max(g.by_strike, key=lambda kv: abs(kv[1]))[0]
        assert 90000 <= peak <= 110000

    def test_reading_mentions_the_regime(self):
        g = CO.dealer_gex([q for q in _chain() if q.kind == "C"], spot=100000.0)
        assert "dealer gamma" in g.reading

    def test_empty_chain_is_none(self):
        assert CO.dealer_gex([]) is None

    def test_zero_open_interest_is_none(self):
        z = _chain(oi_call=0.0, oi_put=0.0)
        assert CO.dealer_gex(z, spot=100000.0) is None


# ── Fetch / circuit breaker ──────────────────────────────────────────────────
class TestGetChain:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        CO.reset_circuit()
        monkeypatch.setattr(CO.storage, "cache_get", lambda k: None)
        monkeypatch.setattr(CO.storage, "cache_set", lambda *a, **k: None)
        yield
        CO.reset_circuit()

    def _stub(self, monkeypatch, result=None, exc=None):
        import types

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"result": result or []}

        mod = types.ModuleType("requests")
        mod.get = (lambda *a, **k: (_ for _ in ()).throw(exc)) if exc else (lambda *a, **k: Resp())
        monkeypatch.setitem(sys.modules, "requests", mod)

    def test_unknown_currency_is_none(self):
        assert CO.get_chain("DOGE") is None

    def test_good_response_parses(self, monkeypatch):
        # get_chain parses against the real clock, so the expiry must be future.
        future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=180)
        tok = future.strftime("%d%b%y").upper()
        self._stub(monkeypatch, result=[_row(f"BTC-{tok}-100000-C", 65.0)])
        chain = CO.get_chain("BTC")
        assert chain and chain[0].strike == 100000.0

    def test_network_error_trips_breaker_and_returns_none(self, monkeypatch):
        self._stub(monkeypatch, exc=OSError("down"))
        assert CO.get_chain("BTC") is None
        assert CO.cooldown_remaining() > 0

    def test_rate_limit_is_a_long_cooldown(self, monkeypatch):
        self._stub(monkeypatch, exc=RuntimeError("HTTP 429"))
        CO.get_chain("BTC")
        assert CO.cooldown_remaining() > CO._ERROR_COOLDOWN
