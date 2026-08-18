"""
Pytest suite for the Polymarket prediction-market reader.

Stubs are shaped like the real Gamma API (list fields arrive as JSON
*strings*, numbers under two possible key names). The honesty properties:
only binary Yes/No markets are read as probabilities, the Yes price is found
by outcome label rather than by position, out-of-range prices are rejected
rather than clamped into fake probabilities, and thin books carry their flag.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.data import prediction_markets as PM       # noqa: E402


def _m(question="Will Bitcoin close above $150,000 by March 31?",
       outcomes='["Yes", "No"]', prices='["0.62", "0.38"]',
       volume=250_000.0, liquidity=40_000.0,
       end="2026-03-31T12:00:00Z", slug="btc-150k-march"):
    return {"question": question, "outcomes": outcomes,
            "outcomePrices": prices, "volumeNum": volume,
            "liquidityNum": liquidity, "endDate": end, "slug": slug}


class TestParseMarket:
    def test_reads_a_binary_market(self):
        m = PM.parse_market(_m())
        assert m is not None
        assert m.yes_prob == pytest.approx(0.62)
        assert m.volume == pytest.approx(250_000)
        assert m.ends == "Mar 31, 2026"
        assert m.url.endswith("/market/btc-150k-march")

    def test_yes_found_by_label_not_position(self):
        m = PM.parse_market(_m(outcomes='["No", "Yes"]',
                               prices='["0.38", "0.62"]'))
        assert m.yes_prob == pytest.approx(0.62)

    def test_multi_outcome_markets_are_refused(self):
        assert PM.parse_market(_m(outcomes='["A", "B", "C"]',
                                  prices='["0.2", "0.3", "0.5"]')) is None

    def test_non_yes_no_binary_is_refused(self):
        assert PM.parse_market(_m(outcomes='["Trump", "Biden"]')) is None

    def test_out_of_range_price_is_refused_not_clamped(self):
        assert PM.parse_market(_m(prices='["1.62", "0.38"]')) is None
        assert PM.parse_market(_m(prices='["-0.1", "1.1"]')) is None

    def test_malformed_fields_are_refused(self):
        assert PM.parse_market(_m(prices='not json')) is None
        assert PM.parse_market(_m(question="")) is None
        assert PM.parse_market("not a dict") is None

    def test_native_lists_accepted_too(self):
        m = PM.parse_market(_m(outcomes=["Yes", "No"], prices=["0.5", "0.5"]))
        assert m is not None

    def test_volume_falls_back_across_key_names(self):
        d = _m()
        del d["volumeNum"]
        d["volume"] = "9000"
        assert PM.parse_market(d).volume == pytest.approx(9000)

    def test_missing_end_date_is_blank_not_invented(self):
        assert PM.parse_market(_m(end=None)).ends == ""

    def test_thin_liquidity_is_flagged(self):
        assert PM.parse_market(_m(liquidity=500.0)).is_thin
        assert not PM.parse_market(_m(liquidity=50_000.0)).is_thin


class TestFilter:
    RAW = [
        _m(question="Will Bitcoin close above $150,000?", volume=900_000),
        _m(question="Will Ethereum flip Bitcoin this year?", volume=100_000,
           slug="eth-flip"),
        _m(question="Will the Fed cut rates in March?", volume=800_000,
           slug="fed-cut"),
        _m(question="Will Solana hit $500?", volume=50_000, slug="sol-500"),
    ]

    def test_keyword_filter_scopes_to_the_coin(self):
        out = PM.filter_markets(self.RAW, PM.keywords_for("SOL"))
        assert [m.question for m in out] == ["Will Solana hit $500?"]

    def test_btc_keywords_catch_both_btc_markets(self):
        out = PM.filter_markets(self.RAW, PM.keywords_for("BTC"))
        qs = " ".join(m.question for m in out)
        assert "150,000" in qs and "flip" in qs
        assert "Fed" not in qs

    def test_sorted_by_volume_and_capped(self):
        out = PM.filter_markets(self.RAW, ["will"], max_n=2)
        assert len(out) == 2
        assert out[0].volume >= out[1].volume

    def test_unmapped_coin_gets_a_generic_net(self):
        kw = PM.keywords_for("PEPE", "Pepe (PEPE)")
        assert "crypto" in kw
        assert PM.filter_markets(self.RAW, kw) == []

    def test_empty_input_is_empty(self):
        assert PM.filter_markets([], ["bitcoin"]) == []
        assert PM.filter_markets(None, ["bitcoin"]) == []


class TestBreaker:
    def test_trip_and_recover_shape(self):
        PM._trip(Exception("429 too many requests"))
        assert PM.cooldown_remaining() > 600        # rate-limit → long cooldown
        PM._COOLDOWN_UNTIL = 0.0                    # reset for other tests
        PM._trip(Exception("connection refused"))
        assert 0 < PM.cooldown_remaining() <= PM._ERROR_COOLDOWN
        PM._COOLDOWN_UNTIL = 0.0

    def test_open_breaker_short_circuits_fetch(self):
        PM._COOLDOWN_UNTIL = __import__("time").time() + 60
        assert PM.fetch_markets() is None
        PM._COOLDOWN_UNTIL = 0.0
