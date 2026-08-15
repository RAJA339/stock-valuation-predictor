"""
Pytest suite for the crypto registry.

The registry is small, but two things must hold or the pane misbehaves silently:
the two symbol systems must stay consistent (a yfinance ticker the data layer can
fetch, a TradingView symbol the chart can embed), and label round-tripping must
be exact, because the selectbox hands a formatted label back and the wrong parse
would quietly analyse the wrong coin.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.data import crypto as C          # noqa: E402


class TestRegistry:
    def test_registry_is_not_empty(self):
        assert len(C.coins()) >= 8

    def test_bitcoin_and_ether_are_present(self):
        syms = {c.symbol for c in C.coins()}
        assert {"BTC", "ETH"} <= syms

    def test_yfinance_tickers_use_the_usd_convention(self):
        for c in C.coins():
            assert c.yf.endswith("-USD"), c.yf
            assert c.yf.split("-")[0] == c.symbol

    def test_tradingview_symbols_are_exchange_qualified(self):
        for c in C.coins():
            assert ":" in c.tv, c.tv
            assert c.tv.endswith("USDT"), c.tv

    def test_symbols_are_unique(self):
        syms = [c.symbol for c in C.coins()]
        assert len(syms) == len(set(syms))

    def test_no_stablecoins(self):
        """A chart of a $1-pegged token has nothing to analyse."""
        syms = {c.symbol for c in C.coins()}
        assert not ({"USDT", "USDC", "DAI", "BUSD", "TUSD"} & syms)


class TestLookup:
    def test_get_is_case_insensitive(self):
        assert C.get("btc").symbol == "BTC"
        assert C.get("BTC").symbol == "BTC"

    def test_get_unknown_is_none(self):
        assert C.get("NOTACOIN") is None
        assert C.get("") is None
        assert C.get(None) is None

    def test_is_supported(self):
        assert C.is_supported("eth")
        assert not C.is_supported("XYZ")


class TestLabels:
    def test_labels_pair_symbol_and_name(self):
        labels = C.labels()
        assert labels[0].startswith("BTC —")
        assert len(labels) == len(C.coins())

    @pytest.mark.parametrize("coin", [c for c in C.coins()])
    def test_every_label_round_trips_to_its_coin(self, coin):
        label = f"{coin.symbol} — {coin.name}"
        assert C.by_label(label) is coin

    def test_by_label_tolerates_a_bare_symbol(self):
        assert C.by_label("ETH").symbol == "ETH"

    def test_by_label_unknown_is_none(self):
        assert C.by_label("ZZZ — Nothing") is None
        assert C.by_label("") is None
