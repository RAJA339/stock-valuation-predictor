"""
Crypto asset registry.
======================

Crypto is a different animal from equities, and this module is deliberately thin
because of it. There is no 10-K, no cash flow, no book value — so none of the
valuation stack (DCF, intrinsic range, Piotroski/Altman/Beneish, accruals)
applies, and this app does not pretend otherwise. Inventing a "fair value" for a
token would be exactly the fabricated number the rest of the project refuses to
produce.

What *does* transfer, unchanged, is everything that reads a price series: the
TradingView chart, the technical indicators, and — the honest centrepiece — each
indicator's *measured* hit rate with a Wilson interval. Those work on any OHLCV
bars, and yfinance serves crypto under the ``<COIN>-USD`` convention, so the
existing intraday / indicator / accuracy pipeline runs on a coin with no new
machinery.

This module only holds the registry that maps a coin to its two symbol systems:
yfinance for the data we compute on, TradingView for the live chart embed.
Markets here are 24/7, so there is no session, no overnight gap, and no
market-hours gating — a fact the analysis copy is careful to reflect rather than
reusing the equity session assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Coin:
    symbol: str          # short ticker, e.g. "BTC"
    name: str            # display name
    yf: str              # yfinance symbol, e.g. "BTC-USD"
    tv: str              # TradingView symbol, e.g. "BINANCE:BTCUSDT"


# Major coins by market cap, using deep, liquid Binance USDT pairs for the
# TradingView feed. Stablecoins are deliberately excluded — a chart of a token
# pegged to $1 has nothing to analyse.
_COINS: tuple[Coin, ...] = (
    Coin("BTC",  "Bitcoin",      "BTC-USD",  "BINANCE:BTCUSDT"),
    Coin("ETH",  "Ethereum",     "ETH-USD",  "BINANCE:ETHUSDT"),
    Coin("SOL",  "Solana",       "SOL-USD",  "BINANCE:SOLUSDT"),
    Coin("BNB",  "BNB",          "BNB-USD",  "BINANCE:BNBUSDT"),
    Coin("XRP",  "XRP",          "XRP-USD",  "BINANCE:XRPUSDT"),
    Coin("ADA",  "Cardano",      "ADA-USD",  "BINANCE:ADAUSDT"),
    Coin("DOGE", "Dogecoin",     "DOGE-USD", "BINANCE:DOGEUSDT"),
    Coin("AVAX", "Avalanche",    "AVAX-USD", "BINANCE:AVAXUSDT"),
    Coin("LINK", "Chainlink",    "LINK-USD", "BINANCE:LINKUSDT"),
    Coin("DOT",  "Polkadot",     "DOT-USD",  "BINANCE:DOTUSDT"),
    Coin("MATIC", "Polygon",     "MATIC-USD", "BINANCE:MATICUSDT"),
    Coin("LTC",  "Litecoin",     "LTC-USD",  "BINANCE:LTCUSDT"),
)

_BY_SYMBOL = {c.symbol: c for c in _COINS}


def coins() -> tuple[Coin, ...]:
    """The full registry, in market-cap order."""
    return _COINS


def labels() -> list[str]:
    """Human-readable labels for a selectbox, e.g. ``'BTC — Bitcoin'``."""
    return [f"{c.symbol} — {c.name}" for c in _COINS]


def by_label(label: str) -> Optional[Coin]:
    """Resolve a ``'BTC — Bitcoin'`` selectbox label back to its Coin."""
    sym = (label or "").split("—", 1)[0].strip().upper()
    return _BY_SYMBOL.get(sym)


def get(symbol: str) -> Optional[Coin]:
    """Look up a coin by its short symbol, case-insensitively."""
    return _BY_SYMBOL.get((symbol or "").strip().upper())


def is_supported(symbol: str) -> bool:
    return (symbol or "").strip().upper() in _BY_SYMBOL
