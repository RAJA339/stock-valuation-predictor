"""
Live market-data pipeline.
==========================

Auto-fetches live ticker price, market cap, shares outstanding, sector/industry
and historical price history. Primary source is ``yfinance``; if it is not
installed or the network is unavailable, an Alpha Vantage path (needs
``ALPHAVANTAGE_API_KEY``) is tried, and finally a deterministic offline stub so
the rest of the app keeps working.

Everything returns a ``MarketData`` dataclass with an ``is_live`` flag the UI
uses to show a LIVE / OFFLINE pill.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:  # optional
    import yfinance as yf  # type: ignore

    _HAS_YF = True
except Exception:  # pragma: no cover
    _HAS_YF = False

from . import storage


@dataclass
class MarketData:
    ticker: str
    price: Optional[float] = None
    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    name: Optional[str] = None
    currency: str = "USD"
    history: pd.DataFrame = field(default_factory=pd.DataFrame)  # Date-indexed OHLC
    source: str = "offline"

    @property
    def is_live(self) -> bool:
        return self.source not in ("offline", "")


def _offline_stub(ticker: str, fallback_price: float = 100.0) -> MarketData:
    """Deterministic pseudo-history so charts/backtests still render offline."""
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    # ~9 years of business days so 5-year forward backtest horizons have room.
    days = pd.date_range(end=pd.Timestamp.today().normalize(), periods=9 * 252, freq="B")
    # Geometric random walk seeded by the ticker for repeatability.
    drift, vol = 0.07 / 252, 0.02
    steps = rng.normal(drift, vol, len(days))
    prices = fallback_price * np.exp(np.cumsum(steps))
    prices *= fallback_price / prices[-1]  # end at fallback_price
    hist = pd.DataFrame({"Close": prices}, index=days)
    hist["Open"] = hist["High"] = hist["Low"] = hist["Close"]
    return MarketData(
        ticker=ticker,
        price=float(prices[-1]),
        market_cap=float(prices[-1]) * 1e9,
        shares_outstanding=1e9,
        sector="Unknown",
        industry="Unknown",
        name=ticker,
        history=hist,
        source="offline",
    )


def _from_yfinance(ticker: str, period: str) -> Optional[MarketData]:
    if not _HAS_YF:
        return None
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.get_info()  # newer yfinance
        except Exception:
            info = getattr(tk, "info", {}) or {}
        hist = tk.history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        hist = hist.rename(columns=str.title)
        price = float(hist["Close"].iloc[-1])
        return MarketData(
            ticker=ticker,
            price=info.get("currentPrice") or price,
            market_cap=info.get("marketCap"),
            shares_outstanding=info.get("sharesOutstanding"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            name=info.get("shortName") or info.get("longName") or ticker,
            currency=info.get("currency", "USD"),
            # Volume rides along when the source provides it — the chart
            # workspace and liquidity studies use it; every consumer selects
            # columns by name, so its presence costs nothing elsewhere.
            history=hist[[c for c in ("Open", "High", "Low", "Close", "Volume")
                          if c in hist.columns]],
            source="yfinance",
        )
    except Exception:
        return None


def _from_alpha_vantage(ticker: str) -> Optional[MarketData]:
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        return None
    try:
        base = "https://www.alphavantage.co/query"
        quote = requests.get(
            base, params={"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": key}, timeout=10
        ).json()
        price = float(quote["Global Quote"]["05. price"])
        daily = requests.get(
            base,
            params={"function": "TIME_SERIES_DAILY", "symbol": ticker, "outputsize": "full", "apikey": key},
            timeout=15,
        ).json()
        ts = daily["Time Series (Daily)"]
        df = pd.DataFrame(ts).T.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().rename(
            columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close"}
        )
        return MarketData(
            ticker=ticker,
            price=price,
            history=df[["Open", "High", "Low", "Close"]],
            name=ticker,
            source="alphavantage",
        )
    except Exception:
        return None


def get_market_data(ticker: str, period: str = "10y", fallback_price: float = 100.0) -> MarketData:
    """
    Return :class:`MarketData` for ``ticker`` from the best available source.

    Never raises — always returns something usable (offline stub as last resort).
    """
    ticker = ticker.upper().strip()
    cache_key = f"market:{ticker}:{period}"

    # Persisted metadata cache (history is re-fetched live but small enough).
    meta = storage.cache_get(cache_key)

    md = _from_yfinance(ticker, period) or _from_alpha_vantage(ticker)
    if md is None:
        md = _offline_stub(ticker, fallback_price)
    else:
        storage.cache_set(
            cache_key,
            {"price": md.price, "market_cap": md.market_cap, "sector": md.sector, "name": md.name},
            ttl=3600,
        )

    # Backfill from a previous live pull if this run is offline.
    if md.source == "offline" and meta:
        md.price = meta.get("price") or md.price
        md.market_cap = meta.get("market_cap") or md.market_cap
        md.sector = meta.get("sector") or md.sector
        md.name = meta.get("name") or md.name
    return md


def returns_over_horizon(history: pd.DataFrame, years: float) -> Optional[float]:
    """Fractional total return of Close over the trailing ``years`` window."""
    if history is None or history.empty or "Close" not in history:
        return None
    end = history["Close"].iloc[-1]
    target = history.index[-1] - pd.Timedelta(days=int(round(years * 365.25)))
    window = history.loc[history.index <= target]
    if window.empty:
        return None
    start = window["Close"].iloc[-1]
    if start <= 0:
        return None
    return float(end / start - 1.0)
