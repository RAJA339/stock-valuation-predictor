"""
Intraday OHLCV bars.
====================

Fetches intraday candles for the charting layer at 5m / 10m / 15m / 30m / 1h.

Yahoo does not serve a native 10-minute bar, so that interval is built by
resampling 5-minute data — the result is a true 10m candle (first open, max
high, min low, last close, summed volume), not an approximation.

Shares the options module's circuit breaker: intraday requests are the most
frequent calls in the app, so if Yahoo is throttling, this backs off with
everything else rather than deepening the block.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore

    _HAS_YF = True
except Exception:  # pragma: no cover
    _HAS_YF = False

from . import options as _opt   # circuit breaker + cooldown helpers

# label -> (yfinance interval, period to request, resample rule or None)
INTERVALS: dict[str, tuple[str, str, Optional[str]]] = {
    "5m":  ("5m",  "1mo", None),
    "10m": ("5m",  "1mo", "10min"),   # Yahoo has no native 10m bar
    "15m": ("15m", "1mo", None),
    "30m": ("30m", "1mo", None),
    "1h":  ("1h",  "3mo", None),
}

# Minutes per bar — used to convert a forward horizon in bars into wall time.
BAR_MINUTES = {"5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60}

_CACHE: dict[tuple, tuple[float, "IntradayData"]] = {}
_CACHE_TTL = 300.0


@dataclass
class IntradayData:
    ticker: str
    interval: str
    df: pd.DataFrame          # index=DatetimeIndex, cols Open/High/Low/Close/Volume
    source: str               # "yfinance" | "synthetic"
    reason: str = ""

    @property
    def is_live(self) -> bool:
        return self.source == "yfinance"

    @property
    def last_price(self) -> Optional[float]:
        return float(self.df["Close"].iloc[-1]) if not self.df.empty else None

    @property
    def session_change(self) -> Optional[float]:
        """Return over the visible window, as a fraction."""
        if len(self.df) < 2:
            return None
        first, last = float(self.df["Close"].iloc[0]), float(self.df["Close"].iloc[-1])
        return (last - first) / first if first else None


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule, label="right", closed="right").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    })
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def _synthetic(ticker: str, interval: str, spot: float, reason: str) -> IntradayData:
    """
    Deterministic intraday bars so the chart always renders.

    Volatility is scaled to the bar length so a 1h candle is genuinely wider
    than a 5m one, and volume follows the usual U-shaped session profile.
    """
    minutes = BAR_MINUTES[interval]
    n = {5: 780, 10: 390, 15: 260, 30: 130, 60: 300}[minutes]
    rng = np.random.default_rng(abs(hash((ticker, interval))) % (2**32))

    ann_vol = 0.35
    per_bar = ann_vol * np.sqrt(minutes / (60 * 6.5 * 252))
    steps = rng.normal(0, per_bar, n)
    close = spot * np.exp(np.cumsum(steps - steps.mean()))

    high = close * (1 + np.abs(rng.normal(0, per_bar * 0.6, n)))
    low = close * (1 - np.abs(rng.normal(0, per_bar * 0.6, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])

    t = np.linspace(0, 1, n)
    vol = (1.6 - 2.4 * t + 2.4 * t ** 2) * rng.uniform(6e5, 1.4e6, n)

    idx = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq=f"{minutes}min")
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                       "Close": close, "Volume": vol.astype(int)}, index=idx)
    return IntradayData(ticker, interval, df, source="synthetic", reason=reason)


def get_intraday(ticker: str, interval: str = "15m", spot_fallback: float = 100.0,
                 use_cache: bool = True) -> IntradayData:
    """Fetch intraday candles for ``ticker`` at ``interval``."""
    ticker = ticker.upper().strip()
    if interval not in INTERVALS:
        interval = "15m"
    key = (ticker, interval)
    now = time.time()

    if use_cache:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < _CACHE_TTL:
            return hit[1]

    yf_interval, period, rule = INTERVALS[interval]
    reason = "yfinance not installed"

    if _HAS_YF and _opt.cooldown_remaining() > 0:
        reason = f"backing off for {_opt.cooldown_remaining() / 60:.0f} more minute(s)"
    elif _HAS_YF:
        try:
            df = yf.Ticker(ticker).history(period=period, interval=yf_interval)
            if df is not None and not df.empty:
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if rule:
                    df = _resample(df, rule)
                if len(df) >= 30:
                    out = IntradayData(ticker, interval, df, source="yfinance")
                    _CACHE[key] = (now, out)
                    return out
                reason = f"only {len(df)} bars returned for {ticker} {interval}"
            else:
                reason = f"no intraday data returned for {ticker}"
        except Exception as exc:
            _opt._trip_breaker(exc)
            reason = f"{type(exc).__name__}: {exc}"[:200]

    out = _synthetic(ticker, interval, spot_fallback, reason)
    _CACHE[key] = (now, out)
    return out


def clear_cache() -> None:
    _CACHE.clear()
