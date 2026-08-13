"""
Technical entry filters.
========================

Before an "Undervalued" fundamental signal is allowed to become an active buy,
it must clear technical confirmation — so we don't catch a falling knife. Uses
only the price/volume ``history`` from :mod:`svp.data.market`, so it works with
live or offline data.

Confirmations:
  * **Trend alignment** — price above a rising 200-day SMA.
  * **RSI(14)** — not deeply oversold-and-still-falling; bullish RSI divergence
    (price lower low, RSI higher low) is flagged as a positive.
  * **Volume support** — current price near a high-volume node (volume-profile
    point of control), i.e. a level the market has defended.
  * **ATR(14)** — reported for downstream stop sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TechnicalSignals:
    price: float
    sma200: Optional[float]
    trend_up: bool                 # price > rising SMA200
    rsi: Optional[float]
    rsi_bullish_divergence: bool
    near_volume_support: bool
    volume_poc: Optional[float]    # price level of highest traded volume
    atr: Optional[float]
    confirmations: dict = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        """Entry confirmed when a majority of available checks pass."""
        checks = [v for v in self.confirmations.values() if v is not None]
        return bool(checks) and sum(bool(v) for v in checks) >= (len(checks) / 2)

    @property
    def score(self) -> float:
        checks = [bool(v) for v in self.confirmations.values() if v is not None]
        return float(np.mean(checks)) if checks else 0.0


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(2, window // 2)).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(history: pd.DataFrame, period: int = 14) -> Optional[float]:
    need = {"High", "Low", "Close"}
    if not need.issubset(history.columns):
        # Approximate ATR from close-to-close range when OHLC is absent.
        if "Close" in history:
            return float(history["Close"].diff().abs().rolling(period).mean().iloc[-1])
        return None
    h, l, c = history["High"], history["Low"], history["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    val = tr.rolling(period, min_periods=period // 2).mean().iloc[-1]
    return float(val) if pd.notna(val) else None


def _bullish_divergence(price: pd.Series, rsi_series: pd.Series, lookback: int = 60) -> bool:
    """Price makes a lower low while RSI makes a higher low → bullish divergence."""
    p = price.iloc[-lookback:]
    r = rsi_series.iloc[-lookback:]
    if len(p.dropna()) < 20 or r.dropna().empty:
        return False
    mid = len(p) // 2
    p1, p2 = p.iloc[:mid].min(), p.iloc[mid:].min()
    r1 = r.iloc[:mid].min()
    r2_idx = p.iloc[mid:].idxmin()
    r2 = r.get(r2_idx, np.nan)
    if np.isnan(r1) or np.isnan(r2):
        return False
    return bool(p2 <= p1 and r2 > r1)


def _volume_poc(history: pd.DataFrame, bins: int = 30, window: int = 252) -> Optional[float]:
    """Volume-profile point of control: price bin with the most traded volume."""
    if "Volume" not in history.columns or "Close" not in history.columns:
        return None
    df = history.iloc[-window:]
    vol = df["Volume"].to_numpy(dtype=float)
    px = df["Close"].to_numpy(dtype=float)
    if vol.sum() <= 0 or len(px) < 10:
        return None
    edges = np.linspace(px.min(), px.max(), bins + 1)
    idx = np.clip(np.digitize(px, edges) - 1, 0, bins - 1)
    profile = np.zeros(bins)
    for i, v in zip(idx, vol):
        profile[i] += v
    poc_bin = int(profile.argmax())
    return float((edges[poc_bin] + edges[poc_bin + 1]) / 2)


def analyze(history: pd.DataFrame) -> Optional[TechnicalSignals]:
    """Compute all technical confirmations from a price/volume history frame."""
    if history is None or history.empty or "Close" not in history:
        return None
    close = history["Close"].dropna()
    if len(close) < 30:
        return None

    price = float(close.iloc[-1])
    s200 = sma(close, 200)
    sma200 = float(s200.iloc[-1]) if pd.notna(s200.iloc[-1]) else None
    # Rising SMA: current above value ~1 month ago.
    trend_up = bool(
        sma200 is not None
        and price > sma200
        and len(s200.dropna()) > 21
        and s200.iloc[-1] > s200.iloc[-21]
    )

    rsi_series = rsi(close)
    rsi_val = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else None
    divergence = _bullish_divergence(close, rsi_series)

    poc = _volume_poc(history)
    near_support = bool(poc is not None and abs(price - poc) / price <= 0.07)

    atr_val = atr(history)

    confirmations = {
        "trend_alignment_200sma": trend_up,
        "rsi_not_falling_knife": None if rsi_val is None else (rsi_val > 30 or divergence),
        "rsi_bullish_divergence": divergence,
        "volume_support": near_support if poc is not None else None,
    }

    return TechnicalSignals(
        price=price,
        sma200=sma200,
        trend_up=trend_up,
        rsi=rsi_val,
        rsi_bullish_divergence=divergence,
        near_volume_support=near_support,
        volume_poc=poc,
        atr=atr_val,
        confirmations=confirmations,
    )
