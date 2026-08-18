"""
Technical snapshot — the chart, summarised in one honest card.
==============================================================

Answers, in a reader's order: what is the trend, how strong is the momentum,
is the oscillator stretched, where are the nearest levels, and how volatile is
the tape right now. One 0–100 score summarises how *aligned* the technical
picture is — alignment, not prophecy: a high score means the trend readings
agree with each other, it does not mean the price will keep going.

Every component of the score is returned alongside it, so the UI can show the
breakdown rather than asking anyone to trust a single number. Language rules
mirror the rest of the app: describe, never instruct — "price is above both
long-term averages", not "buy".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

MIN_BARS = 60          # under this, every reading would be an artefact
_SWING_ORDER = 5       # bars each side that define a swing high/low
_LEVEL_WINDOW = 180    # how far back to look for support/resistance


@dataclass
class Snapshot:
    trend: str                       # Bullish | Neutral | Bearish
    momentum: str                    # Strong | Moderate | Weak
    rsi: Optional[float]
    rsi_status: str                  # Oversold | Neutral | Overbought
    ma_alignment: str                # Stacked bullish | Mixed | Stacked bearish
    rel_volume: Optional[float]      # latest volume vs 20-day average
    support: Optional[float]         # nearest swing low below price
    resistance: Optional[float]      # nearest swing high above price
    volatility: str                  # Low | Normal | Elevated
    atr_pct: Optional[float]         # ATR(14) as % of price
    score: int                       # 0–100 technical alignment
    components: dict[str, int] = field(default_factory=dict)
    narrative: str = ""


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr_pct(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return None
    prev = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    px = float(df["Close"].iloc[-1])
    if pd.isna(atr) or px <= 0:
        return None
    return float(atr) / px * 100


def _swings(df: pd.DataFrame, order: int = _SWING_ORDER):
    """Swing highs/lows: bars that are the extreme of their ±order window."""
    win = df.tail(_LEVEL_WINDOW)
    highs, lows = [], []
    h = win["High"].to_numpy() if "High" in win.columns else win["Close"].to_numpy()
    lo = win["Low"].to_numpy() if "Low" in win.columns else win["Close"].to_numpy()
    for i in range(order, len(win) - order):
        if h[i] == h[i - order:i + order + 1].max():
            highs.append(float(h[i]))
        if lo[i] == lo[i - order:i + order + 1].min():
            lows.append(float(lo[i]))
    return highs, lows


def _nearest_levels(df: pd.DataFrame, price: float):
    highs, lows = _swings(df)
    support = max((p for p in lows if p < price), default=None)
    resistance = min((p for p in highs if p > price), default=None)
    return support, resistance


def compute(df: pd.DataFrame) -> Optional[Snapshot]:
    """Compute the snapshot from daily OHLC(V) history, or ``None``."""
    if df is None or df.empty or "Close" not in df.columns or len(df) < MIN_BARS:
        return None
    df = df.dropna(subset=["Close"]).sort_index()
    if len(df) < MIN_BARS:
        return None

    close = df["Close"].astype(float)
    price = float(close.iloc[-1])

    sma50 = _sma(close, 50).iloc[-1] if len(close) >= 50 else np.nan
    sma200 = _sma(close, 200).iloc[-1] if len(close) >= 200 else np.nan

    # ── Trend: where the price sits relative to its long averages ───────────
    above50 = price > sma50 if not pd.isna(sma50) else None
    above200 = price > sma200 if not pd.isna(sma200) else None
    votes = [v for v in (above50, above200) if v is not None]
    if votes and all(votes):
        trend = "Bullish"
    elif votes and not any(votes):
        trend = "Bearish"
    else:
        trend = "Neutral"

    # ── MA alignment: 20 > 50 > 200 stacked, or inverted, or mixed ──────────
    sma20 = _sma(close, 20).iloc[-1] if len(close) >= 20 else np.nan
    stack = [v for v in (sma20, sma50, sma200) if not pd.isna(v)]
    if len(stack) >= 2 and all(stack[i] > stack[i + 1] for i in range(len(stack) - 1)):
        ma_alignment = "Stacked bullish"
    elif len(stack) >= 2 and all(stack[i] < stack[i + 1] for i in range(len(stack) - 1)):
        ma_alignment = "Stacked bearish"
    else:
        ma_alignment = "Mixed"

    # ── Momentum: 3-month rate of change, sized against its own volatility ──
    look = min(63, len(close) - 1)
    roc = price / float(close.iloc[-1 - look]) - 1
    daily_vol = float(close.pct_change().std())
    # A move worth calling "strong" should be large relative to what this
    # name's ordinary wobble could produce over the same window.
    threshold = daily_vol * np.sqrt(look)
    if abs(roc) > 1.5 * threshold:
        momentum = "Strong"
    elif abs(roc) > 0.5 * threshold:
        momentum = "Moderate"
    else:
        momentum = "Weak"

    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi_val = _rsi(close).iloc[-1]
    rsi_val = None if pd.isna(rsi_val) else float(rsi_val)
    if rsi_val is None:
        rsi_status = "Neutral"
    elif rsi_val >= 70:
        rsi_status = "Overbought"
    elif rsi_val <= 30:
        rsi_status = "Oversold"
    else:
        rsi_status = "Neutral"

    # ── Relative volume ──────────────────────────────────────────────────────
    rel_volume = None
    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        avg20 = vol.rolling(20).mean().iloc[-1]
        if not pd.isna(avg20) and avg20 > 0 and not pd.isna(vol.iloc[-1]):
            rel_volume = float(vol.iloc[-1] / avg20)

    # ── Levels & volatility ──────────────────────────────────────────────────
    support, resistance = _nearest_levels(df, price)
    atr_pct = _atr_pct(df)
    if atr_pct is None:
        volatility = "Normal"
    elif atr_pct > 4.0:
        volatility = "Elevated"
    elif atr_pct < 1.5:
        volatility = "Low"
    else:
        volatility = "Normal"

    # ── Alignment score: transparent components, each 0–100 ─────────────────
    comp: dict[str, int] = {}
    comp["Trend"] = {"Bullish": 100, "Neutral": 50, "Bearish": 0}[trend]
    comp["MA alignment"] = {"Stacked bullish": 100, "Mixed": 50,
                            "Stacked bearish": 0}[ma_alignment]
    if momentum == "Strong":
        comp["Momentum"] = 100 if roc > 0 else 0
    elif momentum == "Moderate":
        comp["Momentum"] = 75 if roc > 0 else 25
    else:
        comp["Momentum"] = 50
    # RSI component rewards room to run in the trend direction, not extremes.
    if rsi_val is None:
        comp["RSI"] = 50
    elif rsi_status == "Overbought":
        comp["RSI"] = 30
    elif rsi_status == "Oversold":
        comp["RSI"] = 30 if trend == "Bearish" else 60
    else:
        comp["RSI"] = 80
    weights = {"Trend": 0.35, "MA alignment": 0.25, "Momentum": 0.25, "RSI": 0.15}
    score = int(round(sum(comp[k] * w for k, w in weights.items())))
    score = max(0, min(100, score))

    # ── Narrative — measured, no instructions ────────────────────────────────
    bits = []
    if above50 is not None and above200 is not None:
        rel = ("above both" if above50 and above200 else
               "below both" if not above50 and not above200 else "between")
        bits.append(f"Price is {rel} the 50- and 200-day averages")
    elif above50 is not None:
        bits.append(f"Price is {'above' if above50 else 'below'} the 50-day average")
    if rsi_status != "Neutral" and rsi_val is not None:
        bits.append(f"RSI at {rsi_val:.0f} reads {rsi_status.lower()}, which can mean "
                    "short-term moves are stretched")
    if support is not None:
        bits.append(f"nearest swing support sits near {support:,.2f}")
    if resistance is not None:
        bits.append(f"nearest swing resistance near {resistance:,.2f}")
    if volatility == "Elevated" and atr_pct is not None:
        bits.append(f"daily range is elevated (ATR ≈ {atr_pct:.1f}% of price)")
    narrative = (". ".join(b[0].upper() + b[1:] for b in bits) + "."
                 if bits else "Not enough structure in the recent tape to describe.")

    return Snapshot(
        trend=trend, momentum=momentum, rsi=rsi_val, rsi_status=rsi_status,
        ma_alignment=ma_alignment, rel_volume=rel_volume,
        support=support, resistance=resistance,
        volatility=volatility, atr_pct=atr_pct,
        score=score, components=comp, narrative=narrative,
    )
