"""
Intraday technical indicators and signals.
==========================================

The indicator set brokerage platforms surface by default — trend (EMA stack,
MACD, Supertrend, ADX), momentum (RSI, Stochastic), volatility (Bollinger,
ATR) and volume (VWAP, OBV) — each reduced to a Bullish / Bearish / Neutral
call plus the value that produced it.

Every calculation is plain pandas/NumPy: no TA-Lib, no pandas-ta, nothing that
needs a C extension on a slim cloud image.

**On accuracy.** No signal here is asserted to be right any fixed share of the
time. ``svp.analytics.accuracy`` measures each one's realised hit rate on the
actual bars in front of you, and that measured number is what the UI shows.
Published "80% accurate" indicator claims do not survive walk-forward testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

BULLISH, BEARISH, NEUTRAL = "Bullish", "Bearish", "Neutral"


@dataclass
class Signal:
    name: str
    category: str          # Trend | Momentum | Volatility | Volume
    value: Optional[float]
    display: str           # formatted value for the UI
    call: str              # BULLISH | BEARISH | NEUTRAL
    note: str              # why it reads that way

    @property
    def score(self) -> int:
        return {BULLISH: 1, BEARISH: -1}.get(self.call, 0)


@dataclass
class IndicatorSet:
    df: pd.DataFrame                       # original bars + indicator columns
    signals: list[Signal] = field(default_factory=list)

    @property
    def bullish(self) -> int:
        return sum(1 for s in self.signals if s.call == BULLISH)

    @property
    def bearish(self) -> int:
        return sum(1 for s in self.signals if s.call == BEARISH)

    @property
    def neutral(self) -> int:
        return sum(1 for s in self.signals if s.call == NEUTRAL)

    @property
    def net_score(self) -> float:
        """Mean signal score in [-1, 1]."""
        return float(np.mean([s.score for s in self.signals])) if self.signals else 0.0

    @property
    def consensus(self) -> str:
        n = self.net_score
        if n >= 0.5:
            return "Strong Buy"
        if n >= 0.15:
            return "Buy"
        if n <= -0.5:
            return "Strong Sell"
        if n <= -0.15:
            return "Sell"
        return "Neutral"

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Indicator": s.name, "Category": s.category,
            "Value": s.display, "Signal": s.call, "Reading": s.note,
        } for s in self.signals])


# ── primitives ────────────────────────────────────────────────────────────────
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(close: pd.Series, period=20, mult=2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return mid + mult * sd, mid, mid - mult * sd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift()
    return pd.concat([df["High"] - df["Low"],
                      (df["High"] - prev).abs(),
                      (df["Low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean(), plus_di, minus_di


def stochastic(df: pd.DataFrame, k=14, d=3):
    low = df["Low"].rolling(k).min()
    high = df["High"].rolling(k).max()
    k_line = 100 * (df["Close"] - low) / (high - low).replace(0, np.nan)
    return k_line, k_line.rolling(d).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP — resets each trading day, as brokers display it."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = df.index.normalize() if isinstance(df.index, pd.DatetimeIndex) else pd.Series(0, index=df.index)
    pv = (tp * df["Volume"]).groupby(day).cumsum()
    vv = df["Volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vv


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff().fillna(0)) * df["Volume"]).cumsum()


def supertrend(df: pd.DataFrame, period=10, mult=3.0):
    """Supertrend line and its direction (+1 up-trend, -1 down-trend)."""
    a = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2
    upper, lower = hl2 + mult * a, hl2 - mult * a
    close = df["Close"].to_numpy()
    # copy=True: to_numpy() can hand back a read-only view, and the bands are
    # ratcheted in place below.
    up, lo = upper.to_numpy(copy=True), lower.to_numpy(copy=True)
    line = np.full(len(df), np.nan)
    direction = np.ones(len(df))

    for i in range(1, len(df)):
        # Bands ratchet: they only tighten while the trend holds.
        up[i] = min(up[i], up[i - 1]) if close[i - 1] <= up[i - 1] else up[i]
        lo[i] = max(lo[i], lo[i - 1]) if close[i - 1] >= lo[i - 1] else lo[i]
        if close[i] > up[i - 1]:
            direction[i] = 1
        elif close[i] < lo[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        line[i] = lo[i] if direction[i] > 0 else up[i]

    return pd.Series(line, index=df.index), pd.Series(direction, index=df.index)


# ── the full set ──────────────────────────────────────────────────────────────
def compute(df: pd.DataFrame) -> IndicatorSet:
    """Compute every indicator and reduce each to a directional call."""
    if df is None or df.empty or len(df) < 30:
        return IndicatorSet(df if df is not None else pd.DataFrame())

    d = df.copy()
    close = d["Close"]
    last = float(close.iloc[-1])

    d["EMA9"], d["EMA20"], d["EMA50"] = ema(close, 9), ema(close, 20), ema(close, 50)
    d["SMA200"] = close.rolling(min(200, max(len(d) // 2, 20))).mean()
    d["RSI"] = rsi(close)
    d["MACD"], d["MACD_SIG"], d["MACD_HIST"] = macd(close)
    d["BB_UP"], d["BB_MID"], d["BB_LOW"] = bollinger(close)
    d["ATR"] = atr(d)
    d["ADX"], d["DI+"], d["DI-"] = adx(d)
    d["STOCH_K"], d["STOCH_D"] = stochastic(d)
    d["VWAP"] = vwap(d)
    d["OBV"] = obv(d)
    d["ST"], d["ST_DIR"] = supertrend(d)

    def v(col):
        x = d[col].iloc[-1]
        return None if pd.isna(x) else float(x)

    sigs: list[Signal] = []

    # ── Trend ────────────────────────────────────────────────────────────────
    e9, e20, e50 = v("EMA9"), v("EMA20"), v("EMA50")
    if None not in (e9, e20, e50):
        if e9 > e20 > e50:
            call, note = BULLISH, "EMA 9 > 20 > 50 — stacked bullish"
        elif e9 < e20 < e50:
            call, note = BEARISH, "EMA 9 < 20 < 50 — stacked bearish"
        else:
            call, note = NEUTRAL, "EMAs interleaved — no clean trend"
        sigs.append(Signal("EMA Stack (9/20/50)", "Trend", e20, f"{e20:,.2f}", call, note))

    s200 = v("SMA200")
    if s200:
        call = BULLISH if last > s200 else BEARISH
        sigs.append(Signal("Price vs SMA 200", "Trend", s200, f"{s200:,.2f}", call,
                           f"Price {'above' if last > s200 else 'below'} the long trend line"))

    m, ms, mh = v("MACD"), v("MACD_SIG"), v("MACD_HIST")
    if None not in (m, ms, mh):
        if m > ms and mh > 0:
            call, note = BULLISH, "MACD above signal, histogram positive"
        elif m < ms and mh < 0:
            call, note = BEARISH, "MACD below signal, histogram negative"
        else:
            call, note = NEUTRAL, "MACD crossing — direction unresolved"
        sigs.append(Signal("MACD (12/26/9)", "Trend", m, f"{m:,.3f}", call, note))

    st_dir = v("ST_DIR")
    if st_dir is not None:
        call = BULLISH if st_dir > 0 else BEARISH
        stv = v("ST")
        sigs.append(Signal("Supertrend (10, 3×ATR)", "Trend", stv,
                           f"{stv:,.2f}" if stv else "—", call,
                           f"Trend flag {'up' if st_dir > 0 else 'down'}"))

    a, dip, dim = v("ADX"), v("DI+"), v("DI-")
    if None not in (a, dip, dim):
        if a < 20:
            call, note = NEUTRAL, f"ADX {a:.0f} — trend too weak to trade directionally"
        else:
            call = BULLISH if dip > dim else BEARISH
            note = f"ADX {a:.0f} — {'+DI' if dip > dim else '−DI'} dominant"
        sigs.append(Signal("ADX (14)", "Trend", a, f"{a:,.1f}", call, note))

    # ── Momentum ─────────────────────────────────────────────────────────────
    r = v("RSI")
    if r is not None:
        if r < 30:
            call, note = BULLISH, f"RSI {r:.0f} — oversold"
        elif r > 70:
            call, note = BEARISH, f"RSI {r:.0f} — overbought"
        else:
            call = BULLISH if r > 50 else BEARISH
            note = f"RSI {r:.0f} — {'above' if r > 50 else 'below'} midline"
        sigs.append(Signal("RSI (14)", "Momentum", r, f"{r:,.1f}", call, note))

    k, dd = v("STOCH_K"), v("STOCH_D")
    if None not in (k, dd):
        if k < 20:
            call, note = BULLISH, f"%K {k:.0f} — oversold"
        elif k > 80:
            call, note = BEARISH, f"%K {k:.0f} — overbought"
        else:
            call = BULLISH if k > dd else BEARISH
            note = f"%K {'above' if k > dd else 'below'} %D"
        sigs.append(Signal("Stochastic (14, 3)", "Momentum", k, f"{k:,.1f}", call, note))

    # ── Volatility ───────────────────────────────────────────────────────────
    bu, bm, bl = v("BB_UP"), v("BB_MID"), v("BB_LOW")
    if None not in (bu, bm, bl) and bu > bl:
        pct = (last - bl) / (bu - bl)
        if pct > 1:
            call, note = BEARISH, "Closed above the upper band — stretched"
        elif pct < 0:
            call, note = BULLISH, "Closed below the lower band — stretched"
        else:
            call = BULLISH if last > bm else BEARISH
            note = f"{pct * 100:.0f}% of the way up the band"
        sigs.append(Signal("Bollinger (20, 2σ)", "Volatility", pct, f"{pct * 100:,.0f}%", call, note))

    av = v("ATR")
    if av:
        sigs.append(Signal("ATR (14)", "Volatility", av, f"{av:,.2f}", NEUTRAL,
                           f"≈{av / last * 100:.2f}% of price per bar — sizing input, not directional"))

    # ── Volume ───────────────────────────────────────────────────────────────
    vw = v("VWAP")
    if vw:
        call = BULLISH if last > vw else BEARISH
        sigs.append(Signal("VWAP (session)", "Volume", vw, f"{vw:,.2f}", call,
                           f"Price {'above' if last > vw else 'below'} the volume-weighted average"))

    o = d["OBV"]
    if len(o.dropna()) > 20:
        slope = float(o.iloc[-1] - o.iloc[-20])
        call = BULLISH if slope > 0 else (BEARISH if slope < 0 else NEUTRAL)
        sigs.append(Signal("OBV (20-bar)", "Volume", slope, f"{slope:,.0f}", call,
                           f"On-balance volume {'rising' if slope > 0 else 'falling'}"))

    return IndicatorSet(d, sigs)
