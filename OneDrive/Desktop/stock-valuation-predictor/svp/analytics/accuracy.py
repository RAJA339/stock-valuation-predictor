"""
Measured indicator accuracy.
============================

Walk-forward hit rates for each indicator on the bars actually loaded, so the
UI can report what a signal *did*, not what it is advertised to do.

Method, per indicator:

  1. Recompute its directional call at every bar, using only data available at
     that bar (all inputs are causal — EMA/RSI/MACD/ADX etc. look backwards
     only, so there is no look-ahead).
  2. Take the forward return over ``horizon`` bars.
  3. Score a Bullish call correct when the forward return is positive, a
     Bearish call correct when it is negative. Neutral bars are excluded.

``hit_rate`` is therefore directional accuracy on that ticker, interval and
horizon — a sample statistic with a real confidence interval, not a property of
the indicator. A Wilson interval is reported alongside it, because on a few
hundred bars a 55% hit rate and a 45% hit rate are frequently the same thing.

Costs are not modelled. Even a genuinely predictive signal can lose money after
spread and commission, so treat these as evidence about direction only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class IndicatorAccuracy:
    name: str
    n_signals: int
    hit_rate: float          # fraction correct, 0..1
    ci_low: float
    ci_high: float
    avg_forward_return: float
    edge_vs_coin: float      # hit_rate - 0.5

    @property
    def significant(self) -> bool:
        """True only when the confidence interval excludes a coin flip."""
        return self.ci_low > 0.5 or self.ci_high < 0.5

    @property
    def verdict(self) -> str:
        if self.n_signals < 30:
            return "Too few signals"
        if not self.significant:
            return "No better than chance"
        return "Better than chance" if self.hit_rate > 0.5 else "Worse than chance"


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves sensibly at small n, unlike the normal one."""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# Each entry maps a name to a vectorised, causal call series in {-1, 0, +1}.
def _call_series(d: pd.DataFrame) -> dict[str, pd.Series]:
    close = d["Close"]
    out: dict[str, pd.Series] = {}

    stacked_up = (d["EMA9"] > d["EMA20"]) & (d["EMA20"] > d["EMA50"])
    stacked_dn = (d["EMA9"] < d["EMA20"]) & (d["EMA20"] < d["EMA50"])
    out["EMA Stack (9/20/50)"] = stacked_up.astype(int) - stacked_dn.astype(int)

    out["Price vs SMA 200"] = np.sign(close - d["SMA200"])

    macd_up = (d["MACD"] > d["MACD_SIG"]) & (d["MACD_HIST"] > 0)
    macd_dn = (d["MACD"] < d["MACD_SIG"]) & (d["MACD_HIST"] < 0)
    out["MACD (12/26/9)"] = macd_up.astype(int) - macd_dn.astype(int)

    out["Supertrend (10, 3×ATR)"] = d["ST_DIR"]

    trending = d["ADX"] >= 20
    out["ADX (14)"] = np.where(trending, np.sign(d["DI+"] - d["DI-"]), 0)

    r = d["RSI"]
    out["RSI (14)"] = np.where(r < 30, 1, np.where(r > 70, -1, np.sign(r - 50)))

    k, dd = d["STOCH_K"], d["STOCH_D"]
    out["Stochastic (14, 3)"] = np.where(k < 20, 1, np.where(k > 80, -1, np.sign(k - dd)))

    width = (d["BB_UP"] - d["BB_LOW"]).replace(0, np.nan)
    pct = (close - d["BB_LOW"]) / width
    out["Bollinger (20, 2σ)"] = np.where(pct > 1, -1, np.where(pct < 0, 1, np.sign(close - d["BB_MID"])))

    out["VWAP (session)"] = np.sign(close - d["VWAP"])
    out["OBV (20-bar)"] = np.sign(d["OBV"] - d["OBV"].shift(20))

    return {k: pd.Series(v, index=d.index).fillna(0) for k, v in out.items()}


def measure(iset: "ind.IndicatorSet", horizon: int = 6,
            min_signals: int = 30) -> list[IndicatorAccuracy]:
    """
    Score every indicator over ``horizon`` bars forward.

    ``horizon`` is in bars, so its wall-clock meaning follows the interval:
    6 bars is 30 minutes at 5m, and 6 hours at 1h.
    """
    d = iset.df
    if d is None or d.empty or len(d) < horizon + 60:
        return []

    fwd = d["Close"].shift(-horizon) / d["Close"] - 1.0
    results: list[IndicatorAccuracy] = []

    for name, calls in _call_series(d).items():
        mask = (calls != 0) & fwd.notna()
        n = int(mask.sum())
        if n < min_signals:
            results.append(IndicatorAccuracy(name, n, float("nan"), 0.0, 0.0, float("nan"), float("nan")))
            continue

        c, f = calls[mask], fwd[mask]
        hits = int(((c > 0) & (f > 0)).sum() + ((c < 0) & (f < 0)).sum())
        rate = hits / n
        lo, hi = wilson_interval(hits, n)
        # Signed forward return: a bearish call that precedes a fall counts positive.
        signed = float((np.sign(c) * f).mean())
        results.append(IndicatorAccuracy(name, n, rate, lo, hi, signed, rate - 0.5))

    results.sort(key=lambda a: (-(a.hit_rate if a.hit_rate == a.hit_rate else 0), a.name))
    return results


def as_frame(rows: list[IndicatorAccuracy]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Indicator": a.name,
        "Signals": a.n_signals,
        "Hit Rate": "—" if a.hit_rate != a.hit_rate else f"{a.hit_rate * 100:.1f}%",
        "95% CI": "—" if a.hit_rate != a.hit_rate else f"{a.ci_low * 100:.0f}–{a.ci_high * 100:.0f}%",
        "Avg Fwd Return": "—" if a.avg_forward_return != a.avg_forward_return
                          else f"{a.avg_forward_return * 100:+.3f}%",
        "Verdict": a.verdict,
    } for a in rows])


def summary(rows: list[IndicatorAccuracy]) -> dict:
    """Headline numbers for the UI and the report."""
    scored = [a for a in rows if a.hit_rate == a.hit_rate]
    sig = [a for a in scored if a.significant and a.hit_rate > 0.5]
    return {
        "measured": len(scored),
        "better_than_chance": len(sig),
        "best": max(scored, key=lambda a: a.hit_rate).name if scored else None,
        "best_rate": max((a.hit_rate for a in scored), default=float("nan")),
        "mean_rate": float(np.mean([a.hit_rate for a in scored])) if scored else float("nan"),
    }
