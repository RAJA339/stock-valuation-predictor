"""
Five-state regime classifier — Hurst (R/S) crossed with ADX and DI.
===================================================================

Two questions, deliberately kept apart:

- **Is this market trending at all?** Answered by the Hurst exponent, which
  measures persistence: does a move tend to be followed by another in the
  same direction (H > 0.5), or given back (H < 0.5)? At H ≈ 0.5 the series
  is a random walk and no trend-following method has anything to work with.

- **Which way, and how strongly?** Answered by ADX with the directional
  indicators: ADX sizes the trend, the +DI/−DI spread signs it.

Neither alone is enough. ADX is high in a strong trend *and* in a violent
chop; Hurst says a series is persistent without saying up or down. Crossing
them gives the five states this module reports.

**Hurst by rescaled range.** ``hurst_rs`` implements the classical R/S
estimator of Hurst (1951): split the series into windows, and in each measure
the range of the cumulative deviation from the mean divided by the standard
deviation. That statistic grows like n^H, so H is the slope of log(R/S)
against log(n). :mod:`svp.analytics.quant` already estimates H a second way,
by the growth of differences; the two are independent estimators and this
module reports both, because agreement between them is worth more than
either alone and disagreement is a warning the estimate is unstable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

MIN_BARS = 120
ADX_STRONG = 25.0
ADX_WEAK = 20.0
TREND_H = 0.55
REVERT_H = 0.45

State = Literal["Strong Bull Trend", "Weak Bull", "Sideways Consolidation",
                "Weak Bear", "Strong Bear Trend"]


@dataclass
class HurstRS:
    """Rescaled-range estimate with the fit behind it."""
    exponent: float
    r_squared: float
    n_windows: int
    reading: str

    @property
    def is_trending(self) -> bool:
        return self.exponent > TREND_H

    @property
    def is_mean_reverting(self) -> bool:
        return self.exponent < REVERT_H

    @property
    def is_reliable(self) -> bool:
        """A poor log-log fit means the exponent is not measuring much."""
        return self.r_squared >= 0.85


def hurst_rs(series, min_window: int = 8,
             max_window: Optional[int] = None) -> Optional[HurstRS]:
    """
    Classical rescaled-range (R/S) Hurst estimate.

    Returns ``None`` when the series is too short for several window sizes,
    since a slope through two points is not an estimate.
    """
    s = pd.Series(series).dropna().astype(float)
    if len(s) < 64:
        return None
    x = np.log(s.to_numpy())
    rets = np.diff(x)
    rets = rets[np.isfinite(rets)]
    n = len(rets)
    if n < 64:
        return None

    max_window = int(max_window or n // 4)
    if max_window <= min_window:
        return None

    sizes, rs_values = [], []
    w = int(min_window)
    while w <= max_window:
        chunks = n // w
        if chunks < 1:
            break
        vals = []
        for j in range(chunks):
            seg = rets[j * w:(j + 1) * w]
            if seg.size < 2:
                continue
            dev = np.cumsum(seg - seg.mean())
            r = dev.max() - dev.min()
            sd = seg.std(ddof=1)
            if sd > 0 and np.isfinite(r):
                vals.append(r / sd)
        if vals:
            sizes.append(w)
            rs_values.append(float(np.mean(vals)))
        w = int(np.ceil(w * 1.5))

    if len(sizes) < 4:
        return None

    lx = np.log(np.asarray(sizes, dtype=float))
    ly = np.log(np.asarray(rs_values, dtype=float))
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    h = float(np.clip(slope, 0.0, 1.0))

    if h > TREND_H:
        note = ("persistent — moves have tended to be followed by further "
                "moves the same way")
    elif h < REVERT_H:
        note = ("mean-reverting — moves have tended to be given back rather "
                "than extended")
    else:
        note = ("close to a random walk, where trend-following and "
                "mean-reversion both have little to work with")
    return HurstRS(exponent=h, r_squared=float(r2), n_windows=len(sizes),
                   reading=f"H ≈ {h:.3f} by rescaled range: {note}.")


@dataclass
class RegimeState:
    state: State
    hurst_rs: Optional[float]
    hurst_r2: Optional[float]
    hurst_diff: Optional[float]      # the second, independent estimator
    adx: Optional[float]
    plus_di: Optional[float]
    minus_di: Optional[float]
    di_spread: Optional[float]
    estimators_agree: bool
    narrative: str

    @property
    def is_trending(self) -> bool:
        return self.state in ("Strong Bull Trend", "Strong Bear Trend")

    @property
    def direction(self) -> str:
        if "Bull" in self.state:
            return "bullish"
        if "Bear" in self.state:
            return "bearish"
        return "neutral"

    @property
    def badge_class(self) -> str:
        return {"Strong Bull Trend": "signal-buy", "Weak Bull": "signal-hold",
                "Sideways Consolidation": "signal-hold",
                "Weak Bear": "signal-hold",
                "Strong Bear Trend": "signal-sell"}[self.state]


def classify(df: pd.DataFrame, adx_period: int = 14) -> Optional[RegimeState]:
    """
    Classify the current tape into one of five explicit states.

    The rule, in order: a weak or absent ADX means consolidation regardless of
    persistence, because direction without strength is noise. Given strength,
    the DI spread signs it, and Hurst decides whether the trend is called
    *strong* — a high ADX on a mean-reverting series is a violent chop, not a
    trend, and that distinction is the reason both are used.
    """
    if df is None or df.empty or len(df) < MIN_BARS:
        return None
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return None
    data = df.dropna(subset=["High", "Low", "Close"]).sort_index()
    if len(data) < MIN_BARS:
        return None

    from . import indicators as ind
    from . import quant as quant_mod

    try:
        adx_s, plus_s, minus_s = ind.adx(data, period=adx_period)
        adx_v = float(adx_s.iloc[-1])
        plus_v = float(plus_s.iloc[-1])
        minus_v = float(minus_s.iloc[-1])
    except Exception:
        adx_v = plus_v = minus_v = float("nan")

    rs = hurst_rs(data["Close"])
    h_rs = rs.exponent if rs else None
    try:
        h_diff = quant_mod.hurst_exponent(data["Close"]).exponent
    except Exception:
        h_diff = None

    agree = (h_rs is not None and h_diff is not None
             and abs(h_rs - h_diff) <= 0.15)

    spread = (plus_v - minus_v) if np.isfinite(plus_v) and np.isfinite(minus_v) \
        else float("nan")
    persistent = h_rs is not None and h_rs > TREND_H
    reverting = h_rs is not None and h_rs < REVERT_H

    if not np.isfinite(adx_v) or adx_v < ADX_WEAK or reverting:
        state: State = "Sideways Consolidation"
    elif spread > 0:
        state = ("Strong Bull Trend" if (adx_v >= ADX_STRONG and persistent)
                 else "Weak Bull")
    elif spread < 0:
        state = ("Strong Bear Trend" if (adx_v >= ADX_STRONG and persistent)
                 else "Weak Bear")
    else:
        state = "Sideways Consolidation"

    bits = [f"**{state}**."]
    if np.isfinite(adx_v):
        strength = ("strong" if adx_v >= ADX_STRONG
                    else "weak" if adx_v >= ADX_WEAK else "absent")
        bits.append(f"ADX({adx_period}) reads {adx_v:.1f} — trend strength "
                    f"{strength}; +DI {plus_v:.1f} against −DI {minus_v:.1f}.")
    if rs is not None:
        bits.append(rs.reading)
        if not rs.is_reliable:
            bits.append(f"The log-log fit behind that exponent is weak "
                        f"(R² {rs.r_squared:.2f}), so treat it as indicative.")
    if h_rs is not None and h_diff is not None:
        bits.append(
            f"Two independent estimators {'agree' if agree else 'disagree'} "
            f"(R/S {h_rs:.2f} versus differences {h_diff:.2f})"
            + ("." if agree else " — disagreement means the persistence "
               "estimate is unstable, which is itself worth knowing."))
    if state == "Sideways Consolidation" and np.isfinite(adx_v) \
            and adx_v >= ADX_STRONG:
        bits.append("Note the combination: ADX is high but the series is "
                    "mean-reverting, which is the signature of violent chop "
                    "rather than of a trend.")

    return RegimeState(
        state=state, hurst_rs=h_rs,
        hurst_r2=(rs.r_squared if rs else None), hurst_diff=h_diff,
        adx=adx_v if np.isfinite(adx_v) else None,
        plus_di=plus_v if np.isfinite(plus_v) else None,
        minus_di=minus_v if np.isfinite(minus_v) else None,
        di_spread=spread if np.isfinite(spread) else None,
        estimators_agree=bool(agree), narrative=" ".join(bits),
    )
