"""
Advanced quantitative analytics for a price series.
===================================================

These are the tools that read a return series and characterise it — volatility,
persistence, regime — without predicting it. That distinction is the whole point:
each function reports a property the data *has*, measured and interval-aware where
it matters, rather than an edge it claims. Built for the crypto tab, where there
is no valuation to anchor to and the series is all there is, but nothing here is
crypto-specific — it is ordinary quant machinery applied honestly.

What is deliberately NOT here: a black-box price forecaster. A model that outputs
"BTC goes up" is only worth anything once its calibration has been measured over
months, exactly as the prediction ledger measures the valuation model. Shipping
one before that measurement exists would be the confident-sounding noise this
project refuses to sell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Crypto trades every day; equities ~252. The annualiser is a parameter so the
# same function serves both, but the crypto caller passes 365.
CRYPTO_YEAR = 365
EQUITY_YEAR = 252


def _log_returns(close: pd.Series) -> np.ndarray:
    c = pd.Series(close).astype(float).dropna()
    if len(c) < 3:
        return np.array([])
    return np.diff(np.log(c.to_numpy()))


# ── Volatility ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VolProfile:
    realized: Optional[float]        # annualised close-to-close vol, fraction
    ewma: Optional[float]            # annualised EWMA vol, fraction
    n: int
    annualiser: int

    @property
    def realized_pct(self) -> Optional[float]:
        return None if self.realized is None else self.realized * 100.0

    @property
    def ewma_pct(self) -> Optional[float]:
        return None if self.ewma is None else self.ewma * 100.0


def volatility(close: pd.Series, annualiser: int = CRYPTO_YEAR,
               ewma_lambda: float = 0.94) -> VolProfile:
    """
    Realized and EWMA annualised volatility from log returns.

    Realized is the plain standard deviation of returns, scaled by √annualiser.
    EWMA (RiskMetrics, λ=0.94) weights recent returns more heavily, so it reacts
    to a volatility spike that the equal-weighted figure would still be diluting.
    Reporting both shows whether volatility is rising or falling *now* versus its
    trailing average, which a single number hides.
    """
    r = _log_returns(close)
    if r.size < 3:
        return VolProfile(None, None, r.size, annualiser)

    scale = math.sqrt(annualiser)
    realized = float(np.std(r, ddof=1)) * scale

    # EWMA variance recursion, seeded on the sample variance.
    var = float(np.var(r, ddof=1))
    for x in r:
        var = ewma_lambda * var + (1.0 - ewma_lambda) * x * x
    ewma = math.sqrt(var) * scale

    return VolProfile(realized, ewma, r.size, annualiser)


@dataclass(frozen=True)
class VolRegime:
    label: str            # "Calm" | "Normal" | "Elevated" | "Stressed"
    current: float        # current short-window vol, annualised fraction
    percentile: float     # where current sits in the trailing distribution, 0-100
    n: int


def vol_regime(close: pd.Series, window: int = 14, lookback: int = 180,
               annualiser: int = CRYPTO_YEAR) -> Optional[VolRegime]:
    """
    Classify the current volatility regime by its own history.

    Regime is relative, not absolute: 60% annualised vol is calm for one coin
    and a shock for another, so the label comes from where the current
    short-window vol sits in the trailing distribution of that same measure —
    a percentile, not a fixed threshold. This is the honest form of a regime
    signal, because it cannot be gamed by a coin simply being more volatile than
    another in general.
    """
    r = _log_returns(close)
    if r.size < window + 30:
        return None

    scale = math.sqrt(annualiser)
    roll = pd.Series(r).rolling(window).std(ddof=1).dropna() * scale
    if roll.empty:
        return None

    hist = roll.iloc[-lookback:] if len(roll) > lookback else roll
    current = float(roll.iloc[-1])
    pct = float((hist < current).mean()) * 100.0

    if pct < 25:
        label = "Calm"
    elif pct < 60:
        label = "Normal"
    elif pct < 85:
        label = "Elevated"
    else:
        label = "Stressed"
    return VolRegime(label, current, pct, int(len(hist)))


# ── Persistence ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HurstResult:
    exponent: Optional[float]
    n: int

    @property
    def reading(self) -> str:
        if self.exponent is None:
            return "Not enough data to estimate persistence"
        h = self.exponent
        if h < 0.45:
            return "Mean-reverting — moves tend to fade (fade breakouts, buy dips)"
        if h > 0.55:
            return "Trending — moves tend to persist (breakouts have follow-through)"
        return "Near a random walk — no persistence either way, momentum signals are weak"

    @property
    def is_trending(self) -> bool:
        return self.exponent is not None and self.exponent > 0.55

    @property
    def is_mean_reverting(self) -> bool:
        return self.exponent is not None and self.exponent < 0.45


def hurst_exponent(close: pd.Series, min_lag: int = 2, max_lag: int = 40) -> HurstResult:
    """
    Estimate the Hurst exponent via the rescaled-lag (variance-of-differences)
    method.

    For a series whose log-price differences over lag τ have standard deviation
    growing like τ^H, H is recovered as the slope of log-std against log-lag.
    H = 0.5 is a random walk; H > 0.5 trends (a move predicts more of the same);
    H < 0.5 mean-reverts. It is a description of the recent past, not a forecast —
    persistence can and does flip — so the UI frames it as "how this has been
    behaving", never "what it will do".
    """
    series = pd.Series(close).astype(float).dropna()
    if len(series) < max_lag * 2:
        return HurstResult(None, len(series))

    x = np.log(series.to_numpy())
    lags = range(min_lag, min(max_lag, len(x) // 2))
    taus = []
    lag_list = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        sd = float(np.std(diff))
        if sd > 0:
            taus.append(sd)
            lag_list.append(lag)
    if len(taus) < 4:
        return HurstResult(None, len(series))

    slope = float(np.polyfit(np.log(lag_list), np.log(taus), 1)[0])
    # Clamp to [0, 1]; numerical noise can push a near-random series slightly out.
    return HurstResult(max(0.0, min(1.0, slope)), len(series))
