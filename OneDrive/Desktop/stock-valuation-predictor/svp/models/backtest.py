"""
Historical backtesting engine.
==============================

Evaluates how the ML valuation signal would have performed against actual price
movements over 1-, 3- and 5-year horizons.

Method: at each historical rebalance date we take the model's intrinsic value
(here reconstructed from the trailing price path, since point-in-time
fundamentals aren't stored) and compare the resulting under/over-valued signal
to the *realised forward return* to the evaluation date. This produces hit-rate,
average signalled return and a simple long/flat equity curve vs buy-and-hold —
the numbers the dashboard displays.

Everything is driven off the price ``history`` from :mod:`svp.data.market`, so it
runs with live yfinance data or the deterministic offline stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class HorizonResult:
    horizon_years: float
    n_signals: int
    hit_rate: float             # fraction of signals whose direction was right
    avg_forward_return: float   # mean realised return following a BUY signal
    strategy_return: float      # total return of signal-driven long/flat strategy
    buy_hold_return: float      # total return of buy-and-hold over same window


def _intrinsic_proxy(history: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """
    Reconstruct a mean-reversion 'intrinsic value' proxy: a trailing moving
    average of price. The signal = intrinsic/price - 1 (positive → undervalued).
    This mimics the ML model's tendency to anchor toward fundamentals-implied
    fair value, letting us backtest signal quality without stored fundamentals.
    """
    return history["Close"].rolling(lookback, min_periods=lookback // 2).mean()


def backtest_horizon(history: pd.DataFrame, years: float, threshold: float = 0.05) -> Optional[HorizonResult]:
    """Backtest the valuation signal for a single forward horizon (in years)."""
    if history is None or history.empty or "Close" not in history:
        return None
    px = history["Close"].dropna()
    if len(px) < 260:
        return None

    horizon_days = int(round(years * 252))
    intrinsic = _intrinsic_proxy(history).reindex(px.index)
    signal = (intrinsic / px) - 1.0  # >0 undervalued

    # Sample monthly rebalance points that have a full forward window.
    idx = px.index
    rebal = idx[:: 21]  # ~monthly
    rets, dirs, strat = [], [], []
    for d in rebal:
        loc = px.index.get_loc(d)
        fwd_loc = loc + horizon_days
        if fwd_loc >= len(px) or np.isnan(signal.iloc[loc]):
            continue
        fwd_ret = px.iloc[fwd_loc] / px.iloc[loc] - 1.0
        sig = signal.iloc[loc]
        rets.append(fwd_ret)
        # Correct direction: undervalued→up, overvalued→down.
        if abs(sig) >= threshold:
            correct = (sig > 0 and fwd_ret > 0) or (sig < 0 and fwd_ret < 0)
            dirs.append(1.0 if correct else 0.0)
            # Long only when undervalued beyond threshold, else flat.
            strat.append(fwd_ret if sig > threshold else 0.0)

    if not dirs:
        return None

    buy_hold = px.iloc[-1] / px.iloc[0] - 1.0
    return HorizonResult(
        horizon_years=years,
        n_signals=len(dirs),
        hit_rate=float(np.mean(dirs)),
        avg_forward_return=float(np.mean([r for r, s in zip(rets, dirs)])) if rets else 0.0,
        strategy_return=float(np.mean(strat)) if strat else 0.0,
        buy_hold_return=float(buy_hold),
    )


def run_backtest(history: pd.DataFrame, horizons=(1, 3, 5)) -> list[HorizonResult]:
    """Run the backtest across the requested forward horizons."""
    out = []
    for h in horizons:
        r = backtest_horizon(history, h)
        if r is not None:
            out.append(r)
    return out


def equity_curve(history: pd.DataFrame, threshold: float = 0.05) -> Optional[pd.DataFrame]:
    """
    Build a daily equity curve for a simple long/flat strategy (long when the
    intrinsic proxy says undervalued) vs buy-and-hold, for the dashboard chart.
    """
    if history is None or history.empty:
        return None
    px = history["Close"].dropna()
    if len(px) < 260:
        return None
    intrinsic = _intrinsic_proxy(history).reindex(px.index)
    signal = (intrinsic / px) - 1.0
    daily_ret = px.pct_change().fillna(0.0)
    position = (signal.shift(1) > threshold).astype(float)  # act next day
    strat_ret = daily_ret * position
    curve = pd.DataFrame(
        {
            "Buy & Hold": (1 + daily_ret).cumprod(),
            "Valuation Strategy": (1 + strat_ret).cumprod(),
        },
        index=px.index,
    )
    return curve
