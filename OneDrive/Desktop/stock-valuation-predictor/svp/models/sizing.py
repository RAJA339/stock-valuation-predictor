"""
Risk-adjusted position sizing & execution parameters.
=====================================================

Turns a binary buy/sell flag into an actual portfolio allocation and risk floors:

  * **Fractional Kelly** — bet fraction from the Monte-Carlo win probability and
    payoff ratio, scaled by a conservative Kelly fraction (default ¼-Kelly).
  * **Volatility parity** — alternative weight so each position contributes a
    target risk budget, using realised volatility.
  * **Dynamic risk floors** — ATR-based stop-loss and a support floor at the
    model's p10 intrinsic value; take-profit anchored on p90 / intrinsic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SizingResult:
    kelly_fraction: float          # raw full-Kelly fraction
    fractional_kelly: float        # scaled (e.g. 0.25x) — the recommended weight
    vol_parity_weight: Optional[float]
    win_probability: float
    payoff_ratio: float
    stop_loss: float
    take_profit: float
    risk_per_share: float
    reward_risk_ratio: Optional[float]


def kelly_from_mc(mc_samples: np.ndarray, price: float) -> tuple[float, float, float]:
    """
    Derive (win_prob, payoff_ratio, full_kelly) from the Monte-Carlo intrinsic
    distribution relative to the current price.
    """
    samples = np.asarray(mc_samples, dtype=float)
    rel = samples / price - 1.0            # implied return per scenario
    wins = rel[rel > 0]
    losses = rel[rel < 0]
    p = len(wins) / len(rel) if len(rel) else 0.0
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else np.nan
    if not avg_loss or np.isnan(avg_loss) or avg_loss == 0:
        b = 3.0  # cap payoff when there are ~no losing scenarios
    else:
        b = avg_win / avg_loss
    # Kelly: f* = p - (1 - p) / b
    kelly = p - (1 - p) / b if b > 0 else 0.0
    return float(p), float(b), float(max(kelly, 0.0))


def volatility_parity_weight(annual_vol: Optional[float], target_vol: float = 0.15) -> Optional[float]:
    """Weight so the position contributes ``target_vol`` of annualised risk."""
    if not annual_vol or annual_vol <= 0:
        return None
    return float(min(target_vol / annual_vol, 1.0))


def compute_sizing(
    price: float,
    mc_samples: np.ndarray,
    intrinsic_low: float,
    intrinsic_high: float,
    atr: Optional[float],
    annual_vol: Optional[float] = None,
    kelly_scale: float = 0.25,
    atr_stop_mult: float = 2.0,
    max_weight: float = 0.20,
) -> SizingResult:
    """
    Produce the full sizing + risk-floor package.

    ``atr`` and ``annual_vol`` come from the technical layer / price history;
    ``intrinsic_low``/``high`` are the model p10/p90.
    """
    p, b, kelly = kelly_from_mc(mc_samples, price)
    frac_kelly = float(np.clip(kelly * kelly_scale, 0.0, max_weight))
    vpw = volatility_parity_weight(annual_vol)

    # ── Risk floors ──────────────────────────────────────────────────────────
    atr_stop = price - atr_stop_mult * atr if atr else None
    # Stop = the higher (tighter) of ATR stop and p10 support, but below price.
    candidates = [x for x in (atr_stop, intrinsic_low) if x is not None and x < price]
    stop_loss = max(candidates) if candidates else price * 0.9
    take_profit = max(intrinsic_high, price * 1.05)

    risk_per_share = max(price - stop_loss, 1e-9)
    reward = take_profit - price
    rr = float(reward / risk_per_share) if risk_per_share > 0 else None

    return SizingResult(
        kelly_fraction=kelly,
        fractional_kelly=frac_kelly,
        vol_parity_weight=vpw,
        win_probability=p,
        payoff_ratio=b,
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        risk_per_share=float(risk_per_share),
        reward_risk_ratio=rr,
    )


def annualized_vol(daily_close) -> Optional[float]:
    """Annualised volatility from a daily close series."""
    import pandas as pd

    if daily_close is None or len(daily_close) < 20:
        return None
    ret = pd.Series(daily_close).pct_change().dropna()
    if ret.empty:
        return None
    return float(ret.std() * np.sqrt(252))
