"""
Order-flow toxicity — VPIN via Bulk-Volume Classification.
=========================================================

VPIN (Volume-Synchronised Probability of Informed Trading; Easley, López de Prado
and O'Hara, 2012) estimates how *toxic* the order flow is — how one-sided, and so
how exposed a liquidity provider is to being adversely selected. A rising VPIN is
the microstructure fingerprint that preceded the 2010 Flash Crash in their work.

A crucial and often-missed point: VPIN was designed to run on **volume bars**, not
the Level-2 order book. Its Bulk-Volume Classification step infers the buy/sell
split of each volume bucket from the *price change* across it, via a normal CDF —
precisely so it can be computed without tick-level trade signing or an order book
feed. That is what makes it honest to build here, on OHLCV bars, when true L2
order-flow imbalance (OFI) cannot be — OFI needs the book, which this app has no
feed for. VPIN-BVC is the OHLCV-native cousin, computed exactly as its authors
intended, and labelled as the approximation it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    # Vectorised standard-normal CDF via erf; no scipy dependency.
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class VPINResult:
    value: float                  # latest VPIN, 0-1
    series: pd.Series             # VPIN over time (index = bucket end position)
    percentile: float             # where the latest value sits in its own history
    n_buckets: int

    @property
    def label(self) -> str:
        if self.percentile >= 90:
            return "Toxic — flow is extremely one-sided versus its own history"
        if self.percentile >= 70:
            return "Elevated — order flow is more one-sided than usual"
        if self.percentile <= 20:
            return "Balanced — two-sided flow, low adverse-selection pressure"
        return "Normal — flow imbalance around its typical level"

    @property
    def is_toxic(self) -> bool:
        return self.percentile >= 90


def vpin_bvc(df: pd.DataFrame, n_buckets: int = 250, window: int = 50) -> Optional[VPINResult]:
    """
    VPIN from OHLCV bars via Bulk-Volume Classification.

    Total traded volume is sliced into ``n_buckets`` equal-*volume* buckets
    (a volume clock, not a wall clock — the whole point of "volume-synchronised"),
    so ``n_buckets`` must comfortably exceed ``window`` for the rolling mean to
    have anything to roll over. Within each bucket the
    buy share is ``Φ(ΔP / σ_ΔP)`` and the sell share its complement, so a bucket
    that ran up is mostly classified as buys. VPIN is the rolling mean over
    ``window`` buckets of ``|V_buy − V_sell| / V`` — the average absolute order
    imbalance. Returns ``None`` when there is too little history to fill the
    buckets and the window.
    """
    if df is None or df.empty or not {"Close", "Volume"} <= set(df.columns):
        return None
    d = df[["Close", "Volume"]].dropna()
    if len(d) < max(n_buckets * 2, window + 5):
        return None

    close = d["Close"].to_numpy(dtype=float)
    vol = d["Volume"].to_numpy(dtype=float)
    dP = np.diff(close, prepend=close[0])
    sigma = float(np.std(dP[1:], ddof=1))
    if sigma <= 0:
        return None

    total_vol = float(vol.sum())
    if total_vol <= 0:
        return None
    bucket_size = total_vol / n_buckets

    # Walk the bars, emitting a bucket each time accumulated volume crosses the
    # bucket size. Buy/sell volume within a bucket is BVC-weighted by that bar's
    # standardized price move.
    buys = sells = filled = 0.0
    imbalances: list[float] = []
    for i in range(1, len(close)):
        b_frac = float(_norm_cdf(np.array([dP[i] / sigma]))[0])
        v = vol[i]
        remaining = v
        while remaining > 0:
            room = bucket_size - filled
            take = min(remaining, room)
            buys += take * b_frac
            sells += take * (1.0 - b_frac)
            filled += take
            remaining -= take
            if filled >= bucket_size - 1e-9:
                imbalances.append(abs(buys - sells) / bucket_size)
                buys = sells = filled = 0.0

    if len(imbalances) < window + 1:
        return None

    imb = pd.Series(imbalances)
    vpin = imb.rolling(window).mean().dropna()
    if vpin.empty:
        return None

    current = float(vpin.iloc[-1])
    pct = float((vpin < current).mean()) * 100.0
    return VPINResult(value=current, series=vpin.reset_index(drop=True),
                      percentile=pct, n_buckets=len(imbalances))
