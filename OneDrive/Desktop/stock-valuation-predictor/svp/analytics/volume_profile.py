"""
Fixed-range volume profile — where the volume actually traded.
==============================================================

A price chart shows *when* volume happened; a volume profile shows *at what
price*. Over a chosen range it bins traded volume by price level and reports
the three levels desks actually quote:

- **POC** (point of control) — the price with the most volume traded. The
  range's centre of gravity, and the level price tends to revisit.
- **Value area** (VAH / VAL) — the band around the POC holding 70% of the
  range's volume. Price inside the value area is "accepted"; price outside
  it is being auctioned somewhere the market has not yet agreed on.
- **Low-volume nodes** — thin shelves the market moved through quickly.

Two honesty points separate this from the crude version it replaces (which
binned each bar's whole volume at its close, in
:func:`svp.analytics.technical._volume_poc`):

1. **Volume is spread across the bar's high-low span**, weighted by how much
   of each price bin the bar overlaps. A bar that ranged ten dollars did not
   trade all of its volume at the close, and pretending otherwise puts the
   POC wherever closes happened to cluster.
2. **Intraday bars are used when available.** The profile of a range is only
   as honest as the resolution underneath it; the caller passes whatever it
   has and the result reports its own ``bar_count`` so the reader can judge.

Nothing here forecasts. A POC is a description of where trade happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

MIN_BARS = 10
DEFAULT_BINS = 48
VALUE_AREA = 0.70          # the conventional 70% of volume


@dataclass
class VolumeProfile:
    poc: float
    vah: float
    val: float
    bin_prices: list[float] = field(default_factory=list)   # bin midpoints
    bin_volumes: list[float] = field(default_factory=list)
    total_volume: float = 0.0
    bar_count: int = 0
    low: float = 0.0                                        # range extremes
    high: float = 0.0
    lvn: list[float] = field(default_factory=list)          # low-volume nodes

    def position_of(self, price: float) -> str:
        """Where a price sits relative to the value area."""
        if price > self.vah:
            return "above value"
        if price < self.val:
            return "below value"
        return "inside value"

    def reading(self, price: float) -> str:
        """One measured sentence. Describes acceptance, never predicts."""
        pos = self.position_of(price)
        base = (f"Most volume in this range traded at {self.poc:,.2f}, with "
                f"70% of it between {self.val:,.2f} and {self.vah:,.2f}.")
        if pos == "inside value":
            return (base + " Price is inside that value area — the market is "
                    "trading where it has already agreed on price.")
        if pos == "above value":
            return (base + " Price is above the value area: it is being "
                    "auctioned higher than where this range's volume "
                    "accepted it. Acceptance would mean volume building at "
                    "the new level; rejection would mean a return toward "
                    "the POC.")
        return (base + " Price is below the value area: it is being "
                "auctioned lower than where this range's volume accepted it. "
                "Acceptance would mean volume building at the new level; "
                "rejection would mean a return toward the POC.")


def _spread_bar(low: float, high: float, volume: float,
                edges: np.ndarray, out: np.ndarray) -> None:
    """
    Add one bar's volume to the bins its high-low span overlaps.

    Weighted by overlap, so a bar spanning three bins contributes to each in
    proportion to how much of the bar sits in it. A zero-range bar (high ==
    low) drops all its volume in the single bin containing that price.
    """
    if volume <= 0:
        return
    if high <= low:
        idx = int(np.clip(np.searchsorted(edges, low, side="right") - 1,
                          0, len(out) - 1))
        out[idx] += volume
        return
    # Overlap of [low, high] with each bin [edges[i], edges[i+1]].
    lo = np.maximum(edges[:-1], low)
    hi = np.minimum(edges[1:], high)
    overlap = np.clip(hi - lo, 0.0, None)
    total = overlap.sum()
    if total <= 0:
        idx = int(np.clip(np.searchsorted(edges, low, side="right") - 1,
                          0, len(out) - 1))
        out[idx] += volume
        return
    out += volume * (overlap / total)


def _value_area(prices: np.ndarray, volumes: np.ndarray,
                poc_idx: int, fraction: float = VALUE_AREA):
    """
    Expand from the POC, always taking the heavier neighbour, until the
    captured volume reaches ``fraction`` of the total. This is the standard
    market-profile construction.
    """
    total = volumes.sum()
    if total <= 0:
        return poc_idx, poc_idx
    lo = hi = poc_idx
    captured = volumes[poc_idx]
    target = total * fraction
    while captured < target and (lo > 0 or hi < len(volumes) - 1):
        below = volumes[lo - 1] if lo > 0 else -1.0
        above = volumes[hi + 1] if hi < len(volumes) - 1 else -1.0
        if above >= below:
            hi += 1
            captured += volumes[hi]
        else:
            lo -= 1
            captured += volumes[lo]
    return lo, hi


def profile(df: pd.DataFrame, bins: int = DEFAULT_BINS,
            lvn_quantile: float = 0.15) -> Optional[VolumeProfile]:
    """
    Build the volume profile of ``df`` (already sliced to the fixed range).

    Returns ``None`` when the frame is too short, lacks volume, or the range
    has no traded volume at all — never an invented profile.
    """
    if df is None or df.empty or len(df) < MIN_BARS:
        return None
    if "Volume" not in df.columns or "Close" not in df.columns:
        return None

    high_s = df["High"] if "High" in df.columns else df["Close"]
    low_s = df["Low"] if "Low" in df.columns else df["Close"]
    frame = pd.DataFrame({
        "high": pd.to_numeric(high_s, errors="coerce"),
        "low": pd.to_numeric(low_s, errors="coerce"),
        "vol": pd.to_numeric(df["Volume"], errors="coerce"),
    }).dropna()
    frame = frame[frame["vol"] > 0]
    if len(frame) < MIN_BARS:
        return None

    lo_px = float(frame["low"].min())
    hi_px = float(frame["high"].max())
    if not np.isfinite(lo_px) or not np.isfinite(hi_px) or hi_px <= lo_px:
        return None

    bins = max(4, int(bins))
    edges = np.linspace(lo_px, hi_px, bins + 1)
    volumes = np.zeros(bins, dtype=float)
    for low, high, vol in frame[["low", "high", "vol"]].itertuples(index=False):
        _spread_bar(float(low), float(high), float(vol), edges, volumes)

    total = float(volumes.sum())
    if total <= 0:
        return None

    mids = (edges[:-1] + edges[1:]) / 2.0
    poc_idx = int(volumes.argmax())
    lo_i, hi_i = _value_area(mids, volumes, poc_idx)

    # Low-volume nodes: thin shelves price moved through, reported only when
    # the range has enough structure for "thin" to mean anything.
    thin = float(np.quantile(volumes, lvn_quantile)) if bins >= 12 else -1.0
    lvn = [float(m) for m, v in zip(mids, volumes) if 0 < v <= thin]

    return VolumeProfile(
        poc=float(mids[poc_idx]),
        vah=float(mids[hi_i]),
        val=float(mids[lo_i]),
        bin_prices=[float(m) for m in mids],
        bin_volumes=[float(v) for v in volumes],
        total_volume=total,
        bar_count=int(len(frame)),
        low=lo_px,
        high=hi_px,
        lvn=lvn,
    )
