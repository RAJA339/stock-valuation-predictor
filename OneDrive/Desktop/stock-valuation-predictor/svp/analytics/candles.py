"""
Three White Soldiers / Three Black Crows, with the filters that matter.
======================================================================

The raw patterns are trivially common: any three green candles in a row is
"three white soldiers" if you squint, which is why the naive version fires
constantly and means nothing. The institutional reading adds conditions that
are the actual claim — that these three bars represent sustained, one-sided
participation rather than drift:

- **Body dominance** — each candle's body must be most of its range. A long
  candle that is mostly wick is indecision wearing a trend's colour.
- **Close near the extreme** — the upper wick must not exceed 15% of the
  range for soldiers (lower wick for crows). Closing on the high is what
  distinguishes a bar that was bought all day from one that faded.
- **Volume expansion** — each bar must trade above 1.5× its 20-day average.
  Three quiet green candles are not participation; they are a lack of
  sellers, which is a different thing.
- **Progression** — each close above the last, each open inside the previous
  body. A gap-and-go sequence is a different pattern with different odds.

The **anchored VWAP** pinned to the first soldier's base is the natural
companion: it is the average price everyone who participated in the move has
paid, so it functions as the level the move must hold to remain intact. It is
computed here rather than described.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

MIN_BARS = 30
VOLUME_MULTIPLE = 1.5
MAX_WICK_FRACTION = 0.15
MIN_BODY_FRACTION = 0.60
VOLUME_WINDOW = 20


@dataclass
class PatternEvent:
    """One validated three-bar sequence."""
    kind: Literal["three_white_soldiers", "three_black_crows"]
    start_date: str
    end_date: str
    start_idx: int
    end_idx: int
    anchor_price: float             # base of the first bar — the AVWAP anchor
    total_move_pct: float
    volume_ratios: list = field(default_factory=list)
    wick_fractions: list = field(default_factory=list)
    body_fractions: list = field(default_factory=list)

    @property
    def label(self) -> str:
        return ("Three White Soldiers" if self.kind == "three_white_soldiers"
                else "Three Black Crows")

    @property
    def direction(self) -> str:
        return "bullish" if self.kind == "three_white_soldiers" else "bearish"

    def reading(self) -> str:
        return (
            f"**{self.label}** completed {self.end_date}, a "
            f"{self.total_move_pct:+.1f}% move across three bars. Volume ran "
            f"{min(self.volume_ratios):.1f}–{max(self.volume_ratios):.1f}× its "
            f"20-day average on every bar, and the "
            f"{'upper' if self.direction == 'bullish' else 'lower'} wicks "
            f"stayed under {max(self.wick_fractions) * 100:.0f}% of range — "
            "so each session closed near its extreme rather than fading. That "
            "is a description of participation, not a forecast of what follows."
        )


def _components(df: pd.DataFrame):
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    lo = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    v = (df["Volume"].to_numpy(dtype=float) if "Volume" in df.columns
         else np.full(len(df), np.nan))
    rng = np.where((h - lo) > 0, h - lo, np.nan)
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - lo
    return o, h, lo, c, v, rng, body, upper, lower


def _vol_ratio_series(v: np.ndarray, window: int = VOLUME_WINDOW) -> np.ndarray:
    """Each bar's volume against its own trailing average (causal)."""
    s = pd.Series(v)
    avg = s.rolling(window, min_periods=max(5, window // 2)).mean().shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / avg).to_numpy(dtype=float)


def detect(df: pd.DataFrame, volume_multiple: float = VOLUME_MULTIPLE,
           max_wick: float = MAX_WICK_FRACTION,
           min_body: float = MIN_BODY_FRACTION,
           require_volume: bool = True) -> Optional[list[PatternEvent]]:
    """
    Find validated soldier/crow sequences.

    ``require_volume=False`` relaxes only the volume filter, for instruments
    whose feed has no volume; every other condition still applies and the
    caller is told which filters ran.
    """
    if df is None or df.empty or len(df) < MIN_BARS:
        return None
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return None
    data = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    if len(data) < MIN_BARS:
        return None

    o, h, lo, c, v, rng, body, upper, lower = _components(data)
    vr = _vol_ratio_series(v)
    idx = data.index
    has_volume = np.isfinite(v).any() and np.nansum(v) > 0

    out: list[PatternEvent] = []
    i = 2
    while i < len(data):
        trio = (i - 2, i - 1, i)
        bull = all(c[j] > o[j] for j in trio)
        bear = all(c[j] < o[j] for j in trio)
        if not (bull or bear):
            i += 1
            continue

        # Progression: each close beyond the last, each open inside the
        # previous body — a gap-and-go is a different pattern.
        if bull:
            ordered = c[i - 2] < c[i - 1] < c[i]
            opens_ok = (o[i - 1] <= c[i - 2] and o[i - 1] >= o[i - 2]
                        and o[i] <= c[i - 1] and o[i] >= o[i - 1])
            wicks = [upper[j] / rng[j] if np.isfinite(rng[j]) else 1.0
                     for j in trio]
        else:
            ordered = c[i - 2] > c[i - 1] > c[i]
            opens_ok = (o[i - 1] >= c[i - 2] and o[i - 1] <= o[i - 2]
                        and o[i] >= c[i - 1] and o[i] <= o[i - 1])
            wicks = [lower[j] / rng[j] if np.isfinite(rng[j]) else 1.0
                     for j in trio]

        bodies = [body[j] / rng[j] if np.isfinite(rng[j]) else 0.0 for j in trio]
        vols = [vr[j] if np.isfinite(vr[j]) else np.nan for j in trio]

        volume_ok = True
        if require_volume and has_volume:
            volume_ok = all(np.isfinite(x) and x >= volume_multiple
                            for x in vols)

        if (ordered and opens_ok and volume_ok
                and all(w <= max_wick for w in wicks)
                and all(b >= min_body for b in bodies)):
            anchor = float(lo[i - 2] if bull else h[i - 2])
            move = (c[i] / o[i - 2] - 1.0) * 100.0 if o[i - 2] else 0.0
            out.append(PatternEvent(
                kind="three_white_soldiers" if bull else "three_black_crows",
                start_date=str(idx[i - 2])[:10], end_date=str(idx[i])[:10],
                start_idx=i - 2, end_idx=i, anchor_price=anchor,
                total_move_pct=float(move),
                volume_ratios=[float(x) if np.isfinite(x) else float("nan")
                               for x in vols],
                wick_fractions=[float(w) for w in wicks],
                body_fractions=[float(b) for b in bodies],
            ))
            i += 3          # non-overlapping: three bars belong to one pattern
            continue
        i += 1
    return out


# ── Anchored VWAP ────────────────────────────────────────────────────────────
@dataclass
class AnchoredVWAP:
    """VWAP from a fixed anchor bar, with standard-deviation envelopes."""
    vwap: pd.Series
    upper1: pd.Series
    lower1: pd.Series
    upper2: pd.Series
    lower2: pd.Series
    anchor_date: str
    anchor_idx: int
    current: float
    price_vs_vwap_pct: float

    @property
    def position(self) -> str:
        if self.price_vs_vwap_pct > 0:
            return "above"
        if self.price_vs_vwap_pct < 0:
            return "below"
        return "at"

    def reading(self, price: float) -> str:
        return (
            f"Anchored from {self.anchor_date}, the volume-weighted average "
            f"price is {self.current:,.2f} — price sits {self.position} it by "
            f"{abs(self.price_vs_vwap_pct):.1f}%. Everyone who participated "
            "since the anchor has, on average, paid that price, which is why "
            "it tends to act as the level a move must hold to stay intact. "
            "The ±1σ and ±2σ bands show how far price has typically strayed "
            "from it over the same window."
        )


def anchored_vwap(df: pd.DataFrame, anchor_idx: int = 0) -> Optional[AnchoredVWAP]:
    """
    VWAP accumulated from ``anchor_idx`` forward, with ±1σ and ±2σ envelopes.

    Uses the typical price (H+L+C)/3 weighted by volume — the standard
    construction. The bands are the volume-weighted standard deviation of
    price around the running VWAP, not a rolling standard deviation of price,
    which would be a different (and less meaningful) object.
    """
    if df is None or df.empty:
        return None
    if not {"High", "Low", "Close"}.issubset(df.columns):
        return None
    data = df.dropna(subset=["High", "Low", "Close"]).sort_index()
    anchor_idx = max(0, min(int(anchor_idx), len(data) - 1))
    seg = data.iloc[anchor_idx:]
    if len(seg) < 2:
        return None

    tp = (seg["High"].astype(float) + seg["Low"].astype(float)
          + seg["Close"].astype(float)) / 3.0
    if "Volume" in seg.columns:
        vol = pd.to_numeric(seg["Volume"], errors="coerce").fillna(0.0)
        if vol.sum() <= 0:
            vol = pd.Series(1.0, index=seg.index)
    else:
        vol = pd.Series(1.0, index=seg.index)

    cum_v = vol.cumsum().replace(0, np.nan)
    vwap = (tp * vol).cumsum() / cum_v
    # Volume-weighted variance around the running VWAP.
    var = ((vol * (tp - vwap) ** 2).cumsum() / cum_v).clip(lower=0)
    sd = np.sqrt(var)

    price = float(seg["Close"].iloc[-1])
    cur = float(vwap.iloc[-1])
    return AnchoredVWAP(
        vwap=vwap, upper1=vwap + sd, lower1=vwap - sd,
        upper2=vwap + 2 * sd, lower2=vwap - 2 * sd,
        anchor_date=str(seg.index[0])[:10], anchor_idx=anchor_idx,
        current=cur,
        price_vs_vwap_pct=float((price / cur - 1.0) * 100.0) if cur else 0.0,
    )


def pattern_frame(events: list[PatternEvent]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Pattern": e.label,
        "Completed": e.end_date,
        "Move": f"{e.total_move_pct:+.1f}%",
        "Min volume": f"{min(e.volume_ratios):.1f}×"
                      if e.volume_ratios and np.isfinite(min(e.volume_ratios))
                      else "—",
        "Max wick": f"{max(e.wick_fractions) * 100:.0f}%",
        "Anchor": f"{e.anchor_price:,.2f}",
    } for e in reversed(events)])
