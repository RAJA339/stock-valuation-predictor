"""
Market structure — swings, BOS, MSS, and order blocks.
======================================================

Price makes highs and lows; structure is the story those points tell. This
module reads it mechanically.

- **Swings** are pivots: a bar whose high is the highest of the ``k`` bars
  either side of it (mirror for lows). A pivot is only knowable ``k`` bars
  after it forms, and this module never pretends otherwise — every event
  carries the bar that *confirmed* it, not the bar it points at.

- **BOS** (break of structure) is a close beyond the last swing **in the
  direction the market was already going**: continuation.

- **MSS** (market structure shift, also CHoCH) is a close beyond the swing on
  the *other* side — the first evidence that the sequence of higher highs and
  higher lows has failed. The distinction is the whole point: a trader who
  cannot tell continuation from reversal has no structure at all, only lines.

- **Order blocks** are the last opposing candle before the impulse that broke
  structure — the bullish demand block is the final down-candle before the
  leg up. Each is scored by the **volume expansion** of the impulse that left
  it, because a block left by an unremarkable move is an unremarkable block,
  and each tracks its own **mitigation**: whether price has since traded back
  into it. An untested block and a thrice-tested one are different objects
  and are labelled as such.

Nothing here forecasts. These are named descriptions of what price did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

MIN_BARS = 60
DEFAULT_SWING = 3
#: Impulse volume below this multiple of its trailing average is not expansion.
VOLUME_EXPANSION = 1.3

Bias = Literal["bullish", "bearish", "neutral"]


@dataclass
class Swing:
    kind: Literal["high", "low"]
    price: float
    date: str
    idx: int
    confirmed_idx: int              # the bar at which this pivot became knowable


@dataclass
class StructureEvent:
    """A break of structure or a shift in it."""
    kind: Literal["BOS", "MSS"]
    direction: Literal["bullish", "bearish"]
    level: float                    # the swing level that was broken
    date: str                       # the bar that closed beyond it, ISO
    idx: int
    prior_bias: Bias
    #: The original index value of the breaking bar, carried verbatim.
    #: ``date`` is a display string; re-parsing it drops the timezone, and
    #: comparing that against a tz-aware price index raises. Keeping the raw
    #: label means callers plot against the same object the frame is indexed
    #: by and the question never arises.
    ts: object = None

    @property
    def label(self) -> str:
        arrow = "↑" if self.direction == "bullish" else "↓"
        return f"{self.kind} {arrow}"

    @property
    def meaning(self) -> str:
        if self.kind == "BOS":
            return ("continuation — price extended the structure it was "
                    "already building")
        return ("shift — the first close against the prevailing sequence of "
                "swings, which is what a reversal looks like at its start")


@dataclass
class OrderBlock:
    """The last opposing candle before a structure-breaking impulse."""
    kind: Literal["demand", "supply"]
    top: float
    bottom: float
    date: str
    idx: int
    impulse_volume_ratio: float     # impulse volume vs its trailing average
    mitigated: bool = False
    mitigation_date: str = ""
    touches: int = 0

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def status(self) -> str:
        if not self.mitigated:
            return "untested"
        return f"tested ×{self.touches}"

    @property
    def strength(self) -> str:
        """How exceptional the impulse that left this block was."""
        r = self.impulse_volume_ratio
        if r >= 2.0:
            return "high"
        if r >= VOLUME_EXPANSION:
            return "expanded"
        return "ordinary"

    @property
    def label(self) -> str:
        side = "Demand" if self.kind == "demand" else "Supply"
        return f"{side} block ({self.status})"


@dataclass
class StructureMap:
    swings: list[Swing] = field(default_factory=list)
    events: list[StructureEvent] = field(default_factory=list)
    blocks: list[OrderBlock] = field(default_factory=list)
    bias: Bias = "neutral"
    swing_k: int = DEFAULT_SWING

    @property
    def last_event(self) -> Optional[StructureEvent]:
        return self.events[-1] if self.events else None

    def untested_blocks(self, price: float, n: int = 4) -> list[OrderBlock]:
        live = [b for b in self.blocks if not b.mitigated]
        return sorted(live, key=lambda b: abs(b.mid - price))[:n]

    def nearest_blocks(self, price: float, n: int = 6) -> list[OrderBlock]:
        return sorted(self.blocks, key=lambda b: abs(b.mid - price))[:n]

    def reading(self, price: float) -> str:
        if not self.events:
            return ("No confirmed breaks of structure in this history — there "
                    "is no directional structure to describe.")
        ev = self.events[-1]
        base = (f"Most recent structural event: **{ev.label}** on {ev.date} "
                f"through {ev.level:,.2f} — {ev.meaning}. Current bias reads "
                f"**{self.bias}**.")
        live = self.untested_blocks(price, n=1)
        if live:
            b = live[0]
            side = "below" if b.mid < price else "above"
            base += (f" Nearest untested {b.kind} block sits {side} price at "
                     f"{b.bottom:,.2f}–{b.top:,.2f}, left by an impulse on "
                     f"{b.impulse_volume_ratio:.1f}× its average volume.")
        return base


def find_swings(df: pd.DataFrame, k: int = DEFAULT_SWING) -> list[Swing]:
    """
    Pivot highs and lows confirmed by ``k`` bars either side.

    A pivot at bar i is only *knowable* at bar i+k, which is recorded in
    ``confirmed_idx``. Reading a pivot as available on the bar it occurs is
    the most common way a structure backtest lies to itself.
    """
    if df is None or len(df) < 2 * k + 1:
        return []
    if not {"High", "Low"}.issubset(df.columns):
        return []
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    idx = df.index
    out: list[Swing] = []
    for i in range(k, len(df) - k):
        window_h = high[i - k:i + k + 1]
        window_l = low[i - k:i + k + 1]
        if high[i] == window_h.max() and (window_h.argmax() == k):
            out.append(Swing("high", float(high[i]), str(idx[i])[:10], i, i + k))
        elif low[i] == window_l.min() and (window_l.argmin() == k):
            out.append(Swing("low", float(low[i]), str(idx[i])[:10], i, i + k))
    return out


def _volume_ratio(vol: np.ndarray, i: int, window: int = 20) -> float:
    """Bar ``i``'s volume against its trailing average. 1.0 when unknowable."""
    lo = max(0, i - window)
    if i <= lo or vol is None:
        return 1.0
    trail = vol[lo:i]
    trail = trail[np.isfinite(trail) & (trail > 0)]
    if trail.size == 0 or not np.isfinite(vol[i]) or vol[i] <= 0:
        return 1.0
    return float(vol[i] / trail.mean())


def _find_order_block(o: np.ndarray, h: np.ndarray, low_a: np.ndarray,
                      c: np.ndarray, vol: Optional[np.ndarray], idx,
                      break_i: int, direction: str,
                      lookback: int = 12) -> Optional[OrderBlock]:
    """
    The last opposing candle before the impulse that broke structure.

    Searched backwards from the breaking bar: for a bullish break, the most
    recent down-candle before it. That candle is where the move originated,
    which is the whole idea of an order block.
    """
    want_down = direction == "bullish"
    start = max(0, break_i - lookback)
    for j in range(break_i - 1, start - 1, -1):
        is_down = c[j] < o[j]
        if is_down == want_down:
            ratio = _volume_ratio(vol, break_i) if vol is not None else 1.0
            return OrderBlock(
                kind="demand" if want_down else "supply",
                top=float(max(o[j], c[j], h[j] if want_down else h[j])),
                bottom=float(min(o[j], c[j], low_a[j])),
                date=str(idx[j])[:10], idx=j,
                impulse_volume_ratio=float(ratio),
            )
    return None


def _mark_mitigation(blocks: list[OrderBlock], h: np.ndarray,
                     low_a: np.ndarray, idx) -> None:
    """Flag blocks price has traded back into, and count the visits."""
    n = len(h)
    for b in blocks:
        for j in range(b.idx + 1, n):
            if low_a[j] <= b.top and h[j] >= b.bottom:
                b.touches += 1
                if not b.mitigated:
                    b.mitigated = True
                    b.mitigation_date = str(idx[j])[:10]


def analyse(df: pd.DataFrame, k: int = DEFAULT_SWING,
            block_lookback: int = 12) -> Optional[StructureMap]:
    """
    Map swings, structure events and order blocks across ``df``.

    Returns ``None`` when the history is too short for pivots to mean
    anything.
    """
    if df is None or df.empty or len(df) < MIN_BARS:
        return None
    need = ("Open", "High", "Low", "Close")
    if not all(col in df.columns for col in need):
        return None
    data = df.dropna(subset=list(need)).sort_index()
    if len(data) < MIN_BARS:
        return None

    o = data["Open"].to_numpy(dtype=float)
    h = data["High"].to_numpy(dtype=float)
    low_a = data["Low"].to_numpy(dtype=float)
    c = data["Close"].to_numpy(dtype=float)
    vol = (data["Volume"].to_numpy(dtype=float)
           if "Volume" in data.columns else None)
    idx = data.index

    swings = find_swings(data, k=k)
    events: list[StructureEvent] = []
    blocks: list[OrderBlock] = []
    bias: Bias = "neutral"

    # Walk forward bar by bar, using only pivots already confirmed at that bar.
    last_high: Optional[Swing] = None
    last_low: Optional[Swing] = None
    s_ptr = 0
    for i in range(len(data)):
        while s_ptr < len(swings) and swings[s_ptr].confirmed_idx <= i:
            s = swings[s_ptr]
            if s.kind == "high":
                last_high = s
            else:
                last_low = s
            s_ptr += 1

        if last_high is not None and c[i] > last_high.price:
            kind = "BOS" if bias == "bullish" else "MSS"
            events.append(StructureEvent(
                kind=kind, direction="bullish", level=last_high.price,
                date=str(idx[i])[:10], idx=i, prior_bias=bias, ts=idx[i]))
            ob = _find_order_block(o, h, low_a, c, vol, idx, i, "bullish",
                                   block_lookback)
            if ob is not None:
                blocks.append(ob)
            bias = "bullish"
            last_high = None            # consumed; wait for the next pivot
        elif last_low is not None and c[i] < last_low.price:
            kind = "BOS" if bias == "bearish" else "MSS"
            events.append(StructureEvent(
                kind=kind, direction="bearish", level=last_low.price,
                date=str(idx[i])[:10], idx=i, prior_bias=bias, ts=idx[i]))
            ob = _find_order_block(o, h, low_a, c, vol, idx, i, "bearish",
                                   block_lookback)
            if ob is not None:
                blocks.append(ob)
            bias = "bearish"
            last_low = None

    _mark_mitigation(blocks, h, low_a, idx)
    return StructureMap(swings=swings, events=events, blocks=blocks,
                        bias=bias, swing_k=k)


def event_frame(events: list[StructureEvent], limit: int = 20) -> pd.DataFrame:
    """Tabular view, most recent first."""
    return pd.DataFrame([{
        "Event": e.label,
        "Type": "Continuation" if e.kind == "BOS" else "Structure shift",
        "Date": e.date,
        "Level broken": f"{e.level:,.2f}",
        "Bias before": e.prior_bias,
    } for e in reversed(events[-limit:])])


def block_frame(blocks: list[OrderBlock]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Block": "Demand" if b.kind == "demand" else "Supply",
        "Zone": f"{b.bottom:,.2f} – {b.top:,.2f}",
        "Formed": b.date,
        "Impulse volume": f"{b.impulse_volume_ratio:.1f}×",
        "Strength": b.strength,
        "Status": b.status,
    } for b in blocks])
