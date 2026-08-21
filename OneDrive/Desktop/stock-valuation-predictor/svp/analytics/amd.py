"""
AMD — accumulation, manipulation, distribution.
===============================================

The framework says a move has three legs: price coils in a range
(**accumulation**), breaks out of it just far enough to trigger the stops
resting beyond it and then closes back inside (**manipulation**), and only
then makes the real move — in the direction opposite the fake one
(**distribution**).

That is a *claim*, and this module treats it as one. Detection is mechanical:

- **Accumulation** — a window whose high-low range is contracted relative to
  that instrument's own recent ranges. Contraction is measured against the
  median, so "tight" means tight for this name rather than tight in dollars.
- **Manipulation** — a later bar whose low undercuts the accumulation low but
  whose close recovers back inside it (or the mirror above the high). This is
  the same false-break test :func:`svp.analytics.microstructure.liquidity_sweeps`
  applies to single bars, anchored here to a defined range.
- **Distribution** — whatever price actually did over the following bars.

The last leg is measured, never assumed. Every detected sequence is scored
for whether distribution ran the way the framework says it should, and the
hit rate is reported with a Wilson interval: unless the interval clears 50%,
this reads as a coin flip and says so. Fed a random walk it must report
chance — that property is what the test suite pins, because a pattern
language that always finds a pattern is worth nothing.

Events are taken non-overlapping so the interval is not built from the same
move counted several times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .accuracy import wilson_interval

MIN_BARS = 120
MIN_EVENTS = 20            # below this, no interval is worth quoting


@dataclass
class AMDEvent:
    """One detected accumulation → manipulation → distribution sequence."""
    acc_start: str                  # ISO date of the accumulation window start
    acc_end: str
    sweep_date: str
    direction: str                  # "bullish" (low swept) | "bearish" (high swept)
    acc_low: float
    acc_high: float
    sweep_extreme: float            # how far past the range the sweep reached
    forward_return_pct: float       # over the distribution horizon
    followed_through: bool


@dataclass
class AMDStudy:
    events: list[AMDEvent] = field(default_factory=list)
    n_events: int = 0
    n_followed: int = 0
    rate: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    acc_bars: int = 20
    horizon: int = 10
    current_phase: str = "None"     # Accumulation | Manipulation | None
    current_note: str = ""

    @property
    def is_better_than_chance(self) -> bool:
        """The interval clears 50%. Absent that, this is not an edge."""
        return self.n_events >= MIN_EVENTS and self.ci_low > 0.50

    @property
    def verdict(self) -> str:
        if self.n_events < MIN_EVENTS:
            return (f"Only {self.n_events} complete sequences in this history — "
                    "too few to say anything about follow-through. No rate is "
                    "quoted on a sample this small.")
        band = (f"{self.rate * 100:.0f}% of {self.n_events} sequences ran the "
                f"way the framework says (95% CI {self.ci_low * 100:.0f}–"
                f"{self.ci_high * 100:.0f}%)")
        if self.ci_low > 0.50:
            return (band + " — above chance on this instrument's history. "
                    "Measured on the past, not a promise about the next one.")
        if self.ci_high < 0.50:
            return (band + " — below chance: after these sweeps price more "
                    "often continued in the sweep's direction rather than "
                    "reversing, which is the opposite of the framework's claim.")
        return (band + " — the interval straddles 50%, so on this history AMD "
                "sequences are indistinguishable from a coin flip.")


def _contraction_mask(high: pd.Series, low: pd.Series, close: pd.Series,
                      acc_bars: int, contraction: float) -> pd.Series:
    """
    True where the trailing ``acc_bars`` window is a contracted range.

    Range is normalised by price and compared with the median of the same
    measure over the whole history, so the threshold adapts to the
    instrument instead of assuming a dollar width.
    """
    rng = (high.rolling(acc_bars).max() - low.rolling(acc_bars).min()) / close
    median = rng.median()
    if not np.isfinite(median) or median <= 0:
        return pd.Series(False, index=high.index)
    return rng <= median * contraction


def detect(df: pd.DataFrame, acc_bars: int = 20, manip_window: int = 10,
           horizon: int = 10, contraction: float = 0.75) -> Optional[AMDStudy]:
    """
    Find AMD sequences and measure their distribution leg.

    ``contraction`` is the fraction of the median normalised range under which
    a window counts as accumulation (0.75 = at least 25% tighter than usual).
    """
    if df is None or df.empty or len(df) < MIN_BARS:
        return None
    need = ("High", "Low", "Close")
    if not all(c in df.columns for c in need):
        return None
    data = df.dropna(subset=list(need)).sort_index()
    if len(data) < MIN_BARS:
        return None

    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    close = data["Close"].astype(float)
    idx = data.index

    tight = _contraction_mask(high, low, close, acc_bars, contraction)
    h, lo_a, c = high.to_numpy(), low.to_numpy(), close.to_numpy()
    tight_a = tight.to_numpy()

    events: list[AMDEvent] = []
    i = acc_bars
    n = len(data)
    while i < n - horizon - 1:
        if not tight_a[i]:
            i += 1
            continue
        acc_high = float(h[i - acc_bars + 1:i + 1].max())
        acc_low = float(lo_a[i - acc_bars + 1:i + 1].min())
        if acc_high <= acc_low:
            i += 1
            continue

        # Look forward for the manipulation leg: a close-back-inside false break.
        sweep = None
        for j in range(i + 1, min(i + 1 + manip_window, n - horizon)):
            if lo_a[j] < acc_low and c[j] > acc_low:
                sweep = (j, "bullish", float(lo_a[j]))
                break
            if h[j] > acc_high and c[j] < acc_high:
                sweep = (j, "bearish", float(h[j]))
                break
        if sweep is None:
            i += 1
            continue

        j, direction, extreme = sweep
        entry, exit_ = float(c[j]), float(c[j + horizon])
        fwd = (exit_ / entry - 1.0) * 100.0 if entry else 0.0
        followed = fwd > 0 if direction == "bullish" else fwd < 0

        events.append(AMDEvent(
            acc_start=str(idx[i - acc_bars + 1])[:10],
            acc_end=str(idx[i])[:10],
            sweep_date=str(idx[j])[:10],
            direction=direction,
            acc_low=acc_low, acc_high=acc_high, sweep_extreme=extreme,
            forward_return_pct=float(fwd), followed_through=bool(followed),
        ))
        # Non-overlapping: resume after this sequence's distribution window, so
        # one move cannot be counted as several independent events.
        i = j + horizon + 1

    n_ev = len(events)
    n_hit = sum(1 for e in events if e.followed_through)
    rate = n_hit / n_ev if n_ev else 0.0
    ci_low, ci_high = wilson_interval(n_hit, n_ev) if n_ev else (0.0, 0.0)

    phase, note = _current_phase(h, lo_a, c, tight_a, idx, acc_bars, manip_window)

    return AMDStudy(
        events=events, n_events=n_ev, n_followed=n_hit, rate=rate,
        ci_low=ci_low, ci_high=ci_high, acc_bars=acc_bars, horizon=horizon,
        current_phase=phase, current_note=note,
    )


def _current_phase(h, lo_a, c, tight_a, idx, acc_bars, manip_window):
    """
    Describe where the tape sits *now* — structure, not forecast.

    Reported as one of Accumulation (the current window is contracted),
    Manipulation (a recent accumulation was just swept and reclaimed), or
    None. The note never says what happens next.
    """
    n = len(c)
    if n < acc_bars + 2:
        return "None", ""

    if tight_a[-1]:
        hi = float(h[-acc_bars:].max())
        lo = float(lo_a[-acc_bars:].min())
        return "Accumulation", (
            f"The last {acc_bars} bars are a contracted range, "
            f"{lo:,.2f}–{hi:,.2f}. That is the coil the framework calls "
            "accumulation; whether anything comes of it is not knowable from "
            "the range itself.")

    # Was there a contracted window that has since been swept and reclaimed?
    for back in range(1, min(manip_window, n - acc_bars - 1) + 1):
        k = n - 1 - back
        if k - acc_bars < 0 or not tight_a[k]:
            continue
        acc_high = float(h[k - acc_bars + 1:k + 1].max())
        acc_low = float(lo_a[k - acc_bars + 1:k + 1].min())
        for j in range(k + 1, n):
            if lo_a[j] < acc_low and c[j] > acc_low:
                return "Manipulation", (
                    f"The {acc_low:,.2f} floor of a recent contracted range "
                    f"was undercut on {str(idx[j])[:10]} and reclaimed the "
                    "same bar — a downside sweep. The framework reads that as "
                    "manipulation before an upward distribution; the "
                    "follow-through rate below is what actually happened after "
                    "sweeps like it.")
            if h[j] > acc_high and c[j] < acc_high:
                return "Manipulation", (
                    f"The {acc_high:,.2f} ceiling of a recent contracted range "
                    f"was exceeded on {str(idx[j])[:10]} and rejected the same "
                    "bar — an upside sweep. The framework reads that as "
                    "manipulation before a downward distribution; the "
                    "follow-through rate below is what actually happened after "
                    "sweeps like it.")
    return "None", ("No contracted range or recent sweep in the latest bars — "
                    "there is no AMD structure to describe right now.")
