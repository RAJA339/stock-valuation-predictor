"""
Bad prints — finding them, and refusing to analyse them.
========================================================

Price feeds contain errors: unadjusted splits, stray ticks, a placeholder bar
for a session that never traded. They are rare, and that is exactly what
makes them dangerous — a single corrupt bar in a year of data is invisible in
a table and catastrophic in anything that takes a maximum, a range, or a
level from the series.

The failure this module exists to stop was concrete. One bar with a high
several times the prevailing price:

- set a chart's axis, leaving the real price action in a fifth of the height;
- created "order blocks" out of the spike, floating at prices the stock has
  never traded near;
- widened the volume profile's range so every bin was wrong.

Each of those was patched at the point it showed. That was the wrong instinct:
the chart was not broken, the data was, and downstream repairs multiply while
the cause sits untouched.

Detection is a robust z-score — deviation from a rolling **median**, scaled by
the rolling **median absolute deviation**. Median and MAD are used precisely
because the thing being detected would corrupt a mean and a standard
deviation. Structural impossibilities (high below low, close outside the bar's
own range) are caught outright, since those need no statistics.

The bars are reported, never silently deleted: a caller that drops rows must
be able to tell the reader what was dropped and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

#: A bar this many robust deviations from local trend is not a price move.
#: Set high deliberately: real markets gap, halt and limit-move, and calling a
#: genuine 20% gap "bad data" would be a worse error than the one being fixed.
Z_THRESHOLD = 10.0
WINDOW = 21
MIN_BARS = 30
#: 1/Φ⁻¹(0.75) — scales MAD to a standard deviation for a normal sample.
_MAD_TO_SIGMA = 1.4826
#: A final bar has no successor to confirm it. Only a move beyond this — far
#: past any ordinary session, gaps and halts included — is called suspect on
#: its own, because erasing a real last-day move would be the worse error.
_LAST_BAR_MOVE = 0.55           # ~75% up or ~42% down, in log terms


@dataclass
class QualityReport:
    n_bars: int
    n_flagged: int
    flagged_dates: list = field(default_factory=list)
    reasons: dict = field(default_factory=dict)     # date -> why

    @property
    def is_clean(self) -> bool:
        return self.n_flagged == 0

    @property
    def share(self) -> float:
        return self.n_flagged / self.n_bars if self.n_bars else 0.0

    def note(self) -> str:
        if self.is_clean:
            return ""
        head = (f"{self.n_flagged} of {self.n_bars:,} bars look like bad "
                f"prints and were excluded from this analysis")
        sample = "; ".join(f"{d} ({self.reasons.get(d, 'outlier')})"
                           for d in self.flagged_dates[:3])
        tail = (f": {sample}" if sample else "")
        more = (f" and {self.n_flagged - 3} more"
                if self.n_flagged > 3 else "")
        return (head + tail + more + ". A single corrupt bar sets the range "
                "for everything computed from it, so they are removed rather "
                "than worked around.")


def _structural_faults(df: pd.DataFrame) -> pd.Series:
    """Bars that are impossible on their face, no statistics required."""
    o, h = df["Open"].astype(float), df["High"].astype(float)
    lo, c = df["Low"].astype(float), df["Close"].astype(float)
    bad = (h < lo)
    bad |= (h < o) | (h < c)
    bad |= (lo > o) | (lo > c)
    bad |= (o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0)
    bad |= ~np.isfinite(o) | ~np.isfinite(h) | ~np.isfinite(lo) | ~np.isfinite(c)
    return bad.fillna(True)


def _robust_outliers(df: pd.DataFrame, window: int, z: float) -> pd.Series:
    """
    Bars that moved absurdly and did not stay moved.

    The distinguishing property is **reversion**, not size. A trending stock
    is permanently far from its own trailing median, so judging distance from
    a median flags real trends as corruption — an earlier version of this
    function did exactly that, flagging a dozen bars in a clean random walk.
    What separates a bad print from a real repricing is what happens next: a
    genuine gap moves and *stays*, while a stray tick springs back the
    following bar.

    Returns are the unit, scaled by the median absolute deviation of returns,
    because the corrupt bar would inflate a standard deviation and hide
    itself.
    """
    # Non-positive prices are already caught structurally; masking them here
    # keeps log() from warning about them on the way past.
    c = df["Close"].astype(float).where(lambda s: s > 0)
    r = np.log(c).diff()

    scale = float((r - r.median()).abs().median() * _MAD_TO_SIGMA)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(r.abs().median()) or 0.01
    zr = (r - r.median()).abs() / scale

    nxt = r.shift(-1)
    # Reverses: the next bar undoes most of the move, in the other direction.
    reverts = (np.sign(r) != np.sign(nxt)) & (nxt.abs() >= 0.5 * r.abs())
    spike = (zr > z) & reverts.fillna(False)

    # A final bar has no next bar to confirm it, so it is judged only on
    # implausibility: a move this size in one session is a corporate action or
    # a bad tick, and either way it should not silently set every range.
    if len(r) > 2:
        last = r.index[-1]
        if np.isfinite(r.loc[last]) and abs(r.loc[last]) > _LAST_BAR_MOVE:
            spike.loc[last] = True

    # An intrabar spike that never reaches the close still sets the range: a
    # wick many times the typical bar is a print, not a trade anyone made.
    rng = (df["High"].astype(float) - df["Low"].astype(float))
    typical = float(rng.median())
    if np.isfinite(typical) and typical > 0:
        spike |= rng > (z * 2.0 * typical)

    return spike.fillna(False)


def flag(df: pd.DataFrame, window: int = WINDOW,
         z: float = Z_THRESHOLD) -> Optional[pd.Series]:
    """Boolean mask of suspect bars, or ``None`` if the frame cannot be judged."""
    if df is None or df.empty:
        return None
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return None
    if len(df) < MIN_BARS:
        return pd.Series(False, index=df.index)
    return _structural_faults(df) | _robust_outliers(df, window, z)


def inspect(df: pd.DataFrame, window: int = WINDOW,
            z: float = Z_THRESHOLD) -> QualityReport:
    """Report suspect bars without altering anything."""
    mask = flag(df, window=window, z=z)
    if mask is None:
        return QualityReport(n_bars=0, n_flagged=0)
    bad = df.index[mask]
    reasons: dict = {}
    if len(bad):
        struct = _structural_faults(df)
        for ts in bad:
            key = str(ts)[:10]
            reasons[key] = ("impossible OHLC" if bool(struct.loc[ts])
                            else "far from local trend")
    return QualityReport(n_bars=len(df), n_flagged=int(mask.sum()),
                         flagged_dates=[str(t)[:10] for t in bad],
                         reasons=reasons)


def clean(df: pd.DataFrame, window: int = WINDOW,
          z: float = Z_THRESHOLD) -> tuple[pd.DataFrame, QualityReport]:
    """
    Return ``(clean_frame, report)``.

    Bars are dropped rather than repaired. Interpolating a price invents a
    trade that did not happen, and every level derived from it would then be
    fiction presented at full confidence — the failure mode this module is
    written to prevent.
    """
    report = inspect(df, window=window, z=z)
    if df is None or df.empty or report.n_flagged == 0:
        return df, report
    mask = flag(df, window=window, z=z)
    return df[~mask], report
