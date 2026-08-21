"""
Fair value gaps and their inversions (FVG / IFVG).
==================================================

A **fair value gap** is a three-bar imbalance: price moved so fast that the
middle bar's range left a band no trade passed through. Bullish gap — the
third bar's low sits above the first bar's high; bearish — the third bar's
high sits below the first bar's low. The band between them is the gap.

An **inverse fair value gap** is what that band becomes once price closes
clean through it. The framework's claim is a polarity flip: a bullish gap
that gets closed below stops being support and starts acting as resistance,
and the mirror for a bearish gap closed above. Traders watch the *retest* —
price returning to the flipped band — as the level that matters.

The detection here is mechanical and exact; the flip is bookkeeping. What is
a claim is whether a retested IFVG is actually respected, so that is the part
this module **measures**: every retest in the history is scored on which way
price went over a fixed horizon, and the hit rate is reported with a Wilson
interval. Unless the interval clears 50% the study says coin flip, and on a
random walk it must — the property the tests pin.

Two rules keep the bookkeeping honest:

- A gap is only inverted by a **close** beyond its far side, not by a wick.
  A wick through is a test; a close through is a rejection of the level.
- Retests are taken **non-overlapping**, so one long move back through a zone
  is one observation rather than several.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .accuracy import wilson_interval

MIN_BARS = 60
MIN_RETESTS = 20           # below this, no rate is worth quoting
#: A gap thinner than this fraction of price is noise, not an imbalance.
MIN_GAP_PCT = 0.001


@dataclass
class Gap:
    """One fair value gap, and whatever became of it."""
    direction: str             # "bullish" | "bearish" — the ORIGINAL gap
    bottom: float
    top: float
    formed: str                # ISO date of the third bar
    formed_idx: int
    inverted: bool = False
    inverted_date: str = ""
    inverted_idx: Optional[int] = None

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def role(self) -> str:
        """What the band acts as now, under the framework's polarity flip."""
        if not self.inverted:
            return "support" if self.direction == "bullish" else "resistance"
        # Inverted: a bullish gap closed below now caps price, and vice versa.
        return "resistance" if self.direction == "bullish" else "support"

    @property
    def label(self) -> str:
        if not self.inverted:
            return f"{self.direction.title()} FVG"
        flipped = "bearish" if self.direction == "bullish" else "bullish"
        return f"{flipped.title()} IFVG"


@dataclass
class Retest:
    """One measured return to an inverted gap."""
    zone_bottom: float
    zone_top: float
    role: str                  # "support" | "resistance" at the time of test
    date: str
    forward_return_pct: float
    respected: bool


@dataclass
class FVGStudy:
    gaps: list[Gap] = field(default_factory=list)
    retests: list[Retest] = field(default_factory=list)
    n_retests: int = 0
    n_respected: int = 0
    rate: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    horizon: int = 5

    @property
    def open_gaps(self) -> list[Gap]:
        """Gaps that have not been closed through — still original polarity."""
        return [g for g in self.gaps if not g.inverted]

    @property
    def inverted_gaps(self) -> list[Gap]:
        return [g for g in self.gaps if g.inverted]

    @property
    def is_better_than_chance(self) -> bool:
        return self.n_retests >= MIN_RETESTS and self.ci_low > 0.50

    @property
    def verdict(self) -> str:
        if self.n_retests < MIN_RETESTS:
            return (f"Only {self.n_retests} inverted-gap retests in this "
                    "history — too few to quote a rate, so none is quoted.")
        band = (f"Price respected the flipped band in {self.rate * 100:.0f}% of "
                f"{self.n_retests} retests (95% CI {self.ci_low * 100:.0f}–"
                f"{self.ci_high * 100:.0f}%)")
        if self.ci_low > 0.50:
            return (band + " — above chance on this name's history. Measured "
                    "on the past, not a promise about the next retest.")
        if self.ci_high < 0.50:
            return (band + " — below chance: these bands more often gave way "
                    "than held, which is the opposite of the framework's claim.")
        return (band + " — the interval straddles 50%, so on this history an "
                "inverted gap is indistinguishable from a coin flip.")

    def nearest(self, price: float, n: int = 4) -> list[Gap]:
        """The zones closest to ``price``, nearest first."""
        return sorted(self.gaps, key=lambda g: abs(g.mid - price))[:n]


def find_gaps(df: pd.DataFrame, min_gap_pct: float = MIN_GAP_PCT) -> list[Gap]:
    """Every three-bar fair value gap in ``df``, oldest first."""
    if df is None or len(df) < 3:
        return []
    if not all(c in df.columns for c in ("High", "Low")):
        return []
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float) if "Close" in df.columns else high
    idx = df.index

    out: list[Gap] = []
    for i in range(1, len(df) - 1):
        ref = close[i] if close[i] else 1.0
        # Bullish: the third bar's low clears the first bar's high.
        if low[i + 1] > high[i - 1]:
            bottom, top = float(high[i - 1]), float(low[i + 1])
            if (top - bottom) / abs(ref) >= min_gap_pct:
                out.append(Gap("bullish", bottom, top, str(idx[i + 1])[:10], i + 1))
        # Bearish: the third bar's high sits under the first bar's low.
        elif high[i + 1] < low[i - 1]:
            bottom, top = float(high[i + 1]), float(low[i - 1])
            if (top - bottom) / abs(ref) >= min_gap_pct:
                out.append(Gap("bearish", bottom, top, str(idx[i + 1])[:10], i + 1))
    return out


def detect(df: pd.DataFrame, horizon: int = 5,
           min_gap_pct: float = MIN_GAP_PCT) -> Optional[FVGStudy]:
    """
    Find gaps, mark the ones price closed through, and measure their retests.

    ``horizon`` is how many bars after a retest the outcome is read. A retest
    is "respected" when price moved away from the zone in the direction the
    flipped polarity implies — down from resistance, up from support.
    """
    if df is None or df.empty or len(df) < MIN_BARS:
        return None
    if not all(c in df.columns for c in ("High", "Low", "Close")):
        return None
    data = df.dropna(subset=["High", "Low", "Close"]).sort_index()
    if len(data) < MIN_BARS:
        return None

    gaps = find_gaps(data, min_gap_pct=min_gap_pct)
    if not gaps:
        return FVGStudy(horizon=horizon)

    high = data["High"].to_numpy(dtype=float)
    low = data["Low"].to_numpy(dtype=float)
    close = data["Close"].to_numpy(dtype=float)
    idx = data.index
    n = len(data)

    retests: list[Retest] = []
    for g in gaps:
        # ── Inversion: the first CLOSE clean through the far side ────────────
        inv_at = None
        for j in range(g.formed_idx + 1, n):
            if g.direction == "bullish" and close[j] < g.bottom:
                inv_at = j
                break
            if g.direction == "bearish" and close[j] > g.top:
                inv_at = j
                break
        if inv_at is None:
            continue
        g.inverted = True
        g.inverted_idx = inv_at
        g.inverted_date = str(idx[inv_at])[:10]

        # ── Retests of the flipped band, non-overlapping ─────────────────────
        role = g.role                       # after the flip
        k = inv_at + 1
        while k < n - horizon:
            touched = (low[k] <= g.top and high[k] >= g.bottom)
            if not touched:
                k += 1
                continue
            entry, later = float(close[k]), float(close[k + horizon])
            fwd = (later / entry - 1.0) * 100.0 if entry else 0.0
            respected = fwd < 0 if role == "resistance" else fwd > 0
            retests.append(Retest(
                zone_bottom=g.bottom, zone_top=g.top, role=role,
                date=str(idx[k])[:10], forward_return_pct=float(fwd),
                respected=bool(respected)))
            k += horizon + 1                # one move, one observation

    n_re = len(retests)
    n_ok = sum(1 for r in retests if r.respected)
    rate = n_ok / n_re if n_re else 0.0
    ci_low, ci_high = wilson_interval(n_ok, n_re) if n_re else (0.0, 0.0)

    retests.sort(key=lambda r: r.date)
    return FVGStudy(gaps=gaps, retests=retests, n_retests=n_re,
                    n_respected=n_ok, rate=rate, ci_low=ci_low,
                    ci_high=ci_high, horizon=horizon)


def unfilled_gaps(study: FVGStudy, price: float, n: int = 4) -> list[Gap]:
    """Open (never closed through) gaps nearest to ``price``."""
    return sorted(study.open_gaps, key=lambda g: abs(g.mid - price))[:n]


def active_ifvgs(study: FVGStudy, price: float, n: int = 4) -> list[Gap]:
    """Inverted gaps nearest to ``price`` — the live IFVG bands."""
    return sorted(study.inverted_gaps, key=lambda g: abs(g.mid - price))[:n]


def summary_note(study: FVGStudy, price: float) -> str:
    """One descriptive line about the nearest live band. Never a forecast."""
    live = active_ifvgs(study, price, n=1)
    if not live:
        opens = unfilled_gaps(study, price, n=1)
        if not opens:
            return "No fair value gaps near the current price."
        g = opens[0]
        side = "above" if g.mid > price else "below"
        return (f"Nearest untouched gap is a {g.direction} FVG {side} price at "
                f"{g.bottom:,.2f}–{g.top:,.2f}, formed {g.formed}. It has not "
                "been closed through, so its original polarity still stands.")
    g = live[0]
    side = "above" if g.mid > price else "below"
    return (f"Nearest inverted band sits {side} price at {g.bottom:,.2f}–"
            f"{g.top:,.2f}: a {g.direction} FVG from {g.formed} that price "
            f"closed through on {g.inverted_date}, so the framework now reads "
            f"it as {g.role}. Whether it holds is what the rate below measures.")


def zone_levels(gaps: list[Gap]) -> list[dict]:
    """Chart price-lines for a set of gaps (top and bottom of each band)."""
    out = []
    for g in gaps:
        for edge, price in (("top", g.top), ("bottom", g.bottom)):
            out.append({
                "price": round(float(price), 2),
                "style": 1 if g.inverted else 2,
                "title": (f"{g.label} {edge} {price:,.2f}"),
            })
    return out


def gap_frame(gaps: list[Gap]) -> pd.DataFrame:
    """Tabular view for the UI."""
    return pd.DataFrame([{
        "Zone": g.label,
        "Band": f"{g.bottom:,.2f} – {g.top:,.2f}",
        "Formed": g.formed,
        "Closed through": g.inverted_date or "—",
        "Acts as": g.role,
    } for g in gaps])
