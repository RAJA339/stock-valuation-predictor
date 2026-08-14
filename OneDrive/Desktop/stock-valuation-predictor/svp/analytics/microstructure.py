"""
Market microstructure — the documented ways institutional flow moves price.
===========================================================================

This is the honest version of "what do the big players do that retail does not
see." It is *not* a claim to have reverse-engineered anyone's proprietary
algorithm, and it does not detect illegal manipulation — spoofing, layering and
marking-the-close leave their traces in the order book and the tape, neither of
which a delayed OHLCV feed contains. What free daily and intraday bars *can*
show is the aggregate footprint of how institutional execution actually clears:

**Overnight vs intraday decomposition.** The most retail-relevant published
anomaly (Cooper, Cliff & Gulen; Lachance; Bogousslavsky). Historically almost
all of the equity return accrued *overnight* — the close-to-open gap — while the
intraday session, the only part a day-trader can act in, drifted flat or
negative. Decomposed per ticker so the reader sees it on the name they hold.

**Time-of-day profile.** Volume and volatility trace a U across the session
(Admati & Pfleiderer): heavy at the open and into the close, thin at midday.
That shape is the residue of VWAP/TWAP execution schedules and the closing
auction, and it says *when* size actually trades — which is when a retail order
is most and least likely to be run over.

**Liquidity sweeps.** A bar that pierces a recent low then closes back above it
is the observable footprint of stops being taken before a reversal — the "stop
run" retail lore describes, made measurable. The rate at which it precedes a
bounce is *measured* here with a Wilson interval, exactly as
:mod:`svp.analytics.accuracy` measures indicator hit rates, so a pattern that is
really just noise reports as noise.

Every measurement carries its sample size and a confidence interval. Nothing
here asserts an edge; several of these, measured honestly, will not be one after
costs, and the module says so rather than selling it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .accuracy import wilson_interval

# US regular session, exchange time. The intraday feed is tz-aware in
# US/Eastern already, so these bound the session for the time-of-day profile.
_SESSION_OPEN_MIN = 9 * 60 + 30       # 09:30
_SESSION_CLOSE_MIN = 16 * 60          # 16:00


# ── Overnight vs intraday ────────────────────────────────────────────────────
@dataclass
class OvernightSplit:
    """How a name's return divides between the overnight gap and the session."""
    n_days: int
    overnight_total: float            # compounded close→open return, %
    intraday_total: float             # compounded open→close return, %
    total: float                      # compounded close→close return, %
    overnight_mean_bp: float          # mean per-day, basis points
    intraday_mean_bp: float
    overnight_share: Optional[float]  # overnight / total, when total move is up

    @property
    def intraday_is_negative(self) -> bool:
        return self.intraday_mean_bp < 0

    @property
    def verdict(self) -> str:
        if self.n_days < 40:
            return "Not enough history to read the split"
        if self.intraday_mean_bp < 0 < self.overnight_mean_bp:
            return ("The gain accrued overnight; the tradable session lost money "
                    "on average — holding through the day, you fought the tape")
        if self.overnight_mean_bp > self.intraday_mean_bp > 0:
            return "Both were positive, but the overnight gap did the heavier lifting"
        if self.intraday_mean_bp > self.overnight_mean_bp:
            return "Unusually, the intraday session out-earned the overnight gap here"
        return "Overnight and intraday roughly split the move"


def overnight_intraday_split(history: pd.DataFrame) -> Optional[OvernightSplit]:
    """
    Decompose daily returns into the overnight gap and the intraday session.

    ``history`` is the daily OHLC frame from :mod:`svp.data.market`. Needs Open
    and Close; the overnight leg is ``open_t / close_{t-1} - 1`` and the intraday
    leg is ``close_t / open_t - 1``. Their product is the close-to-close return,
    which is the identity the two legs must reconstruct.
    """
    if history is None or history.empty or not {"Open", "Close"} <= set(history.columns):
        return None
    df = history[["Open", "Close"]].dropna()
    if len(df) < 40:
        return None

    open_t = df["Open"].to_numpy()
    close_t = df["Close"].to_numpy()
    prev_close = close_t[:-1]

    overnight = open_t[1:] / prev_close - 1.0       # close_{t-1} -> open_t
    intraday = close_t[1:] / open_t[1:] - 1.0       # open_t -> close_t
    n = len(overnight)

    on_total = float(np.prod(1.0 + overnight) - 1.0) * 100.0
    id_total = float(np.prod(1.0 + intraday) - 1.0) * 100.0
    total = float(np.prod(1.0 + overnight) * np.prod(1.0 + intraday) - 1.0) * 100.0

    share = None
    if on_total + id_total != 0 and total > 0:
        # Share of the up-move carried overnight. Only meaningful when the name
        # actually rose; on a decliner the ratio is not interpretable.
        share = on_total / (on_total + id_total) * 100.0

    return OvernightSplit(
        n_days=n,
        overnight_total=on_total,
        intraday_total=id_total,
        total=total,
        overnight_mean_bp=float(np.mean(overnight)) * 10_000.0,
        intraday_mean_bp=float(np.mean(intraday)) * 10_000.0,
        overnight_share=share,
    )


# ── Time-of-day profile ──────────────────────────────────────────────────────
@dataclass
class TimeOfDayProfile:
    """Per-bar volume and absolute-return shares across the session."""
    labels: list = field(default_factory=list)     # bucket labels, e.g. "09:30"
    volume_share: list = field(default_factory=list)   # % of session volume
    volatility: list = field(default_factory=list)     # mean abs return, bp
    n_sessions: int = 0

    @property
    def busiest_label(self) -> Optional[str]:
        if not self.volume_share:
            return None
        return self.labels[int(np.argmax(self.volume_share))]


def time_of_day_profile(intraday_df: pd.DataFrame) -> Optional[TimeOfDayProfile]:
    """
    Average volume and volatility by time-of-day bucket.

    Buckets are the bars' own timestamps (their minute-of-day), averaged across
    every session in the frame, so the U-shape emerges without assuming it. The
    volume figure is each bucket's share of a session's total, which normalises
    across days of different overall activity.
    """
    if intraday_df is None or intraday_df.empty:
        return None
    if not {"Close", "Volume"} <= set(intraday_df.columns):
        return None

    df = intraday_df.copy()
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return None

    minute = idx.hour * 60 + idx.minute
    df = df.assign(
        _minute=minute,
        _day=idx.normalize(),
        _absret=df["Close"].pct_change().abs(),
    )
    n_sessions = df["_day"].nunique()
    if n_sessions < 2:
        return None

    # Each bucket's share of its own session's volume, then averaged over days.
    day_totals = df.groupby("_day")["Volume"].transform("sum")
    df["_vol_share"] = np.where(day_totals > 0, df["Volume"] / day_totals * 100.0, 0.0)

    grp = df.groupby("_minute")
    vol_share = grp["_vol_share"].mean()
    absret = grp["_absret"].mean() * 10_000.0

    labels = [f"{m // 60:02d}:{m % 60:02d}" for m in vol_share.index]
    return TimeOfDayProfile(
        labels=labels,
        volume_share=[float(v) for v in vol_share.to_numpy()],
        volatility=[float(v) for v in absret.fillna(0.0).to_numpy()],
        n_sessions=int(n_sessions),
    )


# ── Liquidity sweeps ─────────────────────────────────────────────────────────
@dataclass
class SweepStudy:
    """Measured reversal rate after a liquidity-sweep bar."""
    direction: str                    # "downside" | "upside"
    n_events: int
    n_reversed: int
    reversal_rate: float
    ci_low: float
    ci_high: float
    lookback: int
    horizon: int

    @property
    def is_better_than_chance(self) -> bool:
        """The interval clears 50%. Absent that, this is not a signal."""
        return self.n_events >= 20 and self.ci_low > 0.50

    @property
    def verdict(self) -> str:
        if self.n_events < 20:
            return "Too few sweeps to measure"
        if self.ci_low > 0.50:
            return "Reversal rate is above chance on this sample"
        if self.ci_high < 0.50:
            return "Sweeps here tend to continue, not reverse"
        return "No better than a coin flip — indistinguishable from noise"


def _sweeps(df: pd.DataFrame, lookback: int, horizon: int, downside: bool) -> SweepStudy:
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()
    n = len(close)

    events = reversed_ = 0
    for i in range(lookback, n - horizon):
        if downside:
            prior_low = low[i - lookback:i].min()
            swept = low[i] < prior_low and close[i] > prior_low
            if swept:
                events += 1
                reversed_ += close[i + horizon] > close[i]
        else:
            prior_high = high[i - lookback:i].max()
            swept = high[i] > prior_high and close[i] < prior_high
            if swept:
                events += 1
                reversed_ += close[i + horizon] < close[i]

    rate = reversed_ / events if events else 0.0
    lo, hi = wilson_interval(reversed_, events)
    return SweepStudy(
        direction="downside" if downside else "upside",
        n_events=events, n_reversed=reversed_, reversal_rate=rate,
        ci_low=lo, ci_high=hi, lookback=lookback, horizon=horizon,
    )


def liquidity_sweeps(df: pd.DataFrame, lookback: int = 20,
                     horizon: int = 5) -> Optional[dict]:
    """
    Measure how often a stop-run bar precedes a reversal, both directions.

    A downside sweep is a bar whose low undercuts the prior ``lookback`` lows but
    whose close recovers back above that level — stops triggered, then price
    reclaimed. "Reversed" means the close ``horizon`` bars later is higher. The
    upside case is the mirror. Both are returned so the reader can see whether
    the effect is real or just the base rate of an up-drifting series.
    """
    if df is None or df.empty or not {"High", "Low", "Close"} <= set(df.columns):
        return None
    if len(df) < lookback + horizon + 5:
        return None
    return {
        "downside": _sweeps(df, lookback, horizon, downside=True),
        "upside": _sweeps(df, lookback, horizon, downside=False),
    }


# ── Round-number magnetism ───────────────────────────────────────────────────
@dataclass
class RoundNumberStudy:
    step: float
    n_bars: int
    near_rate: float                  # share of closes within the band
    expected_rate: float              # share expected if closes were uniform
    excess: float                     # near_rate - expected_rate, points

    @property
    def is_magnetic(self) -> bool:
        """More closes cluster at round numbers than uniform spacing predicts."""
        return self.n_bars >= 100 and self.excess > 3.0


def round_number_magnetism(df: pd.DataFrame, band_frac: float = 0.001) -> Optional[RoundNumberStudy]:
    """
    Do closes cluster at round price levels more than chance would put them there?

    Round numbers are where resting orders and stops concentrate, so price is
    often said to be "drawn" to them. This measures it: the share of closes
    within ``band_frac`` of a round level, against the share a uniform
    distribution would place in the same band. Excess above that baseline is the
    only honest evidence of clustering — a wide band catches many closes for
    reasons that have nothing to do with magnetism, and subtracting the expected
    rate controls for exactly that.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = df["Close"].dropna().to_numpy()
    if len(close) < 100:
        return None

    typical = float(np.median(close))
    if typical <= 0:
        return None
    # Round-number step scaled to the price: $1 for a $100 stock, $0.10 for $10.
    step = 10.0 ** np.floor(np.log10(typical) - 1)

    nearest = np.round(close / step) * step
    dist = np.abs(close - nearest) / close
    near = float(np.mean(dist < band_frac)) * 100.0

    # Under a uniform close within each step, the fraction landing within
    # band_frac*price of the round level is (2 * band_frac * price) / step.
    expected = float(np.mean(2.0 * band_frac * close / step)) * 100.0
    expected = min(expected, 100.0)

    return RoundNumberStudy(
        step=float(step), n_bars=len(close), near_rate=near,
        expected_rate=expected, excess=near - expected,
    )
