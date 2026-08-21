"""
Tail risk and ownership concentration.
======================================

Two risks a price chart does not show.

**Expected shortfall (CVaR).** Value at Risk answers "how bad is a bad day",
and then stops exactly where the question gets interesting. VaR at 95% is the
threshold the worst 5% of days sit beyond; **CVaR is the average of those
days** — the number that matters, because portfolios are destroyed inside the
tail rather than at its edge. Both are reported: historical, straight from
the realised distribution, and a bootstrap Monte Carlo that resamples the
same history to put an interval around the estimate.

**Ownership concentration (HHI).** A stock whose float sits in a handful of
funds behaves differently when one of them leaves. The Herfindahl-Hirschman
Index measures that, but honestly only if you can see every holder — and the
available feed publishes the largest disclosed holders, not the register. So
this module computes it over the disclosed block, says what share of the
company that block is, and labels the company-wide figure as the **lower
bound** it actually is. Reporting it as *the* HHI would be a fabrication of
precision, and a low number would read as safety when it only means the rest
of the register is unseen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

MIN_RETURNS = 120
DEFAULT_LEVEL = 0.95


# ── Expected shortfall ───────────────────────────────────────────────────────
@dataclass
class TailRisk:
    """VaR and CVaR at one confidence level, over one holding period."""
    level: float
    horizon_days: int
    var_pct: float                  # negative: the loss threshold
    cvar_pct: float                 # negative: mean loss beyond it
    mc_cvar_pct: float
    mc_cvar_low: float              # bootstrap interval on the CVaR estimate
    mc_cvar_high: float
    worst_observed_pct: float
    n_obs: int
    n_tail: int                     # observations the historical CVaR averages

    def dollars(self, position_value: float) -> tuple[float, float]:
        """(VaR, CVaR) in currency for a position of ``position_value``."""
        return (position_value * self.var_pct / 100.0,
                position_value * self.cvar_pct / 100.0)

    def reading(self) -> str:
        days = ("a single day" if self.horizon_days == 1
                else f"{self.horizon_days} trading days")
        return (
            f"Over {days}, {self.level * 100:.0f}% of outcomes were better than "
            f"{self.var_pct:.2f}% (that is VaR). Beyond it, the average outcome "
            f"was **{self.cvar_pct:.2f}%** — the expected shortfall, taken "
            f"across {self.n_tail} tail observations out of {self.n_obs:,}. "
            f"Bootstrap resampling puts that estimate between "
            f"{self.mc_cvar_low:.2f}% and {self.mc_cvar_high:.2f}%. The worst "
            f"period actually observed was {self.worst_observed_pct:.2f}%. "
            "Measured on history — a distribution can always produce something "
            "it has not produced before."
        )


def _horizon_returns(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Compound daily returns into non-overlapping ``horizon``-day blocks.

    Non-overlapping because overlapping windows share days and would make the
    tail look better-sampled than it is.
    """
    if horizon <= 1:
        return returns
    usable = len(returns) - (len(returns) % horizon)
    if usable < horizon:
        return np.array([])
    blocks = returns[:usable].reshape(-1, horizon)
    return np.prod(1.0 + blocks, axis=1) - 1.0


def expected_shortfall(prices, level: float = DEFAULT_LEVEL,
                       horizon_days: int = 1, n_sims: int = 20_000,
                       seed: int = 7) -> Optional[TailRisk]:
    """
    Historical and bootstrap-Monte-Carlo CVaR from a price series.

    Returns ``None`` when there is too little history for a tail to mean
    anything — an expected shortfall computed from three observations is a
    number without a distribution behind it.
    """
    if prices is None:
        return None
    s = pd.Series(prices).dropna().astype(float)
    if len(s) < MIN_RETURNS:
        return None
    rets = s.pct_change().dropna().to_numpy()
    rets = rets[np.isfinite(rets)]
    if len(rets) < MIN_RETURNS:
        return None

    horizon_days = max(1, int(horizon_days))
    block = _horizon_returns(rets, horizon_days)
    if len(block) < 20:
        return None

    q = 1.0 - level
    var = float(np.quantile(block, q))
    tail = block[block <= var]
    if tail.size == 0:
        tail = np.array([var])
    cvar = float(tail.mean())

    # Bootstrap: resample daily returns, recompound, re-measure the tail. This
    # puts an interval on the CVaR estimate rather than quoting a point from
    # whatever handful of days happen to be in this sample's tail.
    rng = np.random.default_rng(seed)
    draws = rng.choice(rets, size=(n_sims, horizon_days), replace=True)
    sim = np.prod(1.0 + draws, axis=1) - 1.0
    sim_var = np.quantile(sim, q)
    sim_tail = sim[sim <= sim_var]
    mc_cvar = float(sim_tail.mean()) if sim_tail.size else float(sim_var)

    # Interval from resampling the simulated tail itself.
    boot = []
    for _ in range(200):
        pick = rng.choice(sim_tail if sim_tail.size else sim,
                          size=max(sim_tail.size, 1), replace=True)
        boot.append(pick.mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])

    return TailRisk(
        level=float(level), horizon_days=horizon_days,
        var_pct=var * 100.0, cvar_pct=cvar * 100.0,
        mc_cvar_pct=mc_cvar * 100.0,
        mc_cvar_low=float(lo) * 100.0, mc_cvar_high=float(hi) * 100.0,
        worst_observed_pct=float(block.min()) * 100.0,
        n_obs=int(len(block)), n_tail=int(tail.size),
    )


# ── Ownership concentration ──────────────────────────────────────────────────
@dataclass
class Concentration:
    """HHI over the *disclosed* holders, with its limits stated."""
    hhi_disclosed_block: float      # 0-10000, within the disclosed block
    hhi_company_lower_bound: float  # 0-10000, company-wide LOWER bound
    effective_holders: float        # equal-sized-holder equivalent in the block
    disclosed_pct: float            # share of the company the block represents
    top_holder_pct: Optional[float]
    top_holder: str
    n_holders: int
    concentration_label: str
    notes: list = field(default_factory=list)

    def reading(self) -> str:
        base = (
            f"The {self.n_holders} largest disclosed holders own "
            f"{self.disclosed_pct:.1f}% of the company. Within that block, "
            f"concentration is equivalent to about "
            f"{self.effective_holders:.1f} equally-sized holders "
            f"(HHI {self.hhi_disclosed_block:,.0f}) — **"
            f"{self.concentration_label.lower()}**."
        )
        if self.top_holder_pct is not None:
            base += (f" The largest single holder, {self.top_holder}, holds "
                     f"{self.top_holder_pct:.1f}%.")
        base += (" 13F data is a quarter stale by construction, and only the "
                 "largest holders are published, so the company-wide figure "
                 f"({self.hhi_company_lower_bound:,.0f}) is a lower bound: "
                 "true concentration can only be higher, never lower.")
        return base


def _label(effective: float) -> str:
    """Interpretation calibrated to ownership, not borrowed from antitrust."""
    if effective <= 3:
        return "Highly concentrated"
    if effective <= 6:
        return "Concentrated"
    if effective <= 12:
        return "Moderate"
    return "Dispersed"


def ownership_hhi(holders: Sequence[dict]) -> Optional[Concentration]:
    """
    Concentration of a 13F holder list.

    ``holders`` is the shape :mod:`svp.data.insider` produces: dicts with
    ``holder`` and ``pct_out`` (percent of shares outstanding). Entries
    without a usable percentage are dropped and counted in ``notes`` rather
    than treated as zero, which would understate concentration.
    """
    if not holders:
        return None
    notes: list = []
    pairs = []
    for h in holders:
        pct = h.get("pct_out")
        name = str(h.get("holder") or "").strip() or "unnamed holder"
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(pct) or pct <= 0:
            continue
        # Some feeds publish fractions (0.084) and some percentages (8.4).
        pairs.append((name, pct * 100.0 if pct <= 1.0 else pct))

    dropped = len(holders) - len(pairs)
    if dropped:
        notes.append(f"{dropped} of {len(holders)} disclosed holders had no "
                     "usable ownership percentage and were left out.")
    if not pairs:
        return None

    pairs.sort(key=lambda kv: -kv[1])
    pcts = np.array([p for _, p in pairs], dtype=float)
    disclosed = float(pcts.sum())
    if disclosed <= 0:
        return None
    if disclosed > 100.5:
        notes.append("Disclosed percentages sum above 100%, which means the "
                     "feed is double-counting or mixing share classes; treat "
                     "the figures as indicative.")

    # Company-wide: squares of raw percentages — a lower bound, since every
    # undisclosed holder would only add to the sum.
    hhi_lower = float((pcts ** 2).sum())
    # Within-block: normalise so the disclosed holders sum to 100%.
    shares = pcts / disclosed * 100.0
    hhi_block = float((shares ** 2).sum())
    effective = 10_000.0 / hhi_block if hhi_block > 0 else float("nan")

    return Concentration(
        hhi_disclosed_block=hhi_block,
        hhi_company_lower_bound=hhi_lower,
        effective_holders=float(effective),
        disclosed_pct=disclosed,
        top_holder_pct=float(pcts[0]),
        top_holder=pairs[0][0],
        n_holders=len(pairs),
        concentration_label=_label(effective),
        notes=notes,
    )
