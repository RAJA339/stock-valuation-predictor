"""
Traditional Discounted-Cash-Flow model + Monte-Carlo scenario engine.
=====================================================================

A textbook two-stage DCF (explicit FCF projection + Gordon-growth terminal
value) driven by user-adjustable **WACC** and **terminal growth** assumptions.
The Monte-Carlo variant randomises the key assumptions to produce a fair-value
distribution that complements the ML valuation range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class DCFInputs:
    fcf0: float                 # most recent free cash flow
    shares: float
    net_debt: float = 0.0
    wacc: float = 0.09          # discount rate
    terminal_growth: float = 0.025
    growth_rate: float = 0.08   # near-term FCF growth
    years: int = 5


@dataclass
class DCFResult:
    intrinsic_per_share: float
    enterprise_value: float
    equity_value: float
    pv_explicit: float
    pv_terminal: float
    projected_fcf: list = field(default_factory=list)
    discounted_fcf: list = field(default_factory=list)


def run_dcf(inp: DCFInputs) -> DCFResult:
    """Run a deterministic two-stage DCF."""
    if inp.wacc <= inp.terminal_growth:
        # Guard the Gordon-growth singularity.
        inp = DCFInputs(**{**inp.__dict__, "wacc": inp.terminal_growth + 0.01})

    projected, discounted = [], []
    fcf = inp.fcf0
    for t in range(1, inp.years + 1):
        fcf = fcf * (1 + inp.growth_rate)
        pv = fcf / (1 + inp.wacc) ** t
        projected.append(fcf)
        discounted.append(pv)

    terminal_fcf = projected[-1] * (1 + inp.terminal_growth)
    terminal_value = terminal_fcf / (inp.wacc - inp.terminal_growth)
    pv_terminal = terminal_value / (1 + inp.wacc) ** inp.years

    pv_explicit = float(np.sum(discounted))
    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value - inp.net_debt
    per_share = equity_value / inp.shares if inp.shares else float("nan")

    return DCFResult(
        intrinsic_per_share=float(per_share),
        enterprise_value=float(enterprise_value),
        equity_value=float(equity_value),
        pv_explicit=pv_explicit,
        pv_terminal=float(pv_terminal),
        projected_fcf=projected,
        discounted_fcf=discounted,
    )


def monte_carlo_dcf(
    inp: DCFInputs,
    n: int = 5000,
    wacc_sd: float = 0.015,
    growth_sd: float = 0.02,
    terminal_sd: float = 0.005,
    seed: int = 3,
) -> dict:
    """
    Randomise WACC, near-term growth and terminal growth to produce a
    per-share fair-value distribution. Returns percentile summary + samples.
    """
    rng = np.random.default_rng(seed)
    waccs = np.clip(rng.normal(inp.wacc, wacc_sd, n), 0.03, 0.30)
    growths = rng.normal(inp.growth_rate, growth_sd, n)
    terminals = np.clip(rng.normal(inp.terminal_growth, terminal_sd, n), 0.0, 0.05)

    out = np.empty(n)
    for i in range(n):
        w = max(waccs[i], terminals[i] + 0.01)
        res = run_dcf(
            DCFInputs(
                fcf0=inp.fcf0, shares=inp.shares, net_debt=inp.net_debt,
                wacc=w, terminal_growth=terminals[i], growth_rate=growths[i], years=inp.years,
            )
        )
        out[i] = res.intrinsic_per_share

    out = out[np.isfinite(out)]
    return {
        "samples": out,
        "mean": float(np.mean(out)),
        "median": float(np.median(out)),
        "p10": float(np.percentile(out, 10)),
        "p90": float(np.percentile(out, 90)),
        "std": float(np.std(out)),
    }


# ── Rate sensitivity ─────────────────────────────────────────────────────────
@dataclass
class RateShock:
    """One point on the fair-value / discount-rate curve."""
    shock_bp: int               # parallel move in the discount rate, basis points
    wacc: float
    intrinsic_per_share: float
    change_pct: float           # vs the unshocked case


def rate_shock(inp: DCFInputs,
               shocks_bp: Sequence[int] = (-200, -100, -50, 0, 50, 100, 200)) -> list[RateShock]:
    """
    Re-run the DCF across a parallel shift in the discount rate.

    "What happens to fair value if the ten-year moves a hundred basis points" is
    the question every desk runs, and a DCF quoted at a single WACC cannot
    answer it. The sensitivity is also the honest way to read a DCF: the output
    is far more sensitive to the discount rate than to the growth assumptions
    people spend their time arguing about, and showing the curve makes that
    impossible to miss.

    Shocks are applied to the discount rate only. Terminal growth is deliberately
    held fixed: in reality a rate move is correlated with growth expectations,
    but modelling that link would bury a large assumption inside what is meant to
    be a sensitivity check on one variable.
    """
    base = run_dcf(inp).intrinsic_per_share
    out: list[RateShock] = []
    for bp in shocks_bp:
        shocked = DCFInputs(**{**inp.__dict__, "wacc": inp.wacc + bp / 10_000.0})
        v = run_dcf(shocked).intrinsic_per_share
        change = ((v - base) / base * 100.0) if base and math.isfinite(base) and base != 0 else float("nan")
        out.append(RateShock(shock_bp=int(bp), wacc=shocked.wacc,
                             intrinsic_per_share=float(v), change_pct=float(change)))
    return out


def duration_estimate(shocks: Sequence[RateShock]) -> Optional[float]:
    """
    Percentage change in fair value per 100bp of discount rate, near the base.

    Reported as a positive number for a value that falls when rates rise, which
    is the ordinary direction. Equity duration is not a standard disclosure, so
    the figure is presented as what it is: the local slope of the curve above,
    measured between the +/-100bp points rather than differentiated.
    """
    by_bp = {s.shock_bp: s.intrinsic_per_share for s in shocks}
    lo, hi = by_bp.get(-100), by_bp.get(100)
    base = by_bp.get(0)
    if not base or lo is None or hi is None or not math.isfinite(base):
        return None
    return float((lo - hi) / (2 * base) * 100.0)
