"""
Traditional Discounted-Cash-Flow model + Monte-Carlo scenario engine.
=====================================================================

A textbook two-stage DCF (explicit FCF projection + Gordon-growth terminal
value) driven by user-adjustable **WACC** and **terminal growth** assumptions.
The Monte-Carlo variant randomises the key assumptions to produce a fair-value
distribution that complements the ML valuation range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
