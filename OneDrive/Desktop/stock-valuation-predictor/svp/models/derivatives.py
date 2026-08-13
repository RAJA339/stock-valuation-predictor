"""
Derivatives pricing: options & futures.
=======================================

Self-contained pricing engines (only NumPy/SciPy — no QuantLib dependency):

  * **Black-Scholes-Merton** European option value + full Greeks
    (Δ, Γ, Θ, Vega, Rho), with continuous dividend yield.
  * **Binomial (Cox-Ross-Rubinstein)** tree — supports American exercise.
  * **Monte-Carlo** GBM pricer (European) as a cross-check.
  * **Implied volatility** solver (Brent) from a market premium.
  * **Cost-of-carry futures** fair value ``F = S·e^{(r+s-c)T}``.
  * **Valuation bridge** — treats the app's ML intrinsic target / DCF fair value
    as the projected underlying at expiry (S_T) to flag mispriced LEAPs and
    score covered-call setups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.stats import norm
    from scipy.optimize import brentq

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# ── Normal helpers (fallback if SciPy missing) ───────────────────────────────
def _N(x: float) -> float:
    if _HAS_SCIPY:
        return float(norm.cdf(x))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _n(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass
class OptionResult:
    price: float
    delta: float
    gamma: float
    theta: float          # per calendar day
    vega: float           # per 1 vol point (0.01)
    rho: float            # per 1% rate
    d1: float
    d2: float
    kind: str             # "call" | "put"


def _d1_d2(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return float("inf"), float("inf")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes(S, K, T, r, sigma, q=0.0, kind="call") -> OptionResult:
    """European Black-Scholes-Merton price + Greeks. Dividend yield ``q``."""
    kind = kind.lower()
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, (S - K) if kind == "call" else (K - S))
        return OptionResult(intrinsic, 1.0 if (kind == "call" and S > K) else 0.0,
                            0.0, 0.0, 0.0, 0.0, float("inf"), float("inf"), kind)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if kind == "call":
        price = S * disc_q * _N(d1) - K * disc_r * _N(d2)
        delta = disc_q * _N(d1)
        theta = (-(S * disc_q * _n(d1) * sigma) / (2 * math.sqrt(T))
                 - r * K * disc_r * _N(d2) + q * S * disc_q * _N(d1))
        rho = K * T * disc_r * _N(d2)
    else:
        price = K * disc_r * _N(-d2) - S * disc_q * _N(-d1)
        delta = -disc_q * _N(-d1)
        theta = (-(S * disc_q * _n(d1) * sigma) / (2 * math.sqrt(T))
                 + r * K * disc_r * _N(-d2) - q * S * disc_q * _N(-d1))
        rho = -K * T * disc_r * _N(-d2)

    gamma = disc_q * _n(d1) / (S * sigma * math.sqrt(T))
    vega = S * disc_q * _n(d1) * math.sqrt(T)

    return OptionResult(
        price=float(price),
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta / 365.0),   # per calendar day
        vega=float(vega / 100.0),     # per 1 vol point
        rho=float(rho / 100.0),       # per 1% rate
        d1=float(d1), d2=float(d2), kind=kind,
    )


def binomial_price(S, K, T, r, sigma, q=0.0, kind="call", steps=200, american=True) -> float:
    """Cox-Ross-Rubinstein binomial price (supports American early exercise)."""
    if T <= 0 or sigma <= 0 or steps < 1:
        return max(0.0, (S - K) if kind == "call" else (K - S))
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)
    disc = math.exp(-r * dt)

    # Terminal payoffs.
    j = np.arange(steps + 1)
    ST = S * (u ** j) * (d ** (steps - j))
    if kind == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    for i in range(steps, 0, -1):
        values = disc * (p * values[1:i + 1] + (1 - p) * values[0:i])
        if american:
            Si = S * (u ** np.arange(i)) * (d ** (i - 1 - np.arange(i)))
            exercise = (Si - K) if kind == "call" else (K - Si)
            values = np.maximum(values, np.maximum(exercise, 0.0))
    return float(values[0])


def monte_carlo_price(S, K, T, r, sigma, q=0.0, kind="call", n=50000, seed=7) -> dict:
    """European option price via GBM Monte-Carlo (terminal sampling)."""
    if T <= 0 or sigma <= 0:
        return {"price": max(0.0, (S - K) if kind == "call" else (K - S)), "stderr": 0.0}
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    disc = math.exp(-r * T)
    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / math.sqrt(n)
    return {"price": float(price), "stderr": float(stderr)}


def implied_volatility(market_price, S, K, T, r, q=0.0, kind="call") -> Optional[float]:
    """Solve for implied volatility from a market premium (Brent)."""
    intrinsic = max(0.0, (S - K) if kind == "call" else (K - S))
    if market_price <= intrinsic or T <= 0:
        return None
    f = lambda sig: black_scholes(S, K, T, r, sig, q, kind).price - market_price
    try:
        if _HAS_SCIPY:
            return float(brentq(f, 1e-4, 6.0, maxiter=200))
        # Bisection fallback.
        lo, hi = 1e-4, 6.0
        if f(lo) * f(hi) > 0:
            return None
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        return float((lo + hi) / 2)
    except Exception:
        return None


# ── Futures: cost-of-carry ────────────────────────────────────────────────────
@dataclass
class FuturesResult:
    fair_value: float
    basis: float                 # fair_value - spot
    annualized_carry: float      # r + s - c

    def mispricing(self, market_future: float) -> float:
        return market_future - self.fair_value


def futures_fair_value(spot, r, T, storage_cost=0.0, convenience_yield=0.0) -> FuturesResult:
    """Cost-of-carry futures fair value: F = S · e^{(r + s - c)·T}."""
    carry = r + storage_cost - convenience_yield
    F = spot * math.exp(carry * T)
    return FuturesResult(float(F), float(F - spot), float(carry))


# ── Valuation → options bridge ────────────────────────────────────────────────
@dataclass
class OptionEdge:
    strike: float
    expiry_T: float
    market_price: float
    model_iv: Optional[float]
    bs_price_market_iv: Optional[float]
    prob_itm: float             # risk-neutral-ish P(S_T > K) using target as drift anchor
    expected_payoff: float      # using projected S_T (ML/DCF target)
    edge: float                 # expected_payoff_pv - market_price
    verdict: str


def bridge_call_edge(
    projected_ST: float,
    strike: float,
    T: float,
    r: float,
    market_price: float,
    sigma: float,
    spot: float,
    kind: str = "call",
) -> OptionEdge:
    """
    Score an option using the app's ML/DCF **projected underlying at expiry**
    (``projected_ST``) as the central estimate of S_T.

    ``edge`` compares the present value of the expected intrinsic payoff at that
    target against the current market premium — positive edge flags a
    potentially mispriced LEAP.
    """
    disc = math.exp(-r * T)
    # Expected terminal payoff at the projected target (point estimate).
    if kind == "call":
        payoff = max(projected_ST - strike, 0.0)
    else:
        payoff = max(strike - projected_ST, 0.0)
    expected_pv = disc * payoff

    # Rough probability ITM from a lognormal centered on the projected target.
    prob_itm = float("nan")
    if sigma and T > 0 and spot > 0:
        mu = math.log(projected_ST) - 0.5 * sigma ** 2 * T
        d = (mu - math.log(strike)) / (sigma * math.sqrt(T))
        prob_itm = _N(d) if kind == "call" else _N(-d)

    iv = implied_volatility(market_price, spot, strike, T, r, kind=kind) if market_price > 0 else None
    bs_market = black_scholes(spot, strike, T, r, sigma, kind=kind).price

    edge = expected_pv - market_price
    if edge > 0.1 * max(market_price, 1e-6):
        verdict = "Underpriced vs target"
    elif edge < -0.1 * max(market_price, 1e-6):
        verdict = "Overpriced vs target"
    else:
        verdict = "Fairly priced"

    return OptionEdge(
        strike=strike, expiry_T=T, market_price=market_price,
        model_iv=iv, bs_price_market_iv=bs_market,
        prob_itm=prob_itm, expected_payoff=payoff, edge=float(edge), verdict=verdict,
    )
