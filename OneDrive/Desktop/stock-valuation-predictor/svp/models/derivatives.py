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

    def f(sig):
        return black_scholes(S, K, T, r, sig, q, kind).price - market_price

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
    prob_itm: float             # P(S_T beyond K) under the assumed terminal law
    expected_payoff: float      # undiscounted expected intrinsic at expiry
    edge: float                 # PV(expected payoff) - market_price
    verdict: str
    method: str = "point"       # "point" | "monte-carlo"
    expected_payoff_pv: float = float("nan")
    payoff_p50: float = float("nan")
    payoff_p90: float = float("nan")
    risk_neutral_price: float = float("nan")   # BS value at the market's own IV
    view_premium: float = float("nan")         # model PV - risk-neutral PV


def terminal_distribution(
    spot: float,
    value_samples,
    T: float,
    sigma: float,
    convergence: float = 1.0,
    seed: int = 11,
) -> "np.ndarray":
    """
    Build a distribution of the underlying at expiry from the valuation model.

    Each Monte-Carlo intrinsic-value draw ``V_i`` defines an anchor

        M_i = spot + convergence * (V_i - spot)

    i.e. how much of the gap between today's price and that fair-value draw is
    assumed to close by expiry (``convergence=1`` means fully closed, ``0``
    means the market never re-rates). Lognormal diffusion is then applied
    around each anchor so that ``E[S_T | M_i] = M_i``:

        S_T = M_i * exp(-0.5*sigma^2*T + sigma*sqrt(T)*Z)

    The result carries both sources of uncertainty the point estimate dropped:
    the model's uncertainty about fair value, and the price's own diffusion.

    Note this is a **real-world** distribution built from your valuation view,
    not the risk-neutral law that sets option prices. Any edge computed against
    it is a statement about your view being right, never an arbitrage.
    """
    v = np.asarray(value_samples, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or spot <= 0 or T <= 0:
        return np.array([], dtype=float)

    anchors = spot + float(convergence) * (v - spot)
    anchors = np.clip(anchors, 1e-6, None)

    if sigma is None or sigma <= 0:
        return anchors

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(anchors.size)
    return anchors * np.exp(-0.5 * sigma ** 2 * T + sigma * math.sqrt(T) * z)


def bridge_edge_mc(
    terminal: "np.ndarray",
    strike: float,
    T: float,
    r: float,
    market_price: float,
    spot: float,
    kind: str = "call",
    market_iv: Optional[float] = None,
) -> OptionEdge:
    """
    Score an option by integrating its payoff over the projected terminal
    distribution, rather than evaluating it at a single target price.

    ``edge`` is the present value of the expected intrinsic payoff minus the
    market premium. ``view_premium`` isolates how much of that edge comes from
    the valuation view rather than from volatility: it compares the model PV
    against the Black-Scholes value at the market's own implied volatility, so
    a large edge with a near-zero view premium means the model is simply
    disagreeing about vol, not about direction.
    """
    terminal = np.asarray(terminal, dtype=float)
    if terminal.size == 0 or T <= 0:
        return bridge_call_edge(spot, strike, T, r, market_price,
                                market_iv or 0.0, spot, kind)

    disc = math.exp(-r * T)
    if kind == "call":
        payoff = np.maximum(terminal - strike, 0.0)
        prob_itm = float((terminal > strike).mean())
    else:
        payoff = np.maximum(strike - terminal, 0.0)
        prob_itm = float((terminal < strike).mean())

    expected_payoff = float(payoff.mean())
    expected_pv = disc * expected_payoff
    edge = expected_pv - market_price

    rn_price = float("nan")
    view_premium = float("nan")
    if market_iv and market_iv > 0:
        rn_price = black_scholes(spot, strike, T, r, market_iv, kind=kind).price
        view_premium = expected_pv - rn_price

    tol = 0.1 * max(market_price, 1e-6)
    if edge > tol:
        verdict = "Underpriced vs view"
    elif edge < -tol:
        verdict = "Overpriced vs view"
    else:
        verdict = "Fairly priced"

    return OptionEdge(
        strike=strike, expiry_T=T, market_price=market_price,
        model_iv=market_iv, bs_price_market_iv=rn_price,
        prob_itm=prob_itm, expected_payoff=expected_payoff,
        edge=float(edge), verdict=verdict, method="monte-carlo",
        expected_payoff_pv=float(expected_pv),
        payoff_p50=float(np.percentile(payoff, 50)),
        payoff_p90=float(np.percentile(payoff, 90)),
        risk_neutral_price=rn_price, view_premium=view_premium,
    )


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
    Point-estimate scorer — superseded by :func:`bridge_edge_mc`.

    Evaluates the payoff at a single projected ``S_T`` instead of integrating
    over its distribution, which carries two errors pulling in opposite
    directions: by Jensen's inequality ``max(E[S]-K, 0) <= E[max(S-K, 0)]``, so
    a point evaluation understates a convex payoff; and treating a directional
    target as the expected terminal price overstates it whenever the target
    sits above spot. Retained for the deterministic fallback in
    ``bridge_edge_mc`` and for comparison.
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
