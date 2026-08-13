"""
Live options-chain pipeline.
=============================

Pulls live strike prices, bids/asks, open interest and implied volatility via
``yfinance``'s ``Ticker.option_chain(date)``. Falls back to a deterministic
synthetic chain (Black-Scholes-priced around the current spot) so the Options
tab always has data to render, even offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore

    _HAS_YF = True
except Exception:  # pragma: no cover
    _HAS_YF = False


def _deriv():
    """
    Lazy handle on ``svp.models.derivatives``.

    Imported on first use rather than at module scope: ``svp.models`` pulls in
    ``svp.features``, which imports ``svp.data`` — a module-level import here
    would close that cycle and break ``import svp.features``.
    """
    from ..models import derivatives

    return derivatives


@dataclass
class OptionChain:
    ticker: str
    expiry: str
    spot: float
    calls: pd.DataFrame
    puts: pd.DataFrame
    source: str

    @property
    def is_live(self) -> bool:
        return self.source == "yfinance"


def list_expiries(ticker: str) -> list[str]:
    """Return available expiry date strings for a ticker."""
    if _HAS_YF:
        try:
            exps = yf.Ticker(ticker).options
            if exps:
                return list(exps)
        except Exception:
            pass
    # Synthetic fallback expiries: next 4 monthly-ish dates + a 1y LEAP.
    today = pd.Timestamp.today().normalize()
    out = [(today + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in (30, 60, 90, 180, 365)]
    return out


def _synthetic_chain(ticker: str, expiry: str, spot: float, r: float = 0.045) -> OptionChain:
    """Black-Scholes-priced synthetic chain so the tab always renders."""
    deriv = _deriv()
    T = max((pd.Timestamp(expiry) - pd.Timestamp.today().normalize()).days, 1) / 365.0
    rng = np.random.default_rng(abs(hash((ticker, expiry))) % (2**32))
    base_iv = float(np.clip(rng.normal(0.28, 0.05), 0.12, 0.75))
    strikes = np.round(spot * np.linspace(0.7, 1.3, 25) / 0.5) * 0.5

    def build(kind):
        rows = []
        for k in strikes:
            # Volatility smile: higher IV away from the money.
            moneyness = abs(math.log(max(k, 1e-6) / spot))
            iv = base_iv + 0.15 * moneyness
            bs = deriv.black_scholes(spot, float(k), T, r, iv, kind=kind)
            mid = max(bs.price, 0.01)
            spread = max(mid * 0.04, 0.02)
            oi = int(max(rng.integers(50, 5000) * math.exp(-3 * moneyness), 1))
            rows.append({
                "strike": float(k), "lastPrice": round(mid, 2),
                "bid": round(max(mid - spread, 0.0), 2), "ask": round(mid + spread, 2),
                "openInterest": oi, "volume": int(oi * rng.uniform(0.05, 0.4)),
                "impliedVolatility": round(iv, 4),
                "inTheMoney": (k < spot) if kind == "call" else (k > spot),
            })
        return pd.DataFrame(rows)

    return OptionChain(ticker, expiry, spot, build("call"), build("put"), source="synthetic")


def get_option_chain(ticker: str, expiry: Optional[str] = None, spot_fallback: float = 100.0) -> OptionChain:
    """Fetch the live option chain for ``ticker``/``expiry`` (or the nearest one)."""
    ticker = ticker.upper().strip()
    if _HAS_YF:
        try:
            tk = yf.Ticker(ticker)
            expiries = tk.options
            if expiries:
                exp = expiry if (expiry and expiry in expiries) else expiries[0]
                chain = tk.option_chain(exp)
                spot = spot_fallback
                try:
                    spot = float(tk.fast_info.get("lastPrice") or spot_fallback)
                except Exception:
                    pass
                return OptionChain(ticker, exp, spot, chain.calls, chain.puts, source="yfinance")
        except Exception:
            pass

    exp = expiry or list_expiries(ticker)[2]  # default ~90d
    return _synthetic_chain(ticker, exp, spot_fallback)
