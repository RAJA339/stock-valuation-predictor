"""
Crypto futures — perpetual funding and basis.
=============================================

Crypto "futures" are mostly perpetual swaps: contracts with no expiry that hold
their peg to spot through a **funding rate** exchanged between longs and shorts,
typically every eight hours. That funding rate is the single most informative
futures number on a coin — positive funding means longs are paying to stay long,
which is crowded-bullish and a squeeze risk; negative funding is the reverse. It
is the crypto analogue of the equity cost-of-carry the app already computes, and
it is what this module surfaces.

Two things are provided, at two levels of trust:

- **Live funding** from Binance's public futures endpoint, behind the same
  circuit-breaker discipline as the options feed: one failure trips a cooldown so
  a rate-limited deployment stops hammering the API. It degrades to ``None``, and
  the caller shows the basis calculator instead of a wrong number.

- **Basis / annualised carry** from spot and an observed futures price, pure and
  always available. For a dated future this is ``(F/S − 1)`` annualised by days
  to expiry; for a perpetual the same arithmetic on the mark-vs-index gap gives
  the implied carry the funding rate is enforcing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from . import storage

_PREMIUM_URL = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol={sym}"
_HEADERS = {"User-Agent": "StockValuationPredictor/1.0 (educational)"}
_TTL = 120.0                       # funding barely moves within a funding window

# Circuit breaker — mirrors svp.data.options. A rate-limit trips a long cooldown
# so shared-IP deployments (Streamlit Cloud) do not keep getting throttled.
_COOLDOWN_UNTIL = 0.0
_COOLDOWN_REASON = ""
_RATE_LIMIT_COOLDOWN = 900.0
_ERROR_COOLDOWN = 120.0


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "418" in s or "rate" in s or "too many" in s


def _trip(exc: Exception) -> None:
    global _COOLDOWN_UNTIL, _COOLDOWN_REASON
    wait = _RATE_LIMIT_COOLDOWN if _is_rate_limit(exc) else _ERROR_COOLDOWN
    _COOLDOWN_UNTIL = time.time() + wait
    _COOLDOWN_REASON = f"{type(exc).__name__}: {exc}"[:160]


def cooldown_remaining() -> float:
    return max(0.0, _COOLDOWN_UNTIL - time.time())


def reset_circuit() -> None:
    """Test hook: clear the breaker."""
    global _COOLDOWN_UNTIL, _COOLDOWN_REASON
    _COOLDOWN_UNTIL = 0.0
    _COOLDOWN_REASON = ""


@dataclass(frozen=True)
class Funding:
    symbol: str                    # e.g. "BTCUSDT"
    funding_rate: float            # per-interval rate, fraction (e.g. 0.0001)
    mark_price: Optional[float]
    index_price: Optional[float]
    interval_hours: int = 8

    @property
    def annualised(self) -> float:
        """Funding compounded over a year of intervals, as a fraction."""
        periods = 24 / self.interval_hours * 365
        return self.funding_rate * periods

    @property
    def basis_pct(self) -> Optional[float]:
        """Mark-vs-index gap the funding is enforcing, in percent."""
        if not self.mark_price or not self.index_price:
            return None
        return (self.mark_price - self.index_price) / self.index_price * 100.0

    @property
    def crowded_side(self) -> str:
        if self.funding_rate > 0.0005:
            return "Longs paying heavily — crowded long, squeeze risk to the downside"
        if self.funding_rate > 0:
            return "Longs paying — mild bullish lean"
        if self.funding_rate < -0.0005:
            return "Shorts paying heavily — crowded short, squeeze risk to the upside"
        if self.funding_rate < 0:
            return "Shorts paying — mild bearish lean"
        return "Funding flat — no crowded side"


def get_funding(binance_symbol: str) -> Optional[Funding]:
    """
    Live perpetual funding for a Binance USDT-perp symbol, or ``None``.

    ``None`` covers three cases the caller treats the same — the breaker is
    tripped, the network failed, or the payload was unusable — because in all of
    them the honest UI move is the same: show the basis calculator instead of a
    guessed funding number.
    """
    sym = binance_symbol.upper().strip()
    key = f"funding:{sym}"
    cached = storage.cache_get(key)
    if cached is not None:
        return Funding(**cached)

    if cooldown_remaining() > 0:
        return None

    try:
        import requests

        r = requests.get(_PREMIUM_URL.format(sym=sym), headers=_HEADERS, timeout=10)
        r.raise_for_status()
        d = r.json()
        f = Funding(
            symbol=sym,
            funding_rate=float(d["lastFundingRate"]),
            mark_price=float(d.get("markPrice")) if d.get("markPrice") else None,
            index_price=float(d.get("indexPrice")) if d.get("indexPrice") else None,
        )
        storage.cache_set(key, f.__dict__, ttl=_TTL)
        return f
    except Exception as exc:      # noqa: BLE001 — all failures degrade the same
        _trip(exc)
        return None


# ── Basis (always available) ─────────────────────────────────────────────────
@dataclass(frozen=True)
class Basis:
    spot: float
    futures: float
    days: float
    absolute: float               # futures - spot
    basis_pct: float              # (futures/spot - 1) * 100
    annualised_pct: float         # simple-annualised carry, percent

    @property
    def structure(self) -> str:
        if self.basis_pct > 0.05:
            return "Contango — futures above spot; the market pays to be long carry"
        if self.basis_pct < -0.05:
            return "Backwardation — futures below spot; often stress or heavy short demand"
        return "Flat — futures and spot essentially in line"


def basis(spot: float, futures: float, days: float) -> Optional[Basis]:
    """
    Annualised basis / carry between a spot and a futures price.

    ``days`` is calendar days to expiry; for a perpetual, pass the funding
    interval in days (8h ≈ 0.333) to express its mark-vs-index gap as an
    annualised carry on the same footing as a dated contract. Simple, not
    continuous, annualisation — the convention quoted on crypto desks.
    """
    if spot <= 0 or futures <= 0 or days <= 0:
        return None
    b_pct = (futures / spot - 1.0) * 100.0
    annualised = b_pct * (365.0 / days)
    return Basis(
        spot=float(spot), futures=float(futures), days=float(days),
        absolute=float(futures - spot), basis_pct=float(b_pct),
        annualised_pct=float(annualised),
    )
