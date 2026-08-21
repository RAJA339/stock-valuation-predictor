"""
Gamma exposure — where dealer hedging pushes and where it pulls.
================================================================

Market makers are on the other side of retail option flow, and they hedge
that exposure in the underlying. The direction of the hedge depends on the
sign of their gamma, which is why the aggregate matters:

- **Dealers long gamma** — they hedge *against* the move: selling into
  strength, buying weakness. Volatility gets suppressed, and price tends to
  pin near heavy strikes into expiry.
- **Dealers short gamma** — they hedge *with* the move: selling weakness,
  buying strength. Moves are amplified, and this is the arrangement behind
  the phrase "gamma squeeze".

The **zero-gamma flip** (volatility trigger) is where net exposure crosses
from one regime to the other, which is why it is the level worth knowing.

Convention, stated because conventions differ and an unstated one is a lie
of omission: this uses the standard dealer-side assumption that dealers are
**long calls and short puts** relative to customer flow, so call gamma enters
positive and put gamma negative. Per strike,

    ``GEX = Γ × OI × 100 × S² × 0.01``

which expresses exposure as the dollar change in dealer delta per 1% move in
spot. Everything is derived from the published chain — strikes, open interest
and implied volatility — and any strike missing what it needs is dropped and
counted rather than filled with a guess.

This is an estimate built on an assumption about who holds what. Real dealer
inventory is not public, and the module says so wherever it reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

CONTRACT_MULTIPLIER = 100
MIN_STRIKES = 4


@dataclass
class StrikeGamma:
    strike: float
    call_oi: float
    put_oi: float
    call_gex: float
    put_gex: float

    @property
    def net_gex(self) -> float:
        return self.call_gex + self.put_gex


@dataclass
class GexProfile:
    spot: float
    expiry: str
    strikes: list[StrikeGamma] = field(default_factory=list)
    total_gex: float = 0.0
    call_gex: float = 0.0
    put_gex: float = 0.0
    zero_gamma: Optional[float] = None
    call_wall: Optional[float] = None
    put_wall: Optional[float] = None
    dropped: int = 0
    source: str = ""

    @property
    def is_positive_gamma(self) -> bool:
        return self.total_gex > 0

    @property
    def regime(self) -> str:
        return ("Positive gamma — hedging dampens moves"
                if self.is_positive_gamma
                else "Negative gamma — hedging amplifies moves")

    @property
    def badge_class(self) -> str:
        return "signal-buy" if self.is_positive_gamma else "signal-sell"

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Strike": s.strike,
            "Call OI": s.call_oi,
            "Put OI": s.put_oi,
            "Call GEX": s.call_gex,
            "Put GEX": s.put_gex,
            "Net GEX": s.net_gex,
        } for s in self.strikes])

    def reading(self) -> str:
        bits = [
            f"Net gamma exposure is **${self.total_gex / 1e6:,.1f}M** per 1% "
            f"move — {self.regime.lower()}."
        ]
        if self.is_positive_gamma:
            bits.append(
                "With dealers net long gamma, hedging leans against price: "
                "they sell strength and buy weakness, which suppresses "
                "realised volatility and tends to pin price near heavy "
                "strikes into expiry.")
        else:
            bits.append(
                "With dealers net short gamma, hedging leans with price: they "
                "sell weakness and buy strength, which amplifies moves. This "
                "is the arrangement that produces squeezes in both "
                "directions.")
        if self.zero_gamma is not None:
            side = "above" if self.spot > self.zero_gamma else "below"
            bits.append(
                f"The volatility trigger — where net exposure flips sign — "
                f"sits at {self.zero_gamma:,.2f}, and spot is {side} it. "
                "Crossing that level is where the hedging regime changes.")
        if self.call_wall is not None:
            bits.append(f"Heaviest call open interest sits at "
                        f"{self.call_wall:,.2f}")
            if self.put_wall is not None:
                bits.append(f"and the put wall at {self.put_wall:,.2f}; these "
                            "are the strikes with most to hedge around.")
        bits.append(
            "Dealer inventory is not published — this assumes the standard "
            "long-call / short-put dealer position, so it is an estimate of "
            "positioning rather than a measurement of it.")
        return " ".join(bits)


def bs_gamma(S: float, K: float, T: float, sigma: float,
             r: float = 0.045, q: float = 0.0) -> Optional[float]:
    """
    Black-Scholes gamma. ``None`` when the inputs cannot support one.

    Gamma is identical for calls and puts at the same strike, which is why a
    single function serves both sides of the chain.
    """
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None
        d1 = ((math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T)
              / (sigma * math.sqrt(T)))
        pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        g = math.exp(-q * T) * pdf / (S * sigma * math.sqrt(T))
        return float(g) if math.isfinite(g) else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _num(row, *names) -> Optional[float]:
    for n in names:
        if n in row and pd.notna(row[n]):
            try:
                v = float(row[n])
                if math.isfinite(v):
                    return v
            except (TypeError, ValueError):
                continue
    return None


def _side_map(df: Optional[pd.DataFrame], spot: float, T: float,
              r: float) -> tuple[dict, int]:
    """Strike → (open interest, gamma) for one side of the chain."""
    out: dict = {}
    dropped = 0
    if df is None or len(df) == 0:
        return out, 0
    for _, row in df.iterrows():
        k = _num(row, "strike", "Strike")
        oi = _num(row, "openInterest", "open_interest", "OI")
        iv = _num(row, "impliedVolatility", "iv", "IV")
        if k is None or oi is None or oi <= 0 or iv is None or iv <= 0:
            dropped += 1
            continue
        # Some feeds publish IV as a percentage rather than a fraction.
        if iv > 3.0:
            iv = iv / 100.0
        g = bs_gamma(spot, k, T, iv, r=r)
        if g is None:
            dropped += 1
            continue
        prev_oi, prev_g = out.get(k, (0.0, 0.0))
        out[k] = (prev_oi + oi, g if prev_g == 0.0 else (prev_g + g) / 2)
    return out, dropped


def _zero_gamma(strikes: list[StrikeGamma]) -> Optional[float]:
    """
    Where cumulative net gamma crosses zero, by linear interpolation.

    Cumulative rather than per-strike: the flip is the level at which the
    *aggregate* changes sign, which is what governs hedging behaviour.
    """
    if len(strikes) < 2:
        return None
    ks = np.array([s.strike for s in strikes], dtype=float)
    cum = np.cumsum([s.net_gex for s in strikes])
    for i in range(1, len(cum)):
        if cum[i - 1] == 0:
            return float(ks[i - 1])
        if (cum[i - 1] < 0) != (cum[i] < 0):
            span = cum[i] - cum[i - 1]
            if span == 0:
                return float(ks[i])
            frac = -cum[i - 1] / span
            return float(ks[i - 1] + frac * (ks[i] - ks[i - 1]))
    return None


def compute(chain, days_to_expiry: float, r: float = 0.045
            ) -> Optional[GexProfile]:
    """
    Build a :class:`GexProfile` from an option chain.

    ``chain`` is the shape :mod:`svp.data.options` returns: ``spot``,
    ``calls``/``puts`` frames and an ``expiry``. Returns ``None`` when too few
    strikes survive validation for a profile to mean anything.
    """
    if chain is None:
        return None
    spot = float(getattr(chain, "spot", 0.0) or 0.0)
    if spot <= 0:
        return None
    T = max(float(days_to_expiry), 0.5) / 365.0

    calls, d1 = _side_map(getattr(chain, "calls", None), spot, T, r)
    puts, d2 = _side_map(getattr(chain, "puts", None), spot, T, r)
    all_k = sorted(set(calls) | set(puts))
    if len(all_k) < MIN_STRIKES:
        return None

    scale = CONTRACT_MULTIPLIER * spot * spot * 0.01
    rows: list[StrikeGamma] = []
    for k in all_k:
        c_oi, c_g = calls.get(k, (0.0, 0.0))
        p_oi, p_g = puts.get(k, (0.0, 0.0))
        rows.append(StrikeGamma(
            strike=float(k), call_oi=float(c_oi), put_oi=float(p_oi),
            # Dealers long calls, short puts: calls add, puts subtract.
            call_gex=float(c_g * c_oi * scale),
            put_gex=float(-p_g * p_oi * scale),
        ))

    call_total = sum(s.call_gex for s in rows)
    put_total = sum(s.put_gex for s in rows)
    call_wall = max(rows, key=lambda s: s.call_oi).strike if rows else None
    put_wall = max(rows, key=lambda s: s.put_oi).strike if rows else None

    return GexProfile(
        spot=spot, expiry=str(getattr(chain, "expiry", "")),
        strikes=rows, total_gex=float(call_total + put_total),
        call_gex=float(call_total), put_gex=float(put_total),
        zero_gamma=_zero_gamma(rows),
        call_wall=call_wall if call_wall else None,
        put_wall=put_wall if put_wall else None,
        dropped=int(d1 + d2), source=str(getattr(chain, "source", "")),
    )
