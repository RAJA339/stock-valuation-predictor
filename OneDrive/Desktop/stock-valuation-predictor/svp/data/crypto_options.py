"""
Crypto options — volatility skew and dealer gamma exposure (Deribit).
====================================================================

Deribit is where essentially all listed crypto-options liquidity sits, and its
public API needs no key, so this is the one item from the "institutional
pipeline" wish-list that is genuinely buildable here rather than faked. A single
call — ``get_book_summary_by_currency`` — returns the entire option book with a
mark implied volatility and open interest per contract, which is exactly what the
two headline analytics need:

**Volatility skew.** The implied-vol smile across strikes for an expiry, plus a
25-delta-style risk reversal (out-of-the-money put IV minus call IV). A positive
put-skew — puts richer than calls — is the market paying up for downside
protection, the normal state for crypto and a gauge of fear when it steepens.

**Dealer gamma exposure (GEX).** Aggregate option gamma weighted by open
interest, under the standard dealer-positioning convention. Its sign is the
actionable part: net-positive dealer gamma means market-makers hedge *against*
the move (buy dips, sell rips) and dampen volatility; net-negative means they
hedge *with* it and amplify. The strike where net gamma flips sign is the
"gamma flip" level around which that behaviour changes.

The GEX **sign and shape** are robust; the absolute dollar magnitude depends on
an assumption about which side dealers are on, so the UI leads with the regime
and the flip level and treats the notional as indicative. The IV and OI feeding
it are real, live Deribit data.

Everything sits behind the same circuit breaker as the funding feed: a rate limit
trips a cooldown, and any failure degrades to ``None`` rather than raising.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from . import storage

_SUMMARY_URL = ("https://www.deribit.com/api/v2/public/"
                "get_book_summary_by_currency?currency={ccy}&kind=option")
_HEADERS = {"User-Agent": "StockValuationPredictor/1.0 (educational)"}
_TTL = 120.0

_COOLDOWN_UNTIL = 0.0
_RATE_LIMIT_COOLDOWN = 900.0
_ERROR_COOLDOWN = 120.0

# Deribit option names: BTC-27DEC24-100000-C  →  ccy, expiry, strike, C/P
_NAME_RE = re.compile(r"^[A-Z]+-(\d{1,2}[A-Z]{3}\d{2})-(\d+(?:\.\d+)?)-([CP])$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "rate" in s or "too many" in s


def _trip(exc: Exception) -> None:
    global _COOLDOWN_UNTIL
    _COOLDOWN_UNTIL = time.time() + (_RATE_LIMIT_COOLDOWN if _is_rate_limit(exc)
                                     else _ERROR_COOLDOWN)


def cooldown_remaining() -> float:
    return max(0.0, _COOLDOWN_UNTIL - time.time())


def reset_circuit() -> None:
    global _COOLDOWN_UNTIL
    _COOLDOWN_UNTIL = 0.0


@dataclass(frozen=True)
class OptionQuote:
    strike: float
    expiry_days: float            # calendar days to expiry
    kind: str                     # "C" | "P"
    iv: float                     # implied vol, fraction (0.65 = 65%)
    open_interest: float          # contracts
    underlying: float             # underlying price at quote time

    @property
    def T(self) -> float:
        return max(self.expiry_days / 365.0, 1e-6)


def _parse_expiry_days(token: str, now: Optional[float] = None) -> Optional[float]:
    """'27DEC24' → calendar days from now. Deribit expiries settle 08:00 UTC."""
    m = re.match(r"^(\d{1,2})([A-Z]{3})(\d{2})$", token)
    if not m:
        return None
    day, mon, yr = int(m.group(1)), _MONTHS.get(m.group(2)), 2000 + int(m.group(3))
    if not mon:
        return None
    try:
        import datetime as dt
        expiry = dt.datetime(yr, mon, day, 8, 0, tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    now_ts = now if now is not None else time.time()
    return (expiry.timestamp() - now_ts) / 86400.0


def parse_summary(rows: list, now: Optional[float] = None) -> list:
    """
    Turn Deribit's book-summary rows into :class:`OptionQuote` objects.

    Skips anything unparseable or expired rather than guessing — a malformed
    instrument name or a missing IV must not become a fake data point.
    """
    out: list[OptionQuote] = []
    for r in rows or []:
        name = r.get("instrument_name", "")
        m = _NAME_RE.match(name)
        if not m:
            continue
        days = _parse_expiry_days(m.group(1), now)
        if days is None or days <= 0:
            continue
        iv = r.get("mark_iv")
        if iv is None:
            continue
        try:
            out.append(OptionQuote(
                strike=float(m.group(2)),
                expiry_days=float(days),
                kind=m.group(3),
                iv=float(iv) / 100.0,          # Deribit quotes IV in percent
                open_interest=float(r.get("open_interest") or 0.0),
                underlying=float(r.get("underlying_price")
                                 or r.get("estimated_delivery_price") or 0.0),
            ))
        except (TypeError, ValueError):
            continue
    return out


def get_chain(currency: str = "BTC") -> Optional[list]:
    """
    Live Deribit option chain for ``currency`` (BTC or ETH), or ``None``.

    ``None`` covers a tripped breaker, a network failure, or an unusable payload
    — in every case the caller shows nothing rather than a guessed chain.
    """
    ccy = currency.upper().strip()
    if ccy not in {"BTC", "ETH"}:
        return None
    key = f"deribit_chain:{ccy}"
    cached = storage.cache_get(key)
    if cached is not None:
        return parse_summary(cached)

    if cooldown_remaining() > 0:
        return None
    try:
        import requests

        r = requests.get(_SUMMARY_URL.format(ccy=ccy), headers=_HEADERS, timeout=12)
        r.raise_for_status()
        rows = r.json().get("result") or []
        storage.cache_set(key, rows, ttl=_TTL)
        return parse_summary(rows)
    except Exception as exc:      # noqa: BLE001 — all failures degrade the same
        _trip(exc)
        return None


# ── Black-Scholes gamma (self-contained, testable) ───────────────────────────
def _bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return pdf / (S * sigma * math.sqrt(T))


# ── Skew ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SkewResult:
    expiry_days: float
    atm_iv: float
    rr_25: float                  # 25-delta risk reversal: OTM put IV − OTM call IV
    strikes: list = field(default_factory=list)
    call_iv: list = field(default_factory=list)
    put_iv: list = field(default_factory=list)
    spot: float = 0.0

    @property
    def reading(self) -> str:
        if self.rr_25 > 0.02:
            return ("Put skew — downside protection is bid over upside; the market "
                    "is paying up for a fall, the normal crypto state, steepening in fear")
        if self.rr_25 < -0.02:
            return ("Call skew — upside calls richer than puts; a sign of chase / "
                    "squeeze positioning, less common in crypto")
        return "Roughly symmetric smile — no strong directional vol demand"


def nearest_expiry_skew(chain: list, min_days: float = 3.0) -> Optional[SkewResult]:
    """
    Volatility skew for the nearest expiry beyond ``min_days``.

    The very front expiry is noisy and often illiquid, so a small floor keeps the
    smile meaningful. The 25-delta risk reversal is approximated at ±10% moneyness
    — a robust proxy that needs no per-option delta solve.
    """
    quotes = [q for q in chain or [] if q.expiry_days >= min_days and q.iv > 0]
    if not quotes:
        return None
    spot = max((q.underlying for q in quotes), default=0.0)
    if spot <= 0:
        return None

    target = min(quotes, key=lambda q: q.expiry_days).expiry_days
    exp = [q for q in quotes if abs(q.expiry_days - target) < 1.5]
    calls = sorted([q for q in exp if q.kind == "C"], key=lambda q: q.strike)
    puts = sorted([q for q in exp if q.kind == "P"], key=lambda q: q.strike)
    if len(calls) < 2 or len(puts) < 2:
        return None

    def _iv_at(quotes_, target_strike):
        return min(quotes_, key=lambda q: abs(q.strike - target_strike)).iv

    atm_iv = (_iv_at(calls, spot) + _iv_at(puts, spot)) / 2.0
    otm_put_iv = _iv_at(puts, spot * 0.90)
    otm_call_iv = _iv_at(calls, spot * 1.10)
    rr = otm_put_iv - otm_call_iv

    strikes = sorted({q.strike for q in exp})
    return SkewResult(
        expiry_days=target, atm_iv=atm_iv, rr_25=rr,
        strikes=strikes,
        call_iv=[_iv_at(calls, k) for k in strikes],
        put_iv=[_iv_at(puts, k) for k in strikes],
        spot=spot,
    )


# ── Dealer gamma exposure ────────────────────────────────────────────────────
@dataclass(frozen=True)
class GEXResult:
    total: float                  # net dealer gamma exposure, $ per 1% move (indicative)
    flip_strike: Optional[float]  # spot level where net gamma crosses zero
    spot: float
    by_strike: list = field(default_factory=list)   # [(strike, net_gamma_$), ...]

    @property
    def regime(self) -> str:
        if self.total > 0:
            return "Positive"
        if self.total < 0:
            return "Negative"
        return "Neutral"

    @property
    def reading(self) -> str:
        if self.total > 0:
            base = ("Positive dealer gamma — market-makers hedge against moves "
                    "(buy dips, sell rips), which dampens volatility and pins price")
        elif self.total < 0:
            base = ("Negative dealer gamma — market-makers hedge with moves, "
                    "amplifying them; expect larger, trendier swings")
        else:
            base = "Balanced dealer gamma"
        if self.flip_strike:
            base += (f". Gamma flips around {self.flip_strike:,.0f}: below it the "
                     "regime tends to invert")
        return base


def dealer_gex(chain: list, spot: Optional[float] = None,
               contract_size: float = 1.0) -> Optional[GEXResult]:
    """
    Net dealer gamma exposure across the chain, by strike.

    Convention (the common dealer-positioning assumption): market-makers are long
    call gamma and short put gamma, so per contract the dealer gamma is
    ``+Γ·OI`` for calls and ``−Γ·OI`` for puts. Each is scaled to dollars per 1%
    spot move by ``Γ · OI · size · S² · 0.01``. The **sign and the flip level**
    are the robust, actionable outputs; the notional is indicative because the
    long/short assumption is exactly that.
    """
    quotes = [q for q in chain or [] if q.iv > 0 and q.open_interest > 0]
    if not quotes:
        return None
    S = spot if spot and spot > 0 else max((q.underlying for q in quotes), default=0.0)
    if S <= 0:
        return None

    per_strike: dict = {}
    for q in quotes:
        g = _bs_gamma(S, q.strike, q.T, q.iv)
        dollar = g * q.open_interest * contract_size * (S ** 2) * 0.01
        signed = dollar if q.kind == "C" else -dollar
        per_strike[q.strike] = per_strike.get(q.strike, 0.0) + signed

    by_strike = sorted(per_strike.items())
    total = float(sum(v for _, v in by_strike))

    # Gamma flip: the strike where the cumulative net gamma changes sign.
    flip = None
    cum = 0.0
    prev_cum = 0.0
    prev_k = None
    for k, v in by_strike:
        cum += v
        if prev_k is not None and (prev_cum <= 0 < cum or prev_cum >= 0 > cum):
            flip = (prev_k + k) / 2.0
            break
        prev_cum, prev_k = cum, k

    return GEXResult(total=total, flip_strike=flip, spot=S,
                     by_strike=[(float(k), float(v)) for k, v in by_strike])
