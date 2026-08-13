"""
Live options-chain pipeline.
=============================

Pulls live strike prices, bids/asks, open interest and implied volatility via
``yfinance``'s ``Ticker.option_chain(date)``. Falls back to a deterministic
synthetic chain (Black-Scholes-priced around the current spot) so the Options
tab always has data to render, even offline.

The fallback is never silent: ``OptionChain.reason`` records why the live fetch
did not happen, so the UI can say "no network" rather than quietly showing
made-up strikes as though they were real quotes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
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


# Columns the UI and the report expect, in display order.
CHAIN_COLUMNS = [
    "strike", "lastPrice", "bid", "ask", "volume",
    "openInterest", "impliedVolatility", "inTheMoney", "ivRepaired",
]

# Plausible bounds for an equity implied vol. Yahoo returns a placeholder near
# zero (~1e-5) for many contracts — especially near-dated ones — and that value
# is truthy, so it silently poisons any downstream sigma. Anything outside this
# band is treated as missing and re-solved from the quoted premium.
IV_MIN, IV_MAX = 0.01, 5.0

# Short-lived in-process cache. Streamlit reruns the whole script on every
# widget interaction (expiry dropdown, vol slider, strike input), so without
# this a single session would hammer Yahoo with a request per keystroke and
# get itself rate-limited.
_CACHE: dict[tuple, tuple[float, "OptionChain"]] = {}
_CACHE_TTL = 300.0  # seconds
_EXPIRY_CACHE: dict[str, tuple[float, list[str]]] = {}


@dataclass
class OptionChain:
    ticker: str
    expiry: str
    spot: float
    calls: pd.DataFrame
    puts: pd.DataFrame
    source: str
    reason: str = ""          # why the live fetch was not used, if it wasn't

    @property
    def is_live(self) -> bool:
        return self.source == "yfinance"


def _normalize(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Coerce a yfinance chain frame into the columns the app expects.

    Live frames carry extra columns (contractSymbol, lastTradeDate, change,
    percentChange, contractSize, currency) and can carry NaN volume/open
    interest for illiquid strikes. Missing columns are added as NaN so callers
    can index them unconditionally.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    out = df.copy()
    for col in CHAIN_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    for col in ("strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Volume and open interest are genuinely zero when untraded, not unknown.
    out["volume"] = out["volume"].fillna(0)
    out["openInterest"] = out["openInterest"].fillna(0)

    out["ivRepaired"] = False
    out = out.dropna(subset=["strike"]).sort_values("strike").reset_index(drop=True)
    return out[CHAIN_COLUMNS]


def _mid_price(row) -> Optional[float]:
    """Mid of the quoted spread, falling back to the last traded price."""
    bid, ask = row.get("bid"), row.get("ask")
    if bid and ask and bid > 0 and ask > 0:
        return float((bid + ask) / 2)
    last = row.get("lastPrice")
    if last and last > 0:
        return float(last)
    return None


def repair_iv(df: pd.DataFrame, spot: float, T: float, kind: str,
              r: float = 0.045) -> pd.DataFrame:
    """
    Replace implausible quoted implied vols by re-solving from the premium.

    Yahoo frequently reports ~1e-5 rather than a real IV. Left alone that value
    flows into the pricer as sigma and produces a near-zero-vol world: bogus
    probabilities of finishing in the money and a meaningless edge ranking.
    Rows that cannot be re-solved (premium at or below intrinsic) are left as
    NaN so callers can skip them rather than trust a placeholder.
    """
    if df.empty or T <= 0 or spot <= 0:
        return df

    deriv = _deriv()
    out = df.copy()
    iv = out["impliedVolatility"]
    bad = iv.isna() | (iv < IV_MIN) | (iv > IV_MAX)
    if not bad.any():
        return out

    for idx in out.index[bad]:
        row = out.loc[idx]
        premium = _mid_price(row)
        solved = None
        if premium:
            solved = deriv.implied_volatility(
                premium, spot, float(row["strike"]), T, r, kind=kind
            )
        if solved is not None and IV_MIN <= solved <= IV_MAX:
            out.at[idx, "impliedVolatility"] = float(solved)
            out.at[idx, "ivRepaired"] = True
        else:
            out.at[idx, "impliedVolatility"] = np.nan
    return out


def _spot_from_chain(chain, tk, fallback: float) -> float:
    """
    Resolve spot from the option_chain payload first.

    ``option_chain()`` returns an ``underlying`` dict already fetched in the
    same request, so preferring it avoids a second network round-trip to
    fast_info and keeps spot consistent with the quotes in the chain.
    """
    def _ok(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) \
            and v and v > 0 and not pd.isna(v)

    underlying = chain.underlying if isinstance(getattr(chain, "underlying", None), dict) else {}

    # A live traded price first, from whichever source has one. Only then fall
    # back to stale-but-real quotes (previous close, bid/ask) — yesterday's
    # close is a worse spot than fast_info's last price, so it ranks below it.
    if _ok(underlying.get("regularMarketPrice")):
        return float(underlying["regularMarketPrice"])
    try:
        v = tk.fast_info.get("lastPrice")
        if _ok(v):
            return float(v)
    except Exception:
        pass
    for key in ("regularMarketPreviousClose", "bid", "ask"):
        if _ok(underlying.get(key)):
            return float(underlying[key])
    return float(fallback)


def list_expiries(ticker: str, use_cache: bool = True) -> list[str]:
    """Return available expiry date strings for a ticker."""
    ticker = ticker.upper().strip()
    now = time.time()
    if use_cache:
        hit = _EXPIRY_CACHE.get(ticker)
        if hit and now - hit[0] < _CACHE_TTL:
            return hit[1]

    if _HAS_YF:
        try:
            exps = yf.Ticker(ticker).options
            if exps:
                out = list(exps)
                _EXPIRY_CACHE[ticker] = (now, out)
                return out
        except Exception:
            pass

    # Synthetic fallback expiries: next 4 monthly-ish dates + a 1y LEAP.
    today = pd.Timestamp.today().normalize()
    return [(today + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in (30, 60, 90, 180, 365)]


def _synthetic_chain(ticker: str, expiry: str, spot: float, r: float = 0.045,
                     reason: str = "") -> OptionChain:
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
                "ivRepaired": False,
            })
        return pd.DataFrame(rows)[CHAIN_COLUMNS]

    return OptionChain(ticker, expiry, spot, build("call"), build("put"),
                       source="synthetic", reason=reason)


def get_option_chain(
    ticker: str,
    expiry: Optional[str] = None,
    spot_fallback: float = 100.0,
    use_cache: bool = True,
) -> OptionChain:
    """
    Fetch the live option chain for ``ticker`` / ``expiry`` (or the nearest one).

    Falls back to a synthetic chain on any failure, recording the cause in
    ``OptionChain.reason``. Results are cached briefly so Streamlit reruns do
    not re-request the same chain.
    """
    ticker = ticker.upper().strip()
    key = (ticker, expiry)
    now = time.time()

    if use_cache:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < _CACHE_TTL:
            return hit[1]

    reason = "yfinance not installed"
    if _HAS_YF:
        try:
            tk = yf.Ticker(ticker)
            expiries = tk.options
            if not expiries:
                reason = f"no listed options for {ticker}"
            else:
                exp = expiry if (expiry and expiry in expiries) else expiries[0]
                chain = tk.option_chain(exp)

                # option_chain() returns calls=puts=None when the payload is
                # empty — indexing those would raise, so treat it as a miss.
                calls = _normalize(getattr(chain, "calls", None))
                puts = _normalize(getattr(chain, "puts", None))
                if calls.empty and puts.empty:
                    reason = f"empty chain returned for {ticker} {exp}"
                else:
                    spot = _spot_from_chain(chain, tk, spot_fallback)
                    T = max((pd.Timestamp(exp) - pd.Timestamp.today().normalize()).days, 0) / 365.0
                    calls = repair_iv(calls, spot, T, "call")
                    puts = repair_iv(puts, spot, T, "put")
                    live = OptionChain(ticker, exp, spot, calls, puts, source="yfinance")
                    _CACHE[key] = (now, live)
                    return live
        except Exception as exc:  # network, rate-limit, schema drift
            reason = f"{type(exc).__name__}: {exc}"[:200]

    # Default to roughly the third expiry (~90d) but never index past the end —
    # thinly traded names can list only one or two.
    available = list_expiries(ticker) or [
        (pd.Timestamp.today().normalize() + pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    ]
    exp = expiry or available[min(2, len(available) - 1)]
    out = _synthetic_chain(ticker, exp, spot_fallback, reason=reason)
    _CACHE[key] = (now, out)
    return out


def clear_cache() -> None:
    """Drop cached chains/expiries — used by the UI's manual refresh."""
    _CACHE.clear()
    _EXPIRY_CACHE.clear()
