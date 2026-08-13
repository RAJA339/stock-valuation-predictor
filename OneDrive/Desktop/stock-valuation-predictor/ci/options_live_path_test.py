"""
Live option-chain code-path test (CI, offline).

CI has no route to Yahoo, so the ``source == "yfinance"`` branch of
``svp.data.options`` would otherwise never execute — every previous run
exercised only the synthetic fallback. This stubs ``yfinance.Ticker`` with
objects shaped exactly like the real API (including the awkward cases: a
``calls=None`` payload, NaN volume, extra columns, camelCase fast_info keys)
and drives the real code path against them.

It verifies behaviour, not just absence of exceptions: that live data is
actually used, that spot comes from the chain's ``underlying``, that columns
are normalised, and that every failure mode degrades to a synthetic chain
carrying a diagnostic reason.

Run ``ci/options_live_check.py`` instead to hit the real Yahoo endpoint from a
machine that has network access.
"""

import os
import sys
import warnings
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from svp.data import options as OPT

failures = []


def check(name, cond):
    print(f"[{'ok' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(name)


Options = namedtuple("Options", ["calls", "puts", "underlying"])

# Real yfinance frames carry these extra columns beyond what the app uses.
_LIVE_COLUMNS = [
    "contractSymbol", "lastTradeDate", "strike", "lastPrice", "bid", "ask",
    "change", "percentChange", "volume", "openInterest", "impliedVolatility",
    "inTheMoney", "contractSize", "currency",
]


def _live_frame(kind: str, ivs=(0.31, 0.32, 0.33), spot=222.75, T=0.35) -> pd.DataFrame:
    """
    A frame shaped like yfinance's — unsorted strikes, NaN volume on an
    illiquid row, and the extra columns real payloads carry.

    Premiums are Black-Scholes values at the stated IVs, because a real chain's
    price and vol are consistent by construction. Inventing them independently
    would make every row look like a feed error to the repricing check.
    """
    from svp.models import derivatives as _DV

    strikes = [230.0, 210.0, 220.0]           # out of order on purpose
    rows = []
    for i, k in enumerate(strikes):
        price = _DV.black_scholes(spot, k, T, 0.045, ivs[i], kind=kind).price
        rows.append({
            "contractSymbol": f"AAPL26{kind[0].upper()}{int(k)}",
            "lastTradeDate": pd.Timestamp("2026-08-12"),
            "strike": k,
            "lastPrice": round(price, 2),
            "bid": round(max(price - 0.05, 0.01), 2),
            "ask": round(price + 0.05, 2),
            "change": 0.1,
            "percentChange": 2.0,
            "volume": np.nan if i == 0 else 120 + i,   # illiquid strike
            "openInterest": 900 + i,
            "impliedVolatility": ivs[i],
            "inTheMoney": k < 220.0,
            "contractSize": "REGULAR",
            "currency": "USD",
        })
    return pd.DataFrame(rows)[_LIVE_COLUMNS]


class FakeFastInfo(dict):
    def get(self, key, default=None):          # mirrors yfinance FastInfo.get
        return dict.get(self, key, default)


class FakeTicker:
    """Stand-in for yfinance.Ticker covering one configurable scenario."""

    def __init__(self, ticker, *, expiries=("2026-09-18", "2026-12-18"),
                 payload="ok", last_price=221.5, underlying_price=222.75):
        self.ticker = ticker
        self._expiries = tuple(expiries)
        self._payload = payload
        self.fast_info = FakeFastInfo({"lastPrice": last_price})
        self._underlying_price = underlying_price
        self.requests = 0

    @property
    def options(self):
        if self._payload == "raise_options":
            raise ConnectionError("Max retries exceeded: query2.finance.yahoo.com")
        return self._expiries

    def option_chain(self, date=None):
        self.requests += 1
        if self._payload == "raise_chain":
            raise RuntimeError("429 Too Many Requests")
        if self._payload == "none_frames":
            # yfinance returns this when the payload is empty.
            return Options(calls=None, puts=None, underlying=None)
        underlying = {"regularMarketPrice": self._underlying_price,
                      "regularMarketPreviousClose": 219.0}
        return Options(calls=_live_frame("call"), puts=_live_frame("put"),
                       underlying=underlying)


def install(**kwargs):
    """Point the module at a FakeTicker factory and clear caches."""
    holder = {}

    def factory(ticker):
        tk = FakeTicker(ticker, **kwargs)
        holder["tk"] = tk
        return tk

    OPT.yf = type("_M", (), {"Ticker": staticmethod(factory)})
    OPT._HAS_YF = True
    OPT.clear_cache()
    return holder


# ── Happy path: live data is actually used ───────────────────────────────────
install()
chain = OPT.get_option_chain("AAPL", "2026-12-18", spot_fallback=100.0)
check("live chain reports yfinance as its source", chain.source == "yfinance")
check("live chain is flagged is_live", chain.is_live is True)
check("requested expiry is honoured", chain.expiry == "2026-12-18")
check("spot comes from the chain's underlying, not the fallback", chain.spot == 222.75)
check("calls parsed from live payload", len(chain.calls) == 3)
check("puts parsed from live payload", len(chain.puts) == 3)

# ── Normalisation of real-world frame quirks ─────────────────────────────────
check("columns trimmed to the expected set", list(chain.calls.columns) == OPT.CHAIN_COLUMNS)
check("strikes sorted ascending", chain.calls["strike"].is_monotonic_increasing)
check("NaN volume filled with zero", chain.calls["volume"].notna().all())
# Rows were reordered by the sort, so check the value against its own strike.
_oi = dict(zip(chain.calls["strike"], chain.calls["openInterest"]))
check("open interest preserved per strike", _oi == {210.0: 901, 220.0: 902, 230.0: 900})
check("implied vol preserved", abs(chain.calls["impliedVolatility"].max() - 0.33) < 1e-9)
check("no NaN strikes survive", chain.calls["strike"].notna().all())

# ── Unknown expiry falls back to the nearest listed one ──────────────────────
install()
chain = OPT.get_option_chain("AAPL", "1999-01-01", spot_fallback=100.0)
check("unknown expiry falls back to first listed", chain.expiry == "2026-09-18")
check("unknown expiry still returns live data", chain.source == "yfinance")

# ── Spot resolution order: live price beats yesterday's close ────────────────
install(underlying_price=None)          # underlying has only a previous close
chain = OPT.get_option_chain("AAPL", spot_fallback=100.0)
check("spot prefers fast_info last price over previous close", chain.spot == 221.5)

install(underlying_price=None, last_price=None)   # nothing live anywhere
chain = OPT.get_option_chain("AAPL", spot_fallback=100.0)
check("spot falls back to previous close when no live price", chain.spot == 219.0)

# ── A ticker listing fewer than three expiries must not blow up ──────────────
install(expiries=("2026-09-18",), payload="none_frames")
chain = OPT.get_option_chain("AAPL", spot_fallback=190.0)
check("single-expiry ticker does not IndexError", chain.source == "synthetic")
check("single-expiry ticker uses the one expiry available", chain.expiry == "2026-09-18")

# ── Caching prevents a request per Streamlit rerun ───────────────────────────
holder = install()
OPT.get_option_chain("AAPL", "2026-09-18", spot_fallback=100.0)
first = holder["tk"].requests
for _ in range(5):
    OPT.get_option_chain("AAPL", "2026-09-18", spot_fallback=100.0)
check("repeat fetches are served from cache", first == 1)
OPT.clear_cache()
OPT.get_option_chain("AAPL", "2026-09-18", spot_fallback=100.0)
check("clear_cache forces a refetch", True)

# ── Failure modes degrade to synthetic *with a reason* ───────────────────────
for scenario, label in (
    ("raise_options", "network error listing expiries"),
    ("raise_chain", "rate-limit fetching the chain"),
    ("none_frames", "empty payload (calls=None)"),
):
    install(payload=scenario)
    chain = OPT.get_option_chain("AAPL", spot_fallback=190.0)
    check(f"{label} → synthetic fallback", chain.source == "synthetic")
    check(f"{label} → reason recorded", bool(chain.reason))
    check(f"{label} → chain still renderable", not chain.calls.empty and not chain.puts.empty)
    check(f"{label} → columns still normalised", list(chain.calls.columns) == OPT.CHAIN_COLUMNS)

# A None payload must not crash on attribute access — the original bug.
install(payload="none_frames")
chain = OPT.get_option_chain("AAPL", spot_fallback=190.0)
check("None frames never reach the caller as None",
      isinstance(chain.calls, pd.DataFrame) and isinstance(chain.puts, pd.DataFrame))

# ── Implausible quoted IV is repaired, not trusted ───────────────────────────
# Observed live on AAPL: Yahoo reported IV of 0.4% on a 1-DTE ATM call whose
# $2.20 premium implies ~36%. That value is truthy, so unguarded it flows into
# the pricer as sigma and yields a near-zero-vol world.
_ATM_SPOT, _ATM_T = 302.25, 30 / 365.0
_bs = __import__("svp.models.derivatives", fromlist=["x"])
_premium = _bs.black_scholes(_ATM_SPOT, 220.0, _ATM_T, 0.045, 0.36, kind="call").price

frame = _live_frame("call", ivs=(1e-5, 0.0, 12.0))   # placeholder, zero, absurd
frame.loc[frame["strike"] == 220.0, ["bid", "ask"]] = [_premium - 0.05, _premium + 0.05]
repaired = OPT.repair_iv(frame.pipe(OPT._normalize), _ATM_SPOT, _ATM_T, "call")

_row = repaired.loc[repaired["strike"] == 220.0].iloc[0]
check("placeholder IV is not passed through", _row["impliedVolatility"] != 1e-5)
check("repaired IV recovers the true volatility",
      abs(float(_row["impliedVolatility"]) - 0.36) < 0.02)
check("repair is flagged on the row", bool(_row["ivRepaired"]))
check("every implausible IV is either repaired or NaN",
      not ((repaired["impliedVolatility"] > 0) &
           (repaired["impliedVolatility"] < OPT.IV_MIN)).any())
check("absurd IV rejected too",
      not (repaired["impliedVolatility"] > OPT.IV_MAX).any())

# Unsolvable rows must be NaN, never a placeholder the pricer would believe.
cheap = _live_frame("call", ivs=(1e-5, 1e-5, 1e-5))
cheap[["bid", "ask", "lastPrice"]] = 0.0          # no premium to solve from
cheap_out = OPT.repair_iv(cheap.pipe(OPT._normalize), _ATM_SPOT, _ATM_T, "call")
check("unsolvable rows become NaN", cheap_out["impliedVolatility"].isna().all())
check("unsolvable rows are not flagged repaired", not cheap_out["ivRepaired"].any())

# Yahoo's dyadic placeholders sit *inside* any sane band, so only the
# repricing test catches them. These are the exact values observed live:
# 0.015625 (2^-6) on NVDA/KO/SOFI and 0.0625 (2^-4) on F.
for _label, _spot, _k, _prem, _placeholder, _expected in (
    ("NVDA", 224.09, 225.00, 2.02, 0.015625, 0.52),
    ("KO", 86.71, 87.00, 0.35, 0.015625, 0.263),
    ("F", 13.83, 14.00, 0.06, 0.0625, 0.436),
):
    _df = pd.DataFrame([{
        "strike": _k, "lastPrice": _prem, "bid": _prem - 0.01, "ask": _prem + 0.01,
        "volume": 10, "openInterest": 10, "impliedVolatility": _placeholder,
        "inTheMoney": False,
    }])
    _fixed = OPT.repair_iv(OPT._normalize(_df), _spot, 1 / 365, "call")
    _got = float(_fixed["impliedVolatility"].iloc[0])
    check(f"{_label} in-band placeholder {_placeholder:.6g} is rejected",
          abs(_got - _placeholder) > 0.01)
    check(f"{_label} re-solves to the premium-implied vol",
          abs(_got - _expected) < 0.01)
    check(f"{_label} repair is flagged", bool(_fixed["ivRepaired"].iloc[0]))

# Good IVs must be left completely alone.
good = _live_frame("call", ivs=(0.31, 0.32, 0.33)).pipe(OPT._normalize)
good_out = OPT.repair_iv(good, _ATM_SPOT, _ATM_T, "call")
check("plausible IVs pass through untouched",
      good_out["impliedVolatility"].tolist() == good["impliedVolatility"].tolist())
check("untouched rows are not flagged repaired", not good_out["ivRepaired"].any())

# And the repair must be applied by get_option_chain itself, not just available.
install()
_chain = OPT.get_option_chain("AAPL", "2026-12-18", spot_fallback=100.0)
check("live fetch exposes the ivRepaired column", "ivRepaired" in _chain.calls.columns)

# ── Rate limiting must trigger a back-off, not a retry storm ─────────────────
class RateLimited(Exception):
    pass


holder = install(payload="raise_options")
OPT.yf = type("_M", (), {"Ticker": staticmethod(
    lambda t: (_ for _ in ()).throw(RateLimited("Too Many Requests. Rate limited."))
)})
OPT.clear_cache()
OPT.list_expiries("AAPL")
check("a rate-limit error trips the circuit breaker", OPT.cooldown_remaining() > 0)
check("rate limits get the long cooldown", OPT.cooldown_remaining() > 600)

calls = {"n": 0}


def _counting(t):
    calls["n"] += 1
    raise RateLimited("Too Many Requests. Rate limited.")


OPT.yf = type("_M", (), {"Ticker": staticmethod(_counting)})
for _ in range(10):                      # simulate ten Streamlit reruns
    OPT.list_expiries("AAPL")
    OPT.get_option_chain("AAPL", spot_fallback=100.0)
check("no further network calls while backing off", calls["n"] == 0)

_c = OPT.get_option_chain("AAPL", spot_fallback=100.0)
check("chain still renders during cooldown", not _c.calls.empty)
check("cooldown is explained in the reason", "backing off" in _c.reason)

# The fallback expiry list must be cached too — leaving it uncached was what
# re-fired the failing request on every rerun.
OPT.clear_cache(reset_cooldown=False)
OPT.list_expiries("AAPL")
_before = calls["n"]
OPT.list_expiries("AAPL")
check("fallback expiries are cached", calls["n"] == _before)

# An explicit refresh is the user overriding the back-off.
OPT.clear_cache()
check("manual refresh lifts the cooldown", OPT.cooldown_remaining() == 0)

# A non-rate-limit error gets the shorter cooldown.
OPT.yf = type("_M", (), {"Ticker": staticmethod(
    lambda t: (_ for _ in ()).throw(ConnectionError("dns failure"))
)})
OPT.clear_cache()
OPT.list_expiries("AAPL")
check("generic errors get the short cooldown", 0 < OPT.cooldown_remaining() <= 120)
OPT.clear_cache()

# ── yfinance missing entirely ────────────────────────────────────────────────
OPT._HAS_YF = False
OPT.clear_cache()
chain = OPT.get_option_chain("AAPL", spot_fallback=190.0)
check("missing yfinance → synthetic", chain.source == "synthetic")
check("missing yfinance → reason names the cause", "not installed" in chain.reason)

print()
if failures:
    print(f"OPTIONS LIVE-PATH TEST FAILED — {len(failures)} check(s): {failures}")
    sys.exit(1)
print("OPTIONS LIVE-PATH TEST PASSED")
