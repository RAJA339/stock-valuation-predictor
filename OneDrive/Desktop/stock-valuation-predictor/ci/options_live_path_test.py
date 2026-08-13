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


def _live_frame(kind: str) -> pd.DataFrame:
    """A frame shaped like yfinance's, deliberately unsorted with NaN volume."""
    strikes = [230.0, 210.0, 220.0]           # out of order on purpose
    rows = []
    for i, k in enumerate(strikes):
        rows.append({
            "contractSymbol": f"AAPL26{kind[0].upper()}{int(k)}",
            "lastTradeDate": pd.Timestamp("2026-08-12"),
            "strike": k,
            "lastPrice": 5.0 + i,
            "bid": 4.9 + i,
            "ask": 5.1 + i,
            "change": 0.1,
            "percentChange": 2.0,
            "volume": np.nan if i == 0 else 120 + i,   # illiquid strike
            "openInterest": 900 + i,
            "impliedVolatility": 0.31 + 0.01 * i,
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
