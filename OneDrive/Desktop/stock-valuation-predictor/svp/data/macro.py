"""
Macro-economic data pipeline (FRED + BLS).
==========================================

Real-time macro indicators — CPI, interest rates and the yield curve — from the
FRED API (``FRED_API_KEY``), with the original BLS public API and static
long-run averages as graceful fallbacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests

try:
    from fredapi import Fred  # type: ignore

    _HAS_FREDAPI = True
except Exception:  # pragma: no cover
    _HAS_FREDAPI = False

from . import storage

# FRED series ids.
FRED_SERIES = {
    "cpi": "CPIAUCSL",          # CPI-U (index)
    "fed_funds": "FEDFUNDS",    # effective federal funds rate (%)
    "t10y": "DGS10",            # 10-year treasury yield (%)
    "t2y": "DGS2",              # 2-year treasury yield (%)
    "t3m": "DGS3MO",            # 3-month treasury yield (%)
    "unemployment": "UNRATE",   # unemployment rate (%)
}

# Static long-run fallbacks used when every API is unreachable.
_DEFAULTS = {
    "cpi": 314.0,
    "fed_funds": 4.3,
    "t10y": 4.2,
    "t2y": 4.3,
    "t3m": 4.4,
    "unemployment": 4.1,
}


@dataclass
class MacroSnapshot:
    cpi: float
    fed_funds: float
    t10y: float
    t2y: float
    t3m: float
    unemployment: float
    source: str = "defaults"

    @property
    def is_live(self) -> bool:
        return self.source not in ("defaults", "")

    @property
    def yield_curve_10y_2y(self) -> float:
        """10y-2y spread (bp of recession signal when negative)."""
        return self.t10y - self.t2y

    @property
    def curve_inverted(self) -> bool:
        return self.yield_curve_10y_2y < 0


def _fred_key() -> Optional[str]:
    return os.environ.get("FRED_API_KEY") or os.environ.get("SVP_FRED_API_KEY")


def _fred_latest(series_id: str) -> Optional[float]:
    key = _fred_key()
    if not key:
        return None
    try:
        if _HAS_FREDAPI:
            fred = Fred(api_key=key)
            s = fred.get_series(series_id).dropna()
            return float(s.iloc[-1]) if len(s) else None
        # Raw REST fallback if fredapi missing but key present.
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        }
        obs = requests.get(url, params=params, timeout=10).json().get("observations", [])
        for o in obs:
            if o.get("value") not in (".", None, ""):
                return float(o["value"])
    except Exception:
        return None
    return None


def _bls_latest(series_id: str) -> Optional[float]:
    """Original BLS public API path (no key required for basic usage)."""
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {"seriesid": [series_id], "startyear": "2022", "endyear": "2025"}
    try:
        r = requests.post(url, json=payload, timeout=10).json()
        data = r["Results"]["series"][0]["data"]
        data.sort(key=lambda x: (x["year"], x["period"]), reverse=True)
        return float(data[0]["value"])
    except Exception:
        return None


def get_macro() -> MacroSnapshot:
    """Return the latest macro snapshot from FRED → BLS → static defaults."""
    cached = storage.cache_get("macro:snapshot")
    if cached:
        return MacroSnapshot(**cached)

    vals: dict[str, Optional[float]] = {}
    live_any = False
    for name, sid in FRED_SERIES.items():
        v = _fred_latest(sid)
        if v is not None:
            live_any = True
        vals[name] = v

    # BLS backfill for the two it can serve.
    if vals.get("cpi") is None:
        vals["cpi"] = _bls_latest("CUUR0000SA0")
    if vals.get("unemployment") is None:
        vals["unemployment"] = _bls_latest("LNS14000000")
    if vals.get("cpi") is not None or vals.get("unemployment") is not None:
        live_any = live_any or True

    filled = {k: (vals.get(k) if vals.get(k) is not None else _DEFAULTS[k]) for k in _DEFAULTS}
    source = "FRED/BLS" if live_any else "defaults"
    snap = MacroSnapshot(**filled, source=source)

    if live_any:
        storage.cache_set("macro:snapshot", {**filled, "source": source}, ttl=6 * 3600)
    return snap
