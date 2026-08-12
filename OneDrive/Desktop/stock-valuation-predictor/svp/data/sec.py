"""
SEC EDGAR data pipeline.
========================

Fetches company facts from the SEC EDGAR XBRL API and exposes helpers to pull
both the latest annual value *and* a time-ordered history for a concept (needed
for QoQ / YoY trend features). Parsed filings are persisted via ``svp.data.storage``
so repeat lookups are fast.
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from . import storage

_HEADERS = {"User-Agent": "StockValuationApp contact@example.com"}
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Module-level memo for the (large) ticker→CIK map within a single process.
_ticker_map: Optional[dict] = None


def get_cik(ticker: str) -> Optional[str]:
    """Map a ticker symbol to its zero-padded 10-digit SEC CIK."""
    global _ticker_map
    ticker = ticker.upper().strip()

    cached = storage.cache_get(f"cik:{ticker}")
    if cached:
        return cached

    if _ticker_map is None:
        try:
            data = requests.get(_TICKERS_URL, headers=_HEADERS, timeout=10).json()
            _ticker_map = {e["ticker"].upper(): str(e["cik_str"]).zfill(10) for e in data.values()}
        except Exception:
            _ticker_map = {}

    cik = _ticker_map.get(ticker)
    if cik:
        storage.cache_set(f"cik:{ticker}", cik, ttl=7 * 24 * 3600)
    return cik


def get_financials(cik: str) -> dict:
    """Fetch (and persist) company facts JSON for a CIK."""
    key = f"secfacts:{cik}"
    cached = storage.cache_get(key)
    if cached is not None:
        return cached
    try:
        payload = requests.get(_FACTS_URL.format(cik=cik), headers=_HEADERS, timeout=20).json()
    except Exception:
        return {}
    if payload and "facts" in payload:
        # Persist for a day — filings change quarterly at most.
        storage.cache_set(key, payload, ttl=24 * 3600)
    return payload or {}


def _entries(facts: dict, concept: str, unit: str = "USD") -> list:
    try:
        return facts["facts"]["us-gaap"][concept]["units"][unit]
    except (KeyError, TypeError):
        return []


def extract_latest(facts: dict, concept: str, unit: str = "USD") -> Optional[float]:
    """Most recent *annual* (10-K) value for a concept."""
    annual = [e for e in _entries(facts, concept, unit) if e.get("form") == "10-K" and "end" in e]
    if not annual:
        return None
    annual.sort(key=lambda x: x["end"], reverse=True)
    return annual[0]["val"]


def extract_history(facts: dict, concept: str, unit: str = "USD", forms=("10-K", "10-Q")) -> list[dict]:
    """
    Return a de-duplicated, chronologically sorted history for a concept.

    Each item is ``{"end": <date str>, "val": <float>, "form": <str>}``. Used to
    derive QoQ / YoY growth-velocity features.
    """
    seen: dict[str, dict] = {}
    for e in _entries(facts, concept, unit):
        if e.get("form") in forms and "end" in e and e.get("val") is not None:
            # Keep the latest-filed value for a given period end.
            key = e["end"]
            if key not in seen or e.get("filed", "") > seen[key].get("filed", ""):
                seen[key] = {"end": e["end"], "val": float(e["val"]), "form": e["form"]}
    return sorted(seen.values(), key=lambda x: x["end"])


def concept_first(facts: dict, concepts: list[str], unit: str = "USD") -> Optional[float]:
    """Return the first non-None ``extract_latest`` across candidate concept names."""
    for c in concepts:
        v = extract_latest(facts, c, unit)
        if v is not None:
            return v
    return None


def history_first(facts: dict, concepts: list[str], unit: str = "USD") -> list[dict]:
    """Return the first non-empty history across candidate concept names."""
    for c in concepts:
        h = extract_history(facts, c, unit)
        if h:
            return h
    return []


def company_name(facts: dict) -> Optional[str]:
    return facts.get("entityName") if isinstance(facts, dict) else None
