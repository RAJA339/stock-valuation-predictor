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

# SEC requires a descriptive User-Agent with contact info; generic agents get
# 403-ed. See https://www.sec.gov/os/webmaster-faq#developers
# No Host header here on purpose. requests derives Host from the URL, and this
# package talks to two SEC hosts: www.sec.gov for the ticker map and the
# Archives, data.sec.gov for the XBRL and submissions APIs. Pinning
# Host: www.sec.gov sent a mismatched header to data.sec.gov on every
# companyfacts request — the endpoint behind every valuation — which the CDN
# may reject. The failure then surfaced as "could not parse usable
# fundamentals", blaming the filing for a request we had malformed.
_HEADERS = {
    "User-Agent": "StockValuationPredictor/1.0 (educational; contact@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

# Concept-name variants. Filers tag the same line item differently — notably
# equity, where a company with minority interests uses the long form — so a
# single tag name silently returns None for a large share of real companies.
NET_INCOME_TAGS = [
    "NetIncomeLoss",
    "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
    "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest",
]
EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "Equity",
    "EquityAttributableToOwnersOfParent",
]
ASSETS_TAGS = ["Assets"]
REVENUE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "Revenue",
]
OCF_TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    "CashFlowsFromUsedInOperatingActivities",
]
CAPEX_TAGS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PurchaseOfPropertyPlantAndEquipment",
]
DEBT_TAGS = ["LongTermDebt", "LongTermDebtNoncurrent", "LongTermBorrowings"]

# Forms that carry a full year. 20-F is filed by foreign private issuers and
# 40-F by Canadian ones — excluding them shut out every non-US-domiciled
# company on the exchange.
ANNUAL_FORMS = ("10-K", "20-F", "40-F", "10-K/A", "20-F/A")

# XBRL taxonomies to search, in order. IFRS filers publish under ifrs-full.
_TAXONOMIES = ("us-gaap", "ifrs-full")
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Module-level memo for the (large) ticker→CIK map within a single process.
# _ticker_map_at records when it was last *successfully* built, so a transient
# SEC failure does not poison the process: previously a single failed fetch
# stored {} and the `is None` guard then skipped every retry, making every
# ticker unresolvable until restart.
_ticker_map: Optional[dict] = None
_ticker_map_at: float = 0.0
_TICKER_MAP_RETRY = 120.0


def _ticker_variants(ticker: str) -> list[str]:
    """
    Symbol spellings to try.

    Share classes are written three ways in the wild — BRK.B, BRK-B, BRKB —
    and SEC's file uses the hyphen form.
    """
    t = ticker.upper().strip()
    out = [t, t.replace(".", "-"), t.replace("-", "."), t.replace(".", "").replace("-", "")]
    seen, uniq = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def get_cik(ticker: str) -> Optional[str]:
    """Map a ticker symbol to its zero-padded 10-digit SEC CIK."""
    global _ticker_map, _ticker_map_at
    ticker = ticker.upper().strip()

    cached = storage.cache_get(f"cik:{ticker}")
    if cached:
        return cached

    # Rebuild when we have never succeeded, or the last attempt failed and the
    # retry window has passed. Never leave a failed fetch cached indefinitely.
    stale = (not _ticker_map) and (time.time() - _ticker_map_at > _TICKER_MAP_RETRY)
    if _ticker_map is None or stale:
        try:
            resp = requests.get(_TICKERS_URL, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            built = {e["ticker"].upper(): str(e["cik_str"]).zfill(10) for e in data.values()}
            if built:
                _ticker_map = built
        except Exception:
            if _ticker_map is None:
                _ticker_map = {}
        finally:
            _ticker_map_at = time.time()

    for variant in _ticker_variants(ticker):
        cik = (_ticker_map or {}).get(variant)
        if cik:
            storage.cache_set(f"cik:{ticker}", cik, ttl=7 * 24 * 3600)
            return cik
    return None


# Why the last companyfacts fetch produced nothing, keyed by CIK. An empty dict
# has three quite different causes — the request failed, the company has no
# XBRL facts, or the response was not JSON — and telling the user the wrong one
# sends them to check the wrong thing.
_LAST_FETCH_ERROR: dict[str, str] = {}


def last_fetch_error(cik: str) -> Optional[str]:
    """The reason :func:`get_financials` last returned nothing for this CIK."""
    return _LAST_FETCH_ERROR.get(cik)


def get_financials(cik: str) -> dict:
    """
    Fetch (and persist) company facts JSON for a CIK.

    Returns ``{}`` on any failure, recording the reason for
    :func:`last_fetch_error` so callers can distinguish a transport problem
    from a company that genuinely publishes no XBRL.
    """
    key = f"secfacts:{cik}"
    cached = storage.cache_get(key)
    if cached is not None:
        _LAST_FETCH_ERROR.pop(cik, None)
        return cached

    try:
        resp = requests.get(_FACTS_URL.format(cik=cik), headers=_HEADERS, timeout=20)
    except Exception as exc:
        _LAST_FETCH_ERROR[cik] = f"network error contacting SEC ({type(exc).__name__})"
        return {}

    if resp.status_code == 404:
        _LAST_FETCH_ERROR[cik] = "the SEC publishes no XBRL company facts for this CIK"
        return {}
    if resp.status_code == 429:
        _LAST_FETCH_ERROR[cik] = "SEC EDGAR is rate-limiting this deployment"
        return {}
    if resp.status_code >= 400:
        _LAST_FETCH_ERROR[cik] = f"SEC EDGAR returned HTTP {resp.status_code}"
        return {}

    try:
        payload = resp.json()
    except Exception:
        # A non-JSON body with a 200 is usually an edge/CDN interstitial.
        _LAST_FETCH_ERROR[cik] = "SEC EDGAR returned a non-JSON response"
        return {}

    if not payload or "facts" not in payload:
        _LAST_FETCH_ERROR[cik] = "the SEC response carried no XBRL facts"
        return {}

    # Persist for a day — filings change quarterly at most.
    storage.cache_set(key, payload, ttl=24 * 3600)
    _LAST_FETCH_ERROR.pop(cik, None)
    return payload


def _entries(facts: dict, concept: str, unit: str = "USD") -> list:
    """
    All reported values for a concept.

    Searches every taxonomy, not just us-gaap, so IFRS filers resolve. If the
    requested unit is absent the first available unit is used — some filers
    report in USD, others in their reporting currency under a different key,
    and returning nothing is worse than returning the value they filed.
    """
    try:
        blocks = facts["facts"]
    except (KeyError, TypeError):
        return []

    for tax in _TAXONOMIES:
        try:
            units = blocks[tax][concept]["units"]
        except (KeyError, TypeError):
            continue
        if unit in units:
            return units[unit]
        for key, vals in units.items():
            if key.startswith(unit[:3]) or unit == "USD":
                return vals
    return []


def extract_latest(facts: dict, concept: str, unit: str = "USD") -> Optional[float]:
    """
    Most recent annual value for a concept.

    Tries the annual forms first (10-K plus 20-F / 40-F for foreign issuers),
    then falls back to the most recent filing of any form. Restricting this to
    10-K alone returned None for every foreign private issuer and for any
    company whose facts carried only quarterlies — which read to the user as
    "no data for this company" when the data was there all along.
    """
    entries = [e for e in _entries(facts, concept, unit) if "end" in e and e.get("val") is not None]
    if not entries:
        return None

    annual = [e for e in entries if e.get("form") in ANNUAL_FORMS]
    pool = annual or entries
    pool.sort(key=lambda x: (x["end"], x.get("filed", "")), reverse=True)
    return pool[0]["val"]


def extract_history(facts: dict, concept: str, unit: str = "USD",
                    forms=ANNUAL_FORMS + ("10-Q", "6-K")) -> list[dict]:
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
