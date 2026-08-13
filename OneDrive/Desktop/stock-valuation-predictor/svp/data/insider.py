"""
Insider activity & short-interest pipeline.
===========================================

Corroborates a valuation discrepancy with "smart money" positioning:

  * **SEC Form 4 net insider flow** — parses recent Form 4 filings for the
    company and nets purchase (transaction code ``P``) vs. sale (``S``) share
    counts over a trailing window.
  * **Short interest** — short percent of float (via ``yfinance`` ``info``),
    where a high value signals bearish institutional conviction.

Both degrade gracefully: if the SEC submissions API or yfinance is unreachable,
fields return ``None`` and the UI shows "n/a" rather than failing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import requests

try:
    import yfinance as yf  # type: ignore

    _HAS_YF = True
except Exception:  # pragma: no cover
    _HAS_YF = False

from . import storage

_HEADERS = {"User-Agent": "StockValuationApp contact@example.com"}
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"


@dataclass
class InsiderSignal:
    net_shares: Optional[float]        # + = net buying, - = net selling
    buy_count: int
    sell_count: int
    window_days: int
    short_percent_float: Optional[float]
    source: str

    @property
    def net_direction(self) -> str:
        if self.net_shares is None:
            return "n/a"
        if self.net_shares > 0:
            return "Net buying"
        if self.net_shares < 0:
            return "Net selling"
        return "Neutral"

    @property
    def bullish(self) -> Optional[bool]:
        if self.net_shares is None:
            return None
        return self.net_shares > 0

    @property
    def high_short_interest(self) -> Optional[bool]:
        if self.short_percent_float is None:
            return None
        return self.short_percent_float > 0.10


def _recent_form4_accessions(cik: str, limit: int = 40) -> list[str]:
    """Return recent Form 4 filing accession numbers from the submissions API."""
    try:
        data = requests.get(_SUBMISSIONS.format(cik=cik), headers=_HEADERS, timeout=15).json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accns = recent.get("accessionNumber", [])
        out = [accns[i].replace("-", "") for i, f in enumerate(forms) if f == "4"]
        return out[:limit]
    except Exception:
        return []


def _form4_net(cik: str, accession: str) -> Optional[float]:
    """
    Best-effort parse of a Form 4 XML: net (P - S) non-derivative shares.
    Returns None if the document can't be fetched/parsed.
    """
    cik_int = str(int(cik))
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}"
    try:
        idx = requests.get(url + "/", headers=_HEADERS, timeout=10).text
        m = re.search(r'href="([^"]+\.xml)"', idx)
        if not m:
            return None
        xml_url = "https://www.sec.gov" + m.group(1) if m.group(1).startswith("/") else url + "/" + m.group(1).split("/")[-1]
        xml = requests.get(xml_url, headers=_HEADERS, timeout=10).text
    except Exception:
        return None

    net = 0.0
    # Non-derivative transactions: pair transactionShares with transactionCode.
    for block in re.findall(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", xml, re.S):
        code_m = re.search(r"<transactionCode>\s*([A-Z])\s*</transactionCode>", block)
        shares_m = re.search(r"<transactionShares>.*?<value>\s*([\d.]+)\s*</value>", block, re.S)
        if not code_m or not shares_m:
            continue
        code = code_m.group(1)
        shares = float(shares_m.group(1))
        if code == "P":
            net += shares
        elif code == "S":
            net -= shares
    return net


def _short_percent(ticker: str) -> Optional[float]:
    if not _HAS_YF:
        return None
    try:
        info = yf.Ticker(ticker).get_info()
        v = info.get("shortPercentOfFloat")
        return float(v) if v is not None else None
    except Exception:
        return None


def get_insider_signal(ticker: str, cik: Optional[str], max_filings: int = 12) -> InsiderSignal:
    """Aggregate Form 4 net flow + short interest for a ticker."""
    cache_key = f"insider:{ticker}"
    cached = storage.cache_get(cache_key)
    if cached:
        return InsiderSignal(**cached)

    net_shares: Optional[float] = None
    buys = sells = 0
    source = "unavailable"

    if cik:
        accns = _recent_form4_accessions(cik, limit=max_filings)
        if accns:
            total = 0.0
            got_any = False
            for a in accns:
                n = _form4_net(cik, a)
                if n is None:
                    continue
                got_any = True
                total += n
                if n > 0:
                    buys += 1
                elif n < 0:
                    sells += 1
            if got_any:
                net_shares = total
                source = "sec-form4"

    short_pct = _short_percent(ticker)
    if short_pct is not None and source == "unavailable":
        source = "yfinance"

    sig = InsiderSignal(
        net_shares=net_shares,
        buy_count=buys,
        sell_count=sells,
        window_days=180,
        short_percent_float=short_pct,
        source=source,
    )
    if source != "unavailable":
        storage.cache_set(cache_key, sig.__dict__, ttl=6 * 3600)
    return sig
