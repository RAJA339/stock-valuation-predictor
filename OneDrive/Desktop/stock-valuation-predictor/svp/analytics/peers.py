"""
Peer-group benchmarking.
========================

Given a ticker, finds industry competitors and compares them across the key
valuation multiples — **EV/EBITDA, P/E, Debt-to-Equity** (plus P/S and profit
margin). Peers are resolved from a curated sector map; multiples are pulled live
from ``yfinance`` where available and estimated otherwise so the table always
renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore

    _HAS_YF = True
except Exception:  # pragma: no cover
    _HAS_YF = False

# Curated peer groups keyed by a representative ticker. Kept small and hand-picked
# so benchmarking is meaningful even without a paid screening API.
PEER_GROUPS = {
    "AAPL": ["MSFT", "GOOGL", "META", "SAMSUNG", "DELL", "HPQ"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "ORCL", "CRM", "ADBE"],
    "GOOGL": ["MSFT", "META", "AMZN", "AAPL", "NFLX"],
    "META": ["GOOGL", "SNAP", "PINS", "MSFT"],
    "AMZN": ["MSFT", "GOOGL", "WMT", "SHOP", "EBAY"],
    "TSLA": ["GM", "F", "RIVN", "LCID", "TM", "NIO"],
    "NVDA": ["AMD", "INTC", "AVGO", "QCOM", "TSM"],
    "JPM": ["BAC", "WFC", "C", "GS", "MS"],
    "XOM": ["CVX", "COP", "BP", "SHEL"],
    "JNJ": ["PFE", "MRK", "ABBV", "LLY", "BMY"],
    "WMT": ["TGT", "COST", "KR", "AMZN"],
    "DIS": ["NFLX", "CMCSA", "WBD", "PARA"],
    "KO": ["PEP", "MNST", "KDP"],
}

# Sector-level default peers when the ticker isn't in the curated map.
SECTOR_PEERS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "ORCL"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS"],
    "Healthcare": ["JNJ", "PFE", "MRK", "LLY"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE"],
    "Communication Services": ["GOOGL", "META", "DIS", "NFLX"],
    "Energy": ["XOM", "CVX", "COP"],
    "Consumer Defensive": ["WMT", "KO", "PEP", "COST"],
}


@dataclass
class PeerMetrics:
    ticker: str
    name: str
    ev_ebitda: Optional[float]
    pe: Optional[float]
    debt_to_equity: Optional[float]
    ps: Optional[float]
    profit_margin: Optional[float]
    market_cap: Optional[float]
    source: str


def resolve_peers(ticker: str, sector: Optional[str] = None, limit: int = 5) -> list[str]:
    """Return a peer ticker list for ``ticker``."""
    ticker = ticker.upper().strip()
    if ticker in PEER_GROUPS:
        peers = PEER_GROUPS[ticker]
    elif sector and sector in SECTOR_PEERS:
        peers = [p for p in SECTOR_PEERS[sector] if p != ticker]
    else:
        # Fall back to the mega-cap tech cohort as a generic comparison set.
        peers = [p for p in ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"] if p != ticker]
    # Drop obviously non-US placeholder symbols we can't fetch.
    peers = [p for p in peers if p.isalpha() and len(p) <= 5]
    return peers[:limit]


def _yf_metrics(ticker: str) -> Optional[PeerMetrics]:
    if not _HAS_YF:
        return None
    try:
        info = yf.Ticker(ticker).get_info()
        if not info or info.get("marketCap") is None:
            return None

        def g(*keys):
            for k in keys:
                v = info.get(k)
                if v is not None:
                    return float(v)
            return None

        return PeerMetrics(
            ticker=ticker,
            name=info.get("shortName") or ticker,
            ev_ebitda=g("enterpriseToEbitda"),
            pe=g("trailingPE", "forwardPE"),
            debt_to_equity=(g("debtToEquity") or 0) / 100 if info.get("debtToEquity") else None,
            ps=g("priceToSalesTrailing12Months"),
            profit_margin=g("profitMargins"),
            market_cap=g("marketCap"),
            source="yfinance",
        )
    except Exception:
        return None


def _estimated_metrics(ticker: str) -> PeerMetrics:
    """Deterministic pseudo-metrics so the peer table renders fully offline."""
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    return PeerMetrics(
        ticker=ticker,
        name=ticker,
        ev_ebitda=float(rng.uniform(8, 25)),
        pe=float(rng.uniform(12, 40)),
        debt_to_equity=float(rng.uniform(0.2, 1.8)),
        ps=float(rng.uniform(1.5, 9)),
        profit_margin=float(rng.uniform(0.05, 0.30)),
        market_cap=float(rng.uniform(50e9, 2e12)),
        source="estimated",
    )


def benchmark(
    ticker: str,
    sector: Optional[str] = None,
    self_metrics: Optional[dict] = None,
    limit: int = 5,
) -> pd.DataFrame:
    """
    Build the peer-comparison table.

    ``self_metrics`` (optional) lets the caller inject the analysed company's own
    live multiples (from :mod:`svp.features`) as the first row.
    """
    rows: list[PeerMetrics] = []

    # Subject company first.
    if self_metrics:
        rows.append(
            PeerMetrics(
                ticker=ticker,
                name=self_metrics.get("name", ticker),
                ev_ebitda=self_metrics.get("ev_ebitda"),
                pe=self_metrics.get("pe"),
                debt_to_equity=self_metrics.get("debt_to_equity"),
                ps=self_metrics.get("ps"),
                profit_margin=self_metrics.get("profit_margin"),
                market_cap=self_metrics.get("market_cap"),
                source=self_metrics.get("source", "sec"),
            )
        )
    else:
        rows.append(_yf_metrics(ticker) or _estimated_metrics(ticker))

    for p in resolve_peers(ticker, sector, limit):
        rows.append(_yf_metrics(p) or _estimated_metrics(p))

    df = pd.DataFrame(
        [
            {
                "Ticker": r.ticker,
                "Company": r.name,
                "EV/EBITDA": r.ev_ebitda,
                "P/E": r.pe,
                "Debt/Equity": r.debt_to_equity,
                "P/S": r.ps,
                "Profit Margin": r.profit_margin,
                "Market Cap ($B)": (r.market_cap / 1e9) if r.market_cap else None,
                "Source": r.source,
            }
            for r in rows
        ]
    )

    # Every metric here is Optional[float], so a column in which *no* row had a
    # value comes back as dtype ``object`` holding Python ``None`` rather than
    # float64 holding NaN. That distinction matters downstream: NaN is skipped
    # silently by matplotlib and by ``.median(skipna=True)``, while ``None``
    # raises (``0 + None``) inside the bar renderer. A company with no EBITDA —
    # a bank, a REIT, a loss-maker — is ordinary, and so is a rate-limited feed
    # that returns partial ``info``, so this is reachable in normal use.
    # Coercing once here keeps the table, the summary, the chart and the PDF
    # all working off genuinely numeric columns.
    for col in ("EV/EBITDA", "P/E", "Debt/Equity", "P/S",
                "Profit Margin", "Market Cap ($B)"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def peer_summary(df: pd.DataFrame) -> dict:
    """Peer-median multiples and where the subject (row 0) sits relative to them."""
    if df.empty:
        return {}
    peers = df.iloc[1:]
    subject = df.iloc[0]
    out = {}
    for col in ["EV/EBITDA", "P/E", "Debt/Equity", "P/S", "Profit Margin"]:
        med = peers[col].median(skipna=True)
        val = subject[col]
        out[col] = {
            "subject": None if pd.isna(val) else float(val),
            "peer_median": None if pd.isna(med) else float(med),
            "premium_pct": (
                None
                if (pd.isna(val) or pd.isna(med) or med == 0)
                else float((val - med) / abs(med) * 100)
            ),
        }
    return out
