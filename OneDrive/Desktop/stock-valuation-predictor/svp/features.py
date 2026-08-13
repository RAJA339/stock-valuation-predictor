"""
Feature engineering.
=====================

Builds the model feature vector for a ticker by combining:
  * SEC EDGAR fundamentals (levels + derived ratios)
  * Live market data (price, market cap, shares)
  * Real-time macro indicators (CPI, rates, yield curve)
  * Earnings-call sentiment (FinBERT / lexicon)
  * **Trend & delta features** — QoQ and YoY growth-velocity of revenue, net
    income and free cash flow, passed to the model instead of static totals.

``FEATURE_COLUMNS`` is the canonical, ordered list the ML models train and
predict on, so the training synthetic generator and the live extractor stay in
lock-step.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .data import sec, market as market_mod, macro as macro_mod, sentiment as sent_mod

# Canonical, ordered feature list shared by training + inference.
FEATURE_COLUMNS = [
    "pe_ratio",
    "roe",
    "roa",
    "debt_to_equity",
    "fcf_yield",
    "profit_margin",
    "asset_turnover",
    "revenue_yoy",       # trend: YoY revenue growth
    "revenue_qoq",       # trend: QoQ revenue growth
    "net_income_yoy",    # trend: YoY earnings growth
    "fcf_yoy",           # trend: YoY FCF growth
    "sentiment",         # earnings-call tone (-1..1)
    "cpi",
    "fed_funds",
    "yield_curve",       # 10y-2y spread
    "market_price",
]

# Human-friendly labels for UI / SHAP charts.
FEATURE_LABELS = {
    "pe_ratio": "P/E Ratio",
    "roe": "Return on Equity",
    "roa": "Return on Assets",
    "debt_to_equity": "Debt / Equity",
    "fcf_yield": "FCF Yield",
    "profit_margin": "Profit Margin",
    "asset_turnover": "Asset Turnover",
    "revenue_yoy": "Revenue Growth (YoY)",
    "revenue_qoq": "Revenue Growth (QoQ)",
    "net_income_yoy": "Net Income Growth (YoY)",
    "fcf_yoy": "FCF Growth (YoY)",
    "sentiment": "Earnings Sentiment",
    "cpi": "CPI",
    "fed_funds": "Fed Funds Rate",
    "yield_curve": "Yield Curve (10y-2y)",
    "market_price": "Market Price",
}


def _pct_change(history: list[dict], periods: int) -> Optional[float]:
    """Growth of the latest value vs ``periods`` observations earlier."""
    if len(history) <= periods:
        return None
    latest = history[-1]["val"]
    prior = history[-1 - periods]["val"]
    if prior in (0, None):
        return None
    return (latest - prior) / abs(prior)


def _yoy(history: list[dict]) -> Optional[float]:
    """YoY = 4 quarters back if quarterly data present, else 1 annual step."""
    quarterly = [h for h in history if h["form"] == "10-Q"]
    if len(quarterly) > 4:
        return _pct_change(quarterly, 4)
    return _pct_change(history, 1)


def _qoq(history: list[dict]) -> Optional[float]:
    quarterly = [h for h in history if h["form"] == "10-Q"]
    if len(quarterly) > 1:
        return _pct_change(quarterly, 1)
    return None


def build_features(
    ticker: str,
    market_price: Optional[float] = None,
    transcript: Optional[str] = None,
    md: "market_mod.MarketData | None" = None,
    macro: "macro_mod.MacroSnapshot | None" = None,
) -> Optional[dict]:
    """
    Assemble the full feature dict for ``ticker``.

    Parameters mirror the app flow: ``market_price`` overrides the live price
    when the user types one; ``md`` / ``macro`` can be passed to reuse already
    fetched snapshots. Returns ``None`` only when core fundamentals are missing.
    """
    cik = sec.get_cik(ticker)
    if not cik:
        return None
    facts = sec.get_financials(cik)
    if not facts:
        return None

    if md is None:
        md = market_mod.get_market_data(ticker, fallback_price=market_price or 100.0)
    if macro is None:
        macro = macro_mod.get_macro()

    price = market_price or md.price or 100.0

    # ── Levels ───────────────────────────────────────────────────────────────
    # Every field goes through the tag-variant lists: filers label the same
    # line item differently, and a single tag name silently yields None for a
    # large share of real companies (equity especially, where anything with a
    # minority interest uses the long form).
    revenue = sec.concept_first(facts, sec.REVENUE_TAGS)
    net_income = sec.concept_first(facts, sec.NET_INCOME_TAGS)
    total_assets = sec.concept_first(facts, sec.ASSETS_TAGS)
    equity = sec.concept_first(facts, sec.EQUITY_TAGS)
    op_cash_flow = sec.concept_first(facts, sec.OCF_TAGS)
    capex = sec.concept_first(facts, sec.CAPEX_TAGS)
    long_term_debt = sec.concept_first(facts, sec.DEBT_TAGS)
    shares = md.shares_outstanding or sec.extract_latest(
        facts, "CommonStockSharesOutstanding", unit="shares"
    ) or sec.extract_latest(facts, "EntityCommonStockSharesOutstanding", unit="shares")

    # Only a usable price and *some* fundamental grounding are required. The
    # models are gradient-boosted trees, which handle NaN natively, so demanding
    # all three of net income / equity / assets rejected companies whose filings
    # were merely tagged unusually — the user saw "no data" for a company whose
    # data was present. Missing fields flow through as NaN and the affected
    # ratios are simply absent from the explanation.
    if price <= 0:
        return None
    if net_income is None and total_assets is None and revenue is None:
        return None

    missing = [n for n, v in (("net income", net_income), ("equity", equity),
                              ("total assets", total_assets), ("revenue", revenue),
                              ("operating cash flow", op_cash_flow)) if v is None]

    shares = shares or 1e9
    market_cap = md.market_cap or price * shares

    # ── Derived ratios ───────────────────────────────────────────────────────
    eps = net_income / shares if net_income is not None else None
    pe_ratio = price / eps if (eps and eps > 0) else np.nan
    roe = net_income / equity if (net_income is not None and equity) else np.nan
    roa = net_income / total_assets if (net_income is not None and total_assets) else np.nan
    debt_to_equity = (long_term_debt or 0) / equity if equity else np.nan
    free_cash_flow = ((op_cash_flow or 0) - (capex or 0)) if op_cash_flow is not None else None
    fcf_yield = free_cash_flow / market_cap if (free_cash_flow is not None and market_cap) else np.nan
    asset_turnover = revenue / total_assets if (revenue and total_assets) else np.nan
    profit_margin = net_income / revenue if (net_income is not None and revenue) else np.nan

    # ── Trend / delta features ───────────────────────────────────────────────
    rev_hist = sec.history_first(
        facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
    )
    ni_hist = sec.extract_history(facts, "NetIncomeLoss")
    ocf_hist = sec.extract_history(facts, "NetCashProvidedByUsedInOperatingActivities")

    revenue_yoy = _yoy(rev_hist)
    revenue_qoq = _qoq(rev_hist)
    net_income_yoy = _yoy(ni_hist)
    fcf_yoy = _yoy(ocf_hist)  # OCF proxy for FCF trend

    # ── Sentiment ────────────────────────────────────────────────────────────
    sentiment = sent_mod.score_transcript(transcript) if transcript else None
    sentiment_score = sentiment.score if sentiment else 0.0

    features = {
        "pe_ratio": pe_ratio,
        "roe": roe,
        "roa": roa,
        "debt_to_equity": debt_to_equity,
        "fcf_yield": fcf_yield,
        "profit_margin": profit_margin,
        "asset_turnover": asset_turnover,
        "revenue_yoy": revenue_yoy if revenue_yoy is not None else np.nan,
        "revenue_qoq": revenue_qoq if revenue_qoq is not None else np.nan,
        "net_income_yoy": net_income_yoy if net_income_yoy is not None else np.nan,
        "fcf_yoy": fcf_yoy if fcf_yoy is not None else np.nan,
        "sentiment": sentiment_score,
        "cpi": macro.cpi,
        "fed_funds": macro.fed_funds,
        "yield_curve": macro.yield_curve_10y_2y,
        "market_price": price,
    }

    # Raw fundamentals kept alongside for DCF / peer / report / guardrail modules.
    features["_raw"] = {
        "cik": cik,
        "missing_fields": missing,
        "revenue": revenue,
        "net_income": net_income,
        "total_assets": total_assets,
        "equity": equity,
        "op_cash_flow": op_cash_flow,
        "capex": capex,
        "free_cash_flow": free_cash_flow,
        "long_term_debt": long_term_debt,
        "shares": shares,
        "market_cap": market_cap,
        "eps": eps,
        "sector": md.sector,
        "name": md.name or sec.company_name(facts) or ticker,
        "sentiment_obj": sentiment,
    }
    return features


def to_model_row(features: dict) -> "np.ndarray":
    """Extract the ordered numeric vector (excludes ``_raw``) for the model."""
    import pandas as pd

    row = {c: features.get(c, np.nan) for c in FEATURE_COLUMNS}
    return pd.DataFrame([row])[FEATURE_COLUMNS]
