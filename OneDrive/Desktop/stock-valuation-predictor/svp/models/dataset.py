"""
A training set built from real companies.
=========================================

The model this replaces was fitted on ``rng.uniform`` draws labelled by a
hand-written formula whose base term was the market price itself — see
``docs/model_provenance.md``. Nothing about that arrangement could be repaired
by changing which features it used, so this module builds the thing that was
missing: a dataset of real filings with an observable target.

**Features** are fundamentals only. Every one is computable from an income
statement, a balance sheet, a cash-flow statement or a macro series, and none
requires a price, a market capitalisation, an enterprise value or a multiple.
The provenance table in the docs justifies each individually.

**The target** is ``log(EV / revenue)`` — what the market actually pays for a
company with these fundamentals. It is scale-free, so a $9 stock and a $900
stock are the same problem, and it is *observable*, which "intrinsic value"
is not. The previous model failed precisely because it needed a ground truth
that does not exist and so invented one.

The consequence is worth stating plainly, and the UI repeats it: this is a
**relative** valuation. It answers "does this trade rich or cheap against
companies with similar fundamentals", not "what is this business worth". If
the entire market is expensive, this model is too.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import pandas as pd

#: Fundamentals only. Compare with the old list, which carried market_price,
#: fcf_yield (÷ market cap) and pe_ratio (price ÷ earnings).
FUNDAMENTAL_FEATURES = [
    "fcf_to_revenue",
    "fcf_to_assets",
    "fcf_to_invested_capital",
    "earnings_to_assets",
    "roe",
    "roa",
    "debt_to_equity",
    "profit_margin",
    "gross_margin",
    "asset_turnover",
    "revenue_yoy",
    "net_income_yoy",
    "fcf_yoy",
    "interest_coverage",
    "current_ratio",
    "accruals",
    "log_assets",
]

TARGET = "log_ev_to_revenue"

#: Guards against division blowing up on a near-zero denominator.
_EPS = 1e-9
#: Multiples outside this band are data errors or shells, not valuations.
EV_SALES_FLOOR, EV_SALES_CEILING = 0.05, 100.0


@dataclass
class TrainingRow:
    ticker: str
    features: dict
    target: float
    ev_to_revenue: float
    as_of: str


@dataclass
class BuildReport:
    requested: int = 0
    built: int = 0
    skipped: dict = field(default_factory=dict)      # ticker -> reason

    def note(self) -> str:
        lines = [f"{self.built} of {self.requested} tickers produced a usable row."]
        if self.skipped:
            by_reason: dict = {}
            for reason in self.skipped.values():
                by_reason[reason] = by_reason.get(reason, 0) + 1
            lines.append("Skipped: " + ", ".join(
                f"{n}× {r}" for r, n in sorted(by_reason.items(),
                                               key=lambda kv: -kv[1])))
        return " ".join(lines)


def _safe(num: Optional[float], den: Optional[float],
          lo: float = -1e6, hi: float = 1e6) -> float:
    """A ratio, or NaN. Never an infinity dressed up as a number."""
    try:
        if num is None or den is None:
            return float("nan")
        den = float(den)
        if abs(den) < _EPS:
            return float("nan")
        v = float(num) / den
        return float(v) if lo <= v <= hi and math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def features_from_fundamentals(f: dict) -> dict:
    """
    Build the feature row from raw filing quantities.

    ``f`` carries the plain statement items: revenue, net_income, assets,
    equity, long_term_debt, op_cash_flow, capex, gross_profit,
    interest_expense, current_assets, current_liabilities, plus prior-year
    revenue/net_income/fcf for the growth terms.
    """
    revenue = f.get("revenue")
    assets = f.get("assets")
    equity = f.get("equity")
    ni = f.get("net_income")
    ocf = f.get("op_cash_flow")
    capex = f.get("capex")
    fcf = f.get("free_cash_flow")
    if fcf is None and ocf is not None:
        fcf = ocf - (capex or 0.0)
    ltd = f.get("long_term_debt") or 0.0
    invested = (equity or 0.0) + ltd

    out = {
        "fcf_to_revenue": _safe(fcf, revenue),
        "fcf_to_assets": _safe(fcf, assets),
        "fcf_to_invested_capital": _safe(fcf, invested if invested else None),
        "earnings_to_assets": _safe(ni, assets),
        "roe": _safe(ni, equity),
        "roa": _safe(ni, assets),
        "debt_to_equity": _safe(ltd, equity),
        "profit_margin": _safe(ni, revenue),
        "gross_margin": _safe(f.get("gross_profit"), revenue),
        "asset_turnover": _safe(revenue, assets),
        "revenue_yoy": _growth(revenue, f.get("revenue_prior")),
        "net_income_yoy": _growth(ni, f.get("net_income_prior")),
        "fcf_yoy": _growth(fcf, f.get("fcf_prior")),
        "interest_coverage": _safe(f.get("ebit"), f.get("interest_expense")),
        "current_ratio": _safe(f.get("current_assets"),
                               f.get("current_liabilities")),
        "accruals": _safe((ni - ocf) if (ni is not None and ocf is not None)
                          else None, assets),
        "log_assets": (math.log(assets) if (assets and assets > 0)
                       else float("nan")),
    }
    return {k: out.get(k, float("nan")) for k in FUNDAMENTAL_FEATURES}


def _growth(cur: Optional[float], prior: Optional[float]) -> float:
    """
    Year-over-year growth, defined only where it means something.

    Growth from a negative base is not a percentage anyone can interpret —
    earnings going from −100 to −50 is not "+50% growth" — so it is NaN
    rather than a number that would train the model on nonsense.
    """
    if cur is None or prior is None:
        return float("nan")
    try:
        prior = float(prior)
        if prior <= 0:
            return float("nan")
        return float(cur) / prior - 1.0
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def target_from_market(enterprise_value: Optional[float],
                       revenue: Optional[float]) -> Optional[float]:
    """
    ``log(EV / revenue)``, or ``None`` when the pair cannot support one.

    The log is what makes the target scale-free and its errors symmetric:
    a multiple of 8 misread as 4 and one of 2 misread as 1 are the same
    mistake, and only in logs do they carry the same loss.
    """
    if not enterprise_value or not revenue or revenue <= 0:
        return None
    ratio = enterprise_value / revenue
    if not (EV_SALES_FLOOR <= ratio <= EV_SALES_CEILING):
        return None
    return float(math.log(ratio))


def enterprise_value(market_cap: Optional[float], total_debt: Optional[float],
                     cash: Optional[float]) -> Optional[float]:
    """EV = market cap + debt − cash. Used for the *target* only."""
    if not market_cap or market_cap <= 0:
        return None
    ev = market_cap + (total_debt or 0.0) - (cash or 0.0)
    return float(ev) if ev > 0 else None


def build(tickers: Iterable[str], fetch: Callable[[str], Optional[dict]],
          pause: float = 0.12,
          progress: Optional[Callable[[int, int, str], None]] = None
          ) -> tuple[pd.DataFrame, BuildReport]:
    """
    Assemble the training frame.

    ``fetch(ticker)`` returns the raw fundamentals plus ``market_cap``,
    ``total_debt`` and ``cash`` for the target — or ``None``. Injected rather
    than imported so the builder is testable offline and the network lives in
    one place.

    ``pause`` throttles between calls; SEC asks for well under 10 requests a
    second and a scraper that ignores that gets the whole deployment blocked.
    """
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    report = BuildReport(requested=len(tickers))
    rows: list[TrainingRow] = []

    for i, tkr in enumerate(tickers, 1):
        if progress:
            progress(i, len(tickers), tkr)
        try:
            raw = fetch(tkr)
        except Exception as exc:
            report.skipped[tkr] = f"fetch failed ({type(exc).__name__})"
            continue
        if not raw:
            report.skipped[tkr] = "no data"
            continue

        ev = enterprise_value(raw.get("market_cap"), raw.get("total_debt"),
                              raw.get("cash"))
        y = target_from_market(ev, raw.get("revenue"))
        if y is None:
            report.skipped[tkr] = "no usable EV/revenue"
            continue

        feats = features_from_fundamentals(raw)
        # A row that is mostly holes teaches the imputer, not the model.
        known = sum(1 for v in feats.values() if not _isnan(v))
        if known < len(FUNDAMENTAL_FEATURES) * 0.6:
            report.skipped[tkr] = "too many missing fundamentals"
            continue

        rows.append(TrainingRow(ticker=tkr, features=feats, target=y,
                                ev_to_revenue=math.exp(y),
                                as_of=str(raw.get("as_of", ""))[:10]))
        if pause:
            time.sleep(pause)

    report.built = len(rows)
    if not rows:
        return pd.DataFrame(columns=FUNDAMENTAL_FEATURES + [TARGET]), report

    df = pd.DataFrame([{**r.features, TARGET: r.target, "ticker": r.ticker,
                        "ev_to_revenue": r.ev_to_revenue, "as_of": r.as_of}
                       for r in rows])
    return df, report


def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def winsorize(df: pd.DataFrame, cols: Optional[list] = None,
              lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Clip feature tails to percentiles.

    Real filings contain genuine extremes — a company with near-zero equity
    reports an ROE in the thousands — and a handful of those dominate a
    squared loss. Clipping keeps the row (its other features are informative)
    while stopping one ratio from steering the fit.
    """
    out = df.copy()
    for c in (cols or FUNDAMENTAL_FEATURES):
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        lo, hi = s.quantile(lower), s.quantile(upper)
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            out[c] = s.clip(lo, hi)
    return out


def implied_value_per_share(pred_log_ev_sales: float, revenue: float,
                            total_debt: Optional[float], cash: Optional[float],
                            shares: float) -> Optional[float]:
    """
    Invert the prediction back to dollars per share, at render time only.

    predicted multiple → implied EV → subtract net debt → divide by shares.
    Returns ``None`` rather than a negative price: equity value floors at
    zero, and a negative "fair value" is an artefact of the arithmetic
    running past the point where the model means anything.
    """
    try:
        if revenue is None or revenue <= 0 or not shares or shares <= 0:
            return None
        ev = math.exp(float(pred_log_ev_sales)) * float(revenue)
        equity = ev - (total_debt or 0.0) + (cash or 0.0)
        if equity <= 0:
            return None
        v = equity / float(shares)
        return float(v) if math.isfinite(v) and v > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None
