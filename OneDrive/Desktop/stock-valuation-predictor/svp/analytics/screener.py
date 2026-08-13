"""
Quantile ranking / screening engine.
=====================================

Screens a ticker universe and surfaces buy candidates **only** in the top decile
of margin-of-safety (model p50 intrinsic value vs. market price), optionally
blended with the relative-return model's expected excess return.

Designed to be resilient: each ticker is scored independently and failures are
skipped, so one bad symbol never breaks the screen. When live fundamentals are
unreachable the engine still ranks whatever it could compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..features import build_features
from ..models import valuation as val_mod

# A compact default universe (liquid large caps across sectors).
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC",
    "XOM", "CVX", "JNJ", "PFE", "WMT", "KO", "PEP", "DIS", "NFLX", "ORCL", "AMD",
]


@dataclass
class ScreenRow:
    ticker: str
    market_price: float
    intrinsic_p50: float
    intrinsic_low: float
    intrinsic_high: float
    margin_of_safety: float        # (p50 - price) / price
    expected_excess: Optional[float]
    signal: str


def _score_one(ticker: str, vm, rm=None) -> Optional[ScreenRow]:
    try:
        feats = build_features(ticker)
        if feats is None:
            return None
        price = feats.get("market_price") or feats["_raw"].get("market_cap", 0) / max(feats["_raw"].get("shares", 1), 1)
        if not price or price <= 0:
            return None
        res = val_mod.predict(feats, vm, n_mc=200)
        mos = (res.median - price) / price
        excess = None
        if rm is not None:
            from ..models.relative import predict_excess_return
            excess = predict_excess_return(feats, rm)
        label, _ = val_mod.valuation_signal(res.median, price)
        return ScreenRow(ticker, float(price), res.median, res.low, res.high,
                         float(mos), excess, label)
    except Exception:
        return None


def screen(
    universe: Optional[list[str]],
    vm,
    rm=None,
    top_decile_only: bool = True,
    progress=None,
) -> pd.DataFrame:
    """
    Score a universe and return a ranked DataFrame.

    ``progress`` is an optional callback ``fn(done, total, ticker)`` for a UI
    progress bar. When ``top_decile_only`` is True, only names in the top decile
    of margin-of-safety are flagged as buys (column ``TopDecile``).
    """
    universe = universe or DEFAULT_UNIVERSE
    rows: list[ScreenRow] = []
    total = len(universe)
    for i, tk in enumerate(universe):
        r = _score_one(tk, vm, rm)
        if r is not None:
            rows.append(r)
        if progress:
            progress(i + 1, total, tk)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "Ticker": r.ticker,
        "Market Price": r.market_price,
        "Intrinsic p50": r.intrinsic_p50,
        "Range Low (p10)": r.intrinsic_low,
        "Range High (p90)": r.intrinsic_high,
        "Margin of Safety": r.margin_of_safety,
        "Exp. Excess Return": r.expected_excess,
        "Signal": r.signal,
    } for r in rows])

    df = df.sort_values("Margin of Safety", ascending=False).reset_index(drop=True)
    if len(df) >= 3:
        cutoff = df["Margin of Safety"].quantile(0.9)
        df["TopDecile"] = df["Margin of Safety"] >= cutoff
    else:
        df["TopDecile"] = df["Margin of Safety"] > 0
    return df
