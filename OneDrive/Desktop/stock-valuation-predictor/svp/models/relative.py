"""
Cross-sectional relative-return model.
======================================

Instead of predicting an absolute dollar intrinsic value, this XGBoost model is
trained to predict **forward excess return relative to a sector benchmark**
(e.g. XLK, SPY) over a 1- or 3-month horizon — the quantity that actually drives
cross-sectional stock selection.

Trained on synthetic-but-realistic data consistent with ``FEATURE_COLUMNS`` so it
plugs into the same feature pipeline. Output is an expected excess return (in %),
which the screener ranks across a universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from ..features import FEATURE_COLUMNS

HORIZONS = {"1M": 21, "3M": 63}


@dataclass
class RelativeModel:
    model: xgb.XGBRegressor
    horizon: str
    benchmark: str
    r2: float
    train_medians: pd.Series


def _synthetic_excess_returns(n: int = 4000, horizon: str = "1M", seed: int = 21):
    """Generate features → forward *excess* return (fraction) with realistic noise."""
    rng = np.random.default_rng(seed + len(horizon))
    pe = rng.uniform(5, 50, n)
    roe = rng.uniform(-0.2, 0.4, n)
    roa = rng.uniform(-0.1, 0.25, n)
    de = rng.uniform(0, 3.0, n)
    fcf_y = rng.uniform(-0.05, 0.15, n)
    pm = rng.uniform(-0.2, 0.4, n)
    at = rng.uniform(0.3, 2.0, n)
    rev_yoy = rng.uniform(-0.2, 0.6, n)
    rev_qoq = rng.uniform(-0.15, 0.25, n)
    ni_yoy = rng.uniform(-0.4, 0.8, n)
    fcf_yoy = rng.uniform(-0.4, 0.8, n)
    sentiment = rng.uniform(-1, 1, n)
    cpi = rng.uniform(250, 340, n)
    fed = rng.uniform(0.25, 6.0, n)
    curve = rng.uniform(-1.5, 2.5, n)
    mkt = rng.uniform(10, 500, n)

    # Excess return driven by value + quality + momentum, scaled by horizon.
    h_scale = HORIZONS[horizon] / 21.0
    excess = (
        fcf_y * 0.6
        + roe * 0.15
        + pm * 0.10
        + rev_yoy * 0.08
        + ni_yoy * 0.05
        + sentiment * 0.02
        - (pe - 15) * 0.002
        - de * 0.01
        + curve * 0.005
    ) * h_scale
    excess = excess + rng.normal(0, 0.05 * h_scale, n)

    X = pd.DataFrame({
        "pe_ratio": pe, "roe": roe, "roa": roa, "debt_to_equity": de,
        "fcf_yield": fcf_y, "profit_margin": pm, "asset_turnover": at,
        "revenue_yoy": rev_yoy, "revenue_qoq": rev_qoq, "net_income_yoy": ni_yoy,
        "fcf_yoy": fcf_yoy, "sentiment": sentiment, "cpi": cpi,
        "fed_funds": fed, "yield_curve": curve, "market_price": mkt,
    })[FEATURE_COLUMNS]
    return X, pd.Series(excess, name="excess_return")


def train_relative_model(horizon: str = "1M", benchmark: str = "SPY", seed: int = 21) -> RelativeModel:
    """Train the forward-excess-return model for a horizon/benchmark."""
    if horizon not in HORIZONS:
        horizon = "1M"
    X, y = _synthetic_excess_returns(horizon=horizon, seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = xgb.XGBRegressor(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, verbosity=0,
    )
    model.fit(X_tr, y_tr, verbose=False)
    r2 = float(r2_score(y_te, model.predict(X_te)))
    return RelativeModel(model, horizon, benchmark, r2, X_tr.median())


def predict_excess_return(features: dict, rm: RelativeModel) -> float:
    """Expected forward excess return (fraction) for one feature vector."""
    row = pd.DataFrame([{c: features.get(c, np.nan) for c in FEATURE_COLUMNS}])[FEATURE_COLUMNS]
    row = row.fillna(rm.train_medians)
    return float(rm.model.predict(row)[0])
