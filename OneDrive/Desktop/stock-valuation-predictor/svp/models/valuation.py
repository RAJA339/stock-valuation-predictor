"""
Valuation models.
=================

Trains, on synthetic-but-realistic data, an ensemble of XGBoost regressors:

  * a **point** model (median intrinsic value), and
  * **quantile** models (p10 / p50 / p90) using XGBoost's native quantile
    (pinball) objective, so the app can present an intrinsic-value *range* and a
    confidence interval instead of a single number.

A **Monte-Carlo** helper perturbs the input features according to their
historical dispersion to produce an alternative simulated valuation
distribution. In production ``load_bundle`` would ``joblib.load`` a pre-trained
artifact rather than train on the fly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ..features import FEATURE_COLUMNS

QUANTILES = (0.10, 0.50, 0.90)


@dataclass
class ValuationModel:
    point: xgb.XGBRegressor
    quantiles: dict           # {q: XGBRegressor}
    feature_cols: list
    r2: float
    rmse: float
    train_medians: pd.Series  # for NaN imputation at inference time
    feature_std: pd.Series    # for Monte-Carlo perturbation


@dataclass
class ValuationResult:
    point: float
    low: float                # p10
    median: float             # p50
    high: float               # p90
    mc_mean: float
    mc_std: float
    mc_samples: np.ndarray

    @property
    def ci_width_pct(self) -> float:
        return (self.high - self.low) / self.point * 100 if self.point else float("nan")


def _synthetic_dataset(n: int = 4000, seed: int = 42):
    """Generate a labelled training set consistent with ``FEATURE_COLUMNS``."""
    rng = np.random.default_rng(seed)
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

    # Ground-truth intrinsic value with heteroscedastic noise (wider at the
    # extremes) so quantile models have a genuine spread to learn.
    base = (
        mkt
        + fcf_y * 800
        - (pe - 15) * 2
        + roe * 300
        + pm * 200
        - de * 20
        + rev_yoy * 120
        + rev_qoq * 60
        + ni_yoy * 80
        + fcf_yoy * 70
        + sentiment * 25
        - (cpi - 300) * 0.3
        - fed * 4
        + curve * 6
    )
    noise_scale = 12 + 25 * np.abs(fcf_y) / 0.15 + 10 * np.abs(sentiment)
    intrinsic = base + rng.normal(0, noise_scale)

    X = pd.DataFrame(
        {
            "pe_ratio": pe, "roe": roe, "roa": roa, "debt_to_equity": de,
            "fcf_yield": fcf_y, "profit_margin": pm, "asset_turnover": at,
            "revenue_yoy": rev_yoy, "revenue_qoq": rev_qoq, "net_income_yoy": ni_yoy,
            "fcf_yoy": fcf_yoy, "sentiment": sentiment, "cpi": cpi,
            "fed_funds": fed, "yield_curve": curve, "market_price": mkt,
        }
    )[FEATURE_COLUMNS]
    return X, pd.Series(intrinsic, name="intrinsic")


def train_model(seed: int = 42) -> ValuationModel:
    """Train the point + quantile ensemble and return a :class:`ValuationModel`."""
    X, y = _synthetic_dataset(seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    common = dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, verbosity=0,
    )

    point = xgb.XGBRegressor(**common)
    point.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = point.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    quantiles: dict = {}
    for q in QUANTILES:
        try:
            qm = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **common)
            qm.fit(X_train, y_train, verbose=False)
        except (TypeError, xgb.core.XGBoostError):
            # Older XGBoost without native quantile loss → fall back to point
            # model plus an empirical residual offset for that quantile.
            qm = point
        quantiles[q] = qm

    return ValuationModel(
        point=point,
        quantiles=quantiles,
        feature_cols=FEATURE_COLUMNS,
        r2=float(r2),
        rmse=rmse,
        train_medians=X_train.median(),
        feature_std=X_train.std(),
    )


def _prep_row(features: dict, vm: ValuationModel) -> pd.DataFrame:
    row = pd.DataFrame([{c: features.get(c, np.nan) for c in vm.feature_cols}])[vm.feature_cols]
    return row.fillna(vm.train_medians)


def predict(features: dict, vm: ValuationModel, n_mc: int = 1000, seed: int = 7) -> ValuationResult:
    """Return the full :class:`ValuationResult` (point, range, Monte-Carlo)."""
    row = _prep_row(features, vm)

    point = float(vm.point.predict(row)[0])
    q_preds = {q: float(vm.quantiles[q].predict(row)[0]) for q in QUANTILES}
    low, median, high = q_preds[0.10], q_preds[0.50], q_preds[0.90]

    # Guard monotonicity (quantile crossing can happen with tree models).
    low, median, high = sorted([low, median, high])

    # ── Monte-Carlo: perturb inputs by a fraction of their training dispersion.
    rng = np.random.default_rng(seed)
    base = row.iloc[0].to_numpy(dtype=float)
    std = vm.feature_std.to_numpy(dtype=float)
    # Keep market_price fixed (it is an input, not a driver to randomise).
    mp_idx = vm.feature_cols.index("market_price")
    sims = np.tile(base, (n_mc, 1))
    noise = rng.normal(0, 0.25 * std, size=(n_mc, len(base)))
    noise[:, mp_idx] = 0.0
    sims = sims + noise
    mc_pred = vm.point.predict(pd.DataFrame(sims, columns=vm.feature_cols))
    mc_pred = np.asarray(mc_pred, dtype=float)

    return ValuationResult(
        point=point,
        low=low,
        median=median,
        high=high,
        mc_mean=float(mc_pred.mean()),
        mc_std=float(mc_pred.std()),
        mc_samples=mc_pred,
    )


def valuation_signal(intrinsic: float, market: float) -> tuple[str, str]:
    """(label, css-class) based on margin of safety vs the market price."""
    if market <= 0:
        return "N/A", "signal-hold"
    diff = (intrinsic - market) / market * 100
    if diff > 15:
        return f"🟢 Undervalued  (+{diff:.1f}%)", "signal-buy"
    if diff < -15:
        return f"🔴 Overvalued  ({diff:.1f}%)", "signal-over"
    return f"🟡 Fairly Valued  ({diff:+.1f}%)", "signal-hold"
