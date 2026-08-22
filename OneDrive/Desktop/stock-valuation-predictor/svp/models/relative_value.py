"""
The relative-value model: what multiple do these fundamentals command?
======================================================================

Trains on the real dataset :mod:`svp.models.dataset` builds. The target is
``log(EV / revenue)`` — scale-free, so a $9 stock and a $900 stock pose the
same problem, and observable, which is the property "intrinsic value" lacks
and the reason the previous model had to invent its labels.

What the output means, stated here because the UI must repeat it: the model
predicts the multiple that companies with a given fundamental profile have
historically been assigned **by the market**. The gap between that and the
multiple a company actually trades at is a *relative* signal — rich or cheap
against comparable fundamentals. It is not a claim about what the business is
worth, and if the whole market is expensive this model is expensive with it.

Quantile heads are trained alongside the point model so the band is produced
in the same units as the target and inverted to dollars only at the end.
Nothing here is allowed to return a negative price: equity floors at zero,
and the p10 of −$16.53 the review found was arithmetic running past the point
where the model means anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .dataset import FUNDAMENTAL_FEATURES, TARGET, implied_value_per_share

QUANTILES = (0.10, 0.50, 0.90)


@dataclass
class RelativeValueModel:
    point: object
    quantiles: dict
    feature_cols: list
    train_medians: pd.Series
    r2: float
    mae_log: float
    n_train: int
    n_test: int
    #: Held-out predictions, kept for conformal calibration.
    cal_actual: Optional[np.ndarray] = None
    cal_point: Optional[np.ndarray] = None
    cal_lo: Optional[np.ndarray] = None
    cal_hi: Optional[np.ndarray] = None
    coverage: Optional[float] = None

    @property
    def has_calibration(self) -> bool:
        return self.cal_actual is not None and len(self.cal_actual) > 0

    @property
    def target_description(self) -> str:
        """One sentence, rendered in the UI. Required by P0-1."""
        return (
            "Target: log(enterprise value ÷ revenue), learned from real "
            "filings. The model predicts the revenue multiple that companies "
            "with these fundamentals have carried in the market — a relative "
            "reading against comparable businesses, not an absolute worth."
        )


@dataclass
class RelativeValueResult:
    implied_multiple: float          # EV/revenue the fundamentals imply
    actual_multiple: Optional[float]
    fair_value: Optional[float]      # dollars per share, or None
    low: Optional[float]             # p10, floored at zero
    high: Optional[float]            # p90
    signal_pct: Optional[float]      # vs the market multiple, not the price
    note: str = ""

    @property
    def is_usable(self) -> bool:
        return self.fair_value is not None and self.fair_value > 0


def train(df: pd.DataFrame, seed: int = 42, test_size: float = 0.2
          ) -> Optional[RelativeValueModel]:
    """
    Fit point and quantile heads on the real dataset.

    Returns ``None`` on too little data rather than a model that would report
    metrics from a handful of rows — the failure this whole exercise is about
    is a confident number with nothing behind it.
    """
    if df is None or df.empty or TARGET not in df.columns:
        return None
    data = df.dropna(subset=[TARGET])
    if len(data) < 60:
        return None

    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    cols = [c for c in FUNDAMENTAL_FEATURES if c in data.columns]
    X = data[cols].astype(float)
    y = data[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    common = dict(n_estimators=400, max_depth=4, learning_rate=0.04,
                  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                  reg_lambda=1.0, random_state=seed, verbosity=0)
    point = xgb.XGBRegressor(**common)
    point.fit(X_train, y_train, verbose=False)
    pred = point.predict(X_test)

    quantiles: dict = {}
    for q in QUANTILES:
        try:
            m = xgb.XGBRegressor(objective="reg:quantileerror",
                                 quantile_alpha=q, **common)
            m.fit(X_train, y_train, verbose=False)
        except Exception:
            m = point
        quantiles[q] = m

    try:
        lo = np.asarray(quantiles[QUANTILES[0]].predict(X_test), dtype=float)
        hi = np.asarray(quantiles[QUANTILES[-1]].predict(X_test), dtype=float)
        cov = float(np.mean((y_test.to_numpy() >= lo) & (y_test.to_numpy() <= hi)))
    except Exception:
        lo = hi = None
        cov = None

    return RelativeValueModel(
        point=point, quantiles=quantiles, feature_cols=cols,
        train_medians=medians,
        r2=float(r2_score(y_test, pred)),
        mae_log=float(mean_absolute_error(y_test, pred)),
        n_train=int(len(X_train)), n_test=int(len(X_test)),
        cal_actual=y_test.to_numpy(dtype=float),
        cal_point=np.asarray(pred, dtype=float),
        cal_lo=lo, cal_hi=hi, coverage=cov,
    )


def predict(features: dict, model: RelativeValueModel, revenue: float,
            total_debt: Optional[float], cash: Optional[float],
            shares: float, market_cap: Optional[float] = None
            ) -> Optional[RelativeValueResult]:
    """
    Predict the multiple and invert it to a per-share value.

    Every returned price is positive or ``None``. The band is produced in log
    space and exponentiated, which is what stops a p10 from ever going
    negative: ``exp`` of anything real is positive, and the subsequent net-debt
    subtraction is checked rather than allowed to run past zero.
    """
    if model is None or revenue is None or revenue <= 0:
        return None
    row = pd.DataFrame([{c: features.get(c, np.nan)
                         for c in model.feature_cols}])[model.feature_cols]
    row = row.fillna(model.train_medians)

    try:
        p = float(model.point.predict(row)[0])
        q_lo = float(model.quantiles[QUANTILES[0]].predict(row)[0])
        q_hi = float(model.quantiles[QUANTILES[-1]].predict(row)[0])
    except Exception:
        return None
    q_lo, _, q_hi = sorted([q_lo, p, q_hi])

    fair = implied_value_per_share(p, revenue, total_debt, cash, shares)
    low = implied_value_per_share(q_lo, revenue, total_debt, cash, shares)
    high = implied_value_per_share(q_hi, revenue, total_debt, cash, shares)
    # Equity value floors at zero. A band whose lower leg is wiped out by net
    # debt is reported as zero, not as a negative price.
    low = 0.0 if low is None else max(0.0, low)
    if high is not None:
        high = max(high, low)

    implied_mult = math.exp(p)
    actual_mult = None
    signal = None
    if market_cap and market_cap > 0:
        ev = market_cap + (total_debt or 0.0) - (cash or 0.0)
        if ev > 0:
            actual_mult = ev / revenue
            # The signal lives in multiple space, where the model works.
            signal = (implied_mult / actual_mult - 1.0) * 100.0

    note = ""
    if fair is None:
        note = ("Net debt exceeds the implied enterprise value, so this "
                "approach yields no positive equity value for the name.")

    return RelativeValueResult(
        implied_multiple=float(implied_mult), actual_multiple=actual_mult,
        fair_value=fair, low=low, high=high,
        signal_pct=float(signal) if signal is not None else None, note=note,
    )


@dataclass
class CalibrationSummary:
    """What a calibration run found. Required by P0-2."""
    n: int
    median_abs_signal: float
    mean_abs_signal: float
    coverage: Optional[float]
    band_excludes_market: float          # share of names
    mae_log: Optional[float] = None
    by_decile: dict = field(default_factory=dict)

    #: Past this, the model is broken rather than the market being wrong.
    THRESHOLD = 40.0

    @property
    def is_broken(self) -> bool:
        return self.median_abs_signal > self.THRESHOLD

    def verdict(self) -> str:
        head = (f"{self.n} names: median |signal| "
                f"{self.median_abs_signal:.1f}%, mean "
                f"{self.mean_abs_signal:.1f}%.")
        if self.coverage is not None:
            head += f" p10–p90 covered {self.coverage * 100:.0f}% (nominal 80%)."
        head += (f" The band excluded the market multiple on "
                 f"{self.band_excludes_market * 100:.0f}% of names.")
        if self.is_broken:
            head += (f" **Median |signal| exceeds {self.THRESHOLD:.0f}%, which "
                     "means the model disagrees with the market on most names "
                     "at once — the likelier explanation is the model.**")
        return head


def summarise_calibration(signals, coverage=None, excluded=None,
                          mae_log=None) -> Optional[CalibrationSummary]:
    """Aggregate a calibration sweep into the numbers P0-2 asks for."""
    s = np.asarray([abs(x) for x in signals if x is not None
                    and np.isfinite(x)], dtype=float)
    if s.size == 0:
        return None
    return CalibrationSummary(
        n=int(s.size), median_abs_signal=float(np.median(s)),
        mean_abs_signal=float(s.mean()), coverage=coverage,
        band_excludes_market=float(excluded if excluded is not None else 0.0),
        mae_log=mae_log,
    )
