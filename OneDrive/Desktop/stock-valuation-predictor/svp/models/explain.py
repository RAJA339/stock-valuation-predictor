"""
Model transparency / explainable AI.
====================================

Turns a single prediction into a per-feature attribution so a user can see
exactly how each feature (net income → ROE, total debt → D/E, macro indicators …)
pushed the valuation up or down for a specific ticker.

Primary engine is **SHAP** (``TreeExplainer`` → waterfall). If SHAP is missing
the code falls back to XGBoost's built-in per-node contribution (``pred_contribs``),
which yields the same additive decomposition, so a waterfall can always be drawn.
**LIME** is offered as an optional secondary local explainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..features import FEATURE_COLUMNS, FEATURE_LABELS

try:
    import shap  # type: ignore

    _HAS_SHAP = True
except Exception:  # pragma: no cover
    _HAS_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore

    _HAS_LIME = True
except Exception:  # pragma: no cover
    _HAS_LIME = False


@dataclass
class Attribution:
    base_value: float
    prediction: float
    features: list           # ordered feature keys
    values: list             # feature input values
    contributions: list      # additive SHAP-style contributions
    source: str              # "shap" | "xgboost-contribs"

    def as_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "feature": [FEATURE_LABELS.get(f, f) for f in self.features],
                "value": self.values,
                "contribution": self.contributions,
            }
        )
        df["abs"] = df["contribution"].abs()
        return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def _row(features: dict, vm) -> pd.DataFrame:
    r = pd.DataFrame([{c: features.get(c, np.nan) for c in FEATURE_COLUMNS}])[FEATURE_COLUMNS]
    return r.fillna(vm.train_medians)


def explain_prediction(features: dict, vm) -> Attribution:
    """Return an additive :class:`Attribution` for the point prediction."""
    row = _row(features, vm)
    values = row.iloc[0].tolist()

    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(vm.point)
            sv = explainer.shap_values(row)
            contribs = np.asarray(sv)[0]
            base = float(explainer.expected_value)
            pred = float(base + contribs.sum())
            return Attribution(base, pred, FEATURE_COLUMNS, values, contribs.tolist(), "shap")
        except Exception:
            pass

    # Fallback: XGBoost SHAP-equivalent contributions (last col = bias/base).
    booster = vm.point.get_booster()
    import xgboost as xgb

    dm = xgb.DMatrix(row, feature_names=FEATURE_COLUMNS)
    contribs_full = booster.predict(dm, pred_contribs=True)[0]
    base = float(contribs_full[-1])
    contribs = contribs_full[:-1]
    pred = float(base + contribs.sum())
    return Attribution(base, pred, FEATURE_COLUMNS, values, list(map(float, contribs)), "xgboost-contribs")


def lime_explanation(features: dict, vm, n_samples: int = 3000) -> Optional[pd.DataFrame]:
    """Optional LIME local explanation. Returns None if LIME is unavailable."""
    if not _HAS_LIME:
        return None
    try:
        from ..models.valuation import _synthetic_dataset

        X_bg, _ = _synthetic_dataset(n=1500, seed=11)
        explainer = LimeTabularExplainer(
            X_bg.to_numpy(),
            feature_names=FEATURE_COLUMNS,
            mode="regression",
            discretize_continuous=True,
        )
        row = _row(features, vm).iloc[0].to_numpy()
        exp = explainer.explain_instance(
            row, lambda x: vm.point.predict(pd.DataFrame(x, columns=FEATURE_COLUMNS)),
            num_features=len(FEATURE_COLUMNS), num_samples=n_samples,
        )
        data = exp.as_list()
        return pd.DataFrame(data, columns=["feature", "weight"]).sort_values(
            "weight", key=lambda s: s.abs(), ascending=False
        ).reset_index(drop=True)
    except Exception:
        return None


def has_shap() -> bool:
    return _HAS_SHAP


def has_lime() -> bool:
    return _HAS_LIME
