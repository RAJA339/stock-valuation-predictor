"""
Macro market-regime classifier.
===============================

Detects the prevailing market regime from yield-curve shifts, VIX levels and
macro indicators, because intrinsic-value signals behave very differently in a
calm regime versus a liquidity crunch.

Primary engine is a **Gaussian Hidden Markov Model** (``hmmlearn``) fit over a
VIX + yield-curve feature series; hidden states are mapped to regime labels by
their mean volatility. If ``hmmlearn`` is unavailable it falls back to a
Gaussian mixture, and finally to a transparent rule-based classifier — all three
expose the same :class:`RegimeResult`.

Each regime carries a ``signal_scaler`` in [0, 1] the execution layer uses to
damp valuation signals when the tape is fragile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore

    _HAS_HMM = True
except Exception:  # pragma: no cover
    _HAS_HMM = False

from sklearn.mixture import GaussianMixture

# Regime labels ordered from calmest to most stressed.
REGIMES = ["Calm", "Neutral", "Stress", "Crisis"]
_SCALERS = {"Calm": 1.0, "Neutral": 0.85, "Stress": 0.5, "Crisis": 0.2}
_DESC = {
    "Calm": "Low volatility, positively sloped curve — value signals reliable.",
    "Neutral": "Normal conditions — standard signal weighting.",
    "Stress": "Elevated VIX / flattening curve — de-rate signals, demand confirmation.",
    "Crisis": "Liquidity crunch / inverted curve — value signals unreliable; capital preservation.",
}


@dataclass
class RegimeResult:
    regime: str
    confidence: float             # 0..1
    signal_scaler: float          # multiplier applied to valuation conviction
    description: str
    vix: float
    yield_curve: float
    timeline: Optional[pd.Series] = None  # historical regime label per date
    source: str = "rule"

    @property
    def is_risk_on(self) -> bool:
        return self.regime in ("Calm", "Neutral")


def _label_from_vix(vix: float, curve: float) -> str:
    """Transparent rule-based mapping used directly and to name HMM states."""
    inverted = curve < 0
    if vix >= 38 or (inverted and vix >= 28):
        return "Crisis"
    if vix >= 26 or (inverted and vix >= 20):
        return "Stress"
    if vix >= 17:
        return "Neutral"
    return "Calm"


def _feature_frame(vix_series: pd.Series, curve: float) -> np.ndarray:
    """Build [level, 1-day change] features for the sequence model."""
    v = vix_series.astype(float).to_numpy()
    dv = np.diff(v, prepend=v[0])
    curve_col = np.full_like(v, curve)
    return np.column_stack([v, dv, curve_col])


def classify(
    vix_level: float,
    yield_curve: float,
    vix_history: Optional[pd.Series] = None,
) -> RegimeResult:
    """
    Classify the current regime.

    If a ``vix_history`` series is supplied, an HMM/GMM is fit to produce a
    historical regime *timeline* and a model-based current label; otherwise the
    robust rule-based label is returned.
    """
    rule_label = _label_from_vix(vix_level, yield_curve)

    # Rule-only path.
    if vix_history is None or len(vix_history) < 60:
        return RegimeResult(
            regime=rule_label,
            confidence=0.6,
            signal_scaler=_SCALERS[rule_label],
            description=_DESC[rule_label],
            vix=vix_level,
            yield_curve=yield_curve,
            source="rule",
        )

    X = _feature_frame(vix_history, yield_curve)
    n_states = 3
    labels = None
    source = "rule"
    try:
        if _HAS_HMM:
            model = GaussianHMM(n_components=n_states, covariance_type="diag",
                                n_iter=100, random_state=42)
            model.fit(X)
            hidden = model.predict(X)
            source = "hmm"
        else:
            model = GaussianMixture(n_components=n_states, random_state=42)
            hidden = model.fit_predict(X)
            source = "gmm"

        # Map hidden states → regime labels by mean VIX of each state.
        state_vix = {s: X[hidden == s, 0].mean() for s in np.unique(hidden)}
        order = sorted(state_vix, key=lambda s: state_vix[s])
        # Lowest-VIX state → Calm, then Neutral, then Stress (3 states).
        state_to_label = {order[0]: "Calm", order[1]: "Neutral", order[-1]: "Stress"}
        labels = pd.Series([state_to_label[h] for h in hidden], index=vix_history.index)

        # Escalate the current label to Crisis via the rule if warranted.
        current = labels.iloc[-1]
        if rule_label == "Crisis":
            current = "Crisis"
        # Confidence: share of recent window in the current state.
        recent = labels.iloc[-20:]
        confidence = float((recent == current).mean())
    except Exception:
        current = rule_label
        confidence = 0.6

    return RegimeResult(
        regime=current,
        confidence=confidence,
        signal_scaler=_SCALERS[current],
        description=_DESC[current],
        vix=vix_level,
        yield_curve=yield_curve,
        timeline=labels,
        source=source,
    )


def detect_from_market(vix_market_data, macro) -> RegimeResult:
    """
    Convenience wrapper: takes a :class:`svp.data.market.MarketData` for ``^VIX``
    and a :class:`svp.data.macro.MacroSnapshot`, returns the current regime.
    """
    hist = getattr(vix_market_data, "history", None)
    vix_series = hist["Close"] if (hist is not None and not hist.empty) else None
    vix_level = float(vix_series.iloc[-1]) if vix_series is not None else 18.0
    return classify(vix_level, macro.yield_curve_10y_2y, vix_series)
