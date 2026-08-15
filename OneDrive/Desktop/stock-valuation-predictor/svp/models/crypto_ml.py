"""
Crypto directional model — a black box that is forced to show its work.
======================================================================

This is the ML price-direction model, built the only way it is honest to build
one: every accuracy number it reports is **out-of-sample**, produced by
walk-forward validation, never by scoring the data it trained on. An in-sample
number on a near-random-walk series like crypto looks impressive and means
nothing; the walk-forward number is the one that survives contact with reality,
and it is usually humbling.

Three guardrails are baked in, not bolted on:

1. **Walk-forward only.** The series is cut into expanding folds; each fold
   trains on the past and predicts the next block it has never seen. The hit
   rate is measured across those held-out blocks, with a Wilson interval — the
   same discipline the indicator-accuracy tool uses.

2. **A baseline it must beat.** "Always predict up" scores well in a bull market
   without any skill at all, so the model's edge is reported *relative to* the
   naive persistence/majority baseline. Beating 50% is not the bar; beating the
   baseline is.

3. **A verdict that can say "no."** When the out-of-sample interval straddles the
   baseline, the model reports that it is no better than chance on this coin — in
   words — instead of showing a confident probability it has not earned. That is
   the whole reason this is safe to ship: it is a model that admits when it knows
   nothing.

The gradient-boosted classifier itself (sklearn ``HistGradientBoostingClassifier``)
is deliberately unremarkable. The value is not the box; it is the measurement
around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..analytics.accuracy import wilson_interval

# Feature lookbacks in bars. Short and medium momentum, plus volatility and
# range context — the ordinary inputs a directional model uses. Nothing here is
# secret; the honesty is in how the output is validated, not in the features.
_MOMENTUM_LAGS = (1, 3, 5, 10, 20)
_VOL_WINDOW = 14
_MIN_ROWS = 260          # enough history to train and still hold out folds


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn an OHLCV frame into a model feature matrix.

    Every feature is causal — computed only from bars at or before its row — so
    there is no lookahead leaking a future value into a past prediction, which is
    the most common way a backtest lies.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    c = df["Close"].astype(float)
    out = pd.DataFrame(index=df.index)

    logret = np.log(c).diff()
    for lag in _MOMENTUM_LAGS:
        out[f"mom_{lag}"] = c.pct_change(lag)
    out["ret_1"] = logret
    out["vol_14"] = logret.rolling(_VOL_WINDOW).std()
    out["vol_ratio"] = (logret.rolling(5).std()
                        / logret.rolling(_VOL_WINDOW).std().replace(0, np.nan))

    # RSI(14), self-contained so this module does not depend on the indicators.
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    if "Volume" in df.columns:
        v = df["Volume"].astype(float)
        out["vol_z"] = ((v - v.rolling(20).mean())
                        / v.rolling(20).std().replace(0, np.nan))
    if {"High", "Low"} <= set(df.columns):
        out["range"] = (df["High"] - df["Low"]) / c

    return out


@dataclass(frozen=True)
class CryptoForecast:
    n_oos: int                       # out-of-sample predictions scored
    hit_rate: float                  # OOS directional accuracy
    ci_low: float
    ci_high: float
    baseline_rate: float             # naive persistence/majority accuracy
    horizon: int
    prob_up: Optional[float]         # model's current call, probability of up
    features_used: list = field(default_factory=list)

    @property
    def beats_baseline(self) -> bool:
        """The interval must clear the baseline, not merely exceed it by a hair."""
        return self.n_oos >= 40 and self.ci_low > self.baseline_rate

    @property
    def edge_pts(self) -> float:
        return (self.hit_rate - self.baseline_rate) * 100.0

    @property
    def verdict(self) -> str:
        if self.n_oos < 40:
            return "Not enough out-of-sample bars to judge the model yet"
        if self.ci_low > max(self.baseline_rate, 0.50):
            return ("Out-of-sample edge over the naive baseline on this coin — "
                    "real on this sample, though never a guarantee forward")
        if self.ci_high < 0.50:
            return "The model is worse than a coin flip here — do not use its call"
        return ("No better than the naive baseline once the interval is honest — "
                "the model has no measurable edge on this coin")

    @property
    def call(self) -> str:
        if self.prob_up is None or not self.beats_baseline:
            return "No actionable call"
        return "Up" if self.prob_up >= 0.5 else "Down"


def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Shallow and lightly regularised: on a near-random-walk series a deep model
    # just memorises noise, which walk-forward then exposes as a wasted fold.
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=120, learning_rate=0.05,
        l2_regularization=1.0, random_state=7,
    )


def evaluate(df: pd.DataFrame, horizon: int = 3, n_folds: int = 6) -> Optional[CryptoForecast]:
    """
    Walk-forward evaluate a directional model on one coin's history.

    ``horizon`` is bars ahead the sign is predicted over. The frame is cut into
    ``n_folds`` expanding blocks; each trains on everything before it and predicts
    the block it has not seen. Accuracy is pooled across the held-out blocks, so
    it is genuinely out-of-sample. Returns ``None`` when there is too little
    history to hold anything out.
    """
    feats = build_features(df)
    if feats.empty or len(feats) < _MIN_ROWS:
        return None

    c = df["Close"].astype(float)
    target = (c.shift(-horizon) > c).astype(int)      # 1 if price is higher in H bars

    data = feats.copy()
    data["_y"] = target
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < _MIN_ROWS:
        return None

    X = data.drop(columns="_y").to_numpy()
    y = data["_y"].to_numpy()
    cols = [c for c in data.columns if c != "_y"]
    n = len(y)

    # Expanding walk-forward: start predicting only after an initial train block,
    # then step forward one fold at a time.
    start = int(n * 0.5)
    step = max(1, (n - start) // n_folds)
    oos_pred, oos_true, oos_persist = [], [], []

    for edge in range(start, n - horizon, step):
        Xtr, ytr = X[:edge], y[:edge]
        if len(np.unique(ytr)) < 2:
            continue
        stop = min(edge + step, n - horizon)
        try:
            clf = _model()
            clf.fit(Xtr, ytr)
            pred = clf.predict(X[edge:stop])
        except Exception:
            return None
        oos_pred.extend(pred.tolist())
        oos_true.extend(y[edge:stop].tolist())
        # Persistence baseline: predict the last observed direction.
        prev_dir = (data["ret_1"].to_numpy()[edge - 1:stop - 1] > 0).astype(int)
        oos_persist.extend(prev_dir.tolist())

    if len(oos_true) < 40:
        return None

    oos_pred = np.array(oos_pred)
    oos_true = np.array(oos_true)
    oos_persist = np.array(oos_persist)

    hits = int((oos_pred == oos_true).sum())
    n_oos = len(oos_true)
    rate = hits / n_oos
    lo, hi = wilson_interval(hits, n_oos)

    # Baseline: the better of "always the majority class" and "persistence".
    majority = max(oos_true.mean(), 1 - oos_true.mean())
    persist_acc = float((oos_persist == oos_true).mean())
    baseline = max(majority, persist_acc)

    # Current live call: train on everything, predict the latest row.
    prob_up = None
    try:
        clf = _model()
        clf.fit(X, y)
        if hasattr(clf, "predict_proba"):
            prob_up = float(clf.predict_proba(X[-1:])[0][1])
    except Exception:
        prob_up = None

    return CryptoForecast(
        n_oos=n_oos, hit_rate=rate, ci_low=lo, ci_high=hi,
        baseline_rate=float(baseline), horizon=horizon, prob_up=prob_up,
        features_used=cols,
    )
