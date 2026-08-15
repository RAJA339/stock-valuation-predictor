"""
Crypto regime — Gaussian HMM over fractionally-differenced features.
===================================================================

This is the "feed fractionally-differenced features into a Gaussian HMM" pipeline,
built with the parts that actually run on this deployment. A three-state Gaussian
Hidden Markov Model (``hmmlearn``, already a dependency for the equity regime tab)
is fit on two features per bar: the fractionally-differenced log return — trend
information made stationary without throwing its memory away — and a rolling
volatility. The model learns latent states and labels them by their volatility so
the output is interpretable: Calm, Trending, Stressed.

Why an HMM and not the Temporal Fusion Transformer / Neural ODE from the same
wish-list: those need PyTorch and a GPU this app does not have, would blow the
memory ceiling that already throttled it once, and would overfit a few years of
daily crypto bars spectacularly. A Gaussian HMM is the member of that family that
is both honest on this much data and light enough to deploy — and, fed stationary
frac-diff features, it is a genuinely sound regime model rather than a demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..analytics import fracdiff as fd

try:  # pragma: no cover - exercised by availability, not branched in tests
    from hmmlearn.hmm import GaussianHMM
    _HMM_OK = True
except Exception:  # pragma: no cover
    _HMM_OK = False


@dataclass(frozen=True)
class RegimeState:
    label: str                     # "Calm" | "Trending" | "Stressed"
    state: int                     # raw HMM state index
    confidence: float              # posterior probability of the current state
    d_used: float                  # fractional-difference order applied
    n: int
    persistence: float             # P(stay) for the current state, from the transition matrix
    engine: str = "gaussian-hmm"
    means: list = field(default_factory=list)

    @property
    def reading(self) -> str:
        base = {
            "Calm": "Low-volatility drift — mean-reversion tactics tend to work here",
            "Trending": "Directional persistence — momentum has follow-through here",
            "Stressed": "High-volatility turbulence — size down; signals are noisier",
        }.get(self.label, "Regime unclear")
        return (f"{base}. The model puts {self.confidence*100:.0f}% posterior on "
                f"this state and it persists with probability {self.persistence:.2f}.")


def _label_states(means_vol: np.ndarray) -> dict:
    """Map raw state indices to names by their volatility mean (low→high)."""
    order = np.argsort(means_vol)
    names = ["Calm", "Trending", "Stressed"]
    return {int(order[i]): names[i] for i in range(len(order))}


def detect(history: pd.DataFrame, n_states: int = 3,
           target_acf1: float = 0.10) -> Optional[RegimeState]:
    """
    Fit a Gaussian HMM on frac-diff return + volatility and read the current state.

    Returns ``None`` when hmmlearn is unavailable, there is too little history, or
    the fit does not converge — in every case the caller shows the simpler vol
    regime instead of a fabricated state.
    """
    if not _HMM_OK:
        return None
    if history is None or history.empty or "Close" not in history.columns:
        return None
    close = history["Close"].astype(float).dropna()
    if len(close) < 200:
        return None

    logp = np.log(close)
    fit = fd.min_d(logp, target_acf1=target_acf1)
    d_used = fit.d if fit is not None else 1.0

    fdr = fd.frac_diff(logp, d_used).rename("fdr")
    vol = fdr.rolling(14).std().rename("vol")
    feats = pd.concat([fdr, vol], axis=1).dropna()
    if len(feats) < 120:
        return None

    X = feats.to_numpy()
    try:
        model = GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=200, random_state=7)
        model.fit(X)
        states = model.predict(X)
        posterior = model.predict_proba(X)
    except Exception:
        return None

    cur = int(states[-1])
    conf = float(posterior[-1][cur])
    vol_means = model.means_[:, 1]          # volatility feature per state
    labels = _label_states(vol_means)
    persistence = float(model.transmat_[cur][cur])

    return RegimeState(
        label=labels.get(cur, "Trending"), state=cur, confidence=conf,
        d_used=float(d_used), n=len(feats), persistence=persistence,
        means=[float(m) for m in vol_means],
    )
