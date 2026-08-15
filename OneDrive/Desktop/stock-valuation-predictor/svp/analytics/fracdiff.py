"""
Fractional differentiation (López de Prado).
============================================

Integer differencing — the ordinary ``diff()`` — makes a price series stationary
but erases its memory: returns barely correlate with the level they came from, so
a model loses the very information that made the level predictive. Fractional
differentiation takes a real-valued order ``d`` between 0 and 1 and removes just
enough of the trend to reach stationarity while keeping as much memory as
possible. It is the standard first step in a serious ML feature pipeline, and it
is pure arithmetic — no external service, no lookahead.

This is the honest, deployable member of the "advanced feature" family. Where the
grander items in that family (Temporal Fusion Transformers, Neural ODEs) need a
GPU and a gigabyte of PyTorch this app cannot carry, fractional differentiation
is a few lines of numpy that genuinely improve the features a model like the
crypto directional classifier learns from.

The fixed-width window ("FFD") variant is used: weights are truncated once they
fall below a threshold, so every output value is computed from the same finite
window and the transform is causal and comparable across time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def ffd_weights(d: float, thresh: float = 1e-4, max_width: int = 10_000) -> np.ndarray:
    """
    Binomial weights for fixed-width fractional differencing of order ``d``.

    The weight recursion is w_k = -w_{k-1} · (d - k + 1) / k, starting at w_0 = 1.
    Terms are accumulated until their magnitude drops below ``thresh``; that fixed
    width is what makes the transform stationary in its own right rather than
    drifting as the window grows. Returned oldest-weight-last so it convolves
    directly against a trailing window.
    """
    w = [1.0]
    k = 1
    while k < max_width:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
        k += 1
    return np.array(w[::-1])


def frac_diff(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    """
    Fixed-width fractionally-differenced series of order ``d``.

    ``d = 0`` returns the series unchanged (full memory, non-stationary);
    ``d = 1`` is close to an ordinary first difference (stationary, little
    memory). The useful values sit in between. The leading ``len(weights)-1``
    points have no full window and come back as NaN, exactly as any windowed
    transform must — filling them would invent data.
    """
    s = pd.Series(series).astype(float)
    if d == 0:
        return s.copy()
    w = ffd_weights(d, thresh)
    width = len(w)
    if width > len(s):
        return pd.Series(np.nan, index=s.index)

    vals = s.to_numpy()
    out = np.full(len(s), np.nan)
    for i in range(width - 1, len(s)):
        window = vals[i - width + 1:i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(np.dot(w, window))
    return pd.Series(out, index=s.index)


@dataclass(frozen=True)
class FracDiffFit:
    d: float                       # chosen differencing order
    memory_retained: float         # |corr(original, differenced)|, 0-1
    acf1_before: float             # lag-1 autocorrelation of the raw series
    acf1_after: float              # lag-1 autocorrelation of the differenced series
    reached_target: bool

    @property
    def reading(self) -> str:
        if not self.reached_target:
            return ("No order below 1 flattened the autocorrelation to target on "
                    "this series — it is unusually persistent")
        return (f"d = {self.d:.2f} removes the trend (lag-1 autocorrelation "
                f"{self.acf1_before:.2f} → {self.acf1_after:.2f}) while keeping "
                f"{self.memory_retained*100:.0f}% of the original memory")


def _acf1(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return float("nan")
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / denom)


def min_d(series: pd.Series, target_acf1: float = 0.10,
          grid: Optional[np.ndarray] = None, thresh: float = 1e-4) -> Optional[FracDiffFit]:
    """
    Smallest ``d`` whose differenced series drops lag-1 autocorrelation to target.

    Statsmodels' Augmented Dickey-Fuller test is the textbook stationarity gauge,
    but it is not a dependency here and pulling it in for one number is not worth
    the weight. Lag-1 autocorrelation is the honest lightweight stand-in: a
    trending (unit-root) series has it near 1, and fractional differencing pushes
    it down. The function reports the raw and differenced autocorrelations so the
    reader sees the actual reduction rather than trusting a hidden p-value.

    Returns the fit at the smallest ``d`` that reaches ``target_acf1`` — the
    López de Prado principle of differencing *just enough*.
    """
    s = pd.Series(series).astype(float).dropna()
    if len(s) < 100:
        return None
    if grid is None:
        grid = np.round(np.arange(0.0, 1.01, 0.1), 2)

    raw = s.to_numpy()
    acf1_before = _acf1(raw)
    best_fallback = None

    for d in grid:
        fd = frac_diff(s, float(d), thresh).dropna()
        if len(fd) < 50:
            continue
        acf1_after = _acf1(fd.to_numpy())
        aligned = s.reindex(fd.index)
        mem = abs(float(np.corrcoef(aligned.to_numpy(), fd.to_numpy())[0, 1])) \
            if len(fd) > 5 else 0.0
        mem = 0.0 if np.isnan(mem) else mem
        fit = FracDiffFit(float(d), mem, acf1_before,
                          0.0 if np.isnan(acf1_after) else acf1_after,
                          reached_target=abs(acf1_after) <= target_acf1)
        if fit.reached_target:
            return fit
        best_fallback = fit

    # Nothing hit target; report the most-differenced attempt so the UI can say so.
    if best_fallback is not None:
        return FracDiffFit(best_fallback.d, best_fallback.memory_retained,
                           acf1_before, best_fallback.acf1_after, reached_target=False)
    return None
