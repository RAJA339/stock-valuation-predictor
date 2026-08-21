"""
Conformal prediction — intervals with a coverage guarantee.
===========================================================

The app already publishes a p10-p50-p90 band from XGBoost quantile
regressors. Those are *fitted* quantiles: the model was trained to put 10% of
the training distribution below p10, and whether it does that on data it has
not seen is an empirical question — which is exactly why this app keeps a
ledger measuring realised coverage.

**Split-conformal prediction** answers the question differently. Hold out a
calibration set the model never trained on, measure the errors it makes
there, and take the (1-α) quantile of those errors as the interval width.
The resulting interval carries a *distribution-free, finite-sample* guarantee:
coverage is at least 1-α for any exchangeable data and any underlying model,
however badly specified. Nothing is assumed about normality, and nothing is
assumed about the model being right.

The guarantee is worth stating precisely, because it is narrower than it
sounds:

- It is **marginal**, not conditional — 90% of intervals cover across the
  whole population, which does not promise 90% for large caps specifically.
- It requires **exchangeability** between calibration and future data. Markets
  regime-shift, which breaks that assumption; the interval is honest about the
  world the calibration set came from.

:func:`conformalised_quantiles` combines both ideas — the CQR method of
Romano, Patterson and Candès (2019): start from the fitted quantile band,
which adapts its width to the input, then conformalise it so the coverage
guarantee holds anyway. That is strictly better than either piece alone, and
it is what the UI uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class ConformalInterval:
    """A prediction interval and the calibration that produced it."""
    point: float
    lower: float
    upper: float
    alpha: float                    # 1 - target coverage
    n_calibration: int
    width: float
    method: str                     # "absolute-residual" | "cqr"
    quantile_used: float            # the conformal quantile of the scores
    finite_sample_coverage: float   # the guarantee this n actually supports

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def reading(self) -> str:
        return (
            f"${self.lower:,.2f} – ${self.upper:,.2f} around a "
            f"${self.point:,.2f} estimate. Calibrated on {self.n_calibration} "
            f"held-out errors, this interval covers the truth at least "
            f"{self.finite_sample_coverage * 100:.1f}% of the time — a "
            "distribution-free guarantee that holds whatever shape the errors "
            "take and however wrong the model is, provided future data looks "
            "like the calibration data. Regime shifts are exactly what breaks "
            "that proviso."
        )


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    The ⌈(n+1)(1-α)⌉/n empirical quantile of the calibration scores.

    The (n+1) correction is what turns an empirical quantile into a genuine
    finite-sample guarantee; using the plain (1-α) quantile under-covers at
    small n, which is precisely where the guarantee is worth having.
    """
    n = len(scores)
    if n == 0:
        return float("nan")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        # Too few points to certify this level — the honest answer is the max
        # observed error, and the caller reports the coverage it really has.
        return float(np.max(scores))
    return float(np.sort(scores)[k - 1])


def achievable_coverage(n: int, alpha: float) -> float:
    """
    The coverage a calibration set of size ``n`` can actually certify.

    With n points the finest attainable level is 1 - 1/(n+1); asking for 95%
    from 12 calibration points is asking for a guarantee the sample cannot
    support, and this reports what it can.
    """
    if n <= 0:
        return 0.0
    return float(min(1.0 - alpha, 1.0 - 1.0 / (n + 1)))


def calibrate_absolute(predictions: Sequence[float], actuals: Sequence[float],
                       alpha: float = 0.10) -> Optional[float]:
    """
    The conformal radius from held-out absolute errors.

    Returns the half-width to place either side of any future point estimate,
    or ``None`` when there is nothing to calibrate on.
    """
    p = np.asarray(list(predictions), dtype=float)
    a = np.asarray(list(actuals), dtype=float)
    if p.shape != a.shape or p.size == 0:
        return None
    scores = np.abs(p - a)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return None
    return _conformal_quantile(scores, alpha)


def interval_from_residuals(point: float, predictions: Sequence[float],
                            actuals: Sequence[float],
                            alpha: float = 0.10) -> Optional[ConformalInterval]:
    """A symmetric conformal interval around ``point``."""
    radius = calibrate_absolute(predictions, actuals, alpha)
    if radius is None or not math.isfinite(radius):
        return None
    n = int(np.size(np.asarray(list(predictions))))
    return ConformalInterval(
        point=float(point), lower=float(point - radius),
        upper=float(point + radius), alpha=float(alpha), n_calibration=n,
        width=float(2 * radius), method="absolute-residual",
        quantile_used=float(radius),
        finite_sample_coverage=achievable_coverage(n, alpha),
    )


def conformalised_quantiles(point: float, lo_hat: float, hi_hat: float,
                            cal_lo: Sequence[float], cal_hi: Sequence[float],
                            cal_actual: Sequence[float],
                            alpha: float = 0.10) -> Optional[ConformalInterval]:
    """
    Conformalised quantile regression (Romano et al., 2019).

    Takes the model's own quantile band for the point being predicted
    (``lo_hat``/``hi_hat``) and the same band on a calibration set, then
    widens — or narrows — it by the conformal quantile of

        ``E_i = max(cal_lo_i - actual_i, actual_i - cal_hi_i)``

    which is how far outside its own band the model fell. A model whose band
    was already too wide gets a *negative* correction and a tighter interval,
    which is the elegance of the method: it repairs miscalibration in either
    direction rather than only padding.
    """
    lo = np.asarray(list(cal_lo), dtype=float)
    hi = np.asarray(list(cal_hi), dtype=float)
    act = np.asarray(list(cal_actual), dtype=float)
    if not (lo.shape == hi.shape == act.shape) or lo.size == 0:
        return None

    scores = np.maximum(lo - act, act - hi)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return None

    q = _conformal_quantile(scores, alpha)
    if not math.isfinite(q):
        return None

    lower, upper = float(lo_hat - q), float(hi_hat + q)
    if lower > upper:                       # a pathological calibration set
        lower, upper = upper, lower
    n = int(scores.size)
    return ConformalInterval(
        point=float(point), lower=lower, upper=upper, alpha=float(alpha),
        n_calibration=n, width=float(upper - lower), method="cqr",
        quantile_used=float(q),
        finite_sample_coverage=achievable_coverage(n, alpha),
    )


def empirical_coverage(intervals: Sequence[tuple], actuals: Sequence[float]
                       ) -> Optional[float]:
    """
    The realised hit rate of a set of (lower, upper) intervals.

    The guarantee is a floor on coverage; this measures what was delivered.
    A large gap between the two is informative in itself — it usually means
    the exchangeability assumption is being violated.
    """
    pairs = list(intervals)
    acts = list(actuals)
    if not pairs or len(pairs) != len(acts):
        return None
    hits = sum(1 for (lo, hi), a in zip(pairs, acts) if lo <= a <= hi)
    return float(hits / len(pairs))
