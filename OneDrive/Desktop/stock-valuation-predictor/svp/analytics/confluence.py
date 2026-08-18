"""
Confluence — do the fundamental and technical reads agree?
==========================================================

One 0–100 score summarising how much the app's independent readings point the
same way: valuation gap, financial quality, balance-sheet risk, technical
alignment, momentum, volatility. Like the technical snapshot's score, this
measures **agreement, not probability of profit** — a high score means the
evidence lines up, not that the outcome is assured, and the takeaway language
stays descriptive ("may be a watchlist candidate"), never imperative.

Two rules keep the number honest:

- **Missing inputs shrink the denominator.** A component the data cannot
  support is skipped and the remaining weights renormalised — never scored
  50 as if mediocrity had been observed.
- **The breakdown ships with the score.** ``components`` carries every
  reading and weight actually used, so the UI can show the arithmetic and a
  test can recompute the total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

#: (weight, label) per component; weights renormalised over what's available.
_WEIGHTS = {
    "Valuation": 0.30,
    "Quality": 0.20,
    "Balance sheet": 0.15,
    "Technical trend": 0.20,
    "Momentum": 0.10,
    "Volatility": 0.05,
}

MIN_COMPONENTS = 3


@dataclass
class Confluence:
    score: int                                  # 0-100
    components: dict[str, tuple[int, float]]    # name -> (reading, weight used)
    strengths: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    takeaway: str = ""


def _clamp(x: float) -> int:
    return int(round(max(0.0, min(100.0, x))))


def _valuation_component(gaps: list[float]) -> int:
    """Average valuation gap mapped to 0-100: -30% → 0, 0 → 50, +50% → 100."""
    g = sum(gaps) / len(gaps)
    if g <= 0:
        return _clamp(50 + g / 0.30 * 50)
    return _clamp(50 + g / 0.50 * 50)


def compute(
    ml_gap: Optional[float] = None,        # (ML fair value − price) / price
    dcf_gap: Optional[float] = None,       # (DCF fair value − price) / price
    piotroski: Optional[int] = None,       # 0-9
    altman_zone: Optional[str] = None,     # Safe | Grey | Distress
    technical_score: Optional[int] = None,  # snapshot alignment 0-100
    trend: Optional[str] = None,           # Bullish | Neutral | Bearish
    momentum: Optional[str] = None,        # Strong | Moderate | Weak
    volatility: Optional[str] = None,      # Low | Normal | Elevated
) -> Optional[Confluence]:
    """The confluence read, or ``None`` when too little is measurable."""
    readings: dict[str, int] = {}

    gaps = [g for g in (ml_gap, dcf_gap) if g is not None]
    if gaps:
        readings["Valuation"] = _valuation_component(gaps)
    if piotroski is not None:
        readings["Quality"] = _clamp(piotroski / 9 * 100)
    zone = (altman_zone or "").strip().title()
    if zone in ("Safe", "Grey", "Distress"):
        readings["Balance sheet"] = {"Safe": 100, "Grey": 50, "Distress": 0}[zone]
    if technical_score is not None:
        readings["Technical trend"] = _clamp(technical_score)
    if momentum in ("Strong", "Moderate", "Weak"):
        up = trend == "Bullish"
        down = trend == "Bearish"
        base = {"Strong": 90, "Moderate": 65, "Weak": 50}[momentum]
        readings["Momentum"] = (base if up
                                else (100 - base) if down
                                else 50)
    if volatility in ("Low", "Normal", "Elevated"):
        readings["Volatility"] = {"Low": 75, "Normal": 60, "Elevated": 30}[volatility]

    if len(readings) < MIN_COMPONENTS or "Valuation" not in readings:
        return None

    total_w = sum(_WEIGHTS[k] for k in readings)
    components = {k: (v, round(_WEIGHTS[k] / total_w, 4))
                  for k, v in readings.items()}
    score = _clamp(sum(v * w for v, w in components.values()))

    strengths = [k for k, (v, _) in components.items() if v >= 70]
    cautions = [k for k, (v, _) in components.items() if v <= 35]

    # ── Takeaway: descriptive, conditional, no instructions ──────────────────
    # Divergence is diagnosed before the overall level: a decent average built
    # out of a cheap price and a broken tape is a disagreement, and calling it
    # "evidence points the same way" would be exactly the false comfort this
    # score exists to avoid.
    val = components["Valuation"][0]
    tech = components.get("Technical trend", (None,))[0]
    if val >= 60 and tech is not None and tech < 45:
        take = ("Valuation appears attractive but the technical picture has "
                "not turned — historically the profile of a watchlist "
                "candidate rather than a confirmed setup.")
    elif val < 45 and tech is not None and tech >= 60:
        take = ("The tape is strong but the valuation models see little "
                "margin of safety — momentum is carrying more of this story "
                "than value is.")
    elif score >= 70:
        take = "Most of the measurable evidence points the same way here."
    elif score <= 35:
        take = "Little in the measurable evidence lines up favourably here."
    else:
        take = ("The readings are mixed — no single dimension dominates, "
                "which usually means the thesis depends on judgement the "
                "models cannot settle.")
    take += " This is a description of agreement, not a recommendation."

    return Confluence(score=score, components=components,
                      strengths=strengths, cautions=cautions, takeaway=take)
