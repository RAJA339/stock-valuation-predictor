"""
One verdict, one place. Every card and caption renders from it.
===============================================================

The review found three bugs of the same shape: a banner reading "the market
sits within the model's range, so the signal is weak" directly above a card
reading **Overvalued (−96.9%)** in red; a caption promising that "agreement
between them strengthens the valuation thesis" printed under three estimates
that agreed about nothing; and a break-even readout of "−200bp" for a company
whose fair value never exceeded its market price at any modelled rate.

None of those was a calculation error. Each was a component deciding for
itself what the numbers meant. The fix is structural rather than textual:
:func:`arbitrate` is the only thing permitted to decide, and every consumer
renders what it returns. A component that re-derives the verdict can drift
from the others; one that reads a field cannot.

The ordering of the checks is deliberate and is the substance of the design.
Disagreement between models is checked *first*, because a confident number
computed from two estimates that point opposite ways is worse than no number.
Width is checked next: an interval wider than the estimate itself cannot
support a direction. Only after both pass does the market's position inside
or outside the band mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Verdict(Enum):
    """The five states. Order matters — see :func:`arbitrate`."""
    NO_ESTIMATE = "no_estimate"
    MODELS_DISAGREE = "models_disagree"
    INSUFFICIENT_SIGNAL = "insufficient_signal"
    INSIDE_RANGE = "inside_range"
    UNDERVALUED = "undervalued"
    OVERVALUED = "overvalued"


#: Wider than this, relative to the point estimate, and the band is not
#: describing a value — it is describing ignorance.
WIDTH_LIMIT = 1.5


@dataclass(frozen=True)
class Arbitration:
    """What the app is allowed to say. Every consumer reads these fields."""
    verdict: Verdict
    headline: str
    detail: str
    #: None wherever a percentage would imply confidence the state denies.
    signal_pct: Optional[float]
    css_class: str
    show_percentage: bool
    show_dcf: bool
    dcf_note: str = ""

    @property
    def is_directional(self) -> bool:
        return self.verdict in (Verdict.UNDERVALUED, Verdict.OVERVALUED)

    @property
    def badge(self) -> str:
        return {
            Verdict.NO_ESTIMATE: "No estimate",
            Verdict.MODELS_DISAGREE: "Models disagree",
            Verdict.INSUFFICIENT_SIGNAL: "Signal too wide",
            Verdict.INSIDE_RANGE: "Inside the range",
            Verdict.UNDERVALUED: "Undervalued",
            Verdict.OVERVALUED: "Overvalued",
        }[self.verdict]


def dcf_is_meaningful(trailing_fcf: Optional[float],
                      dcf_value: Optional[float]) -> tuple[bool, str]:
    """
    Whether a DCF figure may be shown at all.

    A discounted-cash-flow model discounts cash flows. Given a company that
    burns cash, the arithmetic still returns a number — it is just a negative
    equity value with no interpretation, and everything derived from it
    inherits that. The review found the consequence: a rate-sensitivity panel
    reporting "breaks even at −200bp" for a name whose value never exceeded
    its price at any rate on the curve.
    """
    if trailing_fcf is None:
        return False, ("No trailing free-cash-flow figure could be read from "
                       "the filings, so a DCF cannot be built for this name.")
    if trailing_fcf <= 0:
        return False, ("Negative free cash flow — a DCF is not meaningful "
                       "here. Discounting a cash burn produces a negative "
                       "equity value, which is arithmetic rather than a "
                       "valuation. Use the peer multiples and the asset-based "
                       "view instead.")
    if dcf_value is not None and dcf_value <= 0:
        return False, ("The projection produces a negative equity value, so "
                       "no DCF figure is shown. That result says the modelled "
                       "cash flows do not cover the debt, not that the shares "
                       "are worth less than nothing.")
    return True, ""


def arbitrate(ml_value: Optional[float], market_price: Optional[float],
              band_low: Optional[float] = None,
              band_high: Optional[float] = None,
              dcf_value: Optional[float] = None,
              trailing_fcf: Optional[float] = None) -> Arbitration:
    """
    Decide, once, what the app says about this name.

    Checked in this order, and the order is the point:

    1. ``NO_ESTIMATE`` — nothing usable to reason from.
    2. ``MODELS_DISAGREE`` — the ML and DCF signals point opposite ways.
       Two estimates that disagree do not average into a view.
    3. ``INSUFFICIENT_SIGNAL`` — the band is wider than the estimate.
    4. ``INSIDE_RANGE`` — the market sits within the band. **This is the
       branch the review found broken**: it must never emit a directional
       call, because a market price the model cannot distinguish from its own
       estimate is the definition of no signal.
    5. ``UNDERVALUED`` / ``OVERVALUED`` — everything agrees and the market is
       outside the band.
    """
    show_dcf, dcf_note = dcf_is_meaningful(trailing_fcf, dcf_value)

    if not market_price or market_price <= 0 or ml_value is None:
        return Arbitration(
            verdict=Verdict.NO_ESTIMATE,
            headline="No usable estimate",
            detail=("The model could not produce a value for this name from "
                    "the available filings, so no comparison against the "
                    "market price is offered."),
            signal_pct=None, css_class="signal-hold", show_percentage=False,
            show_dcf=show_dcf, dcf_note=dcf_note)

    signal = (ml_value - market_price) / market_price * 100.0

    # 2 — models disagree. Only meaningful when a DCF is showable at all; a
    # suppressed DCF is silent rather than dissenting.
    if show_dcf and dcf_value is not None:
        dcf_signal = dcf_value - market_price
        ml_signal = ml_value - market_price
        if dcf_signal * ml_signal < 0:
            return Arbitration(
                verdict=Verdict.MODELS_DISAGREE,
                headline="Models disagree — no view",
                detail=(f"The cash-flow model puts fair value at "
                        f"${dcf_value:,.2f} and the statistical model at "
                        f"${ml_value:,.2f}, on opposite sides of the "
                        f"${market_price:,.2f} market price. Two estimates "
                        "that point opposite ways do not average into a "
                        "view; the disagreement is the finding."),
                signal_pct=None, css_class="signal-hold",
                show_percentage=False, show_dcf=show_dcf, dcf_note=dcf_note)

    # 3 — the band is too wide to carry a direction.
    if band_low is not None and band_high is not None and ml_value:
        width = abs(band_high - band_low)
        if abs(ml_value) > 0 and width / abs(ml_value) > WIDTH_LIMIT:
            return Arbitration(
                verdict=Verdict.INSUFFICIENT_SIGNAL,
                headline="Estimate too uncertain to call",
                detail=(f"The ${band_low:,.2f}–${band_high:,.2f} range is "
                        f"{width / abs(ml_value) * 100:.0f}% of the "
                        f"${ml_value:,.2f} estimate. An interval wider than "
                        "the estimate it surrounds is describing uncertainty, "
                        "not value, so no direction is claimed."),
                signal_pct=None, css_class="metric-sub",
                show_percentage=False, show_dcf=show_dcf, dcf_note=dcf_note)

    # 4 — inside the band. The broken branch: no directional call here.
    if (band_low is not None and band_high is not None
            and band_low <= market_price <= band_high):
        return Arbitration(
            verdict=Verdict.INSIDE_RANGE,
            headline="Trading inside the estimated range",
            detail=(f"At ${market_price:,.2f} the market sits inside the "
                    f"${band_low:,.2f}–${band_high:,.2f} range, so the model "
                    "cannot distinguish the price from its own estimate. "
                    "That is no signal rather than a weak one — no over- or "
                    "undervalued call is made."),
            signal_pct=None, css_class="signal-hold", show_percentage=False,
            show_dcf=show_dcf, dcf_note=dcf_note)

    # 5 — a real call.
    under = signal > 0
    return Arbitration(
        verdict=Verdict.UNDERVALUED if under else Verdict.OVERVALUED,
        headline=("Trading below the estimated range" if under
                  else "Trading above the estimated range"),
        detail=(f"At ${market_price:,.2f} the market sits "
                f"{'below' if under else 'above'} the model's "
                f"${band_low:,.2f}–${band_high:,.2f} range, a "
                f"{abs(signal):.1f}% gap to the ${ml_value:,.2f} estimate."
                + ("" if not show_dcf else
                   " The cash-flow model agrees in direction.")),
        signal_pct=float(signal),
        css_class="signal-buy" if under else "signal-sell",
        show_percentage=True, show_dcf=show_dcf, dcf_note=dcf_note)


def reconciliation_copy(a: Arbitration, ml_value: Optional[float],
                        dcf_value: Optional[float],
                        market_price: Optional[float]) -> str:
    """
    The prose under the three estimates, selected by the verdict.

    Replaces the fixed sentence the review caught — "agreement between them
    strengthens the valuation thesis" — which was written for a case that was
    not occurring and printed regardless of whether it was.
    """
    if not a.show_dcf:
        return (f"Only the statistical estimate is shown. {a.dcf_note} "
                "Comparing two numbers when one of them is undefined would "
                "manufacture an agreement or a conflict that does not exist.")
    if a.verdict is Verdict.MODELS_DISAGREE:
        return ("The two models point opposite ways. That divergence is worth "
                "more attention than either figure: it usually means the "
                "cash-flow assumptions and the fundamental profile are telling "
                "different stories, and the assumptions are where to look.")
    if a.verdict is Verdict.INSUFFICIENT_SIGNAL:
        return ("The estimates are shown for reference, but the interval "
                "around them is too wide to support a comparison.")
    if (ml_value is not None and dcf_value is not None
            and market_price and market_price > 0):
        spread = abs(ml_value - dcf_value) / market_price * 100.0
        if spread < 20:
            return (f"The two models land within {spread:.0f}% of each other "
                    "relative to the market price. Independent methods "
                    "converging is the strongest evidence this app can offer "
                    "— though both could share an error in the inputs.")
        return (f"The two estimates differ by {spread:.0f}% of the market "
                "price. They agree on direction but not on magnitude, so the "
                "assumptions behind the wider one are worth revisiting.")
    return "Both estimates are shown; compare them against the market price."
