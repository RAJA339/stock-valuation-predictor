"""
Macro-regime dynamic DCF — discount rates that move with the rate environment.
==============================================================================

A DCF quoted at a hand-typed 9% WACC says the same thing in 2021 and in 2023,
which cannot be right: the risk-free leg of every discount rate moved several
hundred basis points between them. This module builds the discount rate from
what the macro feed actually reports, so the valuation responds to the
environment rather than to whatever number was last left in the slider.

The construction, each piece stated so it can be argued with:

``WACC = risk-free + equity risk premium + recession premium``

- **Risk-free** — the effective fed funds rate stands in for the short leg.
  The app's macro feed publishes it; using it means the rate the model
  discounts at moves when policy moves.
- **Equity risk premium** — held at a constant 4.5%. It is not observable and
  a model that pretends to measure it live would be inventing precision.
- **Recession premium** — driven by the 10y-2y spread. An inverted curve has
  preceded every US recession since 1955, and while the timing is useless the
  signal is real, so inversion adds to the premium rather than deciding
  anything on its own.

**Terminal growth** is capped by long-run nominal growth: real trend plus the
inflation the CPI print implies. A terminal rate above nominal GDP growth
implies the firm eventually becomes the economy, which is the single most
common way a DCF is quietly rigged.

Every output carries the inputs that produced it, so the UI can show the
arithmetic instead of asserting a rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Not observable, so it is a stated assumption rather than a measurement.
EQUITY_RISK_PREMIUM = 0.045
#: Long-run US real growth trend; the terminal cap is this plus inflation.
REAL_TREND_GROWTH = 0.018
#: Bounds keep a broken macro print from producing an absurd discount rate.
WACC_FLOOR, WACC_CEILING = 0.05, 0.20
TERMINAL_FLOOR, TERMINAL_CEILING = 0.0, 0.04


@dataclass
class RegimeRates:
    """A discount rate and terminal growth derived from the macro snapshot."""
    wacc: float
    terminal_growth: float
    risk_free: float
    equity_premium: float
    recession_premium: float
    regime: str                    # Easing | Neutral | Restrictive | Inverted
    inflation: float
    yield_curve: float
    source: str                    # macro feed's own source label
    notes: list

    @property
    def is_inverted(self) -> bool:
        return self.yield_curve < 0

    def reading(self) -> str:
        return (
            f"Discount rate {self.wacc * 100:.2f}% = {self.risk_free * 100:.2f}% "
            f"policy rate + {self.equity_premium * 100:.1f}% equity risk premium"
            + (f" + {self.recession_premium * 100:.2f}% recession premium"
               if self.recession_premium > 0 else "")
            + f". Terminal growth capped at {self.terminal_growth * 100:.2f}%, "
            f"which is long-run real trend plus {self.inflation * 100:.1f}% "
            f"inflation — a terminal rate above nominal GDP growth would have "
            "the company eventually becoming the economy. Regime reads "
            f"**{self.regime.lower()}**."
        )


def _regime_label(fed_funds: float, curve: float, inflation: float) -> str:
    if curve < 0:
        return "Inverted"
    if fed_funds >= inflation + 0.015:
        return "Restrictive"
    if fed_funds <= max(inflation - 0.005, 0.0):
        return "Easing"
    return "Neutral"


def _recession_premium(curve: float) -> float:
    """
    Extra discount demanded when the curve inverts.

    Scaled by depth and capped: a 50bp inversion is not the same signal as a
    150bp one, but neither justifies an unbounded premium. Zero when the
    curve is positively sloped — this adds risk, it never subtracts it.
    """
    if curve >= 0:
        return 0.0
    return float(min(abs(curve) / 100.0 * 0.75, 0.02))


def from_macro(macro, base_premium: float = EQUITY_RISK_PREMIUM
               ) -> Optional[RegimeRates]:
    """
    Build :class:`RegimeRates` from a :class:`svp.data.macro.MacroSnapshot`.

    Returns ``None`` if the snapshot lacks the fields needed; the caller then
    keeps its static inputs rather than discounting at an invented rate.
    """
    if macro is None:
        return None
    try:
        fed = float(getattr(macro, "fed_funds"))
        curve = float(getattr(macro, "yield_curve_10y_2y"))
        cpi = float(getattr(macro, "cpi"))
    except (TypeError, ValueError, AttributeError):
        return None

    notes: list = []
    # svp.data.macro publishes rates in percentage points (4.3 means 4.3%), so
    # the conversion is fixed rather than inferred. An earlier version guessed
    # by magnitude — "below 1.0 must already be a fraction" — which silently
    # read a 0.5% policy rate as 50% and produced a 54% discount rate. Unit
    # conventions belong in the contract, not in a heuristic.
    risk_free = fed / 100.0
    curve_pct = curve                      # already in percentage points

    # CPI arrives as an index level (e.g. 314.1), not a rate. An index cannot
    # be read as inflation, so the long-run assumption is used and said so.
    inflation = 0.023
    if 0.0 <= cpi <= 25.0:                 # a rate, not an index
        inflation = cpi / 100.0
    else:
        notes.append("CPI arrives as an index level, so a 2.3% long-run "
                     "inflation assumption is used for the terminal cap "
                     "rather than inferring a rate from one print.")

    rec = _recession_premium(curve_pct)
    if rec > 0:
        notes.append(f"The 10y-2y curve is inverted by {abs(curve_pct):.2f} "
                     "points, which adds a recession premium to the discount "
                     "rate. Inversion has preceded every US recession since "
                     "1955, with timing far too variable to act on alone.")

    wacc = min(max(risk_free + base_premium + rec, WACC_FLOOR), WACC_CEILING)
    terminal = min(max(REAL_TREND_GROWTH + inflation, TERMINAL_FLOOR),
                   TERMINAL_CEILING)
    # Gordon growth diverges as terminal approaches the discount rate.
    if terminal >= wacc - 0.01:
        terminal = max(wacc - 0.02, TERMINAL_FLOOR)
        notes.append("Terminal growth was pulled below the discount rate to "
                     "keep the Gordon-growth term finite.")

    return RegimeRates(
        wacc=float(wacc), terminal_growth=float(terminal),
        risk_free=float(risk_free), equity_premium=float(base_premium),
        recession_premium=float(rec),
        regime=_regime_label(risk_free, curve_pct, inflation),
        inflation=float(inflation), yield_curve=float(curve_pct),
        source=str(getattr(macro, "source", "unknown")), notes=notes,
    )


def sensitivity_to_policy(rates: RegimeRates, shocks_bp=(-100, -50, 0, 50, 100)
                          ) -> list[tuple[int, float]]:
    """
    The discount rate under parallel moves in the policy rate.

    Pairs with the existing DCF rate-shock table: that one shocks the rate
    directly, this one shows what a policy move *implies* for the rate this
    module would build.
    """
    out = []
    for bp in shocks_bp:
        w = min(max(rates.risk_free + bp / 10_000.0 + rates.equity_premium
                    + rates.recession_premium, WACC_FLOOR), WACC_CEILING)
        out.append((int(bp), float(w)))
    return out
