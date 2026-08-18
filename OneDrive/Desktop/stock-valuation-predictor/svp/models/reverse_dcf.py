"""
Reverse DCF — what the market price appears to assume.
======================================================

A DCF asks "given these assumptions, what is it worth?". The reverse question
is often more useful: "given what it trades at, what must be true?" This
module solves the same two-stage DCF the app already uses (:mod:`svp.models
.dcf`) for the near-term FCF growth rate that makes the model's value equal
the market price — using the *same* engine, so the two panes can never
disagree about the math.

Honesty constraints:

- Negative or zero free cash flow has no defined implied growth — say so,
  never coerce.
- The solver has bounds. A price outside what any growth in [-50%, +60%] can
  justify is reported as *capped* at the bound, not as a fabricated number.
- The output is an inference about *expectations embedded in a price under
  this model's other assumptions* (WACC, terminal growth). The verdict text
  keeps that conditionality explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import dcf as dcf_mod

GROWTH_LO = -0.50
GROWTH_HI = 0.60

#: Forms whose values are full-year figures. Quarterly rows are excluded from
#: CAGR — comparing a quarter's revenue to a year's manufactures growth.
ANNUAL_FORMS = ("10-K", "20-F", "40-F")


@dataclass
class ImpliedGrowth:
    implied: Optional[float]          # annual FCF growth the price implies
    capped: Optional[str]             # None | "low" | "high" — hit a solver bound
    wacc: float
    terminal_growth: float
    years: int
    implied_at_wacc_up: Optional[float] = None    # +100bp WACC
    implied_at_wacc_down: Optional[float] = None  # -100bp WACC
    reason: Optional[str] = None      # set when implied is None


def _value_at(growth: float, fcf0: float, shares: float, net_debt: float,
              wacc: float, terminal_growth: float, years: int) -> float:
    return dcf_mod.run_dcf(dcf_mod.DCFInputs(
        fcf0=fcf0, shares=shares, net_debt=net_debt, wacc=wacc,
        terminal_growth=terminal_growth, growth_rate=growth, years=years,
    )).intrinsic_per_share


def _solve(price, fcf0, shares, net_debt, wacc, terminal_growth, years):
    """Bisection on growth. Value is monotonically increasing in growth."""
    lo, hi = GROWTH_LO, GROWTH_HI
    v_lo = _value_at(lo, fcf0, shares, net_debt, wacc, terminal_growth, years)
    v_hi = _value_at(hi, fcf0, shares, net_debt, wacc, terminal_growth, years)
    if price <= v_lo:
        return lo, "low"
    if price >= v_hi:
        return hi, "high"
    for _ in range(80):
        mid = (lo + hi) / 2
        if _value_at(mid, fcf0, shares, net_debt, wacc,
                     terminal_growth, years) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, None


def implied_growth(price: float, fcf0: float, shares: float,
                   net_debt: float = 0.0, wacc: float = 0.09,
                   terminal_growth: float = 0.025,
                   years: int = 5) -> Optional[ImpliedGrowth]:
    """The near-term FCF growth rate embedded in ``price``, or ``None``."""
    if price is None or price <= 0 or shares is None or shares <= 0:
        return None
    if fcf0 is None or fcf0 <= 0:
        return ImpliedGrowth(
            implied=None, capped=None, wacc=wacc,
            terminal_growth=terminal_growth, years=years,
            reason="Reverse DCF is undefined when free cash flow is zero or "
                   "negative — there is no growth rate of a negative number "
                   "that discounts to a positive price.")
    if wacc <= terminal_growth:
        wacc = terminal_growth + 0.01

    g, capped = _solve(price, fcf0, shares, net_debt, wacc,
                       terminal_growth, years)
    up, _ = _solve(price, fcf0, shares, net_debt, wacc + 0.01,
                   terminal_growth, years)
    down, _ = _solve(price, fcf0, shares, net_debt, max(wacc - 0.01,
                     terminal_growth + 0.005), terminal_growth, years)
    return ImpliedGrowth(
        implied=float(g), capped=capped, wacc=wacc,
        terminal_growth=terminal_growth, years=years,
        implied_at_wacc_up=float(up), implied_at_wacc_down=float(down),
    )


def cagr_from_history(hist: list[dict], max_points: int = 6,
                      min_span_years: float = 2.5) -> Optional[tuple[float, float]]:
    """
    Compound annual growth from SEC concept history, or ``None``.

    Takes the records :func:`svp.data.sec.extract_history` produces
    (``{"end", "val", "form"}``), keeps annual filings only, and measures
    endpoint-to-endpoint growth over the most recent ``max_points`` years.
    Returns ``(cagr, span_years)`` so the caller can label the figure with the
    span actually measured instead of a round number it wishes it had.
    """
    from datetime import date

    rows = []
    for r in hist or []:
        if r.get("form") not in ANNUAL_FORMS or not r.get("end"):
            continue
        v = r.get("val")
        if v is None or v <= 0:
            continue
        try:
            rows.append((date.fromisoformat(str(r["end"])[:10]), float(v)))
        except Exception:
            continue
    rows.sort()
    rows = rows[-max_points:]
    if len(rows) < 2:
        return None
    (d0, v0), (d1, v1) = rows[0], rows[-1]
    span = (d1 - d0).days / 365.25
    if span < min_span_years:
        return None
    return float((v1 / v0) ** (1 / span) - 1), float(span)


def verdict(res: ImpliedGrowth, historical_cagr: Optional[float],
            history_label: str = "5-year revenue CAGR") -> str:
    """One measured paragraph. Conditional language, no certainty."""
    if res.implied is None:
        return res.reason or "Not computable."
    pct = res.implied * 100
    if res.capped == "high":
        head = (f"At this price the model cannot find a growth rate below "
                f"{GROWTH_HI * 100:.0f}%/yr that justifies it — the market is "
                f"pricing in more than {GROWTH_HI * 100:.0f}% annual FCF "
                f"growth, or assumptions this model does not capture.")
    elif res.capped == "low":
        head = (f"Even shrinking {abs(GROWTH_LO) * 100:.0f}%/yr, the model "
                f"values the shares above this price — the market appears to "
                f"be pricing in outcomes worse than the model's bounds.")
    else:
        head = (f"At the current price, the market appears to assume about "
                f"{pct:.1f}% annual FCF growth over the next {res.years} "
                f"years (holding this model's {res.wacc * 100:.1f}% discount "
                f"rate and {res.terminal_growth * 100:.1f}% terminal growth).")
    if historical_cagr is not None and res.implied is not None and not res.capped:
        h = historical_cagr * 100
        rel = ("well above" if pct > h + 5 else
               "above" if pct > h + 1 else
               "well below" if pct < h - 5 else
               "below" if pct < h - 1 else "roughly in line with")
        head += (f" That is {rel} the company's {history_label} of {h:.1f}%. "
                 "An implied rate far above delivered history means the price "
                 "leans on acceleration; far below can mean pessimism — or "
                 "risks the model does not see.")
    return head + " Estimated, not guaranteed."
