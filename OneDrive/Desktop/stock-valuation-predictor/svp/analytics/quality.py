"""
Fundamental quality & distress guardrails.
==========================================

Computes three classic forensic-accounting scores so the ML model doesn't rate a
structurally broken company as a bargain:

  * **Piotroski F-Score** (0–9) — financial-health trajectory.
  * **Altman Z-Score** — bankruptcy-distress zone.
  * **Beneish M-Score** — earnings-manipulation likelihood.

Inputs come from SEC EDGAR facts (current + prior-year annual values). Each score
degrades gracefully: components whose inputs are missing are skipped and noted,
and a score returns ``None`` only when too little is available to be meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..data import sec


# ── helpers to pull current & prior annual values ────────────────────────────
def _cur(facts, concepts):
    return sec.concept_first(facts, concepts if isinstance(concepts, list) else [concepts])


def _hist2(facts, concepts):
    """Return (latest, prior) annual values for the first available concept."""
    concepts = concepts if isinstance(concepts, list) else [concepts]
    h = sec.history_first(facts, concepts)
    annual = [e for e in h if e["form"] == "10-K"]
    if len(annual) >= 2:
        return annual[-1]["val"], annual[-2]["val"]
    if len(annual) == 1:
        return annual[-1]["val"], None
    return None, None


@dataclass
class QualityScores:
    piotroski: Optional[int]
    piotroski_detail: dict
    altman_z: Optional[float]
    altman_zone: str
    beneish_m: Optional[float]
    beneish_flag: str
    notes: list = field(default_factory=list)

    @property
    def is_distressed(self) -> bool:
        return self.altman_z is not None and self.altman_z < 1.81

    @property
    def possible_manipulation(self) -> bool:
        return self.beneish_m is not None and self.beneish_m > -1.78

    @property
    def weak_fundamentals(self) -> bool:
        return self.piotroski is not None and self.piotroski <= 3

    def guardrail_triggered(self) -> bool:
        return self.is_distressed or self.possible_manipulation or self.weak_fundamentals


# ── Piotroski F-Score ─────────────────────────────────────────────────────────
def piotroski_f_score(facts) -> tuple[Optional[int], dict]:
    detail: dict = {}
    ni, ni_p = _hist2(facts, "NetIncomeLoss")
    assets, assets_p = _hist2(facts, "Assets")
    ocf, _ = _hist2(facts, "NetCashProvidedByUsedInOperatingActivities")
    ltd, ltd_p = _hist2(facts, ["LongTermDebt", "LongTermDebtNoncurrent"])
    ca, ca_p = _hist2(facts, "AssetsCurrent")
    cl, cl_p = _hist2(facts, "LiabilitiesCurrent")
    rev, rev_p = _hist2(facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"])
    gp, gp_p = _hist2(facts, "GrossProfit")
    shares, shares_p = _hist2(facts, ["CommonStockSharesOutstanding", "CommonStockSharesIssued"])

    def add(name, cond):
        if cond is None:
            detail[name] = None
        else:
            detail[name] = 1 if cond else 0

    # Profitability
    add("ROA positive", (ni / assets > 0) if (ni is not None and assets) else None)
    add("Operating cash flow positive", (ocf > 0) if ocf is not None else None)
    add("ROA improving",
        ((ni / assets) > (ni_p / assets_p))
        if (ni is not None and assets and ni_p is not None and assets_p) else None)
    add("Accruals (CFO > NI)", (ocf > ni) if (ocf is not None and ni is not None) else None)
    # Leverage / liquidity
    add("Lower long-term debt ratio",
        ((ltd / assets) < (ltd_p / assets_p))
        if (ltd is not None and assets and ltd_p is not None and assets_p) else None)
    add("Higher current ratio",
        ((ca / cl) > (ca_p / cl_p)) if (ca and cl and ca_p and cl_p) else None)
    add("No dilution (shares not up)",
        (shares <= shares_p) if (shares is not None and shares_p is not None) else None)
    # Operating efficiency
    add("Higher gross margin",
        ((gp / rev) > (gp_p / rev_p)) if (gp and rev and gp_p and rev_p) else None)
    add("Higher asset turnover",
        ((rev / assets) > (rev_p / assets_p)) if (rev and assets and rev_p and assets_p) else None)

    scored = [v for v in detail.values() if v is not None]
    if len(scored) < 4:
        return None, detail
    total = int(sum(scored))
    return total, detail


# ── Altman Z-Score ────────────────────────────────────────────────────────────
def altman_z_score(facts, market_cap: Optional[float]) -> tuple[Optional[float], str]:
    assets = _cur(facts, "Assets")
    total_liab = _cur(facts, "Liabilities")
    ca = _cur(facts, "AssetsCurrent")
    cl = _cur(facts, "LiabilitiesCurrent")
    retained = _cur(facts, "RetainedEarningsAccumulatedDeficit")
    ebit = _cur(facts, ["OperatingIncomeLoss"])
    rev = _cur(facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"])

    if not assets or not total_liab:
        return None, "Unknown"

    wc = (ca - cl) if (ca is not None and cl is not None) else None
    x1 = (wc / assets) if wc is not None else 0.0
    x2 = (retained / assets) if retained is not None else 0.0
    x3 = (ebit / assets) if ebit is not None else 0.0
    x4 = (market_cap / total_liab) if (market_cap and total_liab) else 0.0
    x5 = (rev / assets) if rev is not None else 0.0

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    if z > 2.99:
        zone = "Safe"
    elif z >= 1.81:
        zone = "Grey"
    else:
        zone = "Distress"
    return float(z), zone


# ── Beneish M-Score ───────────────────────────────────────────────────────────
def beneish_m_score(facts) -> tuple[Optional[float], str]:
    # Needs current + prior for eight indices. Missing pieces default to neutral (1.0).
    rec, rec_p = _hist2(facts, "AccountsReceivableNetCurrent")
    rev, rev_p = _hist2(facts, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"])
    cogs, cogs_p = _hist2(facts, "CostOfGoodsAndServicesSold")
    ca, ca_p = _hist2(facts, "AssetsCurrent")
    ppe, ppe_p = _hist2(facts, "PropertyPlantAndEquipmentNet")
    assets, assets_p = _hist2(facts, "Assets")
    dep, dep_p = _hist2(facts, ["DepreciationDepletionAndAmortization",
                                "DepreciationAmortizationAndAccretionNet"])
    sga, sga_p = _hist2(facts, "SellingGeneralAndAdministrativeExpense")
    ni, ni_p = _hist2(facts, "NetIncomeLoss")
    ocf, ocf_p = _hist2(facts, "NetCashProvidedByUsedInOperatingActivities")
    ltd, ltd_p = _hist2(facts, ["LongTermDebt", "LongTermDebtNoncurrent"])
    cl, cl_p = _hist2(facts, "LiabilitiesCurrent")

    if None in (rev, rev_p, assets, assets_p) or not rev_p or not assets_p:
        return None, "Unknown"

    def safe(n, d, default=1.0):
        try:
            if n is None or d is None or d == 0:
                return default
            return n / d
        except Exception:
            return default

    # Days Sales in Receivables Index
    dsri = safe(safe(rec, rev), safe(rec_p, rev_p))
    # Gross Margin Index
    gm = safe((rev - cogs), rev) if (cogs is not None) else None
    gm_p = safe((rev_p - cogs_p), rev_p) if (cogs_p is not None) else None
    gmi = safe(gm_p, gm) if (gm and gm_p) else 1.0
    # Asset Quality Index
    aqi_cur = 1 - safe((ca or 0) + (ppe or 0), assets)
    aqi_p = 1 - safe((ca_p or 0) + (ppe_p or 0), assets_p)
    aqi = safe(aqi_cur, aqi_p)
    # Sales Growth Index
    sgi = safe(rev, rev_p)
    # Depreciation Index
    depi = (
        safe(safe(dep_p, (dep_p or 0) + (ppe_p or 0)),
             safe(dep, (dep or 0) + (ppe or 0)))
        if (dep and dep_p) else 1.0
    )
    # SGA Index
    sgai = safe(safe(sga, rev), safe(sga_p, rev_p)) if (sga and sga_p) else 1.0
    # Leverage Index
    lvg_cur = safe((ltd or 0) + (cl or 0), assets)
    lvg_p = safe((ltd_p or 0) + (cl_p or 0), assets_p)
    lvgi = safe(lvg_cur, lvg_p)
    # Total Accruals to Total Assets
    tata = safe(((ni or 0) - (ocf or 0)), assets) if (ni is not None and ocf is not None) else 0.0

    m = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
         + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    flag = "Likely manipulator" if m > -1.78 else "Unlikely"
    return float(m), flag


def compute_quality(facts, market_cap: Optional[float]) -> QualityScores:
    """Compute all three scores from SEC facts; returns a :class:`QualityScores`."""
    notes: list = []
    if not facts:
        return QualityScores(None, {}, None, "Unknown", None, "Unknown",
                             ["No SEC facts available."])

    f, f_detail = piotroski_f_score(facts)
    if f is None:
        notes.append("Piotroski: insufficient historical data.")
    z, zone = altman_z_score(facts, market_cap)
    if z is None:
        notes.append("Altman Z: missing balance-sheet inputs.")
    m, m_flag = beneish_m_score(facts)
    if m is None:
        notes.append("Beneish M: insufficient two-year data.")

    return QualityScores(f, f_detail, z, zone, m, m_flag, notes)
