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

import math
from dataclasses import dataclass, field
from typing import Optional

from ..data import sec
from ..data.sec import (
    ASSETS_TAGS, NET_INCOME_TAGS, OCF_TAGS,
)


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
    accruals: Optional[float] = None
    accruals_flag: str = "n/a"
    notes: list = field(default_factory=list)
    #: The M-score's eight components, when the filings supported computing it.
    beneish: Optional["BeneishDetail"] = None
    #: Altman Z'' — the variant appropriate to non-manufacturing and tech.
    altman_zz: Optional["AltmanZZ"] = None

    @property
    def is_distressed(self) -> bool:
        return self.altman_z is not None and self.altman_z < 1.81

    @property
    def possible_manipulation(self) -> bool:
        return self.beneish_m is not None and self.beneish_m > -1.78

    @property
    def weak_fundamentals(self) -> bool:
        return self.piotroski is not None and self.piotroski <= 3

    @property
    def high_accruals(self) -> bool:
        return self.accruals is not None and self.accruals > 0.10

    def guardrail_triggered(self) -> bool:
        return (self.is_distressed or self.possible_manipulation
                or self.weak_fundamentals or self.high_accruals)


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


@dataclass
class AltmanZZ:
    """Altman's Z'' — the non-manufacturing / emerging-market variant."""
    z: float
    zone: str                        # Safe | Grey | Distress
    x1: float                        # working capital / assets
    x2: float                        # retained earnings / assets
    x3: float                        # EBIT / assets
    x4: float                        # book equity / total liabilities
    distress_prob: float             # mapped, not a market-implied default rate

    def reading(self) -> str:
        return (
            f"Z'' of {self.z:.2f} places this in the **{self.zone.lower()}** "
            f"zone. The Z'' variant drops the sales-to-assets term of the "
            "original 1968 score and re-fits the rest, because asset turnover "
            "separates manufacturers from each other but says little about a "
            "software or services business — the original Z systematically "
            "reads asset-light firms as riskier than they are. Mapped distress "
            f"likelihood ≈ {self.distress_prob * 100:.0f}%; that is a "
            "calibration of the score's own zones, not a market-implied "
            "default probability."
        )


def altman_z_double_prime(facts, equity_book: Optional[float] = None
                          ) -> Optional[AltmanZZ]:
    """
    Altman's Z''-score for non-manufacturing and technology firms.

    ``Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4``, with X4 taken on **book**
    equity rather than market capitalisation. Zones: above 2.60 safe, 1.10 to
    2.60 grey, below 1.10 distress.

    Why a second score when :func:`altman_z_score` exists: the original Z
    includes sales/assets, which rewards asset-heavy turnover. Applied to a
    software company — little in the way of assets, and revenue that does not
    pass through a factory — it reports distress that is an artefact of the
    business model rather than of the balance sheet.
    """
    assets = _cur(facts, "Assets")
    total_liab = _cur(facts, "Liabilities")
    ca = _cur(facts, "AssetsCurrent")
    cl = _cur(facts, "LiabilitiesCurrent")
    retained = _cur(facts, "RetainedEarningsAccumulatedDeficit")
    ebit = _cur(facts, ["OperatingIncomeLoss"])
    if equity_book is None:
        equity_book = _cur(facts, ["StockholdersEquity",
                                   "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"])

    if not assets or not total_liab:
        return None

    wc = (ca - cl) if (ca is not None and cl is not None) else None
    x1 = (wc / assets) if wc is not None else 0.0
    x2 = (retained / assets) if retained is not None else 0.0
    x3 = (ebit / assets) if ebit is not None else 0.0
    x4 = (equity_book / total_liab) if (equity_book and total_liab) else 0.0

    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    if z > 2.60:
        zone = "Safe"
    elif z >= 1.10:
        zone = "Grey"
    else:
        zone = "Distress"

    return AltmanZZ(z=float(z), zone=zone, x1=float(x1), x2=float(x2),
                    x3=float(x3), x4=float(x4),
                    distress_prob=_distress_probability(z))


def _distress_probability(z: float) -> float:
    """
    Map a Z'' score to a distress likelihood.

    A logistic mapping anchored so the score's own boundaries carry their
    conventional meaning: the distress cut-off (1.10) sits near even odds and
    the safe cut-off (2.60) near one in ten. This is a monotone restatement of
    the zones on a 0-1 scale for the UI — not a hazard model fitted to
    defaults, and the UI says so.
    """
    return float(1.0 / (1.0 + math.exp(1.45 * (z - 1.10))))


# ── Beneish M-Score ───────────────────────────────────────────────────────────
#: Beneish's fitted coefficients, and what each index is asking. Kept beside
#: the maths so the drill-down can explain itself rather than showing eight
#: acronyms.
BENEISH_TERMS = {
    "DSRI": (0.920, "Days sales in receivables",
             "Receivables growing faster than sales — revenue booked but not collected."),
    "GMI":  (0.528, "Gross margin index",
             "Margins deteriorating year over year, a motive to manage earnings."),
    "AQI":  (0.404, "Asset quality index",
             "More of the balance sheet in soft, non-current, non-PPE assets."),
    "SGI":  (0.892, "Sales growth index",
             "Rapid growth: not fraud itself, but the setting where it appears."),
    "DEPI": (0.115, "Depreciation index",
             "Depreciation slowing — useful lives quietly extended."),
    "SGAI": (-0.172, "SG&A index",
             "Overheads rising faster than sales (enters with a negative weight)."),
    "TATA": (4.679, "Total accruals to total assets",
             "Earnings leaning on accruals rather than cash — the heaviest term."),
    "LVGI": (-0.327, "Leverage index",
             "Leverage rising (enters with a negative weight)."),
}

BENEISH_INTERCEPT = -4.84
BENEISH_THRESHOLD = -1.78


@dataclass
class BeneishDetail:
    """The M-score with every component that produced it."""
    m_score: float
    flag: str
    components: dict                 # name -> raw index value
    contributions: dict              # name -> coefficient * index
    missing: list = field(default_factory=list)   # indices defaulted to neutral

    @property
    def is_flagged(self) -> bool:
        return self.m_score > BENEISH_THRESHOLD

    @property
    def top_drivers(self) -> list[tuple[str, float]]:
        """Components pushing the score up hardest, largest first."""
        return sorted(self.contributions.items(), key=lambda kv: -kv[1])[:3]

    def reading(self) -> str:
        if not self.components:
            return "Not enough filing history to compute an M-score."
        head = (f"M-score {self.m_score:.2f} against Beneish's "
                f"{BENEISH_THRESHOLD} threshold — "
                + ("above it, which the model classes as a manipulation-risk "
                   "profile." if self.is_flagged else
                   "below it, so the profile does not resemble the manipulators "
                   "in Beneish's sample."))
        drivers = ", ".join(f"{k} {v:+.2f}" for k, v in self.top_drivers)
        tail = f" Largest contributions: {drivers}."
        if self.missing:
            tail += (f" {len(self.missing)} of 8 indices lacked filing data and "
                     "were held neutral, which pulls the score toward the mean: "
                     + ", ".join(self.missing) + ".")
        return head + tail + (" A flag is a prompt to read the filings, not a "
                              "finding of fraud.")


def beneish_detail(facts) -> Optional[BeneishDetail]:
    """
    The Beneish M-score **with its eight components exposed**.

    The score alone says "manipulation risk" without saying why; the
    components say which of the eight symptoms is actually present, which is
    the difference between a number and a lead worth following. Indices that
    filings do not support are held at their neutral value of 1.0 and named
    in ``missing`` rather than silently shaping the total.
    """
    m, flag, comps, missing = _beneish_core(facts)
    if m is None:
        return None
    contribs = {k: BENEISH_TERMS[k][0] * v for k, v in comps.items()}
    return BeneishDetail(m_score=float(m), flag=flag, components=comps,
                         contributions=contribs, missing=missing)


def beneish_m_score(facts) -> tuple[Optional[float], str]:
    """The M-score and its flag. See :func:`beneish_detail` for the breakdown."""
    m, flag, _, _ = _beneish_core(facts)
    return m, flag


def _beneish_core(facts):
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
        return None, "Unknown", {}, []

    # Indices with no filing support are held neutral; naming them keeps the
    # drill-down honest about how much of the score is actually measured.
    missing: list = []

    def safe(n, d, default=1.0):
        try:
            if n is None or d is None or d == 0:
                return default
            return n / d
        except Exception:
            return default

    # Days Sales in Receivables Index
    if rec is None or rec_p is None:
        missing.append("DSRI")
    dsri = safe(safe(rec, rev), safe(rec_p, rev_p))
    # Gross Margin Index
    gm = safe((rev - cogs), rev) if (cogs is not None) else None
    gm_p = safe((rev_p - cogs_p), rev_p) if (cogs_p is not None) else None
    if not (gm and gm_p):
        missing.append("GMI")
    gmi = safe(gm_p, gm) if (gm and gm_p) else 1.0
    # Asset Quality Index
    aqi_cur = 1 - safe((ca or 0) + (ppe or 0), assets)
    aqi_p = 1 - safe((ca_p or 0) + (ppe_p or 0), assets_p)
    aqi = safe(aqi_cur, aqi_p)
    # Sales Growth Index
    sgi = safe(rev, rev_p)
    # Depreciation Index
    if not (dep and dep_p):
        missing.append("DEPI")
    depi = (
        safe(safe(dep_p, (dep_p or 0) + (ppe_p or 0)),
             safe(dep, (dep or 0) + (ppe or 0)))
        if (dep and dep_p) else 1.0
    )
    # SGA Index
    if not (sga and sga_p):
        missing.append("SGAI")
    sgai = safe(safe(sga, rev), safe(sga_p, rev_p)) if (sga and sga_p) else 1.0
    # Leverage Index
    lvg_cur = safe((ltd or 0) + (cl or 0), assets)
    lvg_p = safe((ltd_p or 0) + (cl_p or 0), assets_p)
    lvgi = safe(lvg_cur, lvg_p)
    # Total Accruals to Total Assets
    if ni is None or ocf is None:
        missing.append("TATA")
    tata = safe(((ni or 0) - (ocf or 0)), assets) if (ni is not None and ocf is not None) else 0.0

    comps = {"DSRI": float(dsri), "GMI": float(gmi), "AQI": float(aqi),
             "SGI": float(sgi), "DEPI": float(depi), "SGAI": float(sgai),
             "TATA": float(tata), "LVGI": float(lvgi)}
    m = BENEISH_INTERCEPT + sum(BENEISH_TERMS[k][0] * v for k, v in comps.items())
    flag = "Likely manipulator" if m > BENEISH_THRESHOLD else "Unlikely"
    return float(m), flag, comps, missing


def accruals_ratio(facts) -> tuple[Optional[float], str]:
    """
    Sloan's accrual ratio: (net income − operating cash flow) / total assets.

    Earnings and cash are the same thing eventually; the gap between them in any
    one year is accruals. Sloan (1996) found that firms whose earnings lean
    heavily on accruals rather than cash go on to underperform — the accrual
    anomaly, one of the more durable results in the literature and still the
    cheapest earnings-quality check there is.

    It sits beside Beneish deliberately. Beneish asks whether the statements
    look manipulated; this asks a narrower and more common question — whether
    reported profit is turning into cash. A company can be entirely honest and
    still fail it, which is why a high ratio is a flag to investigate rather
    than an accusation.

    Positive means earnings exceed operating cash flow. Above roughly 10% of
    assets is the conventional threshold for "high".
    """
    ni, _ = _hist2(facts, NET_INCOME_TAGS)
    ocf, _ = _hist2(facts, OCF_TAGS)
    assets, _ = _hist2(facts, ASSETS_TAGS)

    if ni is None or ocf is None or not assets:
        return None, "n/a"

    ratio = (ni - ocf) / assets
    if ratio > 0.10:
        flag = "High — earnings well ahead of cash"
    elif ratio > 0.05:
        flag = "Elevated"
    elif ratio < -0.05:
        flag = "Cash exceeds earnings — conservative"
    else:
        flag = "Normal"
    return float(ratio), flag


def compute_quality(facts, market_cap: Optional[float]) -> QualityScores:
    """Compute all three scores from SEC facts; returns a :class:`QualityScores`."""
    notes: list = []
    if not facts:
        return QualityScores(None, {}, None, "Unknown", None, "Unknown",
                             notes=["No SEC facts available."])

    f, f_detail = piotroski_f_score(facts)
    if f is None:
        notes.append("Piotroski: insufficient historical data.")
    z, zone = altman_z_score(facts, market_cap)
    if z is None:
        notes.append("Altman Z: missing balance-sheet inputs.")
    m, m_flag = beneish_m_score(facts)
    if m is None:
        notes.append("Beneish M: insufficient two-year data.")
    acc, acc_flag = accruals_ratio(facts)
    if acc is None:
        notes.append("Accruals: needs net income, operating cash flow and assets.")

    detail = beneish_detail(facts)
    zz = altman_z_double_prime(facts)
    if zz is None:
        notes.append("Altman Z'': missing balance-sheet inputs.")

    return QualityScores(f, f_detail, z, zone, m, m_flag, acc, acc_flag, notes,
                         beneish=detail, altman_zz=zz)
