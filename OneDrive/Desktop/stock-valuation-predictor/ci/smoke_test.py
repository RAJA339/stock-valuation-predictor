"""
Offline pipeline smoke test (CI).

Exercises the full svp stack without touching any external API — training,
quantile + Monte-Carlo prediction, SHAP attribution, DCF, backtest, peers,
sentiment, storage and PDF report generation. Exits non-zero on any failure so
CI fails loudly.
"""

import os
import sys
import warnings

# Ensure the project root (parent of ci/) is importable no matter the cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

import numpy as np

from svp import features as F
from svp.models import valuation as V, explain as X, dcf as D, backtest as B
from svp.analytics import peers as P
from svp.data import storage, sentiment, market
from svp import reports

failures = []


def check(name, cond):
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# ── Model training ────────────────────────────────────────────────────────────
vm = V.train_model()
check("model trains with sane R2", 0.5 < vm.r2 <= 1.0)
check("quantile models present", set(vm.quantiles) == set(V.QUANTILES))

# ── Prediction (point + range + Monte-Carlo) ─────────────────────────────────
feats = {c: np.nan for c in F.FEATURE_COLUMNS}
feats.update({
    "pe_ratio": 28.0, "roe": 0.35, "roa": 0.18, "debt_to_equity": 0.6,
    "fcf_yield": 0.04, "profit_margin": 0.25, "asset_turnover": 0.9,
    "revenue_yoy": 0.12, "revenue_qoq": 0.03, "net_income_yoy": 0.15,
    "fcf_yoy": 0.10, "sentiment": 0.4, "cpi": 314.0, "fed_funds": 4.3,
    "yield_curve": -0.1, "market_price": 190.0,
})
res = V.predict(feats, vm)
check("valuation range is ordered (low<=median<=high)", res.low <= res.median <= res.high)
check("monte-carlo produced samples", len(res.mc_samples) > 100)
check("valuation signal returns a tuple", len(V.valuation_signal(res.point, 190.0)) == 2)

# ── Explainability ────────────────────────────────────────────────────────────
attr = X.explain_prediction(feats, vm)
recon = attr.base_value + sum(attr.contributions)
check("attribution reconstructs prediction", abs(recon - attr.prediction) < 1.0)
check("attribution frame non-empty", not attr.as_frame().empty)

# ── DCF + Monte-Carlo ─────────────────────────────────────────────────────────
dinp = D.DCFInputs(fcf0=1.0e11, shares=1.5e10, net_debt=0.0,
                   wacc=0.09, terminal_growth=0.025, growth_rate=0.08)
dres = D.run_dcf(dinp)
mc = D.monte_carlo_dcf(dinp, n=500)
check("DCF per-share is finite & positive", np.isfinite(dres.intrinsic_per_share) and dres.intrinsic_per_share > 0)
check("DCF Monte-Carlo ordered p10<=median<=p90", mc["p10"] <= mc["median"] <= mc["p90"])

# ── Backtest (offline stub history) ──────────────────────────────────────────
md = market.get_market_data("AAPL", fallback_price=190.0)
check("offline market history generated", len(md.history) > 1000)
bt = B.run_backtest(md.history)
check("backtest returns horizon results", len(bt) >= 1)
check("equity curve builds", B.equity_curve(md.history) is not None)

# ── Sentiment ─────────────────────────────────────────────────────────────────
s = sentiment.score_transcript(sentiment.SAMPLE_TRANSCRIPT)
check("sample transcript scores positive", s.score > 0)

# ── Peers ─────────────────────────────────────────────────────────────────────
pdf = P.benchmark("AAPL", sector="Technology")
check("peer table has rows", len(pdf) >= 2)
check("peer summary computed", len(P.peer_summary(pdf)) > 0)

# ── Storage roundtrip ─────────────────────────────────────────────────────────
storage.cache_set("ci:probe", {"v": 1}, ttl=60)
check("storage roundtrip", storage.cache_get("ci:probe") == {"v": 1})

# ── Report generation ─────────────────────────────────────────────────────────
report = reports.build_report(
    "AAPL", "Apple Inc.", 190.0,
    {"point": res.point, "low": res.low, "median": res.median, "high": res.high},
    "Fairly Valued", attr.as_frame(),
    {"intrinsic_per_share": dres.intrinsic_per_share}, pdf,
    {"cpi": 314.0, "fed_funds": 4.3, "yield_curve": -0.1},
)
check("report produced non-trivial bytes", isinstance(report, (bytes, bytearray)) and len(report) > 500)

# ── Result ────────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"SMOKE TEST FAILED — {len(failures)} check(s): {failures}")
    sys.exit(1)
print("SMOKE TEST PASSED")
