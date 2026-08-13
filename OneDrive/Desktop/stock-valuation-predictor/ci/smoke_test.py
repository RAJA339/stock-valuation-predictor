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
import pandas as pd

from svp import features as F
from svp.models import (
    valuation as V, explain as X, dcf as D, backtest as B,
    regime as RG, sizing as SZ, derivatives as DV,
)
from svp.analytics import peers as P, technical as TA
from svp.data import storage, sentiment, market, filings_nlp as NLP, options as OPT
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

# ── Regime classifier ─────────────────────────────────────────────────────────
calm = RG.classify(vix_level=12.0, yield_curve=1.2)
crisis = RG.classify(vix_level=48.0, yield_curve=-1.0)
check("regime labels are known", {calm.regime, crisis.regime} <= {"Calm", "Neutral", "Stress", "Crisis"})
check("stressed regime scales signals down", crisis.signal_scaler <= calm.signal_scaler)

# ── Technical filters ─────────────────────────────────────────────────────────
tech = TA.analyze(md.history)
check("technical signals computed", tech is not None and tech.sma200 is not None)
check("ATR is positive", tech is not None and tech.atr is not None and tech.atr > 0)

# ── Position sizing ───────────────────────────────────────────────────────────
ann_vol = SZ.annualized_vol(md.history["Close"])
szr = SZ.compute_sizing(190.0, res.mc_samples, res.low, res.high, tech.atr if tech else None, ann_vol)
check("kelly fraction bounded to [0,1]", 0.0 <= szr.kelly_fraction <= 1.0)
check("stop-loss sits below entry", szr.stop_loss is None or szr.stop_loss < 190.0)

# ── Filings NLP ───────────────────────────────────────────────────────────────
fdv = NLP.compare_texts(NLP.SAMPLE_LATEST, NLP.SAMPLE_PRIOR)
check("filing similarity in [0,1]", 0.0 <= fdv.similarity <= 1.0)

# ── Derivatives: Black-Scholes, parity, Greeks ────────────────────────────────
_S, _K, _T, _r, _sig, _q = 100.0, 95.0, 0.75, 0.045, 0.30, 0.01
bs_c = DV.black_scholes(_S, _K, _T, _r, _sig, _q, "call")
bs_p = DV.black_scholes(_S, _K, _T, _r, _sig, _q, "put")
parity_lhs = bs_c.price - bs_p.price
parity_rhs = _S * np.exp(-_q * _T) - _K * np.exp(-_r * _T)
check("put-call parity holds", abs(parity_lhs - parity_rhs) < 1e-6)
check("call delta in (0,1)", 0.0 < bs_c.delta < 1.0)
check("put delta in (-1,0)", -1.0 < bs_p.delta < 0.0)
check("gamma and vega positive", bs_c.gamma > 0 and bs_c.vega > 0)

# ── Derivatives: binomial, Monte-Carlo, implied vol ──────────────────────────
bino_eu = DV.binomial_price(_S, _K, _T, _r, _sig, _q, "call", steps=400, american=False)
check("binomial European converges to Black-Scholes", abs(bino_eu - bs_c.price) < 0.05)
amer_put = DV.binomial_price(_S, _K, _T, _r, _sig, _q, "put", steps=400, american=True)
check("American put >= European put", amer_put >= bs_p.price - 1e-9)
mc_opt = DV.monte_carlo_price(_S, _K, _T, _r, _sig, _q, "call", n=50000)
check("Monte-Carlo option price within 4 standard errors",
      abs(mc_opt["price"] - bs_c.price) < 4 * mc_opt["stderr"] + 0.02)
iv_back = DV.implied_volatility(bs_c.price, _S, _K, _T, _r, _q, "call")
check("implied vol round-trips", iv_back is not None and abs(iv_back - _sig) < 1e-3)

# ── Derivatives: futures cost-of-carry + valuation bridge ────────────────────
fut = DV.futures_fair_value(100.0, 0.045, 1.0, storage_cost=0.01, convenience_yield=0.02)
check("futures carry = r + s - c", abs(fut.annualized_carry - (0.045 + 0.01 - 0.02)) < 1e-12)
check("positive carry puts futures above spot", fut.fair_value > 100.0)
edge = DV.bridge_call_edge(projected_ST=399.43, strike=300.0, T=1.0, r=0.045,
                           market_price=45.0, sigma=0.30, spot=280.0)
check("bridge edge is finite", np.isfinite(edge.edge))
check("bridge P(ITM) in [0,1]", 0.0 <= edge.prob_itm <= 1.0)

# ── Option chain pipeline (synthetic fallback offline) ───────────────────────
chain = OPT.get_option_chain("AAPL", spot_fallback=190.0)
check("option chain has calls and puts", not chain.calls.empty and not chain.puts.empty)
check("chain exposes strike/IV columns",
      {"strike", "impliedVolatility"} <= set(chain.calls.columns))
check("expiry list non-empty", len(OPT.list_expiries("AAPL")) > 0)

# ── Full report covers every tab ─────────────────────────────────────────────
# The report is the only place data from all tabs converges, so assert section
# coverage rather than just byte count — a stubbed section still produces bytes.
_opts = {
    "expiry": chain.expiry, "source": "synthetic", "spot": 190.0, "kind": "call",
    "strike": 195.0, "sigma": 0.28, "bs_price": bs_c.price, "binomial_price": bino_eu,
    "mc_price": mc_opt["price"], "delta": bs_c.delta, "gamma": bs_c.gamma,
    "theta": bs_c.theta, "vega": bs_c.vega, "rho": bs_c.rho, "projected_st": 399.43,
    "futures_fair_value": fut.fair_value, "futures_basis": fut.basis,
    "futures_carry": fut.annualized_carry,
}
_screener = pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Margin of Safety": [0.16, 0.05]})
_full_kwargs = dict(
    features=feats, fundamentals={"sector": "Technology", "revenue": 3.8e11, "eps": 6.5},
    backtest=bt, regime=calm, technical=tech, sizing=szr, filings=fdv,
    screener=_screener, options=_opts, excess_return=0.031,
)
full = reports.build_report(
    "AAPL", "Apple Inc.", 190.0,
    {"point": res.point, "low": res.low, "median": res.median, "high": res.high},
    "Fairly Valued", attr.as_frame(),
    {"intrinsic_per_share": dres.intrinsic_per_share}, pdf,
    {"cpi": 314.0, "fed_funds": 4.3, "yield_curve": -0.1},
    **_full_kwargs,
)
check("full report is larger than the summary-only report", len(full) > len(report))

# Text fallback must carry the same sections as the PDF.
_had_reportlab = reports._HAS_REPORTLAB
reports._HAS_REPORTLAB = False
try:
    text_report = reports.build_report(
        "AAPL", "Apple Inc.", 190.0,
        {"point": res.point, "low": res.low, "median": res.median, "high": res.high},
        "Fairly Valued", attr.as_frame(),
        {"intrinsic_per_share": dres.intrinsic_per_share}, pdf,
        {"cpi": 314.0, "fed_funds": 4.3, "yield_curve": -0.1},
        **_full_kwargs,
    ).decode("utf-8")
finally:
    reports._HAS_REPORTLAB = _had_reportlab

for _section in (
    "VALUATION", "FUNDAMENTALS", "MODEL INPUT FEATURES", "SIGNAL BACKTEST",
    "EXECUTION & TIMING", "POSITION SIZING", "FILING DIVERGENCE",
    "OPTIONS & FUTURES", "MACRO BACKDROP", "PEER BENCHMARKING", "SCREENER",
):
    check(f"report section present: {_section}", _section in text_report)

# ── Result ────────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"SMOKE TEST FAILED — {len(failures)} check(s): {failures}")
    sys.exit(1)
print("SMOKE TEST PASSED")
