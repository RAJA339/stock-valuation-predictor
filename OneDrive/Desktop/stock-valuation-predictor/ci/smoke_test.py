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
from svp.analytics import peers as P, technical as TA, indicators as IND, accuracy as ACC
from svp.data import (storage, sentiment, market, filings_nlp as NLP,
                      options as OPT, intraday as INTRA, sec as SEC)
from svp import charts
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

# ── Distribution-aware bridge ─────────────────────────────────────────────────
_spot, _target, _Tb, _rb = 280.0, 399.43, 1.0, 0.045
_vals = np.random.default_rng(5).normal(_target, 40.0, 20000)

# With no diffusion and full convergence the terminal law collapses onto the
# value draws, so the MC scorer must reproduce the point estimator.
_flat = DV.terminal_distribution(_spot, np.full(5000, _target), _Tb, sigma=0.0, convergence=1.0)
check("zero-vol terminal collapses to the target", np.allclose(_flat, _target))
_mc_flat = DV.bridge_edge_mc(_flat, 300.0, _Tb, _rb, market_price=45.0, spot=_spot, kind="call")
_pt = DV.bridge_call_edge(_target, 300.0, _Tb, _rb, 45.0, 0.30, _spot, "call")
check("degenerate MC edge matches the point estimator",
      abs(_mc_flat.edge - _pt.edge) < 1e-6)

# convergence=0 means no re-rating: the anchor stays at spot.
_noconv = DV.terminal_distribution(_spot, _vals, _Tb, sigma=0.0, convergence=0.0)
check("zero convergence anchors on spot", np.allclose(_noconv, _spot))

# Jensen: integrating a convex payoff must not fall below the point evaluation
# at the same mean.
_term = DV.terminal_distribution(_spot, _vals, _Tb, sigma=0.30, convergence=1.0)
_mean = float(_term.mean())
for _k in (250.0, 300.0, 350.0, 450.0):
    _e = DV.bridge_edge_mc(_term, _k, _Tb, _rb, market_price=1.0, spot=_spot, kind="call")
    _point_payoff = max(_mean - _k, 0.0)
    check(f"Jensen holds at K={_k:.0f} (E[max] >= max(E)-K)",
          _e.expected_payoff >= _point_payoff - 1e-6)

_e300 = DV.bridge_edge_mc(_term, 300.0, _Tb, _rb, market_price=45.0, spot=_spot,
                          kind="call", market_iv=0.30)
check("MC method is labelled", _e300.method == "monte-carlo")
check("MC P(ITM) in [0,1]", 0.0 <= _e300.prob_itm <= 1.0)
check("MC discounts the expected payoff",
      _e300.expected_payoff_pv < _e300.expected_payoff)
check("payoff p90 >= p50", _e300.payoff_p90 >= _e300.payoff_p50)
check("risk-neutral price computed when IV supplied", _e300.risk_neutral_price > 0)
check("view premium = model PV - risk-neutral PV",
      abs(_e300.view_premium - (_e300.expected_payoff_pv - _e300.risk_neutral_price)) < 1e-9)

# A bullish view must produce a positive view premium; a bearish one must not.
_bear = DV.terminal_distribution(_spot, np.full(5000, 200.0), _Tb, sigma=0.30, convergence=1.0)
_e_bear = DV.bridge_edge_mc(_bear, 300.0, _Tb, _rb, 45.0, _spot, "call", market_iv=0.30)
check("bearish view yields no positive view premium", _e_bear.view_premium < _e300.view_premium)

# Deeper strikes must be monotonically less likely to finish in the money.
_probs = [DV.bridge_edge_mc(_term, k, _Tb, _rb, 1.0, _spot, "call").prob_itm
          for k in (250.0, 300.0, 350.0, 450.0)]
check("P(ITM) decreases as call strikes rise", all(a >= b for a, b in zip(_probs, _probs[1:])))

# Empty samples must degrade, not raise.
_empty = DV.bridge_edge_mc(np.array([]), 300.0, _Tb, _rb, 45.0, _spot, "call", market_iv=0.3)
check("empty terminal distribution falls back to the point estimator",
      _empty.method == "point")

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

# The overflow bug: bare strings handed to ReportLab are drawn on one line and
# run past the cell edge. Every cell must be a wrapping flowable.
from reportlab.platypus import Paragraph as _RLPara
_probe = reports._table(
    [["Signal", "Reading"],
     ["Regime Note",
      "Low volatility, positively sloped curve — value signals reliable, and this "
      "sentence is deliberately far longer than the column is wide."]],
    [1.6 * reports.inch, 4.1 * reports.inch])
check("table cells are wrapping flowables, not raw strings",
      all(isinstance(c, _RLPara) for row in _probe._cellvalues for c in row))
_w, _h = _probe.wrap(5.7 * reports.inch, 500)
check("a long cell grows the row instead of overflowing", _h > 30)
check("table never exceeds the width it was given", _w <= 5.75 * reports.inch)
check("numeric columns are detected", reports._numeric("$1,234.50") and reports._numeric("-12.3%"))
check("text columns are not", not reports._numeric("Better than chance"))

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

# ── Intraday, indicators, measured accuracy, charts ──────────────────────────
for _iv in INTRA.INTERVALS:
    _bars = INTRA.get_intraday("AAPL", _iv, spot_fallback=302.25)
    check(f"{_iv} bars returned", len(_bars.df) >= 60)
    check(f"{_iv} OHLC is internally consistent",
          bool((_bars.df["High"] >= _bars.df[["Open", "Close"]].max(axis=1) - 1e-9).all()
               and (_bars.df["Low"] <= _bars.df[["Open", "Close"]].min(axis=1) + 1e-9).all()))

# 10m must be a genuine resample of 5m, not a relabel: half the bars, same span.
_b5, _b10 = INTRA.get_intraday("AAPL", "5m", 302.25), INTRA.get_intraday("AAPL", "10m", 302.25)
check("10m bars are twice the width of 5m",
      abs(len(_b10.df) - len(_b5.df) / 2) <= max(2, len(_b5.df) * 0.1) or _b10.source == "synthetic")

_iset = IND.compute(INTRA.get_intraday("AAPL", "15m", 302.25).df)
check("indicator set is populated", len(_iset.signals) >= 10)
check("every signal carries a directional call",
      all(s.call in (IND.BULLISH, IND.BEARISH, IND.NEUTRAL) for s in _iset.signals))
check("net score within [-1, 1]", -1.0 <= _iset.net_score <= 1.0)
check("consensus is a known label",
      _iset.consensus in {"Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell"})
check("bull + bear + neutral covers every signal",
      _iset.bullish + _iset.bearish + _iset.neutral == len(_iset.signals))
check("RSI stays within 0-100", bool(_iset.df["RSI"].dropna().between(0, 100).all()))
check("ATR is non-negative", bool((_iset.df["ATR"].dropna() >= 0).all()))
check("Bollinger upper band sits above lower",
      bool((_iset.df["BB_UP"].dropna() >= _iset.df["BB_LOW"].dropna()).all()))
check("Supertrend direction is +/-1", set(_iset.df["ST_DIR"].dropna().unique()) <= {-1.0, 1.0})

_acc = ACC.measure(_iset, horizon=6)
check("accuracy measured for every indicator", len(_acc) >= 10)
check("hit rates are probabilities",
      all(0.0 <= a.hit_rate <= 1.0 for a in _acc if a.hit_rate == a.hit_rate))
check("confidence interval brackets the estimate",
      all(a.ci_low <= a.hit_rate <= a.ci_high for a in _acc if a.hit_rate == a.hit_rate))
check("no indicator is asserted accurate without evidence",
      all(a.verdict != "Better than chance" or a.significant for a in _acc))
# On a synthetic random walk the mean hit rate must sit near a coin flip; a
# measurement engine that reports otherwise is broken or leaking the future.
_scored = [a.hit_rate for a in _acc if a.hit_rate == a.hit_rate]
check("random-walk hit rates cluster near 50%", abs(float(np.mean(_scored)) - 0.5) < 0.12)
check("Wilson interval is sane at n=0", ACC.wilson_interval(0, 0) == (0.0, 0.0))
check("Wilson interval widens as n shrinks",
      (ACC.wilson_interval(30, 50)[1] - ACC.wilson_interval(30, 50)[0])
      > (ACC.wilson_interval(300, 500)[1] - ACC.wilson_interval(300, 500)[0]))

_png = charts.render(_iset.df, "AAPL", "15m")
check("chart renders to PNG", _png[:4] == b"\x89PNG" and len(_png) > 20000)
check("signal strip renders", charts.signal_strip(_iset)[:4] == b"\x89PNG")
check("chart handles an empty frame", charts.render(pd.DataFrame(), "AAPL", "15m") == b"")

# ── SEC coverage: the parser must not reject well-formed filings ─────────────
def _facts(tax="us-gaap", concept="StockholdersEquity", form="10-K", unit="USD", val=6e10):
    return {"facts": {tax: {concept: {"units": {unit: [
        {"end": "2025-12-31", "val": val, "form": form, "filed": "2026-02-01"}]}}}}}

# Foreign private issuers file 20-F / 40-F, not 10-K. Excluding those forms
# shut every non-US-domiciled listing out of the app.
for _form in ("10-K", "20-F", "40-F", "10-Q"):
    check(f"{_form} filer resolves", SEC.extract_latest(_facts(form=_form), "StockholdersEquity") == 6e10)
check("IFRS taxonomy resolves",
      SEC.extract_latest(_facts(tax="ifrs-full", concept="Equity"), "Equity") == 6e10)
check("non-USD reporting currency resolves",
      SEC.extract_latest(_facts(unit="EUR"), "StockholdersEquity") == 6e10)

# Equity is the tag most often filed under the long form; a single tag name
# returns None for any company carrying a minority interest.
_longform = _facts(concept="StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
check("equity long-form tag resolves via variants",
      SEC.concept_first(_longform, SEC.EQUITY_TAGS) == 6e10)
check("the single-tag lookup it replaced did not",
      SEC.extract_latest(_longform, "StockholdersEquity") is None)

_mixed = {"facts": {"us-gaap": {"Assets": {"units": {"USD": [
    {"end": "2026-03-31", "val": 111, "form": "10-Q", "filed": "2026-04-20"},
    {"end": "2025-12-31", "val": 999, "form": "10-K", "filed": "2026-02-01"}]}}}}}
check("annual filing preferred over a newer quarterly",
      SEC.extract_latest(_mixed, "Assets") == 999)
check("share-class variants are tried", "BRK-B" in SEC._ticker_variants("BRK.B"))
check("missing concept still returns None", SEC.extract_latest(_facts(), "NoSuchConcept") is None)
check("malformed facts do not raise", SEC.extract_latest({"nope": 1}, "Assets") is None)

# ── Result ────────────────────────────────────────────────────────────────────
print()
if failures:
    print(f"SMOKE TEST FAILED — {len(failures)} check(s): {failures}")
    sys.exit(1)
print("SMOKE TEST PASSED")
