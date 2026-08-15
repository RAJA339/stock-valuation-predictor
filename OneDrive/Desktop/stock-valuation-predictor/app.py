"""
Intrinsic Stock Valuation Predictor  —  v4.0
============================================
Webster University — MS Business Analytics | Group ML Project
Author: Raja Mupparaju

A Streamlit application that predicts a company's intrinsic value from SEC EDGAR
fundamentals, live market data, real-time macro indicators and earnings-call
sentiment, then explains and stress-tests that prediction.

Feature areas (each in its own tab):
  0. Charts           — TradingView + native candles at 5/10/15/30m and 1h, an
                        11-indicator panel, and each indicator's *measured* hit
                        rate on those bars (no accuracy is asserted, only observed)
  1. Valuation        — XGBoost point estimate + quantile intrinsic-value range
  2. Explainability   — SHAP / LIME per-feature attribution (waterfall)
  3. DCF & Scenario   — interactive DCF (WACC / terminal-growth sliders) + Monte-Carlo
  4. Peer Benchmarking— EV/EBITDA, P/E, Debt/Equity vs industry peers
  5. Backtesting      — signal performance over 1/3/5-year horizons
  6. Execution/Timing — HMM market regime + 200-SMA / RSI / volume-POC filters
  7. Guardrails       — Piotroski F, Altman Z, Beneish M, insider & short flow
  8. Screener         — top-decile margin-of-safety ranking across a universe
  9. Filings Δ        — TF-IDF divergence between consecutive 10-K / 10-Q filings
 10. Options & Futures— live chains, Black-Scholes Greeks, cost-of-carry, and a
                        bridge from the intrinsic target to mispriced contracts
 11. Report           — one-click PDF equity research summary, including the
                        chart, indicator panel and measured accuracy table

The heavy lifting lives in the ``svp`` package. Every data/ML dependency degrades
gracefully, so the app runs even without network access or optional libraries.
"""

from __future__ import annotations

import datetime as _dt
import traceback
import warnings
from contextlib import contextmanager

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")  # headless backend — avoids Streamlit worker-thread GUI errors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# st.stop() and st.rerun() raise to unwind the script. On current Streamlit they
# derive from BaseException and so pass through `except Exception` untouched,
# but that has not always been true, and tab_guard() must never swallow them.
# The empty tuple is a never-matching except clause if the import path moves.
try:
    from streamlit.runtime.scriptrunner_utils.exceptions import (
        ScriptControlException as _ScriptControl,
    )
except Exception:                                   # pragma: no cover
    _ScriptControl = ()

from svp import theme
from svp.features import build_features, FEATURE_LABELS
from svp.data import (
    market as market_mod, macro as macro_mod, sentiment as sent_mod, storage,
    sec as sec_mod, insider as insider_mod, filings_nlp as nlp_mod,
    options as options_mod, predictions as pred_mod,
    watchlist as watch_mod, calendar as cal_mod, segments as seg_mod,
    crypto as crypto_mod, _userdb,
)
from svp.models import (
    valuation as val_mod, explain as explain_mod, dcf as dcf_mod, backtest as bt_mod,
    regime as regime_mod, relative as relative_mod, sizing as sizing_mod,
    derivatives as deriv_mod,
)
from svp.analytics import (
    peers as peers_mod, technical as technical_mod, quality as quality_mod,
    screener as screener_mod, indicators as ind_mod, accuracy as acc_mod,
    microstructure as micro_mod,
)
from svp.data import intraday as intraday_mod
from svp import rag as rag_mod
from svp import reports, charts

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + THEME
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intrinsic Stock Valuation Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(theme.CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model() -> val_mod.ValuationModel:
    return val_mod.train_model()


@st.cache_data(ttl=3600, show_spinner=False)
def get_macro_cached():
    return macro_mod.get_macro()


@st.cache_data(ttl=1800, show_spinner=False)
def get_market_cached(ticker: str, fallback_price: float):
    md = market_mod.get_market_data(ticker, fallback_price=fallback_price)
    # Cache-friendly: return primitives + history separately.
    return md


# ──────────────────────────────────────────────────────────────────────────────
# CACHED PER-TICKER ANALYTICS
#
# Streamlit re-runs the whole script on every widget interaction, and tab
# bodies are NOT lazy — the code under all 11 tabs executes every time. Without
# these wrappers, dragging a slider on the Options tab re-ran the backtest, the
# DCF Monte-Carlo and two SEC fetches, which is what got the app CPU-throttled.
# Keyed on the ticker rather than on DataFrames so the cache lookup stays cheap.
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def get_backtest_cached(ticker: str, fallback_price: float, threshold: float = 0.05):
    hist = get_market_cached(ticker, fallback_price).history
    if hist is None or hist.empty:
        return [], None
    return bt_mod.run_backtest(hist), bt_mod.equity_curve(hist, threshold=threshold)


@st.cache_data(ttl=1800, show_spinner=False)
def get_global_calibration():
    """
    The model's cross-user track record, scored and cached for 30 minutes.

    Runs across every prediction in the ledger, not just this visitor's, so the
    number strengthens as the whole user base uses the app — the compounding
    credibility a brokerage cannot reproduce. Cached because it fetches a current
    price per distinct ticker; the 30-minute TTL keeps that off the hot path
    while staying fresh enough for a figure that moves over weeks.
    """
    preds = pred_mod.all_predictions()
    if not preds:
        return {}
    scored = pred_mod.score(preds, lambda t: get_market_cached(t, 0.0).price)
    return pred_mod.global_calibration(scored)


@st.cache_data(ttl=3600, show_spinner=False)
def get_facts_cached(cik: str):
    return sec_mod.get_financials(cik) if cik else {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_quality_cached(cik: str, market_cap: float | None):
    return quality_mod.compute_quality(get_facts_cached(cik), market_cap)


@st.cache_data(ttl=3600, show_spinner=False)
def get_insider_cached(ticker: str, cik: str):
    return insider_mod.get_insider_signal(ticker, cik)


# Live-tab TTLs are deliberately short. Anything longer is lag we are adding
# on top of the feed's own delay; the module-level cache in svp.data.intraday
# is what protects the rate limit.
LIVE_TTL = 20


@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def get_intraday_cached(ticker: str, interval: str, spot: float):
    return intraday_mod.get_intraday(ticker, interval, spot_fallback=spot)


@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def get_indicators_cached(ticker: str, interval: str, spot: float):
    return ind_mod.compute(get_intraday_cached(ticker, interval, spot).df)


@st.cache_data(ttl=LIVE_TTL, show_spinner=False)
def get_chart_png_cached(ticker: str, interval: str, spot: float) -> bytes:
    return charts.render(get_indicators_cached(ticker, interval, spot).df, ticker, interval)


# Accuracy is a walk-forward statistic over hundreds of bars — it barely moves
# tick to tick, and recomputing it on every refresh would be wasted CPU.
@st.cache_data(ttl=600, show_spinner=False)
def get_accuracy_cached(ticker: str, interval: str, spot: float, horizon: int):
    return acc_mod.measure(get_indicators_cached(ticker, interval, spot), horizon=horizon)


@st.cache_data(ttl=1800, show_spinner=False)
def get_dcf_mc_cached(fcf0: float, shares: float, net_debt: float, wacc: float,
                      terminal_growth: float, growth_rate: float, n: int = 4000):
    return dcf_mod.monte_carlo_dcf(
        dcf_mod.DCFInputs(fcf0=fcf0, shares=shares, net_debt=net_debt, wacc=wacc,
                          terminal_growth=terminal_growth, growth_rate=growth_rate),
        n=n,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
# A result has to be addressable before it can be shared or returned to.
# ?t=NVDA loads that ticker on arrival and the URL is rewritten after every
# analysis, so the browser back button, a bookmark and a pasted link all work.
# This is also the app's only organic acquisition channel: a link someone sends
# a friend has to open on the thing they were looking at.
_url_ticker = (st.query_params.get("t") or "").upper().strip()
_default_ticker = _url_ticker or st.session_state.get("last_ticker") or "AAPL"

with st.sidebar:
    st.markdown("## Stock Valuation Predictor")
    st.markdown("*Webster University — MS Business Analytics*")
    st.divider()

    ticker = st.text_input(
        "Stock Ticker Symbol", value=_default_ticker, max_chars=6,
    ).upper().strip()
    use_live_price = st.checkbox("Use live market price (yfinance)", value=True)
    manual_price = st.number_input(
        "Market Price override ($)", min_value=0.0, value=0.0, step=0.5, format="%.2f",
        help="Leave 0 to use the live/last price. Enter a value to override.",
    )

    with st.expander("Earnings-call transcript (sentiment)"):
        transcript = st.text_area(
            "Paste transcript text (optional)", value="", height=120,
            help="Scored with FinBERT if available, else a finance lexicon.",
        )
        if st.checkbox("Use sample transcript"):
            transcript = sent_mod.SAMPLE_TRANSCRIPT

    run_btn = st.button("Analyze", type="primary", width="stretch")

    st.divider()

    # ── Ledger identity ──────────────────────────────────────────────────────
    # An anonymous key, not a login: a sign-up wall in front of a tool nobody
    # has used yet costs more traffic than it captures. The key stays out of the
    # shareable ?t= link on purpose — putting it there would hand a user's whole
    # track record to whoever opened a result they shared.
    if "ledger_key" not in st.session_state:
        st.session_state["ledger_key"] = pred_mod.new_key()

    with st.expander("Track record"):
        st.caption(
            "Every valuation you run is saved and scored later against what the "
            "price actually did. Save this key to keep your record across devices."
        )
        if not _userdb.is_durable():
            st.warning(_userdb.durability_note())
        else:
            st.caption(_userdb.durability_note())
        st.code(st.session_state["ledger_key"], language=None)
        _restore = st.text_input(
            "Restore a key", value="", placeholder="paste a saved key",
            label_visibility="collapsed",
        ).strip()
        if _restore and _restore != st.session_state["ledger_key"]:
            if pred_mod.valid_key(_restore):
                st.session_state["ledger_key"] = _restore.lower()
                st.success("Key restored.")
            else:
                st.warning("That does not look like a ledger key.")

    st.divider()
    st.markdown("**Data & Model**")
    st.caption(
        "SEC EDGAR · yfinance/Alpha Vantage · FRED/BLS · FinBERT · "
        "XGBoost (point + quantile) · SHAP/LIME · DCF Monte-Carlo"
    )
    cache_stats = storage.stats()
    st.caption(f"Cache: {cache_stats['backend']} · {cache_stats['fresh']}/{cache_stats['rows']} fresh rows")
    st.caption("⚠️ Educational use only. Not financial advice.")


# ──────────────────────────────────────────────────────────────────────────────
# HEADER + MODEL BANNER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# Intrinsic Stock Valuation Predictor")
st.markdown("*Model Transparency · Live Data Pipelines · Scenario Analytics*")
st.divider()

with st.spinner("Training / loading XGBoost ensemble..."):
    vm = get_model()

macro_now = get_macro_cached()

b1, b2, b3, b4, b5 = st.columns(5)
b1.metric("Model R²", f"{vm.r2:.3f}", help="Variance explained on held-out synthetic test set")
b2.metric("Model RMSE", f"${vm.rmse:.2f}", help="Root Mean Squared Error")
b3.metric("Quantiles", "p10 · p50 · p90", help="Native XGBoost quantile regressors")
b4.metric("CPI", f"{macro_now.cpi:.1f}", help=f"Macro source: {macro_now.source}")
b5.metric("10y-2y Curve", f"{macro_now.yield_curve_10y_2y:+.2f}",
          help="Negative = inverted (recession signal)")

st.markdown(
    f"Macro data {theme.source_pill(macro_now.is_live, 'FRED/BLS LIVE', 'DEFAULTS')}",
    unsafe_allow_html=True,
)
st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, sub: bool = False) -> str:
    cls = "metric-sub" if sub else "metric-value"
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="{cls}">{value}</div></div>'


def run_analysis(ticker: str, price_override: float, transcript: str, use_live: bool) -> dict | None:
    """Fetch everything for a ticker and stash results in session_state."""
    md = get_market_cached(ticker, price_override or 100.0)
    price = None
    if price_override and price_override > 0:
        price = price_override
    elif use_live and md.price:
        price = md.price
    else:
        price = md.price or 100.0

    feats = build_features(ticker, market_price=price, transcript=transcript or None, md=md, macro=macro_now)
    if feats is None:
        return None

    result = val_mod.predict(feats, vm)
    attribution = explain_mod.explain_prediction(feats, vm)
    signal_text, signal_class = val_mod.valuation_signal(result.point, price)

    # ── v3 execution / guardrail layer ───────────────────────────────────────
    # Market regime from VIX + yield curve.
    vix_md = get_market_cached("^VIX", 18.0)
    regime = regime_mod.detect_from_market(vix_md, macro_now)

    # Technical confirmation from price/volume history.
    technical = technical_mod.analyze(md.history)

    # Position sizing + dynamic risk floors.
    ann_vol = sizing_mod.annualized_vol(md.history["Close"]) if not md.history.empty else None
    atr_val = technical.atr if technical else None
    sizing = sizing_mod.compute_sizing(
        price, result.mc_samples, result.low, result.high, atr_val, ann_vol
    )

    # Regime-adjusted conviction on the raw valuation signal.
    raw_mos = (result.point - price) / price if price else 0.0
    adjusted_mos = raw_mos * regime.signal_scaler

    return {
        "ticker": ticker,
        "price": price,
        "md": md,
        "features": feats,
        "result": result,
        "attribution": attribution,
        "signal_text": signal_text,
        "signal_class": signal_class,
        "regime": regime,
        "technical": technical,
        "sizing": sizing,
        "adjusted_mos": adjusted_mos,
        "raw_mos": raw_mos,
    }


# ──────────────────────────────────────────────────────────────────────────────
# RUN / STATE
# ──────────────────────────────────────────────────────────────────────────────
# Arriving on ?t=NVDA runs that ticker once, without a click. A shared link
# that lands on an empty form has wasted the share; the guard keys on the
# ticker so a later manual change does not re-trigger it.
_auto_run = bool(
    _url_ticker
    and _url_ticker == ticker
    and st.session_state.get("autorun_for") != _url_ticker
    and (st.session_state.get("analysis") or {}).get("ticker") != _url_ticker
)
if _auto_run:
    st.session_state["autorun_for"] = _url_ticker

if run_btn or _auto_run:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Analyzing **{ticker}** — SEC EDGAR · market · macro · sentiment..."):
            analysis = run_analysis(ticker, manual_price, transcript, use_live_price)
        if analysis is None:
            _cik = sec_mod.get_cik(ticker)
            if not _cik:
                st.error(
                    f"❌ **{ticker}** has no SEC CIK — it does not file with the SEC.\n\n"
                    "This covers ETFs and index symbols (SPY, QQQ, ^VIX), most non-US "
                    "listings, and crypto or FX pairs. The valuation model is built on SEC "
                    "filings, so it can only run on SEC registrants — including foreign "
                    "companies that file 20-F or 40-F, such as TSM, SAP, SHOP or BABA.\n\n"
                    "If you expected this ticker to work, check the symbol: share classes "
                    "are written BRK-B on EDGAR rather than BRK.B (both are accepted here)."
                )
            else:
                # Say what actually happened. The old message asserted the
                # filings were unparseable even when the request had failed —
                # which sent the reader to check the company instead of the
                # feed, and was wrong for any well-established filer.
                _why = sec_mod.last_fetch_error(_cik)
                if _why:
                    st.error(
                        f"❌ Found **{ticker}** (CIK {_cik}) at the SEC, but could not "
                        f"retrieve its financial data: {_why}.\n\n"
                        + ("Try again in a few minutes — this is a transient "
                           "condition on the SEC side, not a problem with this "
                           "company's filings."
                           if "rate-limit" in _why or "network" in _why or "HTTP" in _why
                           else "This affects companies that file with the SEC without "
                                "publishing XBRL financial facts — recent IPOs, shells "
                                "and some trusts.")
                    )
                else:
                    st.error(
                        f"❌ Found **{ticker}** (CIK {_cik}) at the SEC and retrieved its "
                        "XBRL facts, but none of revenue, net income or total assets "
                        "could be located under any tag this app recognises.\n\n"
                        "That is a gap in the tag list here rather than in the filing. "
                        "Please report the ticker so the mapping can be extended."
                    )
        else:
            st.session_state["analysis"] = analysis
            st.session_state["last_ticker"] = ticker
            # Log the call to the ledger so it can be scored later. Written
            # here rather than at render time: the tab bodies re-run on every
            # widget interaction, and a valuation is made once per analysis.
            _r = analysis["result"]
            pred_mod.record(
                st.session_state["ledger_key"], ticker, analysis["price"],
                _r.low, _r.point, _r.high, analysis["signal_text"],
            )
            # Put the ticker in the address bar so this result can be linked to.
            if st.query_params.get("t") != ticker:
                st.query_params["t"] = ticker

analysis = st.session_state.get("analysis")

if not analysis:
    st.info("Enter a ticker in the sidebar and click **Analyze** to begin.")
    st.markdown("### What's inside")
    cols = st.columns(3)
    cards = [
        ("Explainable AI", "SHAP / LIME waterfalls show exactly how each feature moved the valuation."),
        ("Valuation Range", "Quantile XGBoost + Monte-Carlo give an intrinsic-value range, not a single point."),
        ("Live Pipelines", "yfinance prices, FRED macro (yield curve, CPI, rates), FinBERT sentiment."),
        ("DCF & Scenario", "Interactive DCF with WACC / terminal-growth sliders, Monte-Carlo fair value."),
        ("Peer Benchmarking", "EV/EBITDA, P/E, Debt/Equity vs auto-selected industry peers."),
        ("PDF Reports", "One-click equity research summary with feature impacts and signals."),
    ]
    for i, (t, b) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f"**{t}**")
            st.caption(b)
    st.stop()


# Unpack analysis
tk = analysis["ticker"]
price = analysis["price"]
md: market_mod.MarketData = analysis["md"]
feats = analysis["features"]
raw = feats["_raw"]
result: val_mod.ValuationResult = analysis["result"]
attribution: explain_mod.Attribution = analysis["attribution"]
signal_text = analysis["signal_text"]
signal_class = analysis["signal_class"]
regime = analysis["regime"]
technical = analysis["technical"]
sizing = analysis["sizing"]

# ── Quote header — the price block a brokerage app opens with ────────────────
_day_change = None
if not md.history.empty and len(md.history) > 1:
    _prev = float(md.history["Close"].iloc[-2])
    _day_change = (price - _prev) / _prev if _prev else None

st.markdown(
    theme.quote_header(
        tk, raw.get("name", tk), price, _day_change,
        pills=(theme.source_pill(md.is_live, f"{md.source.upper()} LIVE", "OFFLINE")
               + f'&nbsp;<span class="pill '
                 f'{"pill-live" if regime.is_risk_on else "pill-offline"}">'
                 f"REGIME: {regime.regime}</span>"),
    ),
    unsafe_allow_html=True,
)

# Following a name has to be one click from the thing you are looking at. A
# watchlist you can only edit on the watchlist tab does not get built.
_follow_key = st.session_state.get("ledger_key", "")
_following = watch_mod.contains(_follow_key, tk)
if st.button(("Following ✓" if _following else "Follow"), key="follow_btn"):
    watch_mod.toggle(_follow_key, tk)
    st.rerun()

# ── Verdict — answered before the tabs, not inside one ───────────────────────
st.markdown(
    theme.verdict_bar(price, result.low, result.point, result.high),
    unsafe_allow_html=True,
)

# ── Live track record for calls like this one ────────────────────────────────
# The single thing a brokerage will never show: how the model's own calls of
# this kind have actually resolved, across everyone. Only shown once a cohort
# has enough independent bets to say something honest.
try:
    _gcal = get_global_calibration()
except Exception:
    _gcal = {}
_cohort_name = pred_mod._signal_bucket(signal_text)
_cohort = _gcal.get(_cohort_name)
if _cohort is not None and _cohort.n >= 10:
    _reads_calibrated = _cohort.ci_low <= _cohort.target <= _cohort.ci_high
    _cls = "verdict-inside" if _reads_calibrated else "verdict-over"
    st.markdown(
        f'<div class="verdict"><div class="verdict-line">'
        f'<span class="section-eyebrow">Track record · {_cohort_name}</span><br>'
        f'Across every user, the model\'s <b>{_cohort_name.lower()}</b> landed '
        f'inside their p10–p90 band <span class="{_cls}">'
        f'{_cohort.coverage*100:.0f}%</span> of the time '
        f'<span class="verdict-tail">(95% CI {_cohort.ci_low*100:.0f}–'
        f'{_cohort.ci_high*100:.0f}%, {_cohort.n} independent resolved bets; '
        f'target {_cohort.target*100:.0f}%). Direction was right '
        f'{_cohort.hit_rate*100:.0f}% of the time.</span>'
        f'</div></div>', unsafe_allow_html=True,
    )

# ── Headline metric row ───────────────────────────────────────────────────────
h1, h2, h3, h4 = st.columns(4)
h1.markdown(metric_card("Intrinsic Value", f"${result.point:.2f}"), unsafe_allow_html=True)
h2.markdown(metric_card("Range (p10–p90)", f"${result.low:.0f} – ${result.high:.0f}", sub=True),
            unsafe_allow_html=True)
h3.markdown(metric_card("Market Price", f"${price:.2f}"), unsafe_allow_html=True)
h4.markdown(
    f'<div class="metric-card"><div class="metric-label">Valuation Signal</div>'
    f'<div class="{signal_class}">{signal_text}</div></div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Navigation ───────────────────────────────────────────────────────────────
# Fifteen peer tabs is not navigation, it is a list. No terminal presents its
# surfaces flat, because a reader cannot hold fifteen unranked choices in mind —
# they scan for the two or three that match what they came to do. Grouping into
# named sections turns "which of these fifteen" into "which of six, then which
# of two-to-five", which is the question people can actually answer.
#
# Six sections rather than the four originally sketched: forcing Report,
# Portfolio and Filings into Market/Value/Risk would have produced tidier
# arithmetic and worse groups. The grouping is by what a reader wants, not by
# what the code shares.
SECTIONS: list[tuple[str, list[str]]] = [
    ("Market",    ["Charts", "Execution & Timing", "Microstructure"]),
    ("Value",     ["Valuation", "Explainability", "DCF & Scenario", "Peers", "Screener"]),
    ("Risk",      ["Guardrails", "Backtesting", "Options & Futures"]),
    ("Filings",   ["Ask the Filings", "Fundamental Δ", "Segments"]),
    ("Portfolio", ["Watchlist", "Track Record"]),
    ("Crypto",    []),
    ("Report",    []),
]

# Panes are addressed by label, not position. Every previous phase that added a
# surface had to renumber every guard below it and the CI constant with them;
# a name does not move when a neighbour is inserted.
_section_panes = st.tabs([name for name, _ in SECTIONS])
PANES: dict = {}
for (_name, _subs), _container in zip(SECTIONS, _section_panes):
    if not _subs:
        PANES[_name] = _container
        continue
    with _container:
        for _label, _pane in zip(_subs, st.tabs(_subs)):
            PANES[_label] = _pane

TAB_LABELS = list(PANES)
# Published for ci/apptest_check.py, which asserts on the pane set rather
# than a flat tab count — the count changes whenever a pane moves section.
st.session_state["pane_labels"] = TAB_LABELS
st.session_state["section_count"] = len(SECTIONS)

# Reset per script run. Every tab body re-executes on each rerun, so a failure
# carried over from a previous run would be reported long after it was fixed.
st.session_state["tab_failures"] = {}


@contextmanager
def tab_guard(label: str):
    """
    Render one pane, containing any failure inside it.

    ``st.tabs`` is not lazy: every pane body executes top to bottom in a single
    script run, so an uncaught exception in an early one aborts the run and
    every later pane renders empty. A missing multiple in Peers could therefore
    take out eight panels with nothing wrong with them. Catching per pane turns
    a total outage into one degraded panel.

    The traceback is shown rather than swallowed. Streamlit redacts uncaught
    exceptions in deployment ("original error message is redacted"), which is
    the right default for an app handling third-party data but leaves the
    operator debugging blind; this app renders only public market data, and the
    detail sits behind a collapsed expander.
    """
    with PANES[label]:
        try:
            yield
        except _ScriptControl:
            raise                       # st.stop()/st.rerun() are control flow
        except Exception as exc:
            failed = st.session_state.setdefault("tab_failures", {})
            failed[label] = f"{type(exc).__name__}: {exc}"
            st.error(
                f"**{label} could not be rendered.** "
                f"{type(exc).__name__}: {exc}\n\n"
                "The other panels are unaffected — this one alone failed."
            )
            with st.expander("Technical detail"):
                st.code("".join(traceback.format_exception(exc)), language="text")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHARTS (TradingView + native candles + measured indicator accuracy)
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Charts"):
    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        tv_interval = st.radio(
            "Interval", list(intraday_mod.INTERVALS.keys()),
            index=2, horizontal=True, key="chart_interval",
        )
    with c2:
        horizon_bars = st.selectbox(
            "Accuracy horizon (bars forward)", [3, 6, 12, 24], index=1, key="chart_horizon",
            help="How far ahead a signal is scored. 6 bars is 30 minutes at 5m, 6 hours at 1h.",
        )
    chart_style = st.radio("Chart", ["TradingView", "Native"], horizontal=True,
                           key="chart_style", label_visibility="collapsed")

    with c3:
        refresh_label = st.selectbox(
            "Auto-refresh", ["Off", "5s", "10s", "30s", "60s"], index=2, key="chart_refresh",
            help="Refreshes only this block, not the whole app — a full rerun would "
                 "recompute every tab and is what previously caused CPU throttling.",
        )
    _REFRESH = {"Off": None, "5s": 5, "10s": 10, "30s": 30, "60s": 60}[refresh_label]

    acc_rows = get_accuracy_cached(tk, tv_interval, price, int(horizon_bars))
    acc_sum = acc_mod.summary(acc_rows)

    # Only this block re-runs on the timer. A whole-script rerun would recompute
    # all 12 tabs, which is exactly what got the app CPU-throttled before, so the
    # live view is scoped to a fragment. st.fragment landed in Streamlit 1.38;
    # older builds degrade to a manual refresh rather than failing.
    _fragment = getattr(st, "fragment", None)
    _wrap = (lambda fn: _fragment(run_every=_REFRESH)(fn)) if (_fragment and _REFRESH) \
        else (lambda fn: fn)

    @_wrap
    def _live_view():
        bars = get_intraday_cached(tk, tv_interval, price)
        iset = get_indicators_cached(tk, tv_interval, price)
        st.session_state["rep_chart_interval"] = tv_interval

        if not bars.is_live:
            st.warning(
                f"⚠️ Intraday candles are **simulated**, not market data — the live feed "
                f"was unavailable, so the bars, indicators and hit rates below describe a "
                f"synthetic series.\n\nReason: `{bars.reason or 'unknown'}`"
            )

        # ── Quote strip ──────────────────────────────────────────────────────────
        last = bars.last_price or price
        chg = bars.session_change or 0.0
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.markdown(metric_card("Last", f"${last:,.2f}"), unsafe_allow_html=True)
        q2.markdown(
            f'<div class="metric-card"><div class="metric-label">Change (window)</div>'
            f'<div class="{"signal-under" if chg >= 0 else "signal-over"}">{chg * 100:+.2f}%</div></div>',
            unsafe_allow_html=True)
        q3.markdown(metric_card("Consensus", iset.consensus, sub=True), unsafe_allow_html=True)
        q4.markdown(metric_card("Bull / Bear", f"{iset.bullish} / {iset.bearish}", sub=True),
                    unsafe_allow_html=True)
        q5.markdown(metric_card("Bars", f"{len(bars.df):,}", sub=True), unsafe_allow_html=True)

        # ── Chart ────────────────────────────────────────────────────────────────
        if chart_style == "TradingView":
            tv_map = {"5m": "5", "10m": "10", "15m": "15", "30m": "30", "1h": "60"}
            st.components.v1.html(f"""
    <div class="tradingview-widget-container" style="height:620px">
      <div id="tv_chart" style="height:620px"></div>
    </div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    new TradingView.widget({{
      "autosize": true, "symbol": "{tk}", "interval": "{tv_map[tv_interval]}",
      "timezone": "America/New_York", "theme": "dark", "style": "1", "locale": "en",
      "toolbar_bg": "#161B22", "enable_publishing": false, "hide_side_toolbar": false,
      "allow_symbol_change": true, "withdateranges": true, "details": true,
      "studies": ["MASimple@tv-basicstudies", "RSI@tv-basicstudies",
                  "MACD@tv-basicstudies", "VWAP@tv-basicstudies"],
      "container_id": "tv_chart"
    }});
    </script>
    """, height=640)
            st.caption(
                "TradingView's own chart, with EMA, RSI, MACD and VWAP loaded. It renders in "
                "your browser, so it needs internet access and **cannot be captured into the "
                "PDF** — the report embeds the Native chart instead. "
                f"TradingView has no 10-minute interval; at 10m it shows {tv_map[tv_interval]}m "
                "while the native chart and every indicator below use true resampled 10m bars."
                if tv_interval == "10m" else
                "TradingView's own chart, with EMA, RSI, MACD and VWAP loaded. It renders in "
                "your browser, so it needs internet access and **cannot be captured into the "
                "PDF** — the report embeds the Native chart instead."
            )
        else:
            png = charts.render(iset.df, tk, tv_interval)
            if png:
                st.image(png, width="stretch")
                st.caption("Native chart — this is exactly what goes into the PDF report.")

        st.divider()

        # ── Indicator panel ──────────────────────────────────────────────────────
        st.markdown(theme.section("Indicator panel", "Charts"), unsafe_allow_html=True)
        ic1, ic2 = st.columns([1.6, 1])
        with ic1:
            st.dataframe(iset.as_frame(), width="stretch", hide_index=True, height=420)
        with ic2:
            strip = charts.signal_strip(iset)
            if strip:
                st.image(strip, width="stretch")
            st.metric("Net score", f"{iset.net_score:+.2f}",
                      help="Mean of +1 bullish / 0 neutral / −1 bearish across all indicators")
            st.caption(
                f"**{iset.bullish}** bullish · **{iset.neutral}** neutral · "
                f"**{iset.bearish}** bearish across {len(iset.signals)} indicators."
            )

        st.divider()

        _stamp = bars.df.index[-1] if len(bars.df) else None
        st.caption(
            f"Last bar **{_stamp:%H:%M:%S}** · fetched {pd.Timestamp.now():%H:%M:%S} · "
            f"refresh {refresh_label.lower()}"
            + ("" if bars.is_live else " · simulated data")
            + "  \n_Yahoo's free intraday feed is delayed roughly 15 minutes; the app adds "
              "at most 20s on top of that. Nothing here is a real-time tick feed._"
            if _stamp is not None else "No bars loaded."
        )

    _live_view()

    if _REFRESH and not _fragment:
        st.info(
            "Auto-refresh needs Streamlit 1.38 or newer for `st.fragment`; this build is "
            f"{st.__version__}. The view still updates whenever you interact with it."
        )

    # ── Measured accuracy ────────────────────────────────────────────────────
    st.markdown(theme.section("Measured signal accuracy", "Validation"),
                unsafe_allow_html=True)
    st.caption(
        f"Each indicator's call is recomputed at every bar and "
        f"scored against the actual move {horizon_bars} bars later. These are realised hit "
        f"rates on **{tk} at {tv_interval}** — not vendor claims."
    )

    if acc_rows:
        a1, a2, a3 = st.columns(3)
        a1.markdown(metric_card("Best measured", acc_sum["best"] or "—", sub=True),
                    unsafe_allow_html=True)
        a2.markdown(metric_card("Best hit rate",
                                f"{acc_sum['best_rate'] * 100:.1f}%"
                                if acc_sum["best_rate"] == acc_sum["best_rate"] else "—"),
                    unsafe_allow_html=True)
        a3.markdown(metric_card("Beat chance (95% CI)",
                                f"{acc_sum['better_than_chance']} of {acc_sum['measured']}", sub=True),
                    unsafe_allow_html=True)
        st.dataframe(acc_mod.as_frame(acc_rows), width="stretch", hide_index=True, height=400)
        st.info(
            "**How to read this.** *Verdict* is decided by the 95% confidence interval, not the "
            "point estimate — an indicator scoring 56% on 200 signals has a CI of roughly 49–63%, "
            "which includes a coin flip, so it is reported as no better than chance. Costs are "
            "not modelled either: spread and commission can turn a genuine directional edge into "
            "a loss. Nothing here is an 80%-accurate signal, and any product advertising one is "
            "not measuring it this way."
        )
    else:
        st.info("Not enough bars at this interval to measure accuracy.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Valuation"):
    _missing = raw.get("missing_fields") or []
    if _missing:
        st.info(
            f"**{tk}** does not tag {', '.join(_missing)} in a form this parser "
            f"recognises, so those inputs are blank. The model handles missing features "
            f"natively — the valuation still runs, but any ratio built on them is absent "
            f"from the feature table and the SHAP breakdown."
        )

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown(theme.section("Extracted features", "Inputs"), unsafe_allow_html=True)
        disp = {}
        pct_feats = {"roe", "roa", "fcf_yield", "profit_margin", "revenue_yoy",
                     "revenue_qoq", "net_income_yoy", "fcf_yoy"}
        for key in ["pe_ratio", "roe", "roa", "debt_to_equity", "fcf_yield", "profit_margin",
                    "asset_turnover", "revenue_yoy", "revenue_qoq", "net_income_yoy", "fcf_yoy",
                    "sentiment", "cpi", "fed_funds", "yield_curve"]:
            v = feats.get(key)
            label = FEATURE_LABELS.get(key, key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                disp[label] = "N/A"
            elif key in pct_feats:
                disp[label] = f"{v*100:.1f}%"
            else:
                disp[label] = f"{v:.2f}"
        feat_df = pd.DataFrame(disp.items(), columns=["Feature", "Value"])
        st.dataframe(feat_df, width="stretch", hide_index=True, height=430)

        sent_obj = raw.get("sentiment_obj")
        if sent_obj is not None:
            st.caption(
                f"Earnings sentiment: **{sent_obj.label}** ({sent_obj.score:+.2f}) "
                f"via {sent_obj.source}"
            )

    with right:
        st.markdown(theme.section("Intrinsic value distribution", "Valuation"), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        theme.style_axes(fig, ax)
        ax.hist(result.mc_samples, bins=40, color=theme.TEAL, alpha=0.7, edgecolor="none")
        ax.axvline(result.point, color=theme.GREEN, lw=2, label=f"Point ${result.point:.0f}")
        ax.axvline(result.low, color=theme.YELLOW, lw=1.5, ls="--", label=f"p10 ${result.low:.0f}")
        ax.axvline(result.high, color=theme.YELLOW, lw=1.5, ls="--", label=f"p90 ${result.high:.0f}")
        ax.axvline(price, color=theme.RED, lw=2, label=f"Market ${price:.0f}")
        ax.set_xlabel("Intrinsic Value ($)")
        ax.set_ylabel("Monte-Carlo frequency")
        ax.legend(facecolor=theme.BG, labelcolor=theme.TEXT, fontsize=7.5)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.markdown(
            f"**Confidence interval:** \\${result.low:.2f} – \\${result.high:.2f} "
            f"(width {result.ci_width_pct:.1f}% of point). "
            f"Monte-Carlo mean \\${result.mc_mean:.2f} ± \\${result.mc_std:.2f}."
        )

    st.markdown(theme.section("Price vs intrinsic range"), unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(9, 1.9))
    theme.style_axes(fig2, ax2)
    lo = min(price, result.low) * 0.9
    hi = max(price, result.high) * 1.1
    ax2.set_xlim(lo, hi)
    ax2.barh(0, result.high - result.low, left=result.low, height=0.35, color=theme.TEAL,
             alpha=0.5, label=f"Intrinsic range ${result.low:.0f}–${result.high:.0f}")
    ax2.axvline(result.point, color=theme.GREEN, lw=2.5, label=f"Point ${result.point:.2f}")
    ax2.axvline(price, color=theme.RED, lw=2.5, ls="--", label=f"Market ${price:.2f}")
    ax2.set_yticks([])
    ax2.legend(facecolor=theme.BG, labelcolor=theme.TEXT, fontsize=8.5, loc="upper right")
    st.pyplot(fig2, width="stretch")
    plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLAINABILITY (SHAP / LIME)
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Explainability"):
    _attr_method = "SHAP" if attribution.source == "shap" else "XGBoost contributions"
    st.markdown(
        theme.section(f"Feature attribution — {_attr_method}", "Explainability"),
        unsafe_allow_html=True,
    )
    st.caption(
        "How each feature pushed the valuation away from the model's base value "
        f"(**${attribution.base_value:.2f}**) toward the final prediction "
        f"(**${attribution.prediction:.2f}**). Green = pushed value up, red = down."
    )

    adf = attribution.as_frame()
    wcol, tcol = st.columns([1.3, 1])

    with wcol:
        # SHAP-style waterfall.
        top = adf.head(10).iloc[::-1].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        theme.style_axes(fig, ax)
        cum = attribution.base_value
        for _, r in top.iterrows():
            c = theme.GREEN if r["contribution"] >= 0 else theme.RED
            ax.barh(r["feature"], r["contribution"], left=cum, color=c, edgecolor="none")
            cum += r["contribution"]
        ax.axvline(attribution.base_value, color=theme.MUTED, ls=":", lw=1, label="Base value")
        ax.axvline(attribution.prediction, color=theme.YELLOW, lw=1.5, label="Prediction")
        ax.set_xlabel("Intrinsic Value ($) contribution")
        ax.legend(facecolor=theme.BG, labelcolor=theme.TEXT, fontsize=8)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with tcol:
        show = adf.copy()
        show["contribution"] = show["contribution"].map(lambda x: f"{x:+.2f}")
        show["value"] = show["value"].map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        st.dataframe(show, width="stretch", hide_index=True, height=430)

    # Optional LIME.
    st.markdown(theme.section("LIME local explanation"), unsafe_allow_html=True)
    if explain_mod.has_lime():
        with st.spinner("Computing LIME explanation..."):
            lime_df = explain_mod.lime_explanation(feats, vm)
        if lime_df is not None:
            st.dataframe(lime_df, width="stretch", hide_index=True)
        else:
            st.caption("LIME explanation unavailable for this instance.")
    else:
        st.caption(
            "LIME is not installed in this environment — install `lime` to enable a second "
            "local explainer. SHAP attributions above provide the primary explanation."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DCF & SCENARIO
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("DCF & Scenario"):
    st.markdown(theme.section("Interactive DCF", "Discounted cash flow"), unsafe_allow_html=True)
    st.caption("Adjust the assumptions; the DCF and its Monte-Carlo distribution update live.")

    fcf0 = raw.get("free_cash_flow") or raw.get("op_cash_flow") or 1e9
    shares = raw.get("shares") or 1e9
    net_debt = (raw.get("long_term_debt") or 0.0)

    c1, c2, c3, c4 = st.columns(4)
    wacc = c1.slider("WACC (%)", 4.0, 20.0, 9.0, 0.25) / 100
    tgrowth = c2.slider("Terminal growth (%)", 0.0, 5.0, 2.5, 0.1) / 100
    ngrowth = c3.slider("Near-term FCF growth (%)", -10.0, 40.0, 8.0, 0.5) / 100
    years = c4.slider("Projection years", 3, 10, 5, 1)

    dcf_in = dcf_mod.DCFInputs(
        fcf0=float(fcf0), shares=float(shares), net_debt=float(net_debt),
        wacc=wacc, terminal_growth=tgrowth, growth_rate=ngrowth, years=int(years),
    )
    dcf_res = dcf_mod.run_dcf(dcf_in)
    st.session_state["rep_dcf"] = dcf_res
    dcf_mc = get_dcf_mc_cached(float(fcf0), float(shares), float(net_debt),
                               wacc, tgrowth, ngrowth)

    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(metric_card("DCF Intrinsic / Share", f"${dcf_res.intrinsic_per_share:.2f}"), unsafe_allow_html=True)
    d2.markdown(metric_card("DCF Range (p10–p90)",
                            f"${dcf_mc['p10']:.0f} – ${dcf_mc['p90']:.0f}", sub=True), unsafe_allow_html=True)
    d3.markdown(metric_card("ML Intrinsic (point)", f"${result.point:.2f}"), unsafe_allow_html=True)
    d4.markdown(metric_card("Market Price", f"${price:.2f}"), unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(theme.section("Projected vs discounted FCF"), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        theme.style_axes(fig, ax)
        yrs = list(range(1, int(years) + 1))
        w = 0.4
        ax.bar([y - w / 2 for y in yrs], np.array(dcf_res.projected_fcf) / 1e9, width=w,
               color=theme.TEAL, label="Projected FCF")
        ax.bar([y + w / 2 for y in yrs], np.array(dcf_res.discounted_fcf) / 1e9, width=w,
               color=theme.GREEN, label="Discounted (PV)")
        ax.set_xlabel("Year"); ax.set_ylabel("FCF ($B)")
        ax.legend(facecolor=theme.BG, labelcolor=theme.TEXT, fontsize=8)
        st.pyplot(fig, width="stretch"); plt.close(fig)

    with g2:
        st.markdown(theme.section("Monte-Carlo fair-value distribution"), unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        theme.style_axes(fig, ax)
        ax.hist(dcf_mc["samples"], bins=45, color=theme.BLUE, alpha=0.75, edgecolor="none")
        ax.axvline(dcf_mc["median"], color=theme.GREEN, lw=2, label=f"Median ${dcf_mc['median']:.0f}")
        ax.axvline(price, color=theme.RED, lw=2, ls="--", label=f"Market ${price:.0f}")
        ax.set_xlabel("Fair value / share ($)"); ax.set_ylabel("Frequency")
        ax.legend(facecolor=theme.BG, labelcolor=theme.TEXT, fontsize=8)
        st.pyplot(fig, width="stretch"); plt.close(fig)

    st.caption(
        "Hybrid view: the ML model and traditional DCF are independent estimates. "
        "Agreement between them strengthens the valuation thesis; divergence flags "
        "assumptions worth revisiting."
    )

    # ── Rate sensitivity ─────────────────────────────────────────────────────
    st.markdown(theme.section("If rates move", "Sensitivity"), unsafe_allow_html=True)
    st.caption(
        "The same DCF re-run across a parallel shift in the discount rate. This "
        "is the honest way to read a DCF: the output moves far more with the "
        "discount rate than with the growth assumptions people spend their time "
        "arguing about. Terminal growth is deliberately held fixed — in reality "
        "a rate move and growth expectations are linked, but modelling that here "
        "would bury a large assumption inside a one-variable sensitivity check."
    )
    _shocks = dcf_mod.rate_shock(dcf_in)
    st.session_state["rep_rate_shock"] = _shocks
    _dur = dcf_mod.duration_estimate(_shocks)

    _rs1, _rs2 = st.columns([1.35, 1])
    with _rs1:
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        theme.style_axes(fig, ax)
        _xs = [s.shock_bp for s in _shocks]
        _ys = [s.intrinsic_per_share for s in _shocks]
        ax.plot(_xs, _ys, color=theme.GREEN, lw=2, marker="o", ms=4)
        ax.axhline(price, color=theme.RED, lw=1.4, ls="--", label=f"Market ${price:.0f}")
        ax.axvline(0, color=theme.FAINT, lw=1)
        ax.set_xlabel("Discount-rate shock (bp)")
        ax.set_ylabel("Fair value / share ($)")
        ax.legend(facecolor=theme.BG, labelcolor=theme.TEXT, fontsize=8)
        st.pyplot(fig, width="stretch"); plt.close(fig)
    with _rs2:
        if _dur is not None:
            st.markdown(
                metric_card("Sensitivity", f"{_dur:.1f}%") +
                '<div class="metric-note">fair-value change per 100bp, '
                'measured across the ±100bp points</div>',
                unsafe_allow_html=True,
            )
        _break = [s for s in _shocks if s.intrinsic_per_share < price]
        if _break:
            st.markdown(
                metric_card("Breaks even at", f"{_break[0].shock_bp:+d}bp", sub=True) +
                '<div class="metric-note">smallest modelled rate rise at which '
                'fair value falls below the market price</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                metric_card("Breaks even at", "beyond +200bp", sub=True) +
                '<div class="metric-note">fair value stays above the market '
                'price across every shock modelled</div>',
                unsafe_allow_html=True,
            )

    st.dataframe(
        pd.DataFrame([{
            "Shock": f"{s.shock_bp:+d}bp",
            "Discount rate": f"{s.wacc*100:.2f}%",
            "Fair value": f"${s.intrinsic_per_share:,.2f}",
            "Change": f"{s.change_pct:+.1f}%",
        } for s in _shocks]),
        width="stretch", hide_index=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PEERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Peers"):
    st.markdown(theme.section("Peer group benchmarking", "Peers"), unsafe_allow_html=True)
    self_metrics = {
        "name": raw.get("name", tk),
        "pe": feats.get("pe_ratio"),
        "debt_to_equity": feats.get("debt_to_equity"),
        "profit_margin": feats.get("profit_margin"),
        "market_cap": raw.get("market_cap"),
        "ps": (raw.get("market_cap") / raw["revenue"]) if raw.get("revenue") else None,
        "ev_ebitda": None,
        "source": "SEC",
    }
    with st.spinner("Fetching peer multiples..."):
        pdf = peers_mod.benchmark(tk, sector=raw.get("sector"), self_metrics=self_metrics)

    styled = pdf.copy()
    for col in ["EV/EBITDA", "P/E", "Debt/Equity", "P/S", "Market Cap ($B)"]:
        styled[col] = styled[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    styled["Profit Margin"] = styled["Profit Margin"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
    st.dataframe(styled, width="stretch", hide_index=True)

    summ = peers_mod.peer_summary(pdf)
    st.markdown(theme.section("Relative positioning vs peer median"), unsafe_allow_html=True)
    # The pill is coloured strictly by the sign of the difference: above peers
    # is green, below peers is red. For valuation multiples and leverage that
    # is the opposite of "good for the buyer" — a P/E far above peers means an
    # expensive stock — so the reading is spelled out underneath rather than
    # being carried by colour alone.
    LOWER_IS_CHEAPER = ("EV/EBITDA", "P/E", "P/S")
    LOWER_IS_SAFER = ("Debt/Equity",)

    def _peer_note(metric: str, pct: float) -> str:
        above = pct >= 0
        if metric in LOWER_IS_CHEAPER:
            return ("Richer than peers — paying more per unit of earnings"
                    if above else "Cheaper than peers on this multiple")
        if metric in LOWER_IS_SAFER:
            return ("More levered than peers" if above else "Less levered than peers")
        return ("Above the peer median" if above else "Below the peer median")

    metric_cols = st.columns(len(summ) or 1)
    for i, (metric, d) in enumerate(summ.items()):
        with metric_cols[i]:
            if d["subject"] is None or d["premium_pct"] is None:
                st.markdown(metric_card(metric, "N/A"), unsafe_allow_html=True)
            else:
                pct = d["premium_pct"]
                cls = "delta-pos" if pct >= 0 else "delta-neg"
                arrow = "↑" if pct >= 0 else "↓"
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{metric}</div>'
                    f'<div class="metric-value">{d["subject"]:.2f}</div>'
                    f'<span class="{cls}">{arrow} {pct:+.0f}% vs peers</span>'
                    f'<div class="metric-note">{_peer_note(metric, pct)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Peer bar chart for EV/EBITDA & P/E. Keep a peer that has one of the two
    # multiples but not the other — dropping on P/E alone silently hid every
    # company whose earnings are negative, which is exactly the case a reader
    # wants to see in the comparison.
    plot_df = pdf.dropna(subset=["P/E", "EV/EBITDA"], how="all").head(8)
    if plot_df.empty:
        st.info("No peer reported a usable P/E or EV/EBITDA, so there is nothing to chart.")
    else:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        theme.style_axes(fig, ax)
        x = np.arange(len(plot_df))
        ax.bar(x - 0.2, plot_df["P/E"], width=0.4, color=theme.TEAL, label="P/E")
        ax.bar(x + 0.2, plot_df["EV/EBITDA"], width=0.4, color=theme.GREEN, label="EV/EBITDA")
        ax.set_xticks(x); ax.set_xticklabels(plot_df["Ticker"], fontsize=8)
        ax.legend(facecolor=theme.BG, labelcolor=theme.TEXT, fontsize=8)
        st.pyplot(fig, width="stretch"); plt.close(fig)
    st.caption(f"Peers auto-selected for **{tk}** (sector: {raw.get('sector') or 'n/a'}). "
               "Multiples marked *estimated* when live data was unavailable.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Backtesting"):
    st.markdown(theme.section("Historical backtesting", "Validation"), unsafe_allow_html=True)
    st.caption(
        "How the valuation signal would have performed against actual price moves over "
        "1-, 3- and 5-year forward horizons (monthly rebalances). "
        f"Price history source: **{md.source}**."
    )

    results, curve = get_backtest_cached(tk, price)
    st.session_state["rep_backtest"] = results
    if not results:
        st.warning("Not enough price history to backtest this ticker.")
    else:
        cols = st.columns(len(results))
        for i, r in enumerate(results):
            with cols[i]:
                st.metric(
                    f"{int(r.horizon_years)}-Year Hit Rate", f"{r.hit_rate*100:.0f}%",
                    delta=f"{r.n_signals} signals", delta_color="off",
                )
                st.caption(f"Avg fwd return after BUY: {r.avg_forward_return*100:+.1f}%")

        if curve is not None:
            st.markdown(theme.section("Strategy vs buy &amp; hold — growth of $1"),
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(9, 3.6))
            theme.style_axes(fig, ax)
            ax.plot(curve.index, curve["Buy & Hold"], color=theme.MUTED, lw=1.5, label="Buy & Hold")
            ax.plot(curve.index, curve["Valuation Strategy"], color=theme.GREEN, lw=1.8,
                    label="Valuation Strategy (long when undervalued)")
            ax.set_ylabel("Growth of $1")
            ax.legend(facecolor=theme.BG, labelcolor=theme.TEXT, fontsize=8.5)
            st.pyplot(fig, width="stretch"); plt.close(fig)
        st.caption(
            "Note: point-in-time fundamentals aren't stored, so the backtest uses a "
            "mean-reversion intrinsic-value proxy (trailing average) to evaluate signal "
            "quality — a conservative stand-in for the full ML signal history."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXECUTION & TIMING (regime + technical filters + sizing/risk floors)
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Execution & Timing"):
    st.markdown(theme.section("Macro regime", "Regime"), unsafe_allow_html=True)
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Regime", regime.regime, help=f"Detected via {regime.source.upper()}")
    rc2.metric("Confidence", f"{regime.confidence*100:.0f}%")
    rc3.metric("VIX", f"{regime.vix:.1f}")
    rc4.metric("Signal Scaler", f"{regime.signal_scaler:.2f}×",
               help="Valuation conviction is multiplied by this in-regime")
    st.caption(regime.description)

    if regime.timeline is not None and len(regime.timeline) > 0:
        st.markdown(theme.section("Regime timeline — VIX-driven HMM"), unsafe_allow_html=True)
        code = {"Calm": 0, "Neutral": 1, "Stress": 2, "Crisis": 3}
        tl = regime.timeline.map(code)
        fig, ax = plt.subplots(figsize=(9, 2.4))
        theme.style_axes(fig, ax)
        colors = {0: theme.GREEN, 1: theme.TEAL, 2: theme.YELLOW, 3: theme.RED}
        ax.scatter(range(len(tl)), tl.values, s=4,
                   c=[colors[v] for v in tl.values], edgecolors="none")
        ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(["Calm", "Neutral", "Stress", "Crisis"])
        ax.set_xlabel("Trading days (history)")
        st.pyplot(fig, width="stretch"); plt.close(fig)

    st.divider()
    st.markdown(theme.section("Technical entry filters", "Execution"), unsafe_allow_html=True)
    if technical is None:
        st.warning("Not enough price history for technical confirmation.")
    else:
        tcols = st.columns(4)
        tcols[0].metric("Price vs 200-SMA", "Above ✅" if technical.trend_up else "Below ❌",
                        help=f"200-SMA: ${technical.sma200:.2f}" if technical.sma200 else "n/a")
        tcols[1].metric("RSI(14)", f"{technical.rsi:.0f}" if technical.rsi else "n/a")
        tcols[2].metric("Bullish Divergence", "Yes ✅" if technical.rsi_bullish_divergence else "No")
        tcols[3].metric("Volume Support", "Near ✅" if technical.near_volume_support else "—",
                        help=f"POC: ${technical.volume_poc:.2f}" if technical.volume_poc else "volume data n/a")

        checks = {k: v for k, v in technical.confirmations.items() if v is not None}
        conf_df = pd.DataFrame(
            [(k.replace("_", " ").title(), "✅ Pass" if v else "❌ Fail") for k, v in checks.items()],
            columns=["Confirmation", "Status"],
        )
        st.dataframe(conf_df, width="stretch", hide_index=True)

        undervalued = analysis["raw_mos"] > 0.15
        if undervalued and technical.confirmed:
            st.success(f"✅ **Entry confirmed** — undervalued signal backed by "
                       f"{technical.score*100:.0f}% of technical checks. Not a falling knife.")
        elif undervalued and not technical.confirmed:
            st.warning("⚠️ Undervalued, but technical confirmation is weak — possible falling knife. "
                       "Consider waiting for trend/volume confirmation.")
        else:
            st.info("No active undervalued signal to confirm right now.")

    st.divider()
    st.markdown(theme.section("Risk-adjusted sizing"), unsafe_allow_html=True)
    scols = st.columns(4)
    scols[0].markdown(metric_card("Fractional Kelly", f"{sizing.fractional_kelly*100:.1f}%", sub=True),
                      unsafe_allow_html=True)
    scols[1].markdown(metric_card("Vol-Parity Weight",
                                  f"{sizing.vol_parity_weight*100:.1f}%" if sizing.vol_parity_weight else "n/a",
                                  sub=True), unsafe_allow_html=True)
    scols[2].markdown(metric_card("Stop-Loss", f"${sizing.stop_loss:.2f}", sub=True), unsafe_allow_html=True)
    scols[3].markdown(metric_card("Take-Profit", f"${sizing.take_profit:.2f}", sub=True), unsafe_allow_html=True)

    s2 = st.columns(4)
    s2[0].metric("Win Probability (MC)", f"{sizing.win_probability*100:.0f}%")
    s2[1].metric("Payoff Ratio", f"{sizing.payoff_ratio:.2f}")
    s2[2].metric("Risk / Share", f"${sizing.risk_per_share:.2f}")
    s2[3].metric("Reward:Risk", f"{sizing.reward_risk_ratio:.1f}" if sizing.reward_risk_ratio else "n/a")
    st.caption(
        "Fractional Kelly (¼-Kelly) sizes the position from the Monte-Carlo win "
        "probability & payoff; the stop-loss is the tighter of a 2×ATR stop and the "
        "model's p10 intrinsic support; take-profit anchors on p90 / intrinsic value."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — GUARDRAILS (quality/distress scores + insider/short)
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Guardrails"):
    st.markdown(theme.section("Value-trap &amp; distress screens", "Guardrails"), unsafe_allow_html=True)
    st.caption("Forensic-accounting scores prevent the model from rating structurally "
               "broken companies as bargains.")

    with st.spinner("Computing quality & distress scores..."):
        qs = get_quality_cached(raw.get("cik") or "", raw.get("market_cap"))
    st.session_state["rep_quality"] = qs

    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        pf = f"{qs.piotroski}/9" if qs.piotroski is not None else "N/A"
        st.markdown(metric_card("Piotroski F-Score", pf), unsafe_allow_html=True)
        st.caption("≥7 strong · ≤3 weak fundamentals")
    with qc2:
        zz = f"{qs.altman_z:.2f}" if qs.altman_z is not None else "N/A"
        st.markdown(metric_card(f"Altman Z ({qs.altman_zone})", zz), unsafe_allow_html=True)
        st.caption(">2.99 safe · <1.81 distress")
    with qc3:
        mm = f"{qs.beneish_m:.2f}" if qs.beneish_m is not None else "N/A"
        st.markdown(metric_card("Beneish M-Score", mm), unsafe_allow_html=True)
        st.caption(f"{qs.beneish_flag} · >-1.78 manipulation risk")
    with qc4:
        aa = f"{qs.accruals*100:.1f}%" if qs.accruals is not None else "N/A"
        st.markdown(metric_card("Accruals / Assets", aa), unsafe_allow_html=True)
        st.caption(f"{qs.accruals_flag} · >10% earnings outrunning cash")

    st.caption(
        "Accruals are the gap between reported profit and operating cash. "
        "Sloan (1996) found firms whose earnings lean on accruals rather than "
        "cash go on to underperform. It asks a narrower question than Beneish — "
        "not whether the statements look manipulated, but whether profit is "
        "turning into cash — so a company can be entirely honest and still fail "
        "it. Treat a high reading as a prompt to look, not as an accusation."
    )

    if qs.guardrail_triggered():
        flags = []
        if qs.weak_fundamentals: flags.append("weak fundamentals (low F-Score)")
        if qs.is_distressed: flags.append("bankruptcy-distress zone (low Z-Score)")
        if qs.possible_manipulation: flags.append("earnings-manipulation risk (high M-Score)")
        if qs.high_accruals: flags.append("earnings running well ahead of cash (high accruals)")
        st.error("🚩 **Guardrail triggered:** " + "; ".join(flags) +
                 ". Treat any 'undervalued' reading with caution — possible value trap.")
    else:
        st.success("✅ No distress/manipulation guardrails triggered.")

    if qs.piotroski_detail:
        st.markdown(theme.section("Piotroski components"), unsafe_allow_html=True)
        pdet = pd.DataFrame(
            [(k, "✅" if v == 1 else ("❌" if v == 0 else "—")) for k, v in qs.piotroski_detail.items()],
            columns=["Criterion", "Pass"],
        )
        st.dataframe(pdet, width="stretch", hide_index=True)
    for n in qs.notes:
        st.caption(f"{n}")

    st.divider()
    st.markdown(theme.section("Institutional ownership", ""), unsafe_allow_html=True)
    _own = insider_mod.get_ownership(tk)
    st.session_state["rep_ownership"] = _own
    if _own.source == "unavailable":
        st.caption(
            "Ownership data unavailable for this ticker right now. 13F filings "
            "are indexed on EDGAR by filer rather than by holding, so this comes "
            "from the market feed and shares its rate limits."
        )
    else:
        _o1, _o2, _o3 = st.columns(3)
        _o1.markdown(metric_card(
            "Held by Institutions",
            f"{_own.pct_institutions:.1f}%" if _own.pct_institutions is not None else "N/A"),
            unsafe_allow_html=True)
        _o2.markdown(metric_card(
            "Held by Insiders",
            f"{_own.pct_insiders:.1f}%" if _own.pct_insiders is not None else "N/A"),
            unsafe_allow_html=True)
        _o3.markdown(metric_card(
            "Top 5 Holders",
            f"{_own.top5_pct:.1f}%" if _own.top5_pct is not None else "N/A"),
            unsafe_allow_html=True)
        if _own.is_concentrated:
            st.warning(
                "Concentrated register — the five largest holders control more "
                "than a third of the shares outstanding. Exits by any one of "
                "them move the price more than the fundamentals do."
            )
        if _own.holders:
            st.dataframe(
                pd.DataFrame([{
                    "Holder": h["holder"],
                    "Shares": f"{h['shares']:,.0f}" if h["shares"] else "—",
                    "% Out": f"{h['pct_out']:.2f}%" if h["pct_out"] is not None else "—",
                    "Value": f"${h['value']:,.0f}" if h["value"] else "—",
                    "As of": h["date"] or "—",
                } for h in _own.holders]),
                width="stretch", hide_index=True,
            )
        st.caption(
            "13F holdings are reported up to 45 days after quarter end, so this "
            "view is stale by up to a quarter by construction — that is a "
            "property of the disclosure, not of this feed."
        )

    st.markdown(theme.section("Insider activity &amp; short interest"), unsafe_allow_html=True)
    with st.spinner("Fetching Form 4 & short interest..."):
        ins = get_insider_cached(tk, raw.get("cik") or "")
    st.session_state["rep_insider"] = ins
    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("Insider Net Flow", ins.net_direction,
               help=f"{ins.buy_count} buy / {ins.sell_count} sell filings (Form 4)")
    ic2.metric("Short % of Float",
               f"{ins.short_percent_float*100:.1f}%" if ins.short_percent_float is not None else "n/a")
    ic3.metric("Source", ins.source)
    if ins.high_short_interest:
        st.warning("⚠️ Elevated short interest — institutional money may disagree with an "
                   "undervalued reading.")
    if ins.bullish:
        st.success("✅ Net insider buying corroborates the valuation discrepancy.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — SCREENER (quantile ranking + relative return)
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Screener"):
    st.markdown(theme.section("Cross-sectional screener", "Screener"), unsafe_allow_html=True)
    st.caption("Rank a universe by margin-of-safety (p50 vs price) and expected excess "
               "return; buys are flagged only in the top decile.")

    horizon = st.selectbox("Excess-return horizon", ["1M", "3M"], index=0)
    universe_txt = st.text_input(
        "Universe (comma-separated tickers)",
        value=", ".join(screener_mod.DEFAULT_UNIVERSE[:12]),
    )
    run_screen = st.button("Run Screen", type="primary")

    if run_screen:
        universe = [t.strip().upper() for t in universe_txt.split(",") if t.strip()]
        rm = relative_mod.train_relative_model(horizon=horizon)
        prog = st.progress(0.0, text="Scoring universe...")

        def _cb(done, total, tkr):
            prog.progress(done / total, text=f"Scored {tkr} ({done}/{total})")

        with st.spinner("Screening..."):
            sdf = screener_mod.screen(universe, vm, rm, progress=_cb)
        st.session_state["rep_screener"] = sdf
        prog.empty()

        if sdf.empty:
            st.warning("No tickers could be scored (fundamentals unavailable in this environment).")
        else:
            show = sdf.copy()
            show["Margin of Safety"] = show["Margin of Safety"].map(lambda x: f"{x*100:+.1f}%")
            show["Exp. Excess Return"] = show["Exp. Excess Return"].map(
                lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "n/a")
            for c in ["Market Price", "Intrinsic p50", "Range Low (p10)", "Range High (p90)"]:
                show[c] = show[c].map(lambda x: f"${x:.2f}")
            st.dataframe(show, width="stretch", hide_index=True)
            top = sdf[sdf["TopDecile"]]
            st.success(f"Top-decile buys: {', '.join(top['Ticker'].tolist()) or '—'}")
    else:
        st.info("Enter a universe and click **Run Screen**. "
                "Note: live fundamentals are needed to score real tickers.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — FILINGS Δ (10-K/10-Q textual divergence)
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Fundamental Δ"):
    st.markdown(theme.section("Fundamental-delta NLP", "Filings"), unsafe_allow_html=True)
    st.caption("Cosine-similarity divergence between consecutive SEC filings flags rapid "
               "changes in risk-factor disclosures before the market digests them.")

    mode = st.radio("Source", ["Fetch latest two filings (SEC)", "Paste two excerpts"],
                    horizontal=True)
    if mode == "Paste two excerpts":
        c1, c2 = st.columns(2)
        latest_txt = c1.text_area("Latest filing text", value=nlp_mod.SAMPLE_LATEST, height=160)
        prior_txt = c2.text_area("Prior filing text", value=nlp_mod.SAMPLE_PRIOR, height=160)
        run_nlp = st.button("Compare", type="primary")
        if run_nlp:
            fd = nlp_mod.compare_texts(latest_txt, prior_txt)
        else:
            fd = None
    else:
        run_nlp = st.button("Fetch & Compare", type="primary")
        fd = None
        if run_nlp:
            with st.spinner("Fetching filings from SEC EDGAR..."):
                fd = nlp_mod.analyze_filings(raw.get("cik"))
            if fd.similarity is None:
                st.warning("Could not fetch/parse filings in this environment. "
                           "Try the 'Paste two excerpts' mode to demo the analysis.")

    if fd is not None and fd.similarity is not None:
        st.session_state["rep_filings"] = fd
        n1, n2, n3 = st.columns(3)
        n1.metric("Cosine Similarity", f"{fd.similarity:.3f}")
        n2.metric("Divergence", f"{fd.divergence:.3f}")
        n3.metric("Material Change", "Yes ⚠️" if fd.alert else "No ✅")
        if fd.alert:
            st.warning("⚠️ Disclosure language shifted materially vs the prior filing.")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Newly emphasized terms**")
            st.write(", ".join(fd.added_terms) or "—")
        with cc2:
            st.markdown("**De-emphasized terms**")
            st.write(", ".join(fd.dropped_terms) or "—")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — OPTIONS & FUTURES
# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — FILINGS RAG (retrieval-augmented Q&A over 10-K / 10-Q text)
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Ask the Filings"):
    st.markdown(theme.section("Ask the filings", "Filings"), unsafe_allow_html=True)
    st.caption(
        "Retrieval-augmented Q&A over this company's SEC filings. Text is pulled from "
        "EDGAR, split on Item boundaries, chunked and indexed in memory, so every answer "
        "quotes the filing directly and cites the Item it came from."
    )

    rc1, rc2, rc3 = st.columns([1, 1, 1.3])
    with rc1:
        rag_form = st.selectbox("Filing type", ["10-K", "10-Q"], key="rag_form")
    with rc2:
        rag_section = st.selectbox(
            "Restrict to section",
            ["Auto", "Risk Factors", "MD&A", "Market Risk", "Business"], key="rag_section",
            help="Auto routes the question to a section from its wording.",
        )
    with rc3:
        rag_k = st.slider("Passages to return", 2, 10, 5, key="rag_k")

    if st.button("Load & index filings", type="primary", key="rag_load"):
        with st.spinner(f"Fetching {rag_form} filings for {tk} from SEC EDGAR..."):
            latest_f, prior_f = rag_mod.filings.latest_pair(tk, form=rag_form)
        if latest_f is None or not latest_f.text:
            st.error(
                f"Could not retrieve {rag_form} text for **{tk}**. The company may not "
                "file this form, or EDGAR may be rate-limiting — try again shortly."
            )
        else:
            with st.spinner("Chunking and indexing..."):
                st.session_state["rag_bundle"] = {
                    "ticker": tk, "form": rag_form,
                    "latest": latest_f, "prior": prior_f,
                    "li": rag_mod.store.build_index(latest_f),
                    "pi": rag_mod.store.build_index(prior_f) if prior_f else None,
                }

    bundle = st.session_state.get("rag_bundle")
    if bundle and bundle.get("ticker") != tk:
        st.info(f"Index loaded for **{bundle['ticker']}**, not {tk}. Reload to switch.")

    if not bundle:
        st.info("Load the filings to begin. Nothing is fetched until you ask.")
    else:
        li, pi = bundle["li"], bundle["pi"]
        latest_f, prior_f = bundle["latest"], bundle["prior"]

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(metric_card("Latest filing", latest_f.label, sub=True), unsafe_allow_html=True)
        m2.markdown(metric_card("Compared with",
                                prior_f.label if prior_f else "— none —", sub=True),
                    unsafe_allow_html=True)
        m3.markdown(metric_card("Indexed passages", f"{len(li.chunks):,}"), unsafe_allow_html=True)
        m4.markdown(metric_card("Retriever", li.backend.upper(), sub=True), unsafe_allow_html=True)

        st.caption("Sections found: " + (", ".join(
            f"{s.item} {s.label} ({s.words:,} words)" for s in latest_f.sections
        ) or "none — the filing layout was not recognised."))

        if not rag_mod.store.has_llm_key():
            st.caption(
                "Answers are **extractive**: every bullet is quoted verbatim from the "
                "filing with its Item citation, so you can check it against EDGAR. "
                "This is the default and needs no configuration. Adding an "
                "`OPENAI_API_KEY` (environment variable or Streamlit secret) plus "
                "`llama-index` layers an optional written summary *above* the same "
                "quotes — it does not change or replace them."
            )

        st.markdown("**Try one of these**")
        qcols = st.columns(3)
        for i, q in enumerate(rag_mod.engine.SUGGESTED_QUESTIONS[:6]):
            if qcols[i % 3].button(q, key=f"rag_sug_{i}", width="stretch"):
                st.session_state["rag_question"] = q

        question = st.text_input(
            "Question", value=st.session_state.get("rag_question", ""),
            placeholder="What newly added risk factors appear in the latest 10-K?",
            key="rag_q_input",
        )

        if question:
            with st.spinner("Retrieving..."):
                ans = rag_mod.engine.answer(
                    question, li, pi, k=int(rag_k),
                    section=None if rag_section == "Auto" else rag_section,
                )
            st.session_state["rep_rag"] = ans

            badge = ("comparing latest vs prior filing" if ans.mode == "diff"
                     else "searching the latest filing")
            st.markdown(
                f'<span class="pill pill-accent">{badge}</span>'
                + (f'<span class="pill pill-accent">{ans.section}</span>' if ans.section else ""),
                unsafe_allow_html=True,
            )
            st.write("")

            if ans.synthesis:
                st.markdown(ans.synthesis)
                st.caption("Synthesis is generated from the quoted passages below only.")
                st.divider()

            if ans.empty:
                st.warning(ans.note or "No matching passage found.")
            else:
                for b in ans.bullets:
                    st.markdown(
                        f'<div class="metric-card" style="margin-bottom:8px">'
                        f'<div style="line-height:1.55">“{b.quote}”</div>'
                        f'<div class="metric-note">— {b.citation} '
                        f'· {"novelty" if ans.mode == "diff" else "relevance"} '
                        f'{b.score:.2f}</div></div>',
                        unsafe_allow_html=True,
                    )
                st.caption(
                    "Quotes are verbatim from the filing. "
                    + ("*Novelty* is 1 − similarity to the closest passage in the prior "
                       "filing, so a high score means the language is genuinely new rather "
                       "than reworded." if ans.mode == "diff" else
                       "*Relevance* is cosine similarity to the question.")
                )

# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Options & Futures"):
    st.markdown(theme.section("Live option chain", "Derivatives"), unsafe_allow_html=True)
    st.caption(
        "Strikes, bid/ask, open interest and implied volatility pulled from "
        "`yfinance.Ticker.option_chain(date)`. Falls back to a Black-Scholes-priced "
        "synthetic chain (with a volatility smile) when the feed is unavailable."
    )

    expiries = options_mod.list_expiries(tk)

    # Default to the first expiry at least ~30 days out. Liquid names list
    # weeklies, so the nearest few are 1-7 DTE — near-worthless for a
    # valuation-driven LEAP scan and the worst-quality IV data on the chain.
    _today = pd.Timestamp.today().normalize()
    _default_idx = next(
        (i for i, e in enumerate(expiries)
         if (pd.Timestamp(e) - _today).days >= 30),
        min(2, len(expiries) - 1),
    )

    ec1, ec2 = st.columns([1, 2])
    with ec1:
        expiry = st.selectbox("Expiry", expiries, index=_default_idx, key="opt_expiry")
        if st.button("Refresh chain", key="opt_refresh"):
            options_mod.clear_cache()
        rf_rate = st.number_input(
            "Risk-free rate (r)", value=float(macro_now.fed_funds or 4.5) / 100.0,
            min_value=0.0, max_value=0.20, step=0.0025, format="%.4f", key="opt_r",
        )
        div_yield = st.number_input(
            "Dividend yield (q)", value=0.0, min_value=0.0, max_value=0.15,
            step=0.0025, format="%.4f", key="opt_q",
        )

    chain = options_mod.get_option_chain(tk, expiry, spot_fallback=price)
    T_exp = max((pd.Timestamp(chain.expiry) - pd.Timestamp.today().normalize()).days, 1) / 365.0
    spot = chain.spot or price

    with ec2:
        oa, ob, oc = st.columns(3)
        oa.markdown(metric_card("Spot", f"${spot:.2f}"), unsafe_allow_html=True)
        ob.markdown(metric_card("Time to Expiry", f"{T_exp * 365:.0f} days", sub=True), unsafe_allow_html=True)
        oc.markdown(
            metric_card("Chain Source", "yfinance LIVE" if chain.is_live else "SYNTHETIC", sub=True),
            unsafe_allow_html=True,
        )

    if not chain.is_live:
        _cool = options_mod.cooldown_remaining()
        _msg = (
            f"⚠️ **These are not real quotes.** The live chain could not be fetched, so the "
            f"strikes and premiums below are Black-Scholes synthetics generated around the "
            f"current spot — useful for exercising the pricer, not for trading decisions.\n\n"
            f"Reason: `{chain.reason or 'unknown'}`"
        )
        if "ratelimit" in chain.reason.lower().replace(" ", "") or _cool > 0:
            _msg += (
                f"\n\nYahoo throttles per IP address, and the block lasts longer the more "
                f"requests you send while it is active — so the app has stopped calling out "
                f"for **{_cool / 60:.0f} more minute(s)** to let it lapse. Avoid the "
                f"Screener and repeated refreshes until then. Everything else in the app "
                f"keeps working; only live quotes are affected."
            )
        st.warning(_msg)

    _cols = ["strike", "lastPrice", "bid", "ask", "openInterest", "volume", "impliedVolatility"]

    def _show(df: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in _cols if c in df.columns]
        out = df[keep].copy()
        if "impliedVolatility" in out.columns:
            out["impliedVolatility"] = (out["impliedVolatility"] * 100).round(1)
            out = out.rename(columns={"impliedVolatility": "IV %"})
        return out

    cl, pu = st.columns(2)
    with cl:
        st.markdown("**Calls**")
        st.dataframe(_show(chain.calls), width="stretch", height=280)
    with pu:
        st.markdown("**Puts**")
        st.dataframe(_show(chain.puts), width="stretch", height=280)

    _repaired = int(chain.calls["ivRepaired"].sum() + chain.puts["ivRepaired"].sum())
    if _repaired:
        st.caption(
            f"{_repaired} contract(s) came back from the feed with an implausible "
            f"implied volatility (Yahoo often reports ~0 on near-dated strikes); their IV "
            f"was re-solved from the quoted premium. Contracts that could not be re-solved "
            f"show a blank IV and fall back to the slider volatility below."
        )

    # ── Volatility smile ─────────────────────────────────────────────────────
    _smile_c = chain.calls.dropna(subset=["impliedVolatility"]) if not chain.calls.empty else chain.calls
    _smile_p = chain.puts.dropna(subset=["impliedVolatility"]) if not chain.puts.empty else chain.puts
    if not _smile_c.empty:
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.plot(_smile_c["strike"], _smile_c["impliedVolatility"] * 100,
                marker="o", ms=3, lw=1.2, label="Calls")
        if not _smile_p.empty:
            ax.plot(_smile_p["strike"], _smile_p["impliedVolatility"] * 100,
                    marker="o", ms=3, lw=1.2, label="Puts")
        ax.axvline(spot, color="#888", ls="--", lw=1, label="Spot")
        ax.set_xlabel("Strike")
        ax.set_ylabel("IV (%)")
        ax.set_title("Implied Volatility Smile")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        st.pyplot(fig, clear_figure=True)

    st.divider()

    # ── Black-Scholes pricer + Greeks ────────────────────────────────────────
    st.markdown(theme.section("Theoretical pricer &amp; Greeks"), unsafe_allow_html=True)
    st.caption(
        "Black-Scholes-Merton analytic value cross-checked against a Cox-Ross-Rubinstein "
        "binomial tree (American exercise) and a GBM Monte-Carlo."
    )

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        opt_kind = st.radio("Type", ["call", "put"], horizontal=True, key="opt_kind")
    with g2:
        strike = st.number_input("Strike (K)", value=float(round(spot / 5) * 5),
                                 min_value=0.01, step=1.0, key="opt_K")
    with g3:
        vol_in = st.slider("Volatility σ (%)", 5.0, 120.0, 30.0, 0.5, key="opt_sigma") / 100.0
    with g4:
        t_days = st.slider("Days to expiry", 1, 1095, int(T_exp * 365), key="opt_days")

    T_use = t_days / 365.0
    bs = deriv_mod.black_scholes(spot, strike, T_use, rf_rate, vol_in, div_yield, opt_kind)
    amer = deriv_mod.binomial_price(spot, strike, T_use, rf_rate, vol_in, div_yield,
                                    opt_kind, steps=300, american=True)
    mc = deriv_mod.monte_carlo_price(spot, strike, T_use, rf_rate, vol_in, div_yield,
                                     opt_kind, n=40000)

    p1, p2, p3 = st.columns(3)
    p1.markdown(metric_card("Black-Scholes (European)", f"${bs.price:.2f}"), unsafe_allow_html=True)
    p2.markdown(metric_card("Binomial CRR (American)", f"${amer:.2f}", sub=True), unsafe_allow_html=True)
    p3.markdown(metric_card("Monte-Carlo", f"${mc['price']:.2f} ± {mc['stderr']:.2f}", sub=True),
                unsafe_allow_html=True)

    greeks = pd.DataFrame([
        {"Greek": "Delta (Δ)", "Value": round(bs.delta, 4), "Interpretation": "Δ option value per $1 move in spot"},
        {"Greek": "Gamma (Γ)", "Value": round(bs.gamma, 5), "Interpretation": "Δ of delta per $1 move in spot"},
        {"Greek": "Theta (Θ)", "Value": round(bs.theta, 4), "Interpretation": "Value decay per calendar day"},
        {"Greek": "Vega", "Value": round(bs.vega, 4), "Interpretation": "Δ value per 1 volatility point"},
        {"Greek": "Rho (ρ)", "Value": round(bs.rho, 4), "Interpretation": "Δ value per 1% move in rates"},
    ])
    st.dataframe(greeks, width="stretch", hide_index=True)

    early_ex = amer - bs.price
    if early_ex > 0.01:
        st.caption(f"Early-exercise premium (American − European): **${early_ex:.2f}**.")

    st.divider()

    # ── Valuation → options bridge ───────────────────────────────────────────
    st.markdown(theme.section("Intrinsic value to options bridge"), unsafe_allow_html=True)
    st.caption(
        "Treats the model's intrinsic target as the **projected underlying at expiry (Sₜ)** "
        "and discounts the resulting payoff back to today. Contracts whose premium sits "
        "well below that present value are flagged as potentially mispriced LEAPs."
    )

    dcf_quick = dcf_mod.run_dcf(dcf_mod.DCFInputs(
        fcf0=float(raw.get("free_cash_flow") or raw.get("op_cash_flow") or 1e9),
        shares=float(raw.get("shares") or 1e9),
        net_debt=float(raw.get("long_term_debt") or 0.0),
    ))

    b1, b2 = st.columns([1, 1])
    with b1:
        target_src = st.radio(
            "Projected Sₜ source",
            ["ML intrinsic (point)", "ML p90 (bull)", "DCF fair value", "Custom"],
            key="bridge_src",
        )
    default_target = {
        "ML intrinsic (point)": result.point,
        "ML p90 (bull)": result.high,
        "DCF fair value": dcf_quick.intrinsic_per_share,
        "Custom": result.point,
    }[target_src]
    with b2:
        projected_ST = st.number_input("Projected Sₜ ($)", value=float(round(default_target, 2)),
                                       min_value=0.01, step=1.0, key="bridge_target")

    b3, b4 = st.columns([1, 1])
    with b3:
        convergence = st.slider(
            "Gap to fair value closed by expiry", 0.0, 1.0, 1.0, 0.05, key="bridge_conv",
            help="How much of the spot→target gap you assume the market closes by "
                 "expiry. 1.0 = fully re-rated; 0.0 = no re-rating at all.",
        )
    with b4:
        scan_vol = st.slider(
            "Terminal volatility σ (%)", 5.0, 120.0, float(round(vol_in * 100, 1)), 0.5,
            key="bridge_vol",
            help="Diffusion applied around each fair-value draw. Defaults to the "
                 "pricer's σ; the chain's ATM implied vol is a reasonable choice.",
        ) / 100.0

    st.markdown(
        f"Projected **Sₜ = ${projected_ST:,.2f}** vs spot **${spot:,.2f}** "
        f"→ implied move **{(projected_ST / spot - 1) * 100:+.1f}%** over {T_exp * 365:.0f} days."
    )

    # Recentre the model's Monte-Carlo intrinsic draws on the chosen target so
    # the distribution keeps the model's dispersion while honouring the target
    # the user actually selected.
    _mc = np.asarray(result.mc_samples, dtype=float)
    _mc = _mc[np.isfinite(_mc)]
    if _mc.size and _mc.mean() > 0:
        value_samples = _mc * (projected_ST / _mc.mean())
    else:
        value_samples = np.array([projected_ST])

    terminal = deriv_mod.terminal_distribution(
        spot=spot, value_samples=value_samples, T=T_exp,
        sigma=scan_vol, convergence=convergence,
    )

    if terminal.size:
        t1, t2, t3, t4 = st.columns(4)
        t1.markdown(metric_card("E[Sₜ]", f"${terminal.mean():,.2f}"), unsafe_allow_html=True)
        t2.markdown(metric_card("Sₜ p10–p90",
                                f"${np.percentile(terminal, 10):,.0f} – ${np.percentile(terminal, 90):,.0f}",
                                sub=True), unsafe_allow_html=True)
        t3.markdown(metric_card("P(Sₜ > spot)", f"{(terminal > spot).mean() * 100:.1f}%", sub=True),
                    unsafe_allow_html=True)
        t4.markdown(metric_card("Risk-neutral E[Sₜ]",
                                f"${spot * float(np.exp(rf_rate * T_exp)):,.2f}", sub=True),
                    unsafe_allow_html=True)

    side = "call" if projected_ST >= spot else "put"
    scan_df = chain.calls if side == "call" else chain.puts
    rows = []
    if not scan_df.empty and terminal.size:
        for _, r_ in scan_df.iterrows():
            mkt = float(r_.get("lastPrice") or 0.0)
            if mkt <= 0:
                mid_bid, mid_ask = float(r_.get("bid") or 0.0), float(r_.get("ask") or 0.0)
                mkt = (mid_bid + mid_ask) / 2 if (mid_bid or mid_ask) else 0.0
            if mkt <= 0:
                continue
            k = float(r_["strike"])
            # Quoted IV may be NaN — either absent, or rejected and unsolvable
            # by the chain loader. Never let that reach the pricer as sigma.
            _iv = r_.get("impliedVolatility")
            iv_row = float(_iv) if (_iv is not None and pd.notna(_iv) and _iv > 0) else None
            e = deriv_mod.bridge_edge_mc(
                terminal=terminal, strike=k, T=T_exp, r=rf_rate,
                market_price=mkt, spot=spot, kind=side, market_iv=iv_row,
            )
            rows.append({
                "Strike": k,
                "Premium": round(mkt, 2),
                "IV %": round(iv_row * 100, 1) if iv_row else None,
                "P(ITM)": round(e.prob_itm, 3),
                "E[payoff] PV": round(e.expected_payoff_pv, 2),
                "Median payoff": round(e.payoff_p50, 2),
                "Risk-neutral": round(e.risk_neutral_price, 2) if e.risk_neutral_price == e.risk_neutral_price else None,
                "Edge ($)": round(e.edge, 2),
                "Edge (x premium)": round(e.edge / mkt, 2) if mkt else None,
                "View premium": round(e.view_premium, 2) if e.view_premium == e.view_premium else None,
                "Verdict": e.verdict,
            })

    if rows:
        edge_df = pd.DataFrame(rows).sort_values("Edge ($)", ascending=False)
        best = edge_df.iloc[0]
        st.success(
            f"Best **{side}** by expected edge: strike **${best['Strike']:.2f}** — "
            f"premium **${best['Premium']:.2f}**, P(ITM) **{best['P(ITM)']:.1%}**, "
            f"expected edge **${best['Edge ($)']:.2f}** "
            f"({best['Edge (x premium)']:.2f}× premium) — _{best['Verdict']}_."
        )
        st.dataframe(edge_df, width="stretch", hide_index=True, height=300)
        st.caption(
            "Edge integrates the payoff across the projected terminal distribution "
            "(model intrinsic draws re-centred on your target, diffused at σ), so it is a "
            "genuine expectation rather than a payoff at one price. **It is still a "
            "view-dependent number, not an arbitrage:** the distribution is real-world, "
            "built from your valuation, while the premium is set off the risk-neutral law. "
            "*View premium* isolates that — it compares the model PV against Black-Scholes "
            "at the market's own IV, so a large edge with a small view premium means the "
            "model is disagreeing about volatility, not direction."
        )
    else:
        st.info("No priced contracts available on this chain to score.")

    st.divider()

    # ── Futures cost-of-carry ────────────────────────────────────────────────
    st.markdown(theme.section("Futures cost-of-carry"), unsafe_allow_html=True)
    st.latex(r"F = S \cdot e^{(r + s - c)\,T}")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        f_spot = st.number_input("Spot (S)", value=float(round(spot, 2)), min_value=0.01,
                                 step=1.0, key="fut_S")
    with f2:
        f_T = st.slider("Tenor T (years)", 0.05, 3.0, 1.0, 0.05, key="fut_T")
    with f3:
        storage_c = st.number_input("Storage cost s (annual)", value=0.0, min_value=0.0,
                                    max_value=0.5, step=0.005, format="%.3f", key="fut_s")
    with f4:
        conv_y = st.number_input("Convenience yield c (annual)", value=0.0, min_value=0.0,
                                 max_value=0.5, step=0.005, format="%.3f", key="fut_c")

    fut = deriv_mod.futures_fair_value(f_spot, rf_rate, f_T, storage_c, conv_y)
    market_fut = st.number_input("Observed futures price (optional)", value=0.0,
                                 min_value=0.0, step=1.0, key="fut_mkt")

    u1, u2, u3 = st.columns(3)
    u1.markdown(metric_card("Fair Value (F)", f"${fut.fair_value:,.2f}"), unsafe_allow_html=True)
    u2.markdown(metric_card("Basis (F − S)", f"${fut.basis:,.2f}", sub=True), unsafe_allow_html=True)
    u3.markdown(metric_card("Net Carry (r + s − c)", f"{fut.annualized_carry * 100:.2f}%", sub=True),
                unsafe_allow_html=True)

    if market_fut > 0:
        mis = fut.mispricing(market_fut)
        state = "contango vs fair value" if mis > 0 else "backwardation vs fair value"
        st.info(f"Observed **${market_fut:,.2f}** is **${mis:+,.2f}** off fair value — {state}.")

    # Snapshot for the PDF report.
    _opt_summary = {
        "expiry": chain.expiry, "days_to_expiry": T_exp * 365,
        "source": "yfinance" if chain.is_live else "synthetic", "spot": spot,
        "kind": opt_kind, "strike": strike, "sigma": vol_in,
        "bs_price": bs.price, "binomial_price": amer, "mc_price": mc["price"],
        "delta": bs.delta, "gamma": bs.gamma, "theta": bs.theta,
        "vega": bs.vega, "rho": bs.rho,
        "projected_st": projected_ST,
        "futures_fair_value": fut.fair_value, "futures_basis": fut.basis,
        "futures_carry": fut.annualized_carry,
    }
    if rows:
        _opt_summary.update({
            "best_strike": float(best["Strike"]),
            "best_premium": float(best["Premium"]),
            "best_edge": float(best["Edge ($)"]),
            "best_verdict": str(best["Verdict"]),
            "best_prob_itm": float(best["P(ITM)"]),
            "edge_method": "Monte-Carlo over projected terminal distribution",
            "convergence": convergence,
            "terminal_vol": scan_vol,
        })
        if best["View premium"] is not None and best["View premium"] == best["View premium"]:
            _opt_summary["best_view_premium"] = float(best["View premium"])
    st.session_state["rep_options"] = _opt_summary


# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Report"):
    st.markdown(theme.section("Equity research report", "Report"), unsafe_allow_html=True)
    st.caption(
        "Generate a formatted PDF covering every tab: valuation range and signal, "
        "fundamentals, SHAP impacts, DCF, peers, backtest, regime and technical timing, "
        "guardrails, insider flow, position sizing, screener, filing divergence and the "
        "options / futures desk."
    )

    SECTIONS = [
        "Fundamentals & model features", "SHAP feature impacts", "DCF",
        "Peer benchmarking", "Backtest", "Execution & timing",
        "Position sizing", "Guardrails & insider flow", "Screener",
        "Filing divergence", "Options & futures", "Macro backdrop",
        "Technical chart & indicators",
    ]
    chosen = st.multiselect(
        "Sections to include", SECTIONS, default=SECTIONS,
        help="Everything is included by default. Deselect anything you want left out.",
    )
    want = set(chosen)

    # Sections sourced from other tabs are only available once those tabs have
    # been computed — flag the gaps rather than silently dropping them.
    pending = []
    if "Screener" in want and st.session_state.get("rep_screener") is None:
        pending.append("Screener (run a screen on the Screener tab)")
    if "Filing divergence" in want and st.session_state.get("rep_filings") is None:
        pending.append("Filing divergence (run a comparison on the Filings Δ tab)")
    if "Options & futures" in want and st.session_state.get("rep_options") is None:
        pending.append("Options & futures (open the Options & Futures tab once)")
    if pending:
        st.info("Not yet computed, so these will be skipped: " + "; ".join(pending))

    # A tab that raised contributed nothing to session_state, so the report
    # would quietly omit its section. Say so — an incomplete research note that
    # does not announce the gap is worse than one that does.
    _failed = st.session_state.get("tab_failures") or {}
    if _failed:
        st.warning(
            "These tabs failed to render, so their sections will be missing or "
            "fall back to defaults: "
            + "; ".join(f"**{k}** ({v})" for k, v in _failed.items())
        )

    _rep_iv = st.session_state.get("chart_interval", "15m")
    _rep_hz = int(st.session_state.get("chart_horizon", 6))
    if "Technical chart & indicators" in want:
        st.caption(
            f"The technical section uses the **{_rep_iv}** interval and a **{_rep_hz}-bar** "
            f"accuracy horizon, following the Charts tab. Change them there to change the report."
        )

    if st.button("Generate Report", type="primary"):
        with st.spinner("Building report..."):
            dcf_payload = None
            if "DCF" in want:
                _d = st.session_state.get("rep_dcf") or dcf_mod.run_dcf(dcf_mod.DCFInputs(
                    fcf0=float(raw.get("free_cash_flow") or raw.get("op_cash_flow") or 1e9),
                    shares=float(raw.get("shares") or 1e9),
                    net_debt=float(raw.get("long_term_debt") or 0.0),
                ))
                dcf_payload = {"intrinsic_per_share": _d.intrinsic_per_share}

            peers_payload = None
            if "Peer benchmarking" in want:
                peers_payload = peers_mod.benchmark(
                    tk, sector=raw.get("sector"),
                    self_metrics={"name": raw.get("name", tk), "pe": feats.get("pe_ratio"),
                                  "debt_to_equity": feats.get("debt_to_equity"),
                                  "profit_margin": feats.get("profit_margin"),
                                  "market_cap": raw.get("market_cap"), "source": "SEC"},
                )

            backtest_payload = None
            if "Backtest" in want:
                backtest_payload = (
                    st.session_state.get("rep_backtest")
                    or (get_backtest_cached(tk, price)[0] or None)
                )

            quality_payload = None
            if "Guardrails & insider flow" in want:
                quality_payload = st.session_state.get("rep_quality")
                if quality_payload is None and raw.get("cik"):
                    quality_payload = get_quality_cached(raw["cik"], raw.get("market_cap"))

            pdf_bytes = reports.build_report(
                ticker=tk,
                company=raw.get("name", tk),
                market_price=price,
                valuation={"point": result.point, "low": result.low,
                           "median": result.median, "high": result.high},
                signal_text=signal_text,
                attribution=attribution.as_frame() if "SHAP feature impacts" in want else None,
                dcf=dcf_payload,
                peers=peers_payload,
                macro=({"cpi": macro_now.cpi, "fed_funds": macro_now.fed_funds,
                        "yield_curve": macro_now.yield_curve_10y_2y}
                       if "Macro backdrop" in want else None),
                features=feats if "Fundamentals & model features" in want else None,
                fundamentals=raw if "Fundamentals & model features" in want else None,
                backtest=backtest_payload,
                regime=regime if "Execution & timing" in want else None,
                technical=technical if "Execution & timing" in want else None,
                sizing=sizing if "Position sizing" in want else None,
                quality=quality_payload,
                insider=(st.session_state.get("rep_insider")
                         if "Guardrails & insider flow" in want else None),
                filings=(st.session_state.get("rep_filings")
                         if "Filing divergence" in want else None),
                screener=(st.session_state.get("rep_screener")
                          if "Screener" in want else None),
                options=(st.session_state.get("rep_options")
                         if "Options & futures" in want else None),
                chart_png=(get_chart_png_cached(tk, _rep_iv, price)
                           if "Technical chart & indicators" in want else None),
                chart_interval=_rep_iv if "Technical chart & indicators" in want else None,
                indicators=(get_indicators_cached(tk, _rep_iv, price)
                            if "Technical chart & indicators" in want else None),
                indicator_accuracy=(get_accuracy_cached(tk, _rep_iv, price, _rep_hz)
                                    if "Technical chart & indicators" in want else None),
                accuracy_horizon=_rep_hz if "Technical chart & indicators" in want else None,
            )

        ext = "pdf" if reports.has_reportlab() else "txt"
        mime = "application/pdf" if ext == "pdf" else "text/plain"
        st.success(f"Report ready ({len(pdf_bytes):,} bytes).")
        st.download_button(
            f"Download {tk} Equity Report (.{ext})",
            data=pdf_bytes,
            file_name=f"{tk}_equity_report.{ext}",
            mime=mime,
            type="primary",
        )
        if ext == "txt":
            st.caption("ReportLab not installed — generated a plain-text report instead of PDF.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 14 — TRACK RECORD
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Track Record"):
    st.markdown(theme.section("Your saved valuations", "Track record"), unsafe_allow_html=True)
    st.caption(
        "Every analysis you run is stored with its p10–p90 band and the price at "
        "the time, then scored against what the price actually did. No brokerage "
        "will show you whether its own price targets were right; this does."
    )

    _key = st.session_state.get("ledger_key", "")
    _preds = pred_mod.for_key(_key)

    if not _preds:
        st.info(
            "Nothing recorded yet. Run an analysis and it appears here — "
            "your first entry is saved the moment this one finishes."
        )
    else:
        _outcomes = pred_mod.score(
            _preds, lambda t: market_mod.get_market_data(t).price
        )
        _cal = pred_mod.calibration(_outcomes)

        st.markdown(theme.section("Calibration", ""), unsafe_allow_html=True)
        st.caption(
            f"If the band is honest, about {_cal.target*100:.0f}% of matured "
            "predictions should land inside it. A band that captures almost "
            "everything is too wide to be useful; one that captures far less is "
            "overconfident. Only predictions past their "
            f"{pred_mod.DEFAULT_HORIZON_DAYS}-day horizon are scored — a "
            "one-day-old call has not had time to be right or wrong."
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(metric_card("Recorded", f"{len(_preds)}"), unsafe_allow_html=True)
        k2.markdown(metric_card("Scored", f"{_cal.n}"), unsafe_allow_html=True)
        if _cal.n:
            k3.markdown(
                metric_card("Band Coverage", f"{_cal.coverage*100:.0f}%",
                            sub=True) +
                f'<div class="metric-note">95% CI '
                f'{_cal.ci_low*100:.0f}–{_cal.ci_high*100:.0f}%</div>',
                unsafe_allow_html=True,
            )
            k4.markdown(
                metric_card("Direction", f"{_cal.hit_rate*100:.0f}%", sub=True) +
                f'<div class="metric-note">{_cal.n_direction_correct}'
                f'/{_cal.n_directional} calls</div>',
                unsafe_allow_html=True,
            )
        else:
            k3.markdown(metric_card("Band Coverage", "—"), unsafe_allow_html=True)
            k4.markdown(metric_card("Direction", "—"), unsafe_allow_html=True)

        st.markdown(
            f'<div class="verdict"><div class="verdict-line">{_cal.verdict}.'
            f'</div></div>', unsafe_allow_html=True,
        )

        st.markdown(theme.section("Ledger", ""), unsafe_allow_html=True)
        _rows = []
        for o in _outcomes:
            p = o.prediction
            _rows.append({
                "Date": _dt.datetime.fromtimestamp(p.created_at).strftime("%Y-%m-%d"),
                "Ticker": p.ticker,
                "Price then": f"${p.price_at:,.2f}",
                "Band": f"${p.p10:,.0f} – ${p.p90:,.0f}",
                "Fair": f"${p.p50:,.2f}",
                "Price now": f"${o.price_now:,.2f}" if o.price_now is not None else "—",
                "Move": f"{o.move_pct:+.1f}%" if o.move_pct is not None else "—",
                "Status": (
                    "Open" if not p.is_mature
                    else "In band" if o.inside_band
                    else "Outside band" if o.inside_band is not None
                    else "No price"
                ),
                "Signal": p.signal or "—",
            })
        st.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True)
        st.caption(
            f"Storage: {pred_mod.backend_name()}. Predictions are keyed to the "
            "anonymous ledger key in the sidebar — save it to keep this record."
        )
        if not _userdb.is_durable():
            st.warning(
                "**This record is not durable.** " + _userdb.durability_note()
                + " Until then, treat anything here as provisional."
            )

    # ── Cross-user calibration ───────────────────────────────────────────────
    # The compounding moat: the model's honesty score across every user, not
    # just this one. A brokerage cannot publish this, because it would mean
    # publishing whether its own calls were right.
    st.markdown(theme.section("The model's record across everyone", "Calibration"),
                unsafe_allow_html=True)
    st.caption(
        "Every valuation ever run, scored the same way — did the outcome land "
        "inside the p10–p90 band. Split by the kind of call, because the honest "
        "question is not 'is the model good' but 'is it calibrated *when it says "
        "undervalued*'. Correlated duplicates are collapsed first: fifty users "
        "valuing the same name on the same day are one bet, not fifty, so the "
        "sample counts distinct calls and the interval cannot be faked wide by "
        "popularity."
    )
    try:
        _global = get_global_calibration()
    except Exception:
        _global = {}

    _all = _global.get("All")
    if _all is None or _all.n < 1:
        st.info(
            "No resolved predictions across the user base yet. This table fills "
            "in as valuations are run and mature past their "
            f"{pred_mod.DEFAULT_HORIZON_DAYS}-day horizon — it is the one number "
            "here that gets stronger the more the app is used."
        )
    else:
        _order = ["All", "Undervalued calls", "Overvalued calls",
                  "Fair-value calls", "Other calls"]
        _crows = []
        for _name in _order:
            _c = _global.get(_name)
            if _c is None or _c.n == 0:
                continue
            _crows.append({
                "Cohort": _name,
                "Resolved": _c.n,
                "In band": f"{_c.coverage*100:.0f}%",
                "95% CI": f"{_c.ci_low*100:.0f}–{_c.ci_high*100:.0f}%",
                "Direction": f"{_c.hit_rate*100:.0f}%" if _c.n_directional else "—",
                "Reads as": _c.verdict,
            })
        st.dataframe(pd.DataFrame(_crows), width="stretch", hide_index=True)
        st.caption(
            "Target coverage is 80% — that is what a p10–p90 band claims. A "
            "cohort whose interval brackets 80% is honestly calibrated; well "
            "above means the bands are too wide to be useful, well below means "
            "the model is overconfident on that kind of call. Either way the "
            "number is measured, not asserted — which is the whole difference "
            "from a price target."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 15 — WATCHLIST
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Watchlist"):
    _key = st.session_state.get("ledger_key", "")
    _list = watch_mod.get(_key)

    st.markdown(theme.section("Names you are following", "Watchlist"), unsafe_allow_html=True)

    _wc1, _wc2 = st.columns([3, 1])
    with _wc1:
        _new = st.text_input(
            "Add a ticker", value="", max_chars=8, placeholder="e.g. MSFT",
            label_visibility="collapsed",
        )
    with _wc2:
        if st.button("Add", width="stretch") and _new:
            if watch_mod.add(_key, _new):
                st.rerun()
            else:
                st.warning(f"'{_new}' is not a symbol this list accepts.")

    if not _list:
        st.info(
            "Your watchlist is empty. Add a ticker above, or use the button under "
            "the quote header to follow the company you are looking at."
        )
    else:
        # ── Where each name sits against its own last valuation ───────────────
        st.markdown(theme.section("Against your last valuation", ""), unsafe_allow_html=True)
        st.caption(
            "For each name you have valued, where the price sits now relative to "
            "the band you recorded then. Names with no saved valuation show a "
            "dash — run one and it fills in."
        )
        _rows = []
        for _t in _list:
            _md = market_mod.get_market_data(_t)
            _last = pred_mod.for_key(_key, ticker=_t, limit=1)
            _p = _last[0] if _last else None
            if _p and _md.price:
                _where = ("Below band" if _md.price < _p.p10
                          else "Above band" if _md.price > _p.p90 else "In band")
                _gap = f"{(_p.p50 - _md.price) / _md.price * 100:+.1f}%"
                _band = f"${_p.p10:,.0f} – ${_p.p90:,.0f}"
                _when = _dt.datetime.fromtimestamp(_p.created_at).strftime("%Y-%m-%d")
            else:
                _where, _gap, _band, _when = "—", "—", "—", "—"
            _rows.append({
                "Ticker": _t,
                "Price": f"${_md.price:,.2f}" if _md.price else "—",
                "Your band": _band,
                "Position": _where,
                "To fair value": _gap,
                "Valued on": _when,
            })
        st.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True)

        # ── Calendar ─────────────────────────────────────────────────────────
        st.markdown(theme.section("What happens next", ""), unsafe_allow_html=True)
        st.caption(
            "Earnings dates come from the market feed and are announced. Filing "
            "dates are **projected** from each company's own filing cadence — "
            "EDGAR publishes only what has already been filed — so treat them as "
            "estimates, not commitments."
        )
        _events = cal_mod.upcoming(_list)
        if not _events:
            st.caption(
                "No dates available right now. The earnings feed is frequently "
                "empty between reporting cycles, and a filing projection needs at "
                "least three past filings to measure a cadence from."
            )
        else:
            st.dataframe(
                pd.DataFrame([{
                    "Date": e.date.isoformat(),
                    "In": f"{e.days_away}d",
                    "Ticker": e.ticker,
                    "Event": e.kind,
                    "Basis": "Projected" if e.projected else "Announced",
                } for e in _events]),
                width="stretch", hide_index=True,
            )

        _drop = st.selectbox("Remove a ticker", ["—"] + _list)
        if _drop != "—" and st.button(f"Remove {_drop}"):
            watch_mod.remove(_key, _drop)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTS — revenue by reportable segment, from the latest 10-K
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Segments"):
    st.markdown(theme.section("Where the revenue comes from", "Filings"),
                unsafe_allow_html=True)
    st.caption(
        "Revenue by reportable segment, read from the segment disclosure in the "
        "latest 10-K. This does not come from the XBRL facts API the rest of the "
        "app uses — that endpoint excludes dimensional facts, and a segment is "
        "dimensional by definition — so it is parsed from the rendered table "
        "EDGAR's own viewer displays. Layouts vary by filer, so this fails "
        "closed: no breakdown rather than a wrong one."
    )

    _seg = seg_mod.get_segments(tk)
    st.session_state["rep_segments"] = _seg

    if _seg.confidence == "parsed" and _seg.segments:
        _shares = _seg.shares()
        _ordered = sorted(_seg.segments.items(), key=lambda kv: kv[1], reverse=True)

        _sm1, _sm2 = st.columns([1, 1])
        _sm1.markdown(
            metric_card("Segments Reported", f"{len(_ordered)}"), unsafe_allow_html=True)
        _top_name, _top_val = _ordered[0]
        _sm2.markdown(
            metric_card("Largest", f"{_shares[_top_name]:.0f}%", sub=True) +
            f'<div class="metric-note">{_top_name}</div>',
            unsafe_allow_html=True,
        )

        if _seg.is_concentrated:
            st.warning(
                f"**{_top_name}** carries {_shares[_top_name]:.0f}% of revenue. "
                "A valuation of this company is largely a valuation of that one "
                "segment, whatever the consolidated multiples suggest."
            )

        fig, ax = plt.subplots(figsize=(8, max(2.2, 0.42 * len(_ordered))))
        theme.style_axes(fig, ax)
        _names = [n for n, _ in _ordered][::-1]
        _vals = [v for _, v in _ordered][::-1]
        ax.barh(_names, _vals, color=theme.GREEN, edgecolor="none")
        ax.set_xlabel("Revenue ($)")
        st.pyplot(fig, width="stretch"); plt.close(fig)

        st.dataframe(
            pd.DataFrame([{
                "Segment": n,
                "Revenue": f"${v:,.0f}",
                "Share": f"{_shares[n]:.1f}%",
            } for n, v in _ordered]),
            width="stretch", hide_index=True,
        )
        st.caption(
            f"Source: *{_seg.source_report}*"
            + (f" · {_seg.period}" if _seg.period else "")
            + (f" · figures {_seg.scale_note}" if _seg.scale_note else "")
            + ". Totals, eliminations and reconciling items are excluded so the "
              "shares sum across operating segments only."
        )
    elif _seg.confidence == "table-only":
        st.info(_seg.note)
        if _seg.table_html:
            with st.expander(f"Table as filed — {_seg.source_report}"):
                st.markdown(_seg.table_html, unsafe_allow_html=True)
    else:
        st.info(
            _seg.note or "No segment disclosure located for this filer."
        )
        st.caption(
            "Common and not an error: a single-segment company has nothing to "
            "disaggregate, and some filers place the disclosure in a table this "
            "parser does not recognise."
        )


# ══════════════════════════════════════════════════════════════════════════════
# MICROSTRUCTURE — the documented footprint of institutional execution
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Microstructure"):
    st.markdown(theme.section("How the flow moves the price", "Market"),
                unsafe_allow_html=True)
    st.caption(
        "The honest version of what institutions do that retail does not see. "
        "Not secret algorithms and not manipulation — a delayed price feed shows "
        "neither — but the aggregate, *documented* footprint of how large "
        "execution actually clears. Every figure below carries its sample size "
        "and, where it is a claim about the future, a confidence interval. "
        "Several of these are real anomalies; none of them is a guaranteed edge."
    )

    # ── Overnight vs intraday ────────────────────────────────────────────────
    _split = micro_mod.overnight_intraday_split(md.history)
    st.markdown(theme.section("Overnight vs the tradable session", ""),
                unsafe_allow_html=True)
    if _split is None:
        st.info("Not enough daily history to decompose the return.")
    else:
        st.caption(
            "Splitting each day into the overnight gap (close→open, when you "
            "cannot trade) and the intraday session (open→close, when you can). "
            "The published anomaly — Cooper–Cliff–Gulen and others — is that "
            "historically most of the equity return accrued overnight while the "
            "session a day-trader lives in drifted flat or negative."
        )
        _c1, _c2, _c3 = st.columns(3)
        _c1.markdown(
            metric_card("Overnight", f"{_split.overnight_mean_bp:+.1f} bp/day", sub=True)
            + f'<div class="metric-note">compounded {_split.overnight_total:+.0f}% over {_split.n_days}d</div>',
            unsafe_allow_html=True)
        _c2.markdown(
            metric_card("Intraday", f"{_split.intraday_mean_bp:+.1f} bp/day", sub=True)
            + f'<div class="metric-note">compounded {_split.intraday_total:+.0f}%</div>',
            unsafe_allow_html=True)
        _c3.markdown(
            metric_card("Close-to-close", f"{_split.total:+.0f}%", sub=True)
            + '<div class="metric-note">the two legs, recombined</div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="verdict"><div class="verdict-line">{_split.verdict}.'
            '</div></div>', unsafe_allow_html=True)

    # ── Time-of-day profile ──────────────────────────────────────────────────
    st.markdown(theme.section("When the size trades", ""), unsafe_allow_html=True)
    _intr = intraday_mod.get_intraday(tk, "30m", spot_fallback=price)
    _prof = micro_mod.time_of_day_profile(_intr.df)
    if _prof is None:
        st.info("Intraday history unavailable, so the time-of-day profile is empty.")
    else:
        st.caption(
            f"Averaged over {_prof.n_sessions} sessions of 30-minute bars. Volume "
            "and volatility trace a U across the day — heavy at the open and into "
            "the close, thin at midday — the residue of VWAP/TWAP schedules and "
            "the closing auction. The open and the last half-hour are where a "
            "retail order is most likely to be run over."
            + ("" if _intr.is_live else " *(synthetic data — live feed unavailable.)*")
        )
        fig, ax = plt.subplots(figsize=(9, 3.0))
        theme.style_axes(fig, ax)
        _x = np.arange(len(_prof.labels))
        ax.bar(_x, _prof.volume_share, color=theme.GREEN, alpha=0.85, label="Volume share (%)")
        ax2 = ax.twinx()
        ax2.plot(_x, _prof.volatility, color=theme.CHAMPAGNE, lw=1.8, marker="o",
                 ms=3, label="Volatility (bp)")
        ax2.tick_params(colors=theme.MUTED)
        ax.set_xticks(_x[::2]); ax.set_xticklabels(_prof.labels[::2], fontsize=7, rotation=45)
        ax.set_ylabel("Volume share (%)"); ax2.set_ylabel("Mean |return| (bp)")
        st.pyplot(fig, width="stretch"); plt.close(fig)
        if _prof.busiest_label:
            st.caption(f"Heaviest bucket: **{_prof.busiest_label}**.")

    # ── Liquidity sweeps ─────────────────────────────────────────────────────
    st.markdown(theme.section("Stop-runs, measured", ""), unsafe_allow_html=True)
    _sweeps = micro_mod.liquidity_sweeps(md.history)
    if _sweeps is None:
        st.info("Not enough daily history to study sweeps.")
    else:
        st.caption(
            "A downside sweep is a day that pierces the prior 20 days' low then "
            "closes back above it — stops taken, then price reclaimed, the "
            "'stop-run' retail lore describes. The reversal rate is measured with "
            "a Wilson interval, so a pattern that is really just noise reports as "
            "noise. Read the confidence interval, not the headline percentage."
        )
        _sc = st.columns(2)
        for _col, _key in zip(_sc, ("downside", "upside")):
            _st = _sweeps[_key]
            with _col:
                _cls = "verdict-under" if _st.is_better_than_chance else "verdict-inside"
                st.markdown(
                    metric_card(f"{_key.title()} sweep reversal",
                                f"{_st.reversal_rate*100:.0f}%", sub=True)
                    + f'<div class="metric-note">95% CI '
                    f'{_st.ci_low*100:.0f}–{_st.ci_high*100:.0f}% · '
                    f'{_st.n_events} events</div>'
                    + f'<span class="{_cls}">{_st.verdict}</span>',
                    unsafe_allow_html=True)

    # ── Round-number magnetism ───────────────────────────────────────────────
    _rn = micro_mod.round_number_magnetism(md.history)
    if _rn is not None:
        st.markdown(theme.section("Round-number magnetism", ""), unsafe_allow_html=True)
        st.caption(
            f"Closes within {0.1:.1f}% of a ${_rn.step:g} level, against what a "
            "uniform distribution would place there. Round numbers concentrate "
            "resting orders and stops, so price is often said to be drawn to "
            "them; the excess over the baseline is the only honest evidence."
        )
        _r1, _r2 = st.columns(2)
        _r1.markdown(metric_card("Closes near round levels", f"{_rn.near_rate:.1f}%"),
                     unsafe_allow_html=True)
        _r2.markdown(
            metric_card("Excess over chance", f"{_rn.excess:+.1f} pts", sub=True)
            + f'<div class="metric-note">{"clustering present" if _rn.is_magnetic else "no clear clustering"}</div>',
            unsafe_allow_html=True)

    st.caption(
        "None of the above is trading advice. The point of measuring these is the "
        "opposite of a signal service: to show which patterns survive an honest "
        "confidence interval and which evaporate — because most do."
    )


# ══════════════════════════════════════════════════════════════════════════════
# CRYPTO — live 24/7 charts and measured technical analysis for major coins
# ══════════════════════════════════════════════════════════════════════════════
with tab_guard("Crypto"):
    st.markdown(theme.section("Live crypto — chart and measured signals", "Crypto"),
                unsafe_allow_html=True)
    st.info(
        "**No intrinsic valuation here, on purpose.** A token has no cash flow, "
        "no filings and no book value, so DCF, the quantile fair-value range and "
        "the forensic scores simply do not apply — and inventing a 'fair value' "
        "for a coin would be the fabricated number this whole app refuses to "
        "produce. What *does* carry over is everything that reads a price series: "
        "the live chart, the indicators, and each indicator's **measured** hit "
        "rate with a confidence interval. Markets here run 24/7, so there is no "
        "session or overnight gap to reason about."
    )

    _cc1, _cc2, _cc3 = st.columns([1.4, 1, 1])
    with _cc1:
        _coin_label = st.selectbox("Coin", crypto_mod.labels(), index=0, key="crypto_coin")
    _coin = crypto_mod.by_label(_coin_label) or crypto_mod.coins()[0]
    with _cc2:
        _cx_interval = st.radio(
            "Interval", list(intraday_mod.INTERVALS.keys()),
            index=2, horizontal=True, key="crypto_interval",
        )
    with _cc3:
        _cx_refresh_label = st.selectbox(
            "Auto-refresh", ["Off", "5s", "10s", "30s", "60s"], index=2,
            key="crypto_refresh",
            help="Refreshes only this block, not the whole app.",
        )
    _CX_REFRESH = {"Off": None, "5s": 5, "10s": 10, "30s": 30, "60s": 60}[_cx_refresh_label]
    _cx_horizon = st.selectbox(
        "Accuracy horizon (bars forward)", [3, 6, 12, 24], index=1, key="crypto_horizon",
        help="How far ahead a signal is scored before checking if it was right.",
    )

    # Live TradingView chart for the coin — inherently 24/7 and streaming.
    _tv_map = {"5m": "5", "10m": "10", "15m": "15", "30m": "30", "1h": "60"}
    st.components.v1.html(f"""
<div class="tradingview-widget-container" style="height:600px">
  <div id="tv_crypto" style="height:600px"></div>
</div>
<script src="https://s3.tradingview.com/tv.js"></script>
<script>
new TradingView.widget({{
  "autosize": true, "symbol": "{_coin.tv}", "interval": "{_tv_map[_cx_interval]}",
  "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "en",
  "toolbar_bg": "#161B22", "enable_publishing": false, "hide_side_toolbar": false,
  "allow_symbol_change": true, "withdateranges": true, "details": true,
  "studies": ["MASimple@tv-basicstudies", "RSI@tv-basicstudies",
              "MACD@tv-basicstudies"],
  "container_id": "tv_crypto"
}});
</script>
""", height=620)
    st.caption(
        f"TradingView live feed for **{_coin.tv}**. It streams in your browser, so "
        "it is always current and needs no market-hours gating — crypto trades "
        "around the clock."
    )

    st.divider()

    # Everything below computes on yfinance OHLCV for the coin, reusing the exact
    # indicator + accuracy pipeline the equity Charts tab uses. Only this block
    # re-runs on the timer, so the whole app is not recomputed on each tick.
    _cx_fragment = getattr(st, "fragment", None)
    _cx_wrap = (lambda fn: _cx_fragment(run_every=_CX_REFRESH)(fn)) \
        if (_cx_fragment and _CX_REFRESH) else (lambda fn: fn)

    @_cx_wrap
    def _crypto_view():
        _cmd = get_market_cached(_coin.yf, 0.0)
        _spot = _cmd.price or 0.0
        _bars = get_intraday_cached(_coin.yf, _cx_interval, _spot)
        _last = _bars.last_price or _spot
        _chg = _bars.session_change

        _m1, _m2, _m3 = st.columns(3)
        _m1.markdown(metric_card(f"{_coin.symbol} Price",
                     f"${_last:,.2f}" if _last else "—"), unsafe_allow_html=True)
        _m2.markdown(
            metric_card("Window Change",
                        f"{_chg*100:+.2f}%" if _chg is not None else "—", sub=True)
            + f'<div class="metric-note">over the visible {_cx_interval} window</div>',
            unsafe_allow_html=True)
        _m3.markdown(
            metric_card("Feed", "LIVE" if _bars.is_live else "SYNTHETIC", sub=True)
            + f'<div class="metric-note">{_bars.source}</div>',
            unsafe_allow_html=True)

        st.markdown(theme.section("Measured signal accuracy", ""), unsafe_allow_html=True)
        st.caption(
            "Each indicator's hit rate on this coin's own recent bars, with a "
            "Wilson interval. Read the interval: when it straddles 50%, the "
            "signal is no better than a coin flip on this coin, and the verdict "
            "says so. This is the honest 'output' — it refuses to dress up noise."
        )
        _acc = get_accuracy_cached(_coin.yf, _cx_interval, _spot, int(_cx_horizon))
        if not _acc:
            st.warning(
                "Not enough bars to measure accuracy yet — try a longer interval, "
                "or the live feed may be briefly rate-limited."
            )
        else:
            _asum = acc_mod.summary(_acc)
            _s1, _s2, _s3 = st.columns(3)
            _s1.markdown(metric_card("Indicators measured", f"{_asum['measured']}"),
                         unsafe_allow_html=True)
            _s2.markdown(
                metric_card("Better than chance", f"{_asum['better_than_chance']}", sub=True)
                + '<div class="metric-note">interval clears 50%</div>',
                unsafe_allow_html=True)
            if _asum["best"]:
                _s3.markdown(
                    metric_card("Best", f"{_asum['best_rate']*100:.0f}%", sub=True)
                    + f'<div class="metric-note">{_asum["best"]}</div>',
                    unsafe_allow_html=True)
            st.dataframe(acc_mod.as_frame(_acc), width="stretch", hide_index=True)

            if _asum["better_than_chance"] == 0:
                st.info(
                    "On this coin, over this window, **no indicator beats a coin "
                    "flip** once the confidence interval is honest. That is a "
                    "finding, not a failure — most short-horizon signals on a "
                    "24/7 market do not survive measurement."
                )

    _crypto_view()

    if _CX_REFRESH and not _cx_fragment:
        st.caption(
            "Auto-refresh needs Streamlit 1.38+ for `st.fragment`; this build "
            "updates when you interact with the page."
        )
