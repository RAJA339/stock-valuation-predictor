"""
Intrinsic Stock Valuation Predictor  —  v2.0
============================================
Webster University — MS Business Analytics | Group ML Project
Author: Raja Mupparaju

A Streamlit application that predicts a company's intrinsic value from SEC EDGAR
fundamentals, live market data, real-time macro indicators and earnings-call
sentiment, then explains and stress-tests that prediction.

Feature areas (each in its own tab):
  1. Valuation        — XGBoost point estimate + quantile intrinsic-value range
  2. Explainability   — SHAP / LIME per-feature attribution (waterfall)
  3. DCF & Scenario   — interactive DCF (WACC / terminal-growth sliders) + Monte-Carlo
  4. Peer Benchmarking— EV/EBITDA, P/E, Debt/Equity vs industry peers
  5. Backtesting      — signal performance over 1/3/5-year horizons
  6. Report           — one-click PDF equity research summary

The heavy lifting lives in the ``svp`` package. Every data/ML dependency degrades
gracefully, so the app runs even without network access or optional libraries.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")  # headless backend — avoids Streamlit worker-thread GUI errors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from svp import theme
from svp.features import build_features, FEATURE_LABELS
from svp.data import market as market_mod, macro as macro_mod, sentiment as sent_mod, storage
from svp.models import valuation as val_mod, explain as explain_mod, dcf as dcf_mod, backtest as bt_mod
from svp.analytics import peers as peers_mod
from svp import reports

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + THEME
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intrinsic Stock Valuation Predictor",
    page_icon="📊",
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
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Stock Valuation Predictor")
    st.markdown("*Webster University — MS Business Analytics*")
    st.divider()

    ticker = st.text_input("🔤 Stock Ticker Symbol", value="AAPL", max_chars=6).upper().strip()
    use_live_price = st.checkbox("Use live market price (yfinance)", value=True)
    manual_price = st.number_input(
        "💵 Market Price override ($)", min_value=0.0, value=0.0, step=0.5, format="%.2f",
        help="Leave 0 to use the live/last price. Enter a value to override.",
    )

    with st.expander("📝 Earnings-call transcript (sentiment)"):
        transcript = st.text_area(
            "Paste transcript text (optional)", value="", height=120,
            help="Scored with FinBERT if available, else a finance lexicon.",
        )
        if st.checkbox("Use sample transcript"):
            transcript = sent_mod.SAMPLE_TRANSCRIPT

    run_btn = st.button("🔍  Analyze", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Data & Model**")
    st.caption(
        "SEC EDGAR · yfinance/Alpha Vantage · FRED/BLS · FinBERT · "
        "XGBoost (point + quantile) · SHAP/LIME · DCF Monte-Carlo"
    )
    cache_stats = storage.stats()
    st.caption(f"💾 Cache: {cache_stats['backend']} · {cache_stats['fresh']}/{cache_stats['rows']} fresh rows")
    st.caption("⚠️ Educational use only. Not financial advice.")


# ──────────────────────────────────────────────────────────────────────────────
# HEADER + MODEL BANNER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Intrinsic Stock Valuation Predictor")
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

    return {
        "ticker": ticker,
        "price": price,
        "md": md,
        "features": feats,
        "result": result,
        "attribution": attribution,
        "signal_text": signal_text,
        "signal_class": signal_class,
    }


# ──────────────────────────────────────────────────────────────────────────────
# RUN / STATE
# ──────────────────────────────────────────────────────────────────────────────
if run_btn:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Analyzing **{ticker}** — SEC EDGAR · market · macro · sentiment..."):
            analysis = run_analysis(ticker, manual_price, transcript, use_live_price)
        if analysis is None:
            st.error(
                f"❌ Could not retrieve sufficient fundamentals for **{ticker}**. "
                "Try a major US-listed company (e.g. AAPL, MSFT, GOOGL, JPM)."
            )
        else:
            st.session_state["analysis"] = analysis

analysis = st.session_state.get("analysis")

if not analysis:
    st.info("👈  Enter a ticker in the sidebar and click **Analyze** to begin.")
    st.markdown("### What's inside")
    cols = st.columns(3)
    cards = [
        ("🔍 Explainable AI", "SHAP / LIME waterfalls show exactly how each feature moved the valuation."),
        ("📊 Valuation Range", "Quantile XGBoost + Monte-Carlo give an intrinsic-value range, not a single point."),
        ("🌐 Live Pipelines", "yfinance prices, FRED macro (yield curve, CPI, rates), FinBERT sentiment."),
        ("🧮 DCF & Scenario", "Interactive DCF with WACC / terminal-growth sliders, Monte-Carlo fair value."),
        ("🏦 Peer Benchmarking", "EV/EBITDA, P/E, Debt/Equity vs auto-selected industry peers."),
        ("📄 PDF Reports", "One-click equity research summary with feature impacts and signals."),
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

st.markdown(
    f"### {raw.get('name', tk)} ({tk}) "
    f"{theme.source_pill(md.is_live, f'{md.source.upper()} LIVE', 'OFFLINE')}",
    unsafe_allow_html=True,
)

# ── Headline metric row ───────────────────────────────────────────────────────
h1, h2, h3, h4 = st.columns(4)
h1.markdown(metric_card("Intrinsic Value (point)", f"${result.point:.2f}"), unsafe_allow_html=True)
h2.markdown(metric_card("Intrinsic Range (p10–p90)", f"${result.low:.0f} – ${result.high:.0f}", sub=True),
            unsafe_allow_html=True)
h3.markdown(metric_card("Market Price", f"${price:.2f}"), unsafe_allow_html=True)
h4.markdown(
    f'<div class="metric-card"><div class="metric-label">Valuation Signal</div>'
    f'<div class="{signal_class}">{signal_text}</div></div>',
    unsafe_allow_html=True,
)

st.divider()

tabs = st.tabs([
    "📊 Valuation", "🔍 Explainability", "🧮 DCF & Scenario",
    "🏦 Peers", "📉 Backtesting", "📄 Report",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="section-header">📋 Extracted Features</div>', unsafe_allow_html=True)
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
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=430)

        sent_obj = raw.get("sentiment_obj")
        if sent_obj is not None:
            st.caption(
                f"🗣️ Earnings sentiment: **{sent_obj.label}** ({sent_obj.score:+.2f}) "
                f"via {sent_obj.source}"
            )

    with right:
        st.markdown('<div class="section-header">📊 Intrinsic Value Distribution</div>', unsafe_allow_html=True)
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
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(
            f"**Confidence interval:** \\${result.low:.2f} – \\${result.high:.2f} "
            f"(width {result.ci_width_pct:.1f}% of point). "
            f"Monte-Carlo mean \\${result.mc_mean:.2f} ± \\${result.mc_std:.2f}."
        )

    st.markdown('<div class="section-header">📉 Price vs Intrinsic Range</div>', unsafe_allow_html=True)
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
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLAINABILITY (SHAP / LIME)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(
        f'<div class="section-header">🔍 Feature Attribution '
        f'({"SHAP" if attribution.source=="shap" else "XGBoost contributions"})</div>',
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
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tcol:
        show = adf.copy()
        show["contribution"] = show["contribution"].map(lambda x: f"{x:+.2f}")
        show["value"] = show["value"].map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        st.dataframe(show, use_container_width=True, hide_index=True, height=430)

    # Optional LIME.
    st.markdown('<div class="section-header">🍋 LIME Local Explanation</div>', unsafe_allow_html=True)
    if explain_mod.has_lime():
        with st.spinner("Computing LIME explanation..."):
            lime_df = explain_mod.lime_explanation(feats, vm)
        if lime_df is not None:
            st.dataframe(lime_df, use_container_width=True, hide_index=True)
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
with tabs[2]:
    st.markdown('<div class="section-header">🧮 Discounted Cash Flow — Interactive</div>', unsafe_allow_html=True)
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
    dcf_mc = dcf_mod.monte_carlo_dcf(dcf_in, n=4000)

    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(metric_card("DCF Intrinsic / Share", f"${dcf_res.intrinsic_per_share:.2f}"), unsafe_allow_html=True)
    d2.markdown(metric_card("DCF Range (p10–p90)",
                            f"${dcf_mc['p10']:.0f} – ${dcf_mc['p90']:.0f}", sub=True), unsafe_allow_html=True)
    d3.markdown(metric_card("ML Intrinsic (point)", f"${result.point:.2f}"), unsafe_allow_html=True)
    d4.markdown(metric_card("Market Price", f"${price:.2f}"), unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="section-header">Projected vs Discounted FCF</div>', unsafe_allow_html=True)
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
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with g2:
        st.markdown('<div class="section-header">Monte-Carlo Fair-Value Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        theme.style_axes(fig, ax)
        ax.hist(dcf_mc["samples"], bins=45, color=theme.BLUE, alpha=0.75, edgecolor="none")
        ax.axvline(dcf_mc["median"], color=theme.GREEN, lw=2, label=f"Median ${dcf_mc['median']:.0f}")
        ax.axvline(price, color=theme.RED, lw=2, ls="--", label=f"Market ${price:.0f}")
        ax.set_xlabel("Fair value / share ($)"); ax.set_ylabel("Frequency")
        ax.legend(facecolor=theme.BG, labelcolor=theme.TEXT, fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.caption(
        "Hybrid view: the ML model and traditional DCF are independent estimates. "
        "Agreement between them strengthens the valuation thesis; divergence flags "
        "assumptions worth revisiting."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PEERS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">🏦 Peer Group Benchmarking</div>', unsafe_allow_html=True)
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
    st.dataframe(styled, use_container_width=True, hide_index=True)

    summ = peers_mod.peer_summary(pdf)
    st.markdown('<div class="section-header">Relative Positioning vs Peer Median</div>', unsafe_allow_html=True)
    metric_cols = st.columns(len(summ) or 1)
    for i, (metric, d) in enumerate(summ.items()):
        with metric_cols[i]:
            if d["subject"] is None or d["premium_pct"] is None:
                st.metric(metric, "N/A")
            else:
                st.metric(
                    metric, f"{d['subject']:.2f}",
                    delta=f"{d['premium_pct']:+.0f}% vs peers",
                    delta_color="inverse" if metric in ("EV/EBITDA", "P/E", "Debt/Equity", "P/S") else "normal",
                )

    # Peer bar chart for EV/EBITDA & P/E.
    fig, ax = plt.subplots(figsize=(9, 3.2))
    theme.style_axes(fig, ax)
    plot_df = pdf.dropna(subset=["P/E"]).head(8)
    x = np.arange(len(plot_df))
    ax.bar(x - 0.2, plot_df["P/E"], width=0.4, color=theme.TEAL, label="P/E")
    ax.bar(x + 0.2, plot_df["EV/EBITDA"], width=0.4, color=theme.GREEN, label="EV/EBITDA")
    ax.set_xticks(x); ax.set_xticklabels(plot_df["Ticker"], fontsize=8)
    ax.legend(facecolor=theme.BG, labelcolor=theme.TEXT, fontsize=8)
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.caption(f"Peers auto-selected for **{tk}** (sector: {raw.get('sector') or 'n/a'}). "
               "Multiples marked *estimated* when live data was unavailable.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">📉 Historical Backtesting Engine</div>', unsafe_allow_html=True)
    st.caption(
        "How the valuation signal would have performed against actual price moves over "
        "1-, 3- and 5-year forward horizons (monthly rebalances). "
        f"Price history source: **{md.source}**."
    )

    results = bt_mod.run_backtest(md.history)
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

        curve = bt_mod.equity_curve(md.history)
        if curve is not None:
            st.markdown('<div class="section-header">Strategy vs Buy & Hold (growth of $1)</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(9, 3.6))
            theme.style_axes(fig, ax)
            ax.plot(curve.index, curve["Buy & Hold"], color=theme.MUTED, lw=1.5, label="Buy & Hold")
            ax.plot(curve.index, curve["Valuation Strategy"], color=theme.GREEN, lw=1.8,
                    label="Valuation Strategy (long when undervalued)")
            ax.set_ylabel("Growth of $1")
            ax.legend(facecolor=theme.BG, labelcolor=theme.TEXT, fontsize=8.5)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.caption(
            "Note: point-in-time fundamentals aren't stored, so the backtest uses a "
            "mean-reversion intrinsic-value proxy (trailing average) to evaluate signal "
            "quality — a conservative stand-in for the full ML signal history."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">📄 One-Click Equity Research Report</div>', unsafe_allow_html=True)
    st.caption(
        "Generate a formatted PDF summary: ticker breakdown, valuation range, signal, "
        "SHAP feature impacts, DCF and peer multiples."
    )

    include_dcf = st.checkbox("Include DCF section", value=True)
    include_peers = st.checkbox("Include peer benchmarking", value=True)

    if st.button("🧾 Generate Report", type="primary"):
        with st.spinner("Building report..."):
            dcf_payload = None
            if include_dcf:
                _d = dcf_mod.run_dcf(dcf_mod.DCFInputs(
                    fcf0=float(raw.get("free_cash_flow") or raw.get("op_cash_flow") or 1e9),
                    shares=float(raw.get("shares") or 1e9),
                    net_debt=float(raw.get("long_term_debt") or 0.0),
                ))
                dcf_payload = {"intrinsic_per_share": _d.intrinsic_per_share}

            peers_payload = None
            if include_peers:
                peers_payload = peers_mod.benchmark(
                    tk, sector=raw.get("sector"),
                    self_metrics={"name": raw.get("name", tk), "pe": feats.get("pe_ratio"),
                                  "debt_to_equity": feats.get("debt_to_equity"),
                                  "profit_margin": feats.get("profit_margin"),
                                  "market_cap": raw.get("market_cap"), "source": "SEC"},
                )

            pdf_bytes = reports.build_report(
                ticker=tk,
                company=raw.get("name", tk),
                market_price=price,
                valuation={"point": result.point, "low": result.low,
                           "median": result.median, "high": result.high},
                signal_text=signal_text,
                attribution=attribution.as_frame(),
                dcf=dcf_payload,
                peers=peers_payload,
                macro={"cpi": macro_now.cpi, "fed_funds": macro_now.fed_funds,
                       "yield_curve": macro_now.yield_curve_10y_2y},
            )

        ext = "pdf" if reports.has_reportlab() else "txt"
        mime = "application/pdf" if ext == "pdf" else "text/plain"
        st.success(f"Report ready ({len(pdf_bytes):,} bytes).")
        st.download_button(
            f"⬇️ Download {tk} Equity Report (.{ext})",
            data=pdf_bytes,
            file_name=f"{tk}_equity_report.{ext}",
            mime=mime,
            type="primary",
        )
        if ext == "txt":
            st.caption("ReportLab not installed — generated a plain-text report instead of PDF.")
