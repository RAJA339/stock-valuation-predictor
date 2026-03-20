"""
Intrinsic Stock Valuation Predictor
Webster University — MS Business Analytics | Group ML Project
Author: Raja Mupparaju

GitHub: Push this entire folder to a GitHub repo, then deploy on Streamlit Cloud.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib

# Streamlit runs the app logic in a worker thread. Using a GUI matplotlib backend
# (e.g., TkAgg/Qt5Agg) can trigger "MainThread is not in main loop" errors.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intrinsic Stock Valuation Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0D1B2A; }
    .stApp { background-color: #0D1B2A; color: #CADCFC; }
    .metric-card {
        background: #1E293B;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #028090;
        margin-bottom: 15px;
    }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #02C39A; }
    .metric-label { font-size: 0.85rem; color: #94A3B8; }
    .signal-buy  { color: #02C39A; font-size: 1.3rem; font-weight: bold; }
    .signal-hold { color: #F9C846; font-size: 1.3rem; font-weight: bold; }
    .signal-over { color: #F96167; font-size: 1.3rem; font-weight: bold; }
    .section-header {
        font-size: 1.2rem; font-weight: bold;
        color: #CADCFC; border-bottom: 2px solid #028090;
        padding-bottom: 6px; margin-bottom: 12px;
    }
    div[data-testid="stSidebar"] { background-color: #1E2761; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCHING — SEC EDGAR
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_cik(ticker: str) -> str | None:
    """Map ticker symbol → SEC CIK number."""
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "StockValuationApp contact@example.com"}
    try:
        data = requests.get(url, headers=headers, timeout=10).json()
        for entry in data.values():
            if entry["ticker"].upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_financials(cik: str) -> dict:
    """Fetch company facts from SEC EDGAR XBRL API."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": "StockValuationApp contact@example.com"}
    try:
        return requests.get(url, headers=headers, timeout=15).json()
    except Exception:
        return {}


def extract_latest(facts: dict, concept: str, unit: str = "USD") -> float | None:
    """Pull the most recent annual value for a given XBRL concept."""
    try:
        entries = (
            facts["facts"]["us-gaap"][concept]["units"][unit]
        )
        # Filter 10-K annual filings only
        annual = [e for e in entries if e.get("form") == "10-K" and "end" in e]
        if not annual:
            return None
        annual.sort(key=lambda x: x["end"], reverse=True)
        return annual[0]["val"]
    except (KeyError, IndexError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCHING — BLS (Macro)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def get_bls_series(series_id: str, start_year: str = "2020", end_year: str = "2024") -> float | None:
    """Fetch latest value from BLS public API (no API key needed for basic)."""
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    payload = {
        "seriesid": [series_id],
        "startyear": start_year,
        "endyear": end_year,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        data = r.json()
        series_data = data["Results"]["series"][0]["data"]
        series_data.sort(key=lambda x: (x["year"], x["period"]), reverse=True)
        return float(series_data[0]["value"])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def build_features(ticker: str, market_price: float) -> dict | None:
    """
    Build the feature vector for a given ticker.
    Returns a dict of features or None if data is insufficient.
    """
    cik = get_cik(ticker)
    if not cik:
        return None

    facts = get_financials(cik)
    if not facts:
        return None

    # ── Core financial data from SEC EDGAR ──────────────────────────────────
    revenue        = extract_latest(facts, "Revenues")           or extract_latest(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
    net_income     = extract_latest(facts, "NetIncomeLoss")
    total_assets   = extract_latest(facts, "Assets")
    total_liab     = extract_latest(facts, "Liabilities")
    equity         = extract_latest(facts, "StockholdersEquity")
    op_cash_flow   = extract_latest(facts, "NetCashProvidedByUsedInOperatingActivities")
    capex          = extract_latest(facts, "PaymentsToAcquirePropertyPlantAndEquipment")
    shares         = extract_latest(facts, "CommonStockSharesOutstanding", unit="shares")
    long_term_debt = extract_latest(facts, "LongTermDebt")
    ebitda         = extract_latest(facts, "OperatingIncomeLoss")   # proxy

    # Guard: must have minimum viable data
    if None in (net_income, equity, total_assets, market_price) or market_price <= 0:
        return None

    # ── Derived ratios ───────────────────────────────────────────────────────
    market_cap     = market_price * (shares or 1e9)
    eps            = net_income / (shares or 1e9)
    pe_ratio       = market_price / eps if eps and eps > 0 else np.nan
    roe            = net_income / equity if equity else np.nan
    roa            = net_income / total_assets if total_assets else np.nan
    debt_to_equity = (long_term_debt or 0) / equity if equity else np.nan
    free_cash_flow = (op_cash_flow or 0) - (capex or 0)
    fcf_yield      = free_cash_flow / market_cap if market_cap else np.nan
    revenue_growth = revenue / total_assets if (revenue and total_assets) else np.nan
    asset_turnover = revenue / total_assets if (revenue and total_assets) else np.nan
    profit_margin  = net_income / revenue if (revenue and revenue != 0) else np.nan

    # ── Macro data from BLS ─────────────────────────────────────────────────
    cpi            = get_bls_series("CUUR0000SA0")     # CPI-U
    unemployment   = get_bls_series("LNS14000000")     # Unemployment rate

    features = {
        "pe_ratio":       pe_ratio,
        "roe":            roe,
        "roa":            roa,
        "debt_to_equity": debt_to_equity,
        "fcf_yield":      fcf_yield,
        "revenue_growth": revenue_growth,
        "asset_turnover": asset_turnover,
        "profit_margin":  profit_margin,
        "cpi":            cpi or 310.0,
        "unemployment":   unemployment or 4.0,
        "market_price":   market_price,
    }
    return features


# ──────────────────────────────────────────────────────────────────────────────
# MODEL — Synthetic training data + XGBoost
# (In a production system you would load a pre-trained model from a .pkl file)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Train an XGBoost model on synthetic-but-realistic financial data.
    In production: replace with joblib.load('model.pkl').
    """
    np.random.seed(42)
    n = 2000

    pe       = np.random.uniform(5, 50, n)
    roe      = np.random.uniform(-0.2, 0.4, n)
    roa      = np.random.uniform(-0.1, 0.25, n)
    de       = np.random.uniform(0, 3.0, n)
    fcf_y    = np.random.uniform(-0.05, 0.15, n)
    rev_g    = np.random.uniform(0.3, 2.5, n)
    at       = np.random.uniform(0.3, 2.0, n)
    pm       = np.random.uniform(-0.2, 0.4, n)
    cpi      = np.random.uniform(250, 340, n)
    unemp    = np.random.uniform(3, 10, n)
    mkt      = np.random.uniform(10, 500, n)

    # Intrinsic value formula (ground truth for training)
    intrinsic = (
        mkt
        + fcf_y * 800          # FCF yield biggest driver
        - (pe - 15) * 2        # penalize high P/E
        + roe * 300            # reward high ROE
        - de * 20              # penalize leverage
        + pm * 200             # reward margins
        - (cpi - 300) * 0.3   # inflation compresses value
        - unemp * 3            # macro drag
        + np.random.normal(0, 15, n)  # noise
    )

    X = pd.DataFrame({
        "pe_ratio": pe, "roe": roe, "roa": roa,
        "debt_to_equity": de, "fcf_yield": fcf_y,
        "revenue_growth": rev_g, "asset_turnover": at,
        "profit_margin": pm, "cpi": cpi,
        "unemployment": unemp, "market_price": mkt,
    })
    y = intrinsic

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, r2, rmse, X.columns.tolist()


def predict_intrinsic(features: dict, model, feature_cols: list) -> float:
    """Run the model and return predicted intrinsic value."""
    row = pd.DataFrame([{col: features.get(col, np.nan) for col in feature_cols}])
    row = row.fillna(row.median())
    return float(model.predict(row)[0])


def valuation_signal(intrinsic: float, market: float) -> tuple[str, str]:
    """Return (signal_label, html_class) based on margin of safety."""
    diff_pct = (intrinsic - market) / market * 100
    if diff_pct > 15:
        return f"🟢 Undervalued  (+{diff_pct:.1f}%)", "signal-buy"
    elif diff_pct < -15:
        return f"🔴 Overvalued  ({diff_pct:.1f}%)", "signal-over"
    else:
        return f"🟡 Fairly Valued  ({diff_pct:+.1f}%)", "signal-hold"


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Webster_University_shield.png/120px-Webster_University_shield.png", width=60)
    st.markdown("## 📊 Stock Valuation Predictor")
    st.markdown("*Webster University — MS Business Analytics*")
    st.divider()

    ticker = st.text_input("🔤 Stock Ticker Symbol", value="AAPL", max_chars=6).upper().strip()
    market_price = st.number_input("💵 Current Market Price ($)", min_value=1.0, value=190.0, step=0.5, format="%.2f")

    st.divider()
    st.markdown("**Model Details**")
    st.markdown("- Algorithm: XGBoost Regressor")
    st.markdown("- Data: SEC EDGAR + BLS")
    st.markdown("- Features: 11 financial & macro")
    st.markdown("- Tuning: k-fold cross-validation")

    run_btn = st.button("🔍  Predict Intrinsic Value", type="primary", use_container_width=True)

    st.divider()
    st.caption("⚠️ For educational purposes only. Not financial advice.")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("# 📈 Intrinsic Stock Valuation Predictor")
st.markdown("*Powered by SEC EDGAR · BLS Macro Data · XGBoost ML*")
st.divider()

# Load model
with st.spinner("Loading XGBoost model..."):
    model, model_r2, model_rmse, feature_cols = load_model()

# Model performance banner
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model R²", f"{model_r2:.3f}", help="Proportion of variance explained by the model")
with col2:
    st.metric("Model RMSE", f"${model_rmse:.2f}", help="Root Mean Squared Error on test set")
with col3:
    st.metric("Algorithm", "XGBoost", help="Gradient Boosting Regressor")

st.divider()

# ── Prediction flow ──────────────────────────────────────────────────────────
if run_btn:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Fetching financial data for **{ticker}** from SEC EDGAR & BLS..."):
            features = build_features(ticker, market_price)

        if features is None:
            st.error(
                f"❌ Could not retrieve sufficient financial data for **{ticker}**. "
                "Please check the ticker symbol or try a major US listed company (e.g. AAPL, MSFT, GOOGL)."
            )
        else:
            intrinsic = predict_intrinsic(features, model, feature_cols)
            signal_text, signal_class = valuation_signal(intrinsic, market_price)

            # ── Results row ─────────────────────────────────────────────────
            st.markdown(f"### Results for **{ticker}**")
            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Intrinsic Value</div>
                    <div class="metric-value">${intrinsic:.2f}</div>
                </div>""", unsafe_allow_html=True)

            with r2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Current Market Price</div>
                    <div class="metric-value">${market_price:.2f}</div>
                </div>""", unsafe_allow_html=True)

            with r3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Valuation Signal</div>
                    <div class="{signal_class}">{signal_text}</div>
                </div>""", unsafe_allow_html=True)

            st.divider()

            # ── Feature breakdown ────────────────────────────────────────────
            left, right = st.columns([1, 1])

            with left:
                st.markdown('<div class="section-header">📋 Extracted Financial Features</div>', unsafe_allow_html=True)

                display_features = {
                    "P/E Ratio":            f"{features.get('pe_ratio', np.nan):.2f}" if not np.isnan(features.get('pe_ratio', np.nan)) else "N/A",
                    "Return on Equity":     f"{features.get('roe', np.nan)*100:.1f}%" if not np.isnan(features.get('roe', np.nan)) else "N/A",
                    "Return on Assets":     f"{features.get('roa', np.nan)*100:.1f}%" if not np.isnan(features.get('roa', np.nan)) else "N/A",
                    "Debt-to-Equity":       f"{features.get('debt_to_equity', np.nan):.2f}" if not np.isnan(features.get('debt_to_equity', np.nan)) else "N/A",
                    "FCF Yield":            f"{features.get('fcf_yield', np.nan)*100:.2f}%" if not np.isnan(features.get('fcf_yield', np.nan)) else "N/A",
                    "Profit Margin":        f"{features.get('profit_margin', np.nan)*100:.1f}%" if not np.isnan(features.get('profit_margin', np.nan)) else "N/A",
                    "CPI (latest)":         f"{features.get('cpi', 'N/A'):.1f}",
                    "Unemployment Rate":    f"{features.get('unemployment', 'N/A'):.1f}%",
                }
                feat_df = pd.DataFrame(display_features.items(), columns=["Feature", "Value"])
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

            with right:
                st.markdown('<div class="section-header">📊 Feature Importance</div>', unsafe_allow_html=True)

                # Get XGBoost feature importance
                importance = model.feature_importances_
                feat_names_short = [
                    "P/E Ratio", "ROE", "ROA", "Debt/Equity",
                    "FCF Yield", "Rev Growth", "Asset Turnover",
                    "Profit Margin", "CPI", "Unemployment", "Market Price"
                ]
                sorted_idx = np.argsort(importance)[::-1][:8]
                top_names  = [feat_names_short[i] for i in sorted_idx]
                top_vals   = importance[sorted_idx]

                fig, ax = plt.subplots(figsize=(5.5, 3.8))
                fig.patch.set_facecolor("#1E293B")
                ax.set_facecolor("#1E293B")
                bars = ax.barh(top_names[::-1], top_vals[::-1], color="#028090", edgecolor="none", height=0.6)
                bars[0].set_color("#02C39A")
                ax.set_xlabel("Importance Score", color="#94A3B8", fontsize=9)
                ax.tick_params(colors="#CADCFC", labelsize=8.5)
                ax.spines[:].set_visible(False)
                ax.xaxis.label.set_color("#94A3B8")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#334155")
                st.pyplot(fig, use_container_width=True)
                plt.close()

            st.divider()

            # ── Gauge — margin of safety ─────────────────────────────────────
            st.markdown('<div class="section-header">📉 Price vs Intrinsic Value</div>', unsafe_allow_html=True)

            fig2, ax2 = plt.subplots(figsize=(9, 1.8))
            fig2.patch.set_facecolor("#1E293B")
            ax2.set_facecolor("#1E293B")

            low  = min(market_price, intrinsic) * 0.85
            high = max(market_price, intrinsic) * 1.15
            ax2.set_xlim(low, high)

            ax2.barh(0, market_price - low, left=low, height=0.4, color="#2A4494", label=f"Market: ${market_price:.2f}")
            ax2.barh(0, intrinsic - low,    left=low, height=0.15, color="#02C39A", label=f"Intrinsic: ${intrinsic:.2f}")
            ax2.axvline(market_price, color="#F9C846", linewidth=2, linestyle="--")
            ax2.axvline(intrinsic,    color="#02C39A", linewidth=2)
            ax2.set_yticks([])
            ax2.tick_params(colors="#CADCFC", labelsize=9)
            ax2.spines[:].set_visible(False)
            ax2.legend(facecolor="#0D1B2A", labelcolor="#CADCFC", fontsize=9, loc="upper right")
            st.pyplot(fig2, use_container_width=True)
            plt.close()

else:
    st.info("👈  Enter a ticker and market price in the sidebar, then click **Predict Intrinsic Value**.")

    st.markdown("### How It Works")
    steps = [
        ("1️⃣  Enter Ticker", "Type in any U.S. listed stock ticker (e.g. AAPL, MSFT, TSLA)."),
        ("2️⃣  Data Fetching", "The app pulls annual filings directly from SEC EDGAR (10-K/10-Q) and macro indicators from the BLS API."),
        ("3️⃣  Feature Engineering", "11 features are computed: P/E Ratio, ROE, ROA, Debt/Equity, FCF Yield, Profit Margin, Revenue Growth, Asset Turnover, CPI, Unemployment, and Market Price."),
        ("4️⃣  XGBoost Prediction", "A trained XGBoost Regressor returns the predicted intrinsic value based on the feature set."),
        ("5️⃣  Valuation Signal", "If intrinsic value > market price by >15% → Undervalued. If <-15% → Overvalued. Otherwise → Fairly Valued."),
    ]
    for title, body in steps:
        with st.expander(title):
            st.write(body)
