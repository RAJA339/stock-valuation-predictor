# 📈 Intrinsic Stock Valuation Predictor — v2.0

*Webster University — MS Business Analytics | Group ML Project*

A Streamlit application that estimates a company's **intrinsic value** from SEC
EDGAR fundamentals, live market data, real-time macro indicators and
earnings-call sentiment — then **explains** and **stress-tests** that estimate.

---

## ✨ Features

### Model Transparency & Explainable AI
- **SHAP / LIME visualizations** — a SHAP waterfall shows exactly how each
  feature (net income → ROE, total debt → D/E, macro indicators …) pushed the
  valuation up or down for a specific ticker. LIME is available as an optional
  second local explainer. *(Falls back to XGBoost `pred_contribs` if SHAP is
  absent, so a waterfall always renders.)*
- **Valuation ranges (quantile regression)** — native XGBoost quantile
  regressors (p10 / p50 / p90) plus a Monte-Carlo simulation present a realistic
  intrinsic-value **range** and confidence interval instead of a single point.

### Live Data Pipelines & Advanced Features
- **Automated market data** — `yfinance` (with an Alpha Vantage fallback)
  auto-fetches live prices, market caps, sector and historical price history.
- **Sentiment & macro indicators** — **FinBERT** sentiment on quarterly
  earnings-call transcripts (compact finance-lexicon fallback when
  `transformers`/`torch` aren't installed) and real-time **FRED** data
  (yield curve, CPI, interest rates), with BLS as a backup.
- **Trend & delta features** — QoQ and YoY growth-velocity of revenue, net
  income and FCF are fed to the model instead of static balance-sheet totals.

### Scenario Modeling & Financial Analytics
- **Hybrid ML + traditional DCF** — a comparative tab with an interactive DCF
  (Streamlit sliders for WACC and terminal growth) and a Monte-Carlo fair-value
  distribution shown alongside the ML valuation.
- **Peer group benchmarking** — auto-selected industry competitors compared
  across EV/EBITDA, P/E, Debt-to-Equity, P/S and profit margin.

### Product & UX Upgrades
- **Historical backtesting engine** — a dashboard showing how the valuation
  signal performed against actual price moves over 1-, 3- and 5-year horizons,
  with a strategy-vs-buy-&-hold equity curve.
- **Automated PDF equity reports** — one-click **ReportLab** PDF with the ticker
  breakdown, feature impacts and valuation signals (plain-text fallback).
- **Data caching & storage** — a database backend (`@st.cache_data` in memory +
  a persistent **SQLite** store, upgradeable to **PostgreSQL** via
  `DATABASE_URL`) speeds up repeat lookups of parsed SEC filings.

---

## 🗂️ Project structure

```
stock-valuation-predictor/
├── app.py                     # Streamlit entry — tabbed UI
├── requirements.txt
├── .streamlit/config.toml
└── svp/                       # application package
    ├── theme.py               # palette / CSS / matplotlib styling
    ├── features.py            # feature engineering incl. QoQ/YoY trend metrics
    ├── reports.py             # ReportLab PDF reports
    ├── data/
    │   ├── sec.py             # SEC EDGAR (levels + history)
    │   ├── market.py          # yfinance / Alpha Vantage live data
    │   ├── macro.py           # FRED + BLS macro indicators
    │   ├── sentiment.py       # FinBERT + lexicon fallback
    │   └── storage.py         # SQLite / PostgreSQL persistent cache
    ├── models/
    │   ├── valuation.py       # XGBoost point + quantile + Monte-Carlo
    │   ├── explain.py         # SHAP / LIME attribution
    │   ├── dcf.py             # DCF + Monte-Carlo scenario engine
    │   └── backtest.py        # 1/3/5-year backtesting engine
    └── analytics/
        └── peers.py           # peer-group benchmarking
```

---

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔑 Optional API keys / config (environment variables)

| Variable | Enables |
|---|---|
| `FRED_API_KEY` | Real-time FRED macro (yield curve, CPI, rates) |
| `ALPHAVANTAGE_API_KEY` | Alpha Vantage market-data fallback |
| `DATABASE_URL` | Persistent PostgreSQL cache (else SQLite) |

Everything works **without** keys — the app degrades gracefully to public
endpoints and deterministic offline fallbacks so no feature ever hard-crashes.

> ⚠️ For educational purposes only. Not financial advice.
