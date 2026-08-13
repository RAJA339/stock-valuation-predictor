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

### Charting & Technical Analysis
- **TradingView charts** embedded at **5m / 10m / 15m / 30m / 1h**, with EMA,
  RSI, MACD and VWAP studies preloaded. Yahoo has no native 10-minute bar, so
  that interval is built by resampling 5m data into true 10m candles.
- **Native brokerage-style candles** — candlesticks with EMA 9/20/50, VWAP and
  Bollinger bands, a direction-coloured volume histogram, and RSI and MACD
  panels. This is the chart embedded in the PDF, since an iframe cannot be.
- **11-indicator panel** across trend (EMA stack, MACD, Supertrend, ADX),
  momentum (RSI, Stochastic), volatility (Bollinger, ATR) and volume (VWAP,
  OBV), each reduced to a Bullish / Bearish / Neutral call with a consensus.
- **Measured signal accuracy** — every indicator's call is recomputed at each
  bar and scored against the actual move N bars later, reported as a hit rate
  with a **Wilson 95% confidence interval**. An indicator is only called
  "better than chance" when that interval excludes 50%.

> **On accuracy claims.** This app does not ship an "80% accurate" indicator,
> because no such thing survives walk-forward testing on liquid US equities.
> It measures what each signal actually did on the bars in front of you and
> reports that, including when the answer is "no better than a coin flip".
> Trading costs are not modelled, so a positive directional edge can still lose
> money after spread and commission.

### Quant Execution Layer
- **Market-regime classifier** — a Hidden Markov Model over VIX and the yield
  curve labels the tape *Calm / Neutral / Stress / Crisis* and scales conviction
  on every valuation signal accordingly (falls back to a Gaussian mixture, then
  to rules, when `hmmlearn` is unavailable).
- **Technical entry filters** — 200-day SMA trend gate, RSI bullish divergence,
  volume point-of-control and ATR, so a cheap stock is only actionable once
  price action confirms.
- **Forensic guardrails** — Piotroski F-Score, Altman Z-Score and Beneish
  M-Score computed straight from SEC EDGAR facts, plus Form 4 insider flow and
  short interest.
- **Relative-return model** — a second XGBoost model trained on forward excess
  return versus a benchmark, complementing the absolute intrinsic estimate.
- **Margin-of-safety screener** — ranks a configurable universe and surfaces the
  top decile by discount to intrinsic value.
- **Position sizing** — fractional Kelly from the Monte-Carlo win probability,
  blended with volatility parity and floored by ATR / p10 stop-loss levels.
- **Filing divergence** — TF-IDF cosine similarity between consecutive 10-K and
  10-Q filings highlights language that materially changed quarter over quarter.

### Derivatives — Options & Futures
- **Live option chains** — strikes, bid/ask, open interest and implied
  volatility via `yfinance.Ticker.option_chain(date)`, with a Black-Scholes
  synthetic chain (volatility smile included) as an offline fallback.
- **Pricing engines** — analytic **Black-Scholes-Merton** with a continuous
  dividend yield, a **Cox-Ross-Rubinstein binomial tree** supporting American
  early exercise, and a **GBM Monte-Carlo** cross-check.
- **Full Greeks** — Δ, Γ, Θ (per calendar day), Vega (per vol point) and ρ (per
  1% rate), displayed beside the theoretical value.
- **Implied volatility solver** — Brent root-finder on the market premium, with
  a bisection fallback when SciPy is absent.
- **Valuation → options bridge** — treats the ML intrinsic target (or the DCF
  fair value) as the projected underlying at expiry `Sₜ`, discounts the
  resulting payoff back to today and ranks the chain to flag mispriced LEAPs.
- **Futures cost-of-carry** — fair value `F = S·e^{(r+s−c)T}` with basis, net
  carry and a contango/backwardation read against an observed futures price.

---

## 🗂️ Project structure

```
stock-valuation-predictor/
├── app.py                     # Streamlit entry — tabbed UI
├── requirements.txt
├── .streamlit/config.toml
└── svp/                       # application package
    ├── theme.py               # palette / CSS / matplotlib styling
    ├── charts.py              # brokerage-style candlestick + indicator panels
    ├── features.py            # feature engineering incl. QoQ/YoY trend metrics
    ├── reports.py             # ReportLab PDF reports
    ├── data/
    │   ├── sec.py             # SEC EDGAR (levels + history)
    │   ├── market.py          # yfinance / Alpha Vantage live data
    │   ├── macro.py           # FRED + BLS macro indicators
    │   ├── sentiment.py       # FinBERT + lexicon fallback
    │   ├── insider.py         # Form 4 insider flow + short interest
    │   ├── filings_nlp.py     # TF-IDF divergence between filings
    │   ├── options.py         # live option chains (+ synthetic fallback)
    │   ├── intraday.py        # 5m/10m/15m/30m/1h OHLCV bars
    │   └── storage.py         # SQLite / PostgreSQL persistent cache
    ├── models/
    │   ├── valuation.py       # XGBoost point + quantile + Monte-Carlo
    │   ├── explain.py         # SHAP / LIME attribution
    │   ├── dcf.py             # DCF + Monte-Carlo scenario engine
    │   ├── backtest.py        # 1/3/5-year backtesting engine
    │   ├── regime.py          # HMM market-regime classifier
    │   ├── relative.py        # forward excess-return model
    │   ├── sizing.py          # fractional Kelly + vol parity + stops
    │   └── derivatives.py     # Black-Scholes / binomial / MC / futures
    └── analytics/
        ├── peers.py           # peer-group benchmarking
        ├── technical.py       # SMA / RSI / volume POC / ATR filters
        ├── quality.py         # Piotroski F, Altman Z, Beneish M
        └── screener.py        # margin-of-safety universe ranking
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
