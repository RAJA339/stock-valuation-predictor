# Model provenance and the P0-1 investigation

This document answers the question the external review asked first: **what is
the training target?** It is written before any fix, because the answer
changes which fixes are even coherent.

## Finding: the model was trained on synthetic data

`svp/models/valuation.py::_synthetic_dataset` generated the entire training
set. Every feature was `rng.uniform(...)` — random numbers with no connection
to any company — and the label was a hand-written linear formula of those
same randoms:

```python
base = (mkt                      # market price, coefficient 1.0
        + fcf_y * 800 - (pe - 15) * 2 + roe * 300 + pm * 200
        - de * 20 + rev_yoy * 120 + ... )
intrinsic = base + rng.normal(0, noise_scale)
```

So the answer to the review's question is none of the three it anticipated.
The target was not a forward price, not an analyst target, and not the app's
own DCF. It was **a formula written by hand, evaluated on random inputs.**

The consequences follow directly:

1. **`market_price` was not leaking into the target — it *was* the target's
   base term**, at coefficient exactly 1.0, contributing 75% of the label's
   mean. "Intrinsic value" was market price plus a hand-picked adjustment.
2. **The model is an emulator of that formula.** A reported R² near 0.9
   measured how well gradient boosting recovers a linear formula from its own
   noise. It said nothing about valuing companies.
3. **`train_medians`**, used to impute missing fundamentals at inference, were
   medians of uniform random draws rather than of real companies.

### The arithmetic confirms it

The review observed a SHAP base value of **$341.50** and a `Market Price`
contribution of **−112.76** for MRNA at $145.13. Both fall out of the formula
without needing to run anything:

| Quantity | Predicted from the formula | Observed in the app |
|---|---|---|
| Mean of the target | $339.00 (analytic, summing each term's mean) | $341.50 |
| SHAP for `market_price` at $145.13 | 145.13 − 255.00 = **−109.87** | **−112.76** |

`255.00` is the mean of `rng.uniform(10, 500)`. The match is the proof.

### Why the extreme signals appeared

Inference feeds **real** fundamentals to a model trained on uniform randoms,
so real companies routinely fall outside the training domain. Trees cannot
extrapolate; they clamp to the nearest learned leaf.

| | Features outside the training range | Result |
|---|---|---|
| MRNA | 9 of 16 (`profit_margin` −0.90 vs a −0.20 floor, `fcf_yield` −0.18 vs −0.05, …) | Clamped to the extreme low leaves |
| KO | 1 of 16 (`roe` 0.42 vs a 0.40 ceiling) | Pulled toward the $341 training mean, from a $91 price |

The −96.9% and +139.0% signals in the review are this effect, not a view about
either company.

## What this means for the review's plan

The review's P0-1 prescribes feature surgery — drop `market_price`, replace
`fcf_yield` and `pe_ratio` — then retrain and report the honest drop in
accuracy. **That plan does not apply to this code.** Removing a term from a
formula you also wrote does not produce a valuation model; it produces a
different formula. There were no real companies in the training set to retrain
on, so "before/after held-out metrics" would compare two synthetic fits.

The fix is therefore not surgery on the features. It is a real training set,
which is what `svp/models/dataset.py` and `scripts/` now build.

## Feature provenance

Required by P0-1 acceptance criterion 2. `price_derived` asks only: does
computing this quantity require a market price, market cap, enterprise value,
or a trading multiple?

| Feature | Definition | Price-derived | Justification |
|---|---|---|---|
| `fcf_to_revenue` | free cash flow ÷ revenue | **no** | Both from the cash-flow and income statements. |
| `fcf_to_assets` | free cash flow ÷ total assets | **no** | Cash-flow statement over balance sheet. |
| `fcf_to_invested_capital` | FCF ÷ (equity + long-term debt) | **no** | Book capital, not market capital. |
| `earnings_to_assets` | net income ÷ total assets | **no** | Replaces `pe_ratio`; ROA by another name. |
| `roe` | net income ÷ shareholders' equity | **no** | **Book** equity from the balance sheet. |
| `roa` | net income ÷ total assets | **no** | Both from filings. |
| `debt_to_equity` | long-term debt ÷ **book** equity | **no** | Book equity, not market cap. |
| `profit_margin` | net income ÷ revenue | **no** | Income statement only. |
| `asset_turnover` | revenue ÷ total assets | **no** | Income statement over balance sheet. |
| `gross_margin` | gross profit ÷ revenue | **no** | Income statement only. |
| `revenue_yoy`, `revenue_qoq` | growth in reported revenue | **no** | Filing-to-filing deltas. |
| `net_income_yoy`, `fcf_yoy` | growth in reported earnings / FCF | **no** | Filing-to-filing deltas. |
| `interest_coverage` | EBIT ÷ interest expense | **no** | Income statement only. |
| `current_ratio` | current assets ÷ current liabilities | **no** | Balance sheet only. |
| `accruals` | (net income − operating cash flow) ÷ assets | **no** | Sloan's ratio; all three from filings. |
| `sentiment` | FinBERT tone of an earnings transcript | **no** | Language, not price. |
| `log_assets` | log of total assets | **no** | A **book** size control, deliberately replacing market cap. |
| `cpi`, `fed_funds`, `yield_curve` | FRED macro levels | **no** | Economy-wide; carries no issuer's price. |

Removed as price-derived: `market_price` (price itself), `fcf_yield`
(÷ market cap), `pe_ratio` (price ÷ earnings).

Beta and realised volatility are named by the review as second-order leakage.
Neither is used, and neither is added.

## The target for the real model

There is no observable ground truth for "intrinsic value", so inventing one is
how the previous model went wrong. The observable quantity is **what the market
pays for a given fundamental profile**, so that is the target:

```
y = log(enterprise value / revenue)          # scale-free by construction
```

The model learns the multiple that companies with a given set of fundamentals
command. Predicting it back yields a **peer-implied** value, and the signal is
the gap between that and the actual multiple.

This is relative valuation and must be labelled as such: it answers *"does this
trade rich or cheap against companies with similar fundamentals?"* and not
*"what is this business worth?"* If the whole market is expensive, so is this
model's output. The UI says so.

The features contain no price. The target does, by construction — that is what
makes it observable, and it is the entire distinction between a market-multiple
model and the circular one it replaces.
