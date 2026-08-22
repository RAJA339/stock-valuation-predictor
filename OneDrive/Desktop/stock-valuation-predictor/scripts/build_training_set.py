#!/usr/bin/env python3
"""
Build the real training set, then train and calibrate the relative-value model.

Run this where SEC EDGAR is reachable — it is blocked from some cloud and
sandbox egress proxies, which is why the fetch lives in a script you run
rather than in a build step that would fail silently and leave the old
synthetic model in place.

    python scripts/build_training_set.py --limit 300
    python scripts/build_training_set.py --tickers AAPL,MSFT,KO,MRNA
    python scripts/build_training_set.py --limit 300 --skip-fetch   # reuse cache

Outputs, all under data/:
    training_set.csv      the assembled rows
    model_metrics.json    held-out R², MAE, quantile coverage
    calibration.json      median |signal| and the rest of the P0-2 numbers

Nothing here writes into the app's runtime path. A model only becomes the
app's model once its metrics are inspected and it is promoted deliberately —
a half-migrated model in the repo is the failure mode the review warned about.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from svp.models import dataset as DS                            # noqa: E402
from svp.models import relative_value as RV                     # noqa: E402

DATA_DIR = os.path.join(_ROOT, "data")

#: A spread of sectors and market-cap deciles. P0-2 asks for 200–300 names;
#: this is the seed list and --limit trims it. Deliberately mixes profitable
#: mega-caps with cash-burning biotech, because a model that only sees healthy
#: companies learns nothing about the ones the app most needs to get right.
DEFAULT_UNIVERSE = """
AAPL MSFT GOOGL AMZN META NVDA TSLA AVGO ORCL CRM ADBE AMD INTC CSCO QCOM
TXN IBM NOW INTU AMAT MU ADI LRCX KLAC SNPS CDNS PANW CRWD DDOG SNOW NET
ZS OKTA TEAM WDAY VEEV HUBS ZM DOCU TWLO SHOP SQ PYPL COIN ABNB UBER LYFT
DASH RBLX U PLTR SNAP PINS SPOT ROKU
JPM BAC WFC GS MS C SCHW BLK SPGI CME ICE AXP V MA COF USB PNC TFC BK
JNJ PFE MRK ABBV LLY BMY AMGN GILD BIIB VRTX REGN MRNA BNTX ILMN ISRG
SYK BSX MDT ABT TMO DHR A ZTS CI CVS UNH ELV HUM CNC
XOM CVX COP SLB EOG PSX VLO MPC OXY HAL DVN FANG HES KMI WMB
PG KO PEP WMT COST TGT HD LOW MCD SBUX NKE DIS CMCSA VZ T TMUS
CAT DE HON GE MMM BA LMT RTX NOC GD UPS FDX UNP CSX NSC
LIN APD SHW ECL NEM FCX DOW DD PPG NUE
NEE DUK SO D AEP EXC XEL ED WEC ES
AMT PLD CCI EQIX SPG O PSA WELL DLR
F GM RIVN LCID NIO
"""


def _default_fetcher():
    """
    Build a fetcher over the app's own SEC and market plumbing.

    Reuses ``svp.data.sec`` and ``svp.data.market`` rather than re-implementing
    EDGAR access, so the training set is assembled from exactly the same
    parsing the app uses at inference. A mismatch between the two is a
    classic and invisible source of train/serve skew.
    """
    from svp.data import market as market_mod, sec as sec_mod

    def fetch(ticker: str):
        cik = sec_mod.get_cik(ticker)
        if not cik:
            return None
        facts = sec_mod.get_financials(cik)
        if not facts:
            return None

        def cur(names):
            return sec_mod.concept_first(
                facts, names if isinstance(names, list) else [names])

        def prior(names):
            names = names if isinstance(names, list) else [names]
            hist = [e for e in sec_mod.history_first(facts, names)
                    if e.get("form") == "10-K"]
            return hist[-2]["val"] if len(hist) >= 2 else None

        rev_tags = ["Revenues",
                    "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet"]
        ocf = cur(["NetCashProvidedByUsedInOperatingActivities"])
        capex = cur(["PaymentsToAcquirePropertyPlantAndEquipment"]) or 0.0
        ocf_p = prior(["NetCashProvidedByUsedInOperatingActivities"])

        md = market_mod.get_market_data(ticker)
        return {
            "revenue": cur(rev_tags),
            "revenue_prior": prior(rev_tags),
            "net_income": cur(["NetIncomeLoss"]),
            "net_income_prior": prior(["NetIncomeLoss"]),
            "assets": cur(["Assets"]),
            "equity": cur(["StockholdersEquity"]),
            "long_term_debt": cur(["LongTermDebt", "LongTermDebtNoncurrent"]),
            "op_cash_flow": ocf,
            "capex": capex,
            "free_cash_flow": (ocf - capex) if ocf is not None else None,
            "fcf_prior": (ocf_p - capex) if ocf_p is not None else None,
            "gross_profit": cur(["GrossProfit"]),
            "ebit": cur(["OperatingIncomeLoss"]),
            "interest_expense": cur(["InterestExpense"]),
            "current_assets": cur(["AssetsCurrent"]),
            "current_liabilities": cur(["LiabilitiesCurrent"]),
            "cash": cur(["CashAndCashEquivalentsAtCarryingValue"]),
            "total_debt": cur(["LongTermDebt", "LongTermDebtNoncurrent"]),
            "market_cap": md.market_cap,
            "shares": md.shares_outstanding,
            "as_of": "",
        }
    return fetch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--tickers", type=str, default="")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="reuse data/training_set.csv instead of refetching")
    ap.add_argument("--out", type=str, default=DATA_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "training_set.csv")

    if args.skip_fetch and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Reusing {csv_path}: {len(df)} rows")
    else:
        universe = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
                    if args.tickers else DEFAULT_UNIVERSE.split())
        universe = universe[:args.limit]
        print(f"Fetching fundamentals for {len(universe)} tickers "
              "(SEC asks for a modest request rate; this is throttled)...")

        t0 = time.time()

        def progress(i, n, tkr):
            if i % 10 == 0 or i == n:
                print(f"  [{i:>4}/{n}] {tkr:<6} "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)

        df, report = DS.build(universe, _default_fetcher(), progress=progress)
        print("\n" + report.note())
        if df.empty:
            print("\nNo rows built. If every ticker failed to fetch, SEC is "
                  "unreachable from this machine — that is the usual cause, "
                  "and it is a network problem rather than a code one.")
            return 1
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path} ({len(df)} rows)")

    df = DS.winsorize(df)
    model = RV.train(df)
    if model is None:
        print(f"\nOnly {len(df)} rows — too few to train on. Widen the "
              "universe; a model fitted on a handful of names would repeat "
              "the mistake this replaces.")
        return 1

    metrics = {
        "n_train": model.n_train, "n_test": model.n_test,
        "target": "log(EV/revenue)",
        "r2_log_target": round(model.r2, 4),
        "mae_log_target": round(model.mae_log, 4),
        "quantile_coverage_p10_p90": (round(model.coverage, 4)
                                      if model.coverage is not None else None),
        "nominal_coverage": 0.80,
        "features": model.feature_cols,
    }
    with open(os.path.join(args.out, "model_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n=== HELD-OUT METRICS (real data, scale-free target) ===")
    for k, v in metrics.items():
        if k != "features":
            print(f"  {k:28s} {v}")
    print(f"  {'mae in multiple terms':28s} "
          f"±{(np.exp(model.mae_log) - 1) * 100:.1f}% on EV/revenue")

    # ── Calibration sweep (P0-2) ─────────────────────────────────────────────
    signals, excluded = [], 0
    for _, row in df.iterrows():
        feats = {c: row.get(c) for c in model.feature_cols}
        rev = row.get("ev_to_revenue")
        if not rev or not np.isfinite(row.get(DS.TARGET, np.nan)):
            continue
        try:
            pred = float(model.point.predict(
                pd.DataFrame([feats])[model.feature_cols]
                .fillna(model.train_medians))[0])
        except Exception:
            continue
        implied, actual = np.exp(pred), row["ev_to_revenue"]
        signals.append((implied / actual - 1.0) * 100.0)
        try:
            lo = float(model.quantiles[RV.QUANTILES[0]].predict(
                pd.DataFrame([feats])[model.feature_cols]
                .fillna(model.train_medians))[0])
            hi = float(model.quantiles[RV.QUANTILES[-1]].predict(
                pd.DataFrame([feats])[model.feature_cols]
                .fillna(model.train_medians))[0])
            if not (np.exp(lo) <= actual <= np.exp(hi)):
                excluded += 1
        except Exception:
            pass

    summary = RV.summarise_calibration(
        signals, coverage=model.coverage,
        excluded=excluded / max(len(signals), 1), mae_log=model.mae_log)
    if summary is None:
        print("\nNo signals could be computed for calibration.")
        return 1

    with open(os.path.join(args.out, "calibration.json"), "w") as fh:
        json.dump({"n": summary.n,
                   "median_abs_signal_pct": round(summary.median_abs_signal, 2),
                   "mean_abs_signal_pct": round(summary.mean_abs_signal, 2),
                   "coverage": summary.coverage,
                   "band_excludes_market_share":
                       round(summary.band_excludes_market, 4),
                   "broken_threshold_pct": summary.THRESHOLD,
                   "is_broken": summary.is_broken}, fh, indent=2)

    print("\n=== CALIBRATION ===")
    print("  " + summary.verdict())
    if summary.is_broken:
        print("\n  Do not promote this model. A median disagreement above "
              f"{summary.THRESHOLD:.0f}% across the universe is the signature "
              "of a broken fit, not of a mispriced market.")
        return 2
    print(f"\nWrote metrics and calibration to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
