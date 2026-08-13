"""
Streamlit AppTest check (CI).

Boots app.py through Streamlit's testing harness, seeds a fully-formed analysis
(so every tab renders without needing network fundamentals), and asserts the
script runs with no uncaught exceptions. Exits non-zero on failure.
"""

import os
import sys
import warnings

# Ensure the project root (parent of ci/) is importable no matter the cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

warnings.filterwarnings("ignore")

import numpy as np
from streamlit.testing.v1 import AppTest

from svp.models import (
    valuation as V, explain as X, regime as R, sizing as SZ,
)
from svp.analytics import technical as TA
from svp.data import market as market_mod, macro as macro_mod
from svp.features import FEATURE_COLUMNS

EXPECTED_TABS = 13


def build_seed_analysis():
    vm = V.train_model()
    feats = {c: np.nan for c in FEATURE_COLUMNS}
    feats.update({
        "pe_ratio": 28.0, "roe": 0.35, "roa": 0.18, "debt_to_equity": 0.6,
        "fcf_yield": 0.04, "profit_margin": 0.25, "asset_turnover": 0.9,
        "revenue_yoy": 0.12, "revenue_qoq": 0.03, "net_income_yoy": 0.15,
        "fcf_yoy": 0.10, "sentiment": 0.4, "cpi": 314.0, "fed_funds": 4.3,
        "yield_curve": -0.1, "market_price": 190.0,
    })
    feats["_raw"] = {
        "cik": "0000320193",
        "revenue": 3.8e11, "net_income": 1.0e11, "total_assets": 3.5e11,
        "equity": 6e10, "op_cash_flow": 1.1e11, "capex": 1e10,
        "free_cash_flow": 1.0e11, "long_term_debt": 1e11, "shares": 1.5e10,
        "market_cap": 2.9e12, "eps": 6.5, "sector": "Technology",
        "name": "Apple Inc.", "sentiment_obj": None,
    }
    md = market_mod.get_market_data("AAPL", fallback_price=190.0)
    res = V.predict(feats, vm)
    attr = X.explain_prediction(feats, vm)
    sig_t, sig_c = V.valuation_signal(res.point, 190.0)

    # v3 execution / guardrail layer, mirroring app.run_analysis().
    regime = R.detect_from_market(market_mod.get_market_data("^VIX", 18.0), macro_mod.get_macro())
    technical = TA.analyze(md.history)
    ann_vol = SZ.annualized_vol(md.history["Close"]) if not md.history.empty else None
    sizing = SZ.compute_sizing(
        190.0, res.mc_samples, res.low, res.high,
        technical.atr if technical else None, ann_vol,
    )
    raw_mos = (res.point - 190.0) / 190.0
    return {
        "ticker": "AAPL", "price": 190.0, "md": md, "features": feats,
        "result": res, "attribution": attr, "signal_text": sig_t, "signal_class": sig_c,
        "regime": regime, "technical": technical, "sizing": sizing,
        "raw_mos": raw_mos, "adjusted_mos": raw_mos * regime.signal_scaler,
    }


def main():
    at = AppTest.from_file(os.path.join(_ROOT, "app.py"), default_timeout=240)
    at.session_state["analysis"] = build_seed_analysis()
    at.run()

    if at.exception:
        print("APPTEST FAILED — uncaught exceptions:")
        for e in at.exception:
            print(" ", repr(e)[:300])
        sys.exit(1)

    n_tabs = len(at.tabs)
    n_metrics = len(at.metric)
    print(f"tabs rendered: {n_tabs}")
    print(f"metrics rendered: {n_metrics}")
    print(f"errors: {[e.value[:80] for e in at.error]}")

    if n_tabs != EXPECTED_TABS:
        print(f"APPTEST FAILED — expected {EXPECTED_TABS} tabs, got {n_tabs}")
        sys.exit(1)
    if at.error:
        print("APPTEST FAILED — app emitted st.error output")
        sys.exit(1)

    print("APPTEST PASSED")


if __name__ == "__main__":
    main()
