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

EXPECTED_TABS = 14


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


def _run(seed):
    at = AppTest.from_file(os.path.join(_ROOT, "app.py"), default_timeout=240)
    at.session_state["analysis"] = seed
    at.run()
    return at


def check_healthy(seed):
    """Baseline: the whole app renders cleanly."""
    at = _run(seed)

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
    return n_metrics


def check_tab_isolation(seed, baseline_metrics):
    """
    A failure in one tab must not take down the others.

    st.tabs is not lazy — every body runs in a single pass — so before
    tab_guard() an exception in Peers (tab 4) aborted the script and left
    Backtesting through Report blank. This drives that exact case: Peers is
    forced to raise, and the rest of the app must still come up.
    """
    from svp.analytics import peers as P

    original = P.benchmark

    def _boom(*a, **k):
        raise ValueError("synthetic peer-feed failure")

    P.benchmark = _boom
    try:
        at = _run(seed)
    finally:
        P.benchmark = original

    if at.exception:
        print("APPTEST FAILED — a failing tab escaped tab_guard():")
        for e in at.exception:
            print(" ", repr(e)[:300])
        sys.exit(1)

    failures = at.session_state["tab_failures"]
    print(f"isolation: tabs={len(at.tabs)} metrics={len(at.metric)} failed={failures}")

    if list(failures) != ["Peers"]:
        print(f"APPTEST FAILED — expected only Peers to fail, got {list(failures)}")
        sys.exit(1)
    if len(at.tabs) != EXPECTED_TABS:
        print(f"APPTEST FAILED — expected {EXPECTED_TABS} tabs, got {len(at.tabs)}")
        sys.exit(1)
    # Peers renders its cards as HTML, not st.metric, so a contained failure
    # there should cost no metrics at all. Downstream tabs must be untouched.
    if len(at.metric) != baseline_metrics:
        print(f"APPTEST FAILED — {baseline_metrics - len(at.metric)} metrics lost "
              "downstream of the failing tab; isolation is leaking")
        sys.exit(1)
    if len(at.error) != 1:
        print(f"APPTEST FAILED — expected exactly 1 scoped error, got {len(at.error)}")
        sys.exit(1)


def check_ledger_populated(seed):
    """
    The Track Record tab with rows in it.

    The default AppTest seeds session_state directly, so no prediction is ever
    written and the tab only ever renders its empty state — which is the half
    that cannot break. This seeds a matured ledger and asserts the table, the
    calibration figures and the verdict all render.
    """
    import time

    from svp.data import predictions as P

    key = P.new_key()
    for tkr in ("NVDA", "AAPL", "MSFT"):
        P.record(key, tkr, 100.0, 90.0, 110.0, 130.0, "Undervalued", dedupe_hours=0)
    # Backdate so the rows count as matured and are actually scored.
    conn = P._sq()
    conn.execute("UPDATE svp_predictions SET created_at = ? WHERE ledger_key = ?",
                 (time.time() - 200 * 86400, key))
    conn.commit()

    at = AppTest.from_file(os.path.join(_ROOT, "app.py"), default_timeout=240)
    at.session_state["analysis"] = seed
    at.session_state["ledger_key"] = key
    at.run()

    if at.exception:
        print("APPTEST FAILED — populated ledger raised:")
        for e in at.exception:
            print(" ", repr(e)[:300])
        sys.exit(1)

    failures = at.session_state["tab_failures"]
    if failures:
        print(f"APPTEST FAILED — tabs failed with a populated ledger: {failures}")
        sys.exit(1)
    if not at.dataframe:
        print("APPTEST FAILED — ledger table did not render")
        sys.exit(1)

    text = " ".join(str(m.value) for m in at.markdown)
    if not any(w in text for w in ("well-calibrated", "Not enough", "confident")):
        print("APPTEST FAILED — no calibration verdict rendered")
        sys.exit(1)
    print(f"ledger: rows={len(P.for_key(key))} dataframes={len(at.dataframe)}")


def main():
    seed = build_seed_analysis()
    baseline_metrics = check_healthy(seed)
    check_tab_isolation(seed, baseline_metrics)
    check_ledger_populated(seed)
    print("APPTEST PASSED")


if __name__ == "__main__":
    main()
