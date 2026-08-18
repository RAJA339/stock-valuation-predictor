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

# The nav is two levels: six sections, and panes nested inside five of them.
# Asserting a flat tab count would have to change every time a pane moves
# between sections, so the check is on the pane set the app actually built.
EXPECTED_SECTIONS = 7
EXPECTED_PANES = {
    "Chart Studio", "Charts", "Execution & Timing", "Microstructure",
    "Valuation", "Explainability", "DCF & Scenario", "Peers", "Screener",
    "Guardrails", "Backtesting", "Options & Futures",
    "Ask the Filings", "Fundamental Δ", "Segments",
    "Watchlist", "Track Record",
    "Crypto",
    "Report",
}


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

    n_metrics = len(at.metric)
    print(f"tab containers: {len(at.tabs)}")
    print(f"metrics rendered: {n_metrics}")
    print(f"errors: {[e.value[:80] for e in at.error]}")

    labels = set(at.session_state["pane_labels"])
    if labels != EXPECTED_PANES:
        print(f"APPTEST FAILED — pane set differs.\n"
              f"  missing: {sorted(EXPECTED_PANES - labels)}\n"
              f"  unexpected: {sorted(labels - EXPECTED_PANES)}")
        sys.exit(1)
    n_sections = at.session_state["section_count"]
    if n_sections != EXPECTED_SECTIONS:
        print(f"APPTEST FAILED — expected {EXPECTED_SECTIONS} sections, got {n_sections}")
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
    if set(at.session_state["pane_labels"]) != EXPECTED_PANES:
        print("APPTEST FAILED — a pane vanished when one of them raised")
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


def check_watchlist_populated(seed):
    """
    The Watchlist tab with names on it, and a deterministic calendar.

    Same gap as the ledger: with an empty list the tab renders a placeholder and
    nothing else executes. The calendar is stubbed rather than fetched so the
    check tests the rendering, not the runner's route to Yahoo.
    """
    import datetime as dt

    from svp.data import _userdb, calendar as C, predictions as P, watchlist as W

    key = _userdb.new_key()
    for t in ("NVDA", "AAPL"):
        W.add(key, t)
    P.record(key, "NVDA", 100.0, 90.0, 110.0, 130.0, "Undervalued", dedupe_hours=0)

    today = dt.date.today()
    real_earn, real_file = C.next_earnings, C.next_filing
    C.next_earnings = lambda t: C.Event(t, "Earnings", today + dt.timedelta(days=12), False)
    C.next_filing = lambda t: C.Event(t, "10-Q", today + dt.timedelta(days=40), True)
    try:
        at = AppTest.from_file(os.path.join(_ROOT, "app.py"), default_timeout=240)
        at.session_state["analysis"] = seed
        at.session_state["ledger_key"] = key
        at.run()
    finally:
        C.next_earnings, C.next_filing = real_earn, real_file

    if at.exception:
        print("APPTEST FAILED — populated watchlist raised:")
        for e in at.exception:
            print(" ", repr(e)[:300])
        sys.exit(1)

    failures = at.session_state["tab_failures"]
    if failures:
        print(f"APPTEST FAILED — tabs failed with a populated watchlist: {failures}")
        sys.exit(1)

    # AAPL is the seeded analysis and is on the list, so the header control must
    # read as already-followed rather than offering to follow it again.
    labels = [b.label for b in at.button]
    if not any("Following" in lb for lb in labels):
        print(f"APPTEST FAILED — follow control did not reflect state: {labels}")
        sys.exit(1)
    print(f"watchlist: names={len(W.get(key))} dataframes={len(at.dataframe)}")


def check_global_calibration(seed):
    """
    The cross-user cohort table and the live track-record readout.

    Seeds distinct matured bets across many keys and tickers, then patches the
    price lookup the scorer uses so every bet reads in-band — the network is not
    reachable in CI. Asserts both the Track Record cohort section and the
    under-verdict readout on the main analysis actually render, which the empty
    state never exercises.
    """
    import streamlit as st

    from svp.data import predictions as P

    tickers = ["NVDA", "AAPL", "MSFT", "KO", "F", "GM", "AMD", "INTC",
               "T", "VZ", "XOM", "CVX", "PG", "JNJ", "WMT"]
    for t in tickers:
        P.record(P.new_key(), t, 100.0, 90.0, 110.0, 130.0, "Undervalued",
                 dedupe_hours=0)
    conn = P._sq()
    conn.execute("UPDATE svp_predictions SET created_at = created_at - ?",
                 (200 * 86400,))
    conn.commit()

    # get_global_calibration is @st.cache_data; an earlier phase already scored
    # an empty global set (real lookup, offline) and cached it. Clear so this
    # phase recomputes against the seeded, patched data.
    st.cache_data.clear()

    original = P.score
    P.score = lambda preds, lookup: original(preds, lambda t: 120.0)
    try:
        at = AppTest.from_file(os.path.join(_ROOT, "app.py"), default_timeout=240)
        at.session_state["analysis"] = seed
        at.run()
    finally:
        P.score = original

    if at.exception:
        print("APPTEST FAILED — global calibration raised:")
        for e in at.exception:
            print(" ", repr(e)[:300])
        sys.exit(1)
    if at.session_state["tab_failures"]:
        print(f"APPTEST FAILED — tabs failed: {at.session_state['tab_failures']}")
        sys.exit(1)

    text = " ".join(str(m.value) for m in at.markdown).lower()
    if "record across everyone" not in text:
        print("APPTEST FAILED — cross-user cohort section did not render")
        sys.exit(1)
    if "across every user" not in text:
        print("APPTEST FAILED — live cohort readout did not render under the verdict")
        sys.exit(1)
    print("global calibration: cohort section and live readout rendered")


def check_identity_persistence(seed):
    """
    The ledger identity is adopted from ?id= and survives a reload.

    The bug this guards: a fresh anonymous key was minted on every load, so one
    person's ledger silently split across keys whenever the page reloaded. The
    key now rides in the URL. This asserts an incoming ?id= is adopted rather
    than replaced, that junk in that slot falls back to a freshly minted valid
    key rather than crashing, and that a no-id load writes a key back to the URL
    so the next reload lands on the same identity.
    """
    from svp.data import predictions as P

    known = P.new_key()
    at = AppTest.from_file(os.path.join(_ROOT, "app.py"), default_timeout=240)
    at.query_params["id"] = known
    at.session_state["analysis"] = seed
    at.run()
    if at.exception:
        print("APPTEST FAILED — identity adoption raised:", repr(at.exception[0])[:200])
        sys.exit(1)
    if at.session_state["ledger_key"] != known:
        print(f"APPTEST FAILED — ?id= not adopted: {at.session_state['ledger_key']}")
        sys.exit(1)

    at2 = AppTest.from_file(os.path.join(_ROOT, "app.py"), default_timeout=240)
    at2.query_params["id"] = "not-a-valid-key"
    at2.session_state["analysis"] = seed
    at2.run()
    minted = at2.session_state["ledger_key"]
    if not P.valid_key(minted) or minted == "not-a-valid-key":
        print(f"APPTEST FAILED — junk ?id= not replaced with a valid key: {minted}")
        sys.exit(1)

    at3 = AppTest.from_file(os.path.join(_ROOT, "app.py"), default_timeout=240)
    at3.session_state["analysis"] = seed
    at3.run()
    # AppTest stores each query param as a list; unwrap before comparing.
    written = at3.query_params.get("id")
    written = written[0] if isinstance(written, list) else written
    if written != at3.session_state["ledger_key"]:
        print("APPTEST FAILED — a no-id load did not write the key back to the URL")
        sys.exit(1)
    print("identity: ?id= adopted, junk replaced, fresh key persisted to URL")


def main():
    seed = build_seed_analysis()
    baseline_metrics = check_healthy(seed)
    check_tab_isolation(seed, baseline_metrics)
    check_ledger_populated(seed)
    check_watchlist_populated(seed)
    check_global_calibration(seed)
    check_identity_persistence(seed)
    print("APPTEST PASSED")


if __name__ == "__main__":
    main()
