"""
Shared report — the read-only view behind a ``?r=`` slug.
=========================================================

What a recipient sees when someone shares a snapshot: the frozen numbers,
stamped with when they were frozen, and a pointer to the live analysis. The
page never resolves the sender's identity, and a revoked, expired or unknown
slug gets the same honest dead-end — no probing which of the three it was.
"""

from __future__ import annotations

import streamlit as st

from svp import theme
from svp.data import shared_reports as sr_mod

_slug = (st.query_params.get("r") or "").strip()
_rep = sr_mod.get(_slug) if _slug else None

if _rep is None:
    st.markdown(theme.section("Shared report", "Not available"),
                unsafe_allow_html=True)
    st.info("This link is no longer available — it may have been revoked by "
            "its owner, expired, or never existed.")
    st.page_link("pages/home.py", label="Go to the app")
else:
    p = _rep.payload
    st.markdown(theme.section(f"{p.get('name', _rep.ticker)} ({_rep.ticker})",
                              "SHARED SNAPSHOT"), unsafe_allow_html=True)
    st.caption(f"Frozen {p.get('as_of', 'at share time')} — the live analysis "
               "may say something different today.")

    c1, c2, c3, c4 = st.columns(4)

    def _card(col, label, value, sub=False):
        cls = "metric-sub" if sub else "metric-value"
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="{cls}">{value}</div></div>', unsafe_allow_html=True)

    _price = p.get("price")
    _card(c1, "Price at share time",
          f"${_price:,.2f}" if isinstance(_price, (int, float)) else "—")
    _pt, _lo, _hi = p.get("point"), p.get("low"), p.get("high")
    _card(c2, "Estimated value (then)",
          f"${_pt:,.2f}" if isinstance(_pt, (int, float)) else "—")
    _card(c3, "Range (p10–p90)",
          f"${_lo:,.0f} – ${_hi:,.0f}"
          if isinstance(_lo, (int, float)) and isinstance(_hi, (int, float))
          else "—", sub=True)
    _card(c4, "Signal (then)", p.get("signal") or "—")

    d1, d2, d3 = st.columns(3)
    _mos = p.get("mos_pct")
    _card(d1, "Margin of safety",
          f"{_mos:+.1f}%" if isinstance(_mos, (int, float)) else "—", sub=True)
    _dcf = p.get("dcf_per_share")
    _card(d2, "DCF / share",
          f"${_dcf:,.2f}" if isinstance(_dcf, (int, float)) else "—", sub=True)
    _cf = p.get("confluence")
    _card(d3, "Confluence",
          f"{_cf}/100" if isinstance(_cf, (int, float)) else "—", sub=True)

    st.divider()
    if st.button("Run the live analysis", type="primary"):
        st.session_state["pending_ticker"] = _rep.ticker
        try:
            st.switch_page("pages/research.py")
        except Exception:
            st.rerun()

st.caption("Scenario-based estimates from the Intrinsic Stock Valuation "
           "Predictor — not predictions, and not financial advice. For "
           "educational purposes only.")
