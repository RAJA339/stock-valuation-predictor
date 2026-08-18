"""
Home — the landing page.
========================

The arrival experience for a visitor who has not asked a question yet: say
what the product does in one line, take a ticker, and get out of the way.
Everything analytical lives in the Research terminal; this page's only job is
the first thirty seconds.

Deliberately no sign-in wall and no feature tour that must be dismissed —
value first, identity later (the anonymous ledger key is already active, the
entry script minted it). The search hands the ticker to the terminal through
session state and switches pages; on the very first load the entry router
does the same thing from the ``?t=`` parameter, so both doors open onto the
same room.
"""

from __future__ import annotations

import streamlit as st

from svp import theme
from svp.data import _userdb

EXAMPLES = ["AAPL", "NVDA", "TSLA", "MSFT", "AAL"]


def _go(ticker: str) -> None:
    st.session_state["pending_ticker"] = ticker.upper().strip()
    try:
        st.switch_page("pages/research.py")
    except Exception:
        # Some harnesses cannot switch pages mid-script; the entry router
        # reads pending_ticker on rerun and lands on the terminal anyway.
        st.rerun()


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="max-width:860px;margin:8vh auto 0;text-align:center;">
      <div class="section-eyebrow">INTRINSIC · STOCK VALUATION PREDICTOR</div>
      <h1 style="font-size:2.6rem;line-height:1.15;margin:.4rem 0 .8rem;">
        See the chart. Understand the value.<br>Know the thesis.</h1>
      <p style="color:{theme.MUTED};font-size:1.05rem;max-width:620px;
                margin:0 auto;">
        Enter a ticker and get an explainable read: what the chart says, what
        the business may be worth under transparent models, and how the two
        agree or conflict — with every assumption on display.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

_pad_l, _mid, _pad_r = st.columns([1, 2, 1])
with _mid:
    with st.form("home_search", border=False):
        _t = st.text_input(
            "Ticker", value="", max_chars=6, placeholder="Try AAPL, NVDA, TSLA…",
            label_visibility="collapsed",
        )
        _submitted = st.form_submit_button("Analyze", type="primary",
                                           width="stretch")
    if _submitted and _t.strip():
        _go(_t)

    _cols = st.columns(len(EXAMPLES))
    for _c, _ex in zip(_cols, EXAMPLES):
        if _c.button(_ex, key=f"ex_{_ex}", width="stretch"):
            _go(_ex)

# Market status lives in the persistent app header (rendered by the entry
# script on every page), so the hero stays a hero rather than repeating it.
st.divider()

# ── What you get ─────────────────────────────────────────────────────────────
_f1, _f2, _f3 = st.columns(3)
_f1.markdown(
    '<div class="metric-card"><div class="metric-label">CHART STUDIO</div>'
    '<div class="metric-sub">Institutional candlestick workspace with the '
    "model's fair-value band drawn on the price axis — see at a glance "
    'whether price sits below, inside, or above the estimated range.</div></div>',
    unsafe_allow_html=True,
)
_f2.markdown(
    '<div class="metric-card"><div class="metric-label">TRANSPARENT VALUATION</div>'
    '<div class="metric-sub">DCF, quantile ML and peer models with every input, '
    'assumption and sensitivity inspectable. Ranges and confidence, never '
    'false precision.</div></div>',
    unsafe_allow_html=True,
)
_f3.markdown(
    '<div class="metric-card"><div class="metric-label">MEASURED TRACK RECORD</div>'
    '<div class="metric-sub">Every call is saved and scored against what the '
    'price actually did — across all users. Accuracy is observed here, '
    'not asserted.</div></div>',
    unsafe_allow_html=True,
)

# ── Trust + disclaimer ───────────────────────────────────────────────────────
_durable = "PostgreSQL" if _userdb.is_durable() else "local"
st.markdown(
    f"""<div style="text-align:center;margin-top:1.6rem;color:{theme.MUTED};
        font-size:.8rem;">
        Your work is saved to an anonymous key in this page's URL — no account,
        no email. We never access your brokerage account or place trades.
        Storage: {_durable}.<br><br>
        <b>For educational purposes only. Not financial advice.</b></div>""",
    unsafe_allow_html=True,
)
