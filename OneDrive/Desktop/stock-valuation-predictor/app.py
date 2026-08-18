"""
Intrinsic Stock Valuation Predictor — entry point and router.
=============================================================
Webster University — MS Business Analytics | Group ML Project
Author: Raja Mupparaju

The app is a multipage shell built on ``st.navigation``: a Home landing page
for arrival and the Research terminal where the analysis lives (see
``pages/research.py`` — the heavy lifting is in the ``svp`` package).

This entry script owns only what must hold on *every* page:

- page config and the global obsidian CSS;
- the anonymous ``?id=`` ledger identity — adopted from the URL and written
  back before any page runs, so a reload keeps the same track record whether
  the visitor lands on Home or deep inside an analysis;
- deep-link routing: a bare visit is an arrival and gets Home, but every
  ``?t=TICKER`` analysis link ever shared still lands directly on the thing
  it points to. A link someone sends a friend has to open on what they were
  looking at.
"""

from __future__ import annotations

import streamlit as st

from svp import auth, theme
from svp.data import predictions as pred_mod, profiles as prof_mod

st.set_page_config(
    page_title="Intrinsic Stock Valuation Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(theme.CSS, unsafe_allow_html=True)

# ── Ledger identity — on every page ──────────────────────────────────────────
# An anonymous key, not a login, persisted in the URL as ?id= so a page reload
# keeps the same identity instead of minting a fresh one every time — which was
# silently splitting one person's ledger across several keys. The key is an
# anonymous handle with no personal data behind it; it is a separate parameter
# from the ?t= ticker, so users can share the ticker without their key.
_url_id = (st.query_params.get("id") or "").strip().lower()
if "ledger_key" not in st.session_state:
    st.session_state["ledger_key"] = (
        _url_id if pred_mod.valid_key(_url_id) else pred_mod.new_key()
    )
if st.query_params.get("id") != st.session_state["ledger_key"]:
    st.query_params["id"] = st.session_state["ledger_key"]

# ── Signed-in continuity ─────────────────────────────────────────────────────
# When native OIDC sign-in is configured and the visitor is logged in, bind
# the account to its ledger key: a first sign-in adopts the session's
# anonymous key (pre-sign-in work is kept), and every later sign-in restores
# the stored key — the continuity an account exists to provide. Unconfigured
# or signed-out deployments skip this entirely and behave as before.
_user = auth.current_user()
if _user is not None and st.session_state.get("profile_sub") != _user.sub:
    _prof = prof_mod.get_or_create(_user.sub, _user.email, _user.name,
                                   st.session_state["ledger_key"])
    if _prof is not None:
        st.session_state["profile_sub"] = _prof.sub
        st.session_state["profile_plan"] = _prof.plan
        st.session_state["profile_name"] = _prof.display_name or _user.name
        if _prof.ledger_key != st.session_state["ledger_key"]:
            st.session_state["ledger_key"] = _prof.ledger_key
            st.query_params["id"] = _prof.ledger_key

# ── Routing ──────────────────────────────────────────────────────────────────
# ?t= in the URL (every link ever shared) or a ticker handed over by the Home
# search sends the visitor straight to the terminal on first load. Once the
# terminal has run in this session, the flag flips and navigation behaves
# normally — Home stays Home when the user clicks it.
_deep = (
    ("t" in st.query_params) or bool(st.session_state.get("pending_ticker"))
) and not st.session_state.get("research_visited", False)
# A ?r= slug is a recipient opening someone's frozen snapshot — that page
# wins the landing outright; the slug stays in the URL, so a reload still
# shows the snapshot until the recipient navigates away themselves.
_shared = bool((st.query_params.get("r") or "").strip())

home = st.Page("pages/home.py", title="Home",
               default=not (_deep or _shared))
research = st.Page("pages/research.py", title="Research terminal",
                   url_path="research", default=_deep and not _shared)
account = st.Page("pages/account.py", title="Account", url_path="account")
shared = st.Page("pages/shared.py", title="Shared report", url_path="shared",
                 default=_shared)
st.navigation([home, research, account, shared]).run()
