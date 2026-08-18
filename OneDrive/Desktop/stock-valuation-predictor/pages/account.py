"""
Account — identity, without a wall.
===================================

Three states, each told plainly:

- **Signed in** — who you are, your plan, the ledger key your account owns,
  and sign-out.
- **Signed out, sign-in available** — the case for an account (continuity
  across devices), provider buttons, and the reassurance that guests lose
  nothing but convenience.
- **Sign-in not configured** — the honest operator state: what is missing and
  exactly how to add it, instead of a dead button. The app is fully usable
  anonymously either way, so this page never blocks anything.
"""

from __future__ import annotations

import streamlit as st

from svp import auth, theme
from svp.data import _userdb

st.markdown(theme.section("Account", "IDENTITY"), unsafe_allow_html=True)

_user = auth.current_user()

if _user is not None:
    # ── Signed in ────────────────────────────────────────────────────────────
    _name = st.session_state.get("profile_name") or _user.name or "—"
    _plan = st.session_state.get("profile_plan", "free")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="metric-card"><div class="metric-label">Signed in as</div>'
        f'<div class="metric-sub">{_name}<br>{_user.email or ""}</div></div>',
        unsafe_allow_html=True)
    c2.markdown(
        f'<div class="metric-card"><div class="metric-label">Plan</div>'
        f'<div class="metric-value">{_plan.title()}</div></div>',
        unsafe_allow_html=True)
    c3.markdown(
        f'<div class="metric-card"><div class="metric-label">Storage</div>'
        f'<div class="metric-sub">{_userdb.backend_name()}</div></div>',
        unsafe_allow_html=True)

    st.markdown("")
    st.markdown(
        "Your track record, watchlist and saved work are bound to this "
        "account — sign in on any device and they follow you. The anonymous "
        "key below is what the account points at; you no longer need to "
        "bookmark it.")
    with st.expander("Ledger key (advanced)"):
        st.code(st.session_state.get("ledger_key", "—"), language=None)
        st.caption("Kept for transparency and manual restore. Treat it as "
                   "private — anyone holding it can read that ledger.")

    if st.button("Sign out", type="secondary"):
        auth.logout()
        st.rerun()

elif auth.configured():
    # ── Signed out, sign-in available ────────────────────────────────────────
    st.markdown(
        f"""
        <div style="max-width:520px;">
          <h3 style="margin-bottom:.3rem;">Invest with a clearer view of value.</h3>
          <p style="color:{theme.MUTED};">Sign in to keep your research, track
          record, watchlists and valuation scenarios with you across devices.
          Everything already works as a guest — an account only adds
          continuity.</p>
        </div>
        """, unsafe_allow_html=True)

    _provs = auth.providers()
    if _provs:
        for _p in _provs:
            if st.button(f"Continue with {_p.title()}", key=f"login_{_p}",
                         type="primary"):
                auth.login(_p)
    else:
        if st.button("Sign in", type="primary"):
            auth.login()

    st.caption("We never access your brokerage account or place trades. "
               "Signing in stores your email and name, nothing else.")

else:
    # ── Not configured — the honest operator state ───────────────────────────
    st.info(
        "Sign-in is not configured on this deployment yet. The app is fully "
        "usable anonymously — your work is saved to the key in this page's "
        "URL. An account adds cross-device continuity once the operator "
        "completes the setup below.")
    with st.expander("Operator setup — enable Google sign-in"):
        st.markdown(
            """
1. **Google Cloud console** → *APIs & Services* → *Credentials* →
   *Create credentials* → *OAuth client ID* → type **Web application**.
2. Add the authorized redirect URI — your deployed app URL plus
   `/oauth2callback`, e.g. `https://YOUR-APP.streamlit.app/oauth2callback`.
3. Copy the client ID and secret into the app's **Secrets** (Streamlit Cloud →
   Manage app → Settings → Secrets):

```toml
[auth]
redirect_uri = "https://YOUR-APP.streamlit.app/oauth2callback"
cookie_secret = "a-long-random-string-you-generate"

[auth.google]
client_id = "....apps.googleusercontent.com"
client_secret = "..."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

4. `Authlib` is already in requirements — redeploy and this page grows a
   *Continue with Google* button. A `[auth.github]` block works the same way
   with GitHub's OIDC endpoints.
            """)

st.divider()
st.caption("For educational purposes only. Not financial advice.")
