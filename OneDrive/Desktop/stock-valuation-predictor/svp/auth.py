"""
Native OIDC sign-in, wrapped so its absence is a mode and not an error.
=======================================================================

Streamlit's ``st.login()`` / ``st.user`` need two things this deployment may
or may not have: the Authlib package, and an ``[auth]`` block in secrets
(client id/secret, redirect URI, cookie secret). Until both exist the app
must run exactly as it always has — anonymous ledger key, no wall, nothing
half-configured on screen. Every capability check and every ``st.user``
access therefore lives here, guarded, so no page ever touches the raw API.

The rule the wrapper enforces: **signing in is additive.** A signed-in
session binds its account to the ledger key via :mod:`svp.data.profiles`;
signing out clears the session's user state and hands back a fresh anonymous
identity. Nothing about being signed out removes ability — it only removes
cross-device continuity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st

#: Session keys that belong to the person, not the page. Cleared on logout so
#: the next visitor at the same browser starts clean. Widget/UI state (chart
#: ranges, open tabs) survives — it is about the screen, not the person.
USER_STATE_KEYS = (
    "ledger_key",
    "profile_sub",
    "profile_plan",
    "profile_name",
    "pending_ticker",
    "last_ticker",
    "analysis",
)


def _authlib_ok() -> bool:
    try:
        import authlib                                    # noqa: F401
        return True
    except Exception:
        return False


def configured() -> bool:
    """True when native sign-in can actually work in this deployment."""
    try:
        return _authlib_ok() and "auth" in st.secrets
    except Exception:
        return False


@dataclass
class UserInfo:
    sub: str
    email: Optional[str]
    name: Optional[str]


def current_user() -> Optional[UserInfo]:
    """The signed-in user, or ``None`` — including when auth is unconfigured.

    ``st.user`` raises on attribute access when no auth is configured, so this
    is the only place allowed to touch it.
    """
    if not configured():
        return None
    try:
        if not st.user.is_logged_in:
            return None
        d = st.user.to_dict()
    except Exception:
        return None
    sub = str(d.get("sub") or d.get("email") or "").strip()
    if not sub:
        return None
    return UserInfo(sub=sub,
                    email=d.get("email"),
                    name=d.get("name") or d.get("given_name"))


def providers() -> list[str]:
    """
    Named identity providers in the secrets, e.g. ``["google", "github"]``.

    An empty list with :func:`configured` True means a single unnamed provider
    block — ``st.login()`` with no argument. Detection is by the presence of a
    ``client_id`` inside a nested table, which is what Streamlit itself needs
    to drive the flow.
    """
    if not configured():
        return []
    try:
        block = st.secrets["auth"]
        out = []
        for key in block:
            try:
                sub = block[key]
                if hasattr(sub, "get") and sub.get("client_id"):
                    out.append(str(key))
            except Exception:
                continue
        return out
    except Exception:
        return []


def login(provider: Optional[str] = None) -> bool:
    """Start the OIDC flow. Returns False (with no crash) when unavailable."""
    if not configured():
        return False
    try:
        if provider:
            st.login(provider)
        else:
            st.login()
        return True
    except Exception:
        return False


def clear_user_state() -> None:
    for key in USER_STATE_KEYS:
        st.session_state.pop(key, None)


def logout() -> None:
    """
    Sign out and forget the person, not their stored work.

    Clears the user-owned session keys and drops the ``?id=`` from the URL so
    the next load mints a fresh anonymous key — the account still points at
    the stored ledger, and signing back in restores it. Database rows are
    never touched from here.
    """
    clear_user_state()
    try:
        if "id" in st.query_params:
            del st.query_params["id"]
    except Exception:
        pass
    if configured():
        try:
            st.logout()                      # ends the OIDC session; reruns
        except Exception:
            pass
