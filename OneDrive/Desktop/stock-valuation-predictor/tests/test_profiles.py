"""
Pytest suite for account profiles and the auth wrapper.

The property that carries the feature: **adoption**. A first sign-in must bind
the account to the anonymous key already in the session (pre-sign-in work is
kept), and every later sign-in must return the stored key no matter what key
the new session arrived with — that is the continuity an account exists to
provide. The auth wrapper's property is the opposite one: with nothing
configured, every call degrades to guest mode instead of raising.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SVP_SQLITE_PATH"] = os.path.join(tempfile.mkdtemp(), "profiles_test.db")
os.environ.pop("SVP_DATABASE_URL", None)
os.environ.pop("DATABASE_URL", None)

from svp.data import _userdb, profiles as PR       # noqa: E402
from svp import auth                                # noqa: E402


class TestProfiles:
    def setup_method(self):
        _userdb.reset_for_tests()
        # Unique subs per run: profiles are keyed by sub, and a fixed string
        # would collide with any row that survived in a shared database.
        self.sub = "sub-" + _userdb.new_key()[:8]

    def test_first_sign_in_adopts_the_session_key(self):
        anon = _userdb.new_key()
        p = PR.get_or_create(self.sub, "a@b.com", "Ada", anon)
        assert p is not None and p.created
        assert p.ledger_key == anon
        assert p.plan == "free"

    def test_later_sign_in_restores_the_stored_key(self):
        anon = _userdb.new_key()
        PR.get_or_create(self.sub, "a@b.com", "Ada", anon)
        other_session = _userdb.new_key()
        p = PR.get_or_create(self.sub, "a@b.com", "Ada", other_session)
        assert not p.created
        assert p.ledger_key == anon          # the stored key wins
        assert p.ledger_key != other_session

    def test_malformed_session_key_gets_a_fresh_valid_one(self):
        p = PR.get_or_create(self.sub, None, None, "not-a-key")
        assert p is not None
        assert _userdb.valid_key(p.ledger_key)

    def test_get_unknown_sub_is_none(self):
        assert PR.get("nobody") is None
        assert PR.get("") is None

    def test_empty_sub_is_refused(self):
        assert PR.get_or_create("", "a@b.com", "X", _userdb.new_key()) is None

    def test_claims_refresh_but_key_never_changes(self):
        anon = _userdb.new_key()
        PR.get_or_create(self.sub, None, None, anon)
        p = PR.get_or_create(self.sub, "new@mail.com", "New Name", _userdb.new_key())
        assert p.ledger_key == anon
        stored = PR.get(self.sub)
        assert stored.email == "new@mail.com"
        assert stored.display_name == "New Name"


class TestAuthGuestMode:
    """With no Authlib and no [auth] secrets, everything degrades, nothing raises."""

    def test_not_configured_here(self):
        assert auth.configured() is False

    def test_current_user_is_none(self):
        assert auth.current_user() is None

    def test_login_returns_false(self):
        assert auth.login() is False
        assert auth.login("google") is False

    def test_providers_empty(self):
        assert auth.providers() == []

    def test_user_state_keys_cover_the_identity(self):
        # Logout must at minimum forget the ledger key and profile binding —
        # leaving either behind would hand the next visitor someone's ledger.
        for must in ("ledger_key", "profile_sub", "analysis"):
            assert must in auth.USER_STATE_KEYS

    def test_logout_never_raises_in_bare_mode(self):
        auth.logout()          # no session, no auth, no query params — still fine
