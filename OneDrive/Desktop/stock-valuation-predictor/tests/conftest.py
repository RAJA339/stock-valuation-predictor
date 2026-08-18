"""
Suite-wide guard: pin the user-data SQLite path to a fresh temp file BEFORE
any test module imports svp.data._userdb.

_userdb freezes its database path at import time, and its default is a
*relative* ``svp_cache.db``. In a full-suite run the first module to import
it wins; if that module never set ``SVP_SQLITE_PATH``, every suite shares one
persistent file in the working directory — and rows written by one pytest
invocation (a profile for a fixed OIDC sub, a matured prediction) leak into
the next, producing failures that only reproduce in full runs. conftest
imports before every test module, so the setdefault here is the one that
sticks; suites that assign their own temp path still take precedence over
this default for the values, but the point is that *someone* always set it.
"""

import os
import tempfile

os.environ.setdefault(
    "SVP_SQLITE_PATH", os.path.join(tempfile.mkdtemp(), "svp_suite.db"))
