"""
Prediction ledger — every valuation the app makes, recorded and later scored.
============================================================================

The model emits a p10 / p50 / p90 band. This module stores each band with the
price at the time, and scores it against what the price actually did.

Two things follow from that, and they are the point of the module:

**A visitor has something to come back to.** Predictions mature on their own
schedule, so value accrues while the user is away. No notification pressure is
required, and none is built here.

**The model gets held to its own standard.** ``svp.analytics.accuracy`` already
refuses to assert an indicator's hit rate and measures it with a Wilson interval
instead. :func:`calibration` applies that to the valuation model: if the p10–p90
band is honest, about 80% of outcomes should land inside it. A band that
captures 98% is too wide to be useful; one that captures 40% is overconfident.
Either way the number is reported rather than asserted.

Storage mirrors :mod:`svp.data.storage` — Postgres when ``SVP_DATABASE_URL`` is
set, SQLite otherwise — but in its own table, because this is user data with no
expiry rather than a cache with a TTL.

Identity is an anonymous key, never a login. See :func:`new_key`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional, Sequence

from . import _userdb
from ..analytics.accuracy import wilson_interval

# A prediction is scored once this much time has passed. Shorter than the
# horizon a valuation actually implies — a DCF is a multi-year claim — but the
# app has to show something before then, so an outcome is marked "open" until
# the window elapses and the reader is told which is which.
DEFAULT_HORIZON_DAYS = 90

_DDL_PG = """
CREATE TABLE IF NOT EXISTS svp_predictions (
    id          TEXT PRIMARY KEY,
    ledger_key  TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    created_at  DOUBLE PRECISION NOT NULL,
    price_at    DOUBLE PRECISION NOT NULL,
    p10         DOUBLE PRECISION NOT NULL,
    p50         DOUBLE PRECISION NOT NULL,
    p90         DOUBLE PRECISION NOT NULL,
    signal      TEXT,
    horizon_days INTEGER NOT NULL
)
"""

_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS svp_predictions (
    id          TEXT PRIMARY KEY,
    ledger_key  TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    price_at    REAL NOT NULL,
    p10         REAL NOT NULL,
    p50         REAL NOT NULL,
    p90         REAL NOT NULL,
    signal      TEXT,
    horizon_days INTEGER NOT NULL
)
"""

_IDX = ("CREATE INDEX IF NOT EXISTS svp_pred_key ON svp_predictions (ledger_key)",
        "CREATE INDEX IF NOT EXISTS svp_pred_tkr ON svp_predictions (ticker)")

_PG_DDL = (_DDL_PG, *_IDX)
_SQ_DDL = (_DDL_SQLITE, *_IDX)

# Identity lives in _userdb because the watchlist keys off the same value: one
# key is one person's saved work, whatever kind it is.
new_key = _userdb.new_key
valid_key = _userdb.valid_key
_lock = _userdb.LOCK


def _pg():
    return _userdb.pg(_PG_DDL)


def _sq():
    return _userdb.sq(_SQ_DDL)


def backend_name() -> str:
    return _userdb.backend_name()


# ── Records ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Prediction:
    id: str
    ledger_key: str
    ticker: str
    created_at: float
    price_at: float
    p10: float
    p50: float
    p90: float
    signal: Optional[str]
    horizon_days: int

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400.0

    @property
    def is_mature(self) -> bool:
        return self.age_days >= self.horizon_days


@dataclass(frozen=True)
class Outcome:
    """A prediction paired with what the price actually did."""
    prediction: Prediction
    price_now: Optional[float]

    @property
    def inside_band(self) -> Optional[bool]:
        if self.price_now is None:
            return None
        return self.prediction.p10 <= self.price_now <= self.prediction.p90

    @property
    def move_pct(self) -> Optional[float]:
        if self.price_now is None or not self.prediction.price_at:
            return None
        return (self.price_now - self.prediction.price_at) / self.prediction.price_at * 100.0

    @property
    def direction_correct(self) -> Optional[bool]:
        """
        Did the price move toward the estimate?

        A separate question from band coverage, and the one a reader intuitively
        cares about. Flat calls — where the model's estimate sat on the price —
        are excluded rather than counted as wins.
        """
        if self.price_now is None:
            return None
        p = self.prediction
        implied_up = p.p50 > p.price_at
        moved_up = self.price_now > p.price_at
        if p.p50 == p.price_at or self.price_now == p.price_at:
            return None
        return implied_up == moved_up


_COLS = ("id, ledger_key, ticker, created_at, price_at, p10, p50, p90, "
         "signal, horizon_days")


def _row_to_prediction(r: Sequence) -> Prediction:
    return Prediction(
        id=r[0], ledger_key=r[1], ticker=r[2], created_at=float(r[3]),
        price_at=float(r[4]), p10=float(r[5]), p50=float(r[6]), p90=float(r[7]),
        signal=r[8], horizon_days=int(r[9]),
    )


def record(
    ledger_key: str,
    ticker: str,
    price: float,
    p10: float,
    p50: float,
    p90: float,
    signal: Optional[str] = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    dedupe_hours: float = 12.0,
) -> Optional[Prediction]:
    """
    Store one valuation.

    Returns the stored record, or ``None`` if it was suppressed as a duplicate.
    Streamlit re-runs the whole script on every widget interaction, so without
    ``dedupe_hours`` a user who moves a slider would write a dozen identical
    rows and quietly corrupt their own calibration statistics.
    """
    if not valid_key(ledger_key) or not ticker:
        return None

    now = time.time()
    if dedupe_hours > 0:
        recent = for_key(ledger_key, ticker=ticker, limit=1)
        if recent and (now - recent[0].created_at) < dedupe_hours * 3600:
            return None

    pred = Prediction(
        id=str(uuid.uuid4()), ledger_key=ledger_key.strip().lower(),
        ticker=ticker.upper().strip(), created_at=now, price_at=float(price),
        p10=float(p10), p50=float(p50), p90=float(p90), signal=signal,
        horizon_days=int(horizon_days),
    )
    row = (pred.id, pred.ledger_key, pred.ticker, pred.created_at, pred.price_at,
           pred.p10, pred.p50, pred.p90, pred.signal, pred.horizon_days)

    with _lock:
        conn = _pg()
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO svp_predictions ({_COLS}) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", row,
                    )
                return pred
            except Exception:
                pass  # fall through to sqlite
        try:
            c = _sq()
            c.execute(
                f"INSERT INTO svp_predictions ({_COLS}) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)", row,
            )
            c.commit()
            return pred
        except Exception:
            return None


def for_key(ledger_key: str, ticker: Optional[str] = None,
            limit: int = 200) -> list[Prediction]:
    """Predictions for one ledger key, newest first."""
    if not valid_key(ledger_key):
        return []
    key = ledger_key.strip().lower()

    where = "ledger_key = {ph}"
    args: list = [key]
    if ticker:
        where += " AND ticker = {ph}"
        args.append(ticker.upper().strip())

    with _lock:
        conn = _pg()
        if conn is not None:
            try:
                with conn, conn.cursor() as cur:
                    cur.execute(
                        f"SELECT {_COLS} FROM svp_predictions "
                        f"WHERE {where.format(ph='%s')} "
                        "ORDER BY created_at DESC LIMIT %s", (*args, limit),
                    )
                    return [_row_to_prediction(r) for r in cur.fetchall()]
            except Exception:
                pass
        try:
            cur = _sq().execute(
                f"SELECT {_COLS} FROM svp_predictions "
                f"WHERE {where.format(ph='?')} "
                "ORDER BY created_at DESC LIMIT ?", (*args, limit),
            )
            return [_row_to_prediction(r) for r in cur.fetchall()]
        except Exception:
            return []


def score(predictions: Sequence[Prediction],
          price_lookup) -> list[Outcome]:
    """
    Pair each prediction with the current price.

    ``price_lookup`` takes a ticker and returns a price or ``None``; it is
    injected rather than imported so this module stays free of the market feed
    and so the tests can run offline.
    """
    cache: dict[str, Optional[float]] = {}
    out = []
    for p in predictions:
        if p.ticker not in cache:
            try:
                cache[p.ticker] = price_lookup(p.ticker)
            except Exception:
                cache[p.ticker] = None
        out.append(Outcome(prediction=p, price_now=cache[p.ticker]))
    return out


@dataclass(frozen=True)
class Calibration:
    n: int
    n_inside: int
    coverage: float
    ci_low: float
    ci_high: float
    n_directional: int
    n_direction_correct: int
    hit_rate: float
    hit_ci_low: float
    hit_ci_high: float
    target: float = 0.80

    @property
    def verdict(self) -> str:
        """
        Plain-language reading of the coverage interval against the target.

        Phrased against the confidence interval rather than the point estimate,
        so a handful of predictions cannot produce a confident-sounding claim.
        """
        if self.n < 10:
            return "Not enough resolved predictions to judge calibration yet"
        if self.ci_low <= self.target <= self.ci_high:
            return "Consistent with a well-calibrated band"
        if self.coverage > self.target:
            return "Band is wider than it needs to be — under-confident"
        return "Band is too narrow — over-confident"


def calibration(outcomes: Sequence[Outcome], target: float = 0.80,
                mature_only: bool = True) -> Calibration:
    """
    How often reality landed inside the p10–p90 band.

    ``mature_only`` restricts the sample to predictions past their horizon.
    Scoring a one-day-old prediction against an 80% band is not a measurement
    of anything — the price has not had time to move — and including them would
    flatter coverage badly.
    """
    scored = [o for o in outcomes
              if o.inside_band is not None and (o.prediction.is_mature or not mature_only)]
    n = len(scored)
    inside = sum(1 for o in scored if o.inside_band)
    lo, hi = wilson_interval(inside, n)

    directional = [o for o in scored if o.direction_correct is not None]
    nd = len(directional)
    correct = sum(1 for o in directional if o.direction_correct)
    dlo, dhi = wilson_interval(correct, nd)

    return Calibration(
        n=n, n_inside=inside, coverage=(inside / n if n else 0.0),
        ci_low=lo, ci_high=hi,
        n_directional=nd, n_direction_correct=correct,
        hit_rate=(correct / nd if nd else 0.0),
        hit_ci_low=dlo, hit_ci_high=dhi,
        target=target,
    )


def stats() -> dict:
    """Row counts, for the sidebar."""
    with _lock:
        try:
            conn = _pg()
            if conn is not None:
                with conn, conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM svp_predictions")
                    return {"backend": "PostgreSQL", "rows": int(cur.fetchone()[0])}
            cur = _sq().execute("SELECT COUNT(*) FROM svp_predictions")
            return {"backend": "SQLite", "rows": int(cur.fetchone()[0])}
        except Exception:
            return {"backend": backend_name(), "rows": 0}
