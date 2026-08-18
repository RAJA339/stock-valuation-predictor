"""
Broker CSV import — Robinhood reports and generic position exports.
===================================================================

Robinhood exposes no official API; the "integrations" that exist ask users to
type brokerage credentials into a third-party app, which this product will
never do — the trust copy says so on three pages. What Robinhood *does* give
every customer is a CSV export of their own history (app → Account →
Statements & History → Reports). This module turns that file into positions:

- **Transaction reports** (Robinhood's format): positions are rebuilt from
  the trades themselves under the average-cost method — buys add shares and
  cost, sells remove shares at the running average, split credits add shares
  at zero cost (which is exactly what a split does to average cost). The
  result is net shares and a true average cost, not a guess.
- **Position exports** (generic): many brokers export current holdings
  directly (symbol / quantity / average cost columns under varying names);
  those are taken as stated.

Everything here is pure text-in → data-out. Rows the parser cannot use are
counted and reported, never silently dropped into wrong numbers; a file that
yields nothing usable says so.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from typing import Optional

from .watchlist import normalise

#: Robinhood trans codes that move share counts. Everything else (dividends,
#: interest, transfers, option events) is skipped and counted.
_BUY_CODES = {"BUY", "B"}
_SELL_CODES = {"SELL", "S"}
_SPLIT_CODES = {"SPL"}

_MONEY_RE = re.compile(r"[,$()\s]")


@dataclass
class ImportedPosition:
    ticker: str
    shares: float
    avg_cost: float


@dataclass
class ImportResult:
    source: str                       # "transactions" | "positions"
    positions: list[ImportedPosition] = field(default_factory=list)
    rows_used: int = 0
    rows_skipped: int = 0
    notes: list[str] = field(default_factory=list)


def _num(raw) -> Optional[float]:
    """Parse '1,234.56', '$12.30', '(45.10)' → float; None when not a number."""
    s = str(raw or "").strip()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = _MONEY_RE.sub("", s)
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _norm_header(h: str) -> str:
    return re.sub(r"[^a-z]", "", (h or "").lower())


def _pick(headers: dict[str, str], *candidates: str) -> Optional[str]:
    """The original header name matching any normalised candidate."""
    for c in candidates:
        if c in headers:
            return headers[c]
    return None


def parse_csv(text: str) -> Optional[ImportResult]:
    """Parse a broker CSV into positions, or ``None`` if it is not one."""
    if not text or not text.strip():
        return None
    try:
        rows = list(csv.DictReader(io.StringIO(text.strip().lstrip("﻿"))))
    except Exception:
        return None
    if not rows:
        return None

    headers = {_norm_header(h): h for h in (rows[0].keys() or []) if h}

    instrument = _pick(headers, "instrument", "symbol", "ticker")
    code = _pick(headers, "transcode", "transactiontype", "type", "side", "action")
    qty = _pick(headers, "quantity", "shares", "qty")
    price = _pick(headers, "price", "averageprice", "avgcost", "averagecost",
                  "costbasispershare", "purchaseprice")

    date_col = _pick(headers, "activitydate", "processdate", "date",
                     "settledate", "tradedate")

    if instrument and code and qty and price is not None:
        return _parse_transactions(rows, instrument, code, qty, price, date_col)
    if instrument and qty and price is not None:
        return _parse_positions(rows, instrument, qty, price)
    return None


def _parse_date(raw) -> Optional[tuple]:
    """'8/18/2026' or '2026-08-18' → sortable (y, m, d); None otherwise."""
    s = str(raw or "").strip()
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        return (int(m.group(3)), int(m.group(1)), int(m.group(2)))
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _parse_transactions(rows, c_inst, c_code, c_qty, c_price,
                        c_date=None) -> ImportResult:
    res = ImportResult(source="transactions")
    # Oldest first: the average-cost method only makes sense applied in the
    # order the trades happened. Sort by the date column when the file has a
    # parseable one (stable, so same-day trades keep their file order);
    # otherwise assume Robinhood's newest-first convention and reverse.
    dates = [_parse_date(r.get(c_date)) if c_date else None for r in rows]
    if c_date and all(d is not None for d in dates):
        rows = [r for _, r in sorted(zip(dates, rows), key=lambda p: p[0])]
    else:
        rows = list(reversed(rows))

    book: dict[str, tuple[float, float]] = {}      # ticker -> (qty, cost)
    for r in rows:
        tkr = normalise(str(r.get(c_inst) or ""))
        code = str(r.get(c_code) or "").strip().upper()
        q = _num(r.get(c_qty))
        px = _num(r.get(c_price))
        if not tkr or q is None or q <= 0:
            res.rows_skipped += 1
            continue

        held_q, held_cost = book.get(tkr, (0.0, 0.0))
        if code in _BUY_CODES and px is not None and px > 0:
            book[tkr] = (held_q + q, held_cost + q * px)
        elif code in _SPLIT_CODES:
            book[tkr] = (held_q + q, held_cost)      # extra shares, same cost
        elif code in _SELL_CODES:
            if held_q <= 0:
                res.rows_skipped += 1                # sell with no known basis
                continue
            sold = min(q, held_q)
            book[tkr] = (held_q - sold,
                         held_cost * (1 - sold / held_q))
        else:
            res.rows_skipped += 1
            continue
        res.rows_used += 1

    for tkr, (q, cost) in sorted(book.items()):
        # Positions closed out (or reduced to dust) are history, not holdings.
        if q > 1e-6 and cost > 0:
            res.positions.append(ImportedPosition(
                ticker=tkr, shares=round(q, 6), avg_cost=round(cost / q, 4)))

    if res.rows_skipped:
        res.notes.append(
            f"{res.rows_skipped} rows skipped (dividends, transfers, options "
            "and anything else that does not move a share count).")
    if not res.positions and res.rows_used == 0:
        res.notes.append("No buy/sell rows found — is this the right report?")
    return res


def _parse_positions(rows, c_inst, c_qty, c_price) -> ImportResult:
    res = ImportResult(source="positions")
    seen: dict[str, ImportedPosition] = {}
    for r in rows:
        tkr = normalise(str(r.get(c_inst) or ""))
        q = _num(r.get(c_qty))
        px = _num(r.get(c_price))
        if not tkr or q is None or q <= 0 or px is None or px <= 0:
            res.rows_skipped += 1
            continue
        if tkr in seen:
            # Two lots of the same name: merge under average cost.
            prev = seen[tkr]
            total_q = prev.shares + q
            avg = (prev.shares * prev.avg_cost + q * px) / total_q
            seen[tkr] = ImportedPosition(tkr, round(total_q, 6), round(avg, 4))
        else:
            seen[tkr] = ImportedPosition(tkr, round(q, 6), round(px, 4))
        res.rows_used += 1
    res.positions = [seen[k] for k in sorted(seen)]
    if res.rows_skipped:
        res.notes.append(f"{res.rows_skipped} rows skipped (no usable symbol, "
                         "quantity or cost).")
    return res
