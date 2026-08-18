"""
Prediction markets — Polymarket's crowd-implied probabilities, read-only.
=========================================================================

A prediction-market price *is* a probability: a "Will BTC close above $150K
by March 31?" contract trading at 62¢ means real-money forecasters, in
aggregate, put roughly 62% on yes. That is a different kind of evidence from
anything else in the crypto desk — not a model output, not an options
surface, but a crowd that loses money when it is wrong — which is exactly
why it earns a panel and also why the caveats ship with it:

- **Resolution risk.** A market pays on its own resolution rules, which can
  differ from what the question seems to ask. The link to the market is
  always shown so the reader can check.
- **Liquidity.** A thin book moves on small orders; those markets are
  flagged rather than quoted as if they carried the same weight.

Data comes from Polymarket's public Gamma API — read-only, keyless, no
account anywhere. This app displays probabilities; it never places bets.
Binary (Yes/No) markets only: a two-outcome price reads directly as a
probability, while multi-outcome books need per-outcome context a summary
row would flatten dishonestly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

_BASE = "https://gamma-api.polymarket.com/markets"
_PAGE_LIMIT = 500
_MAX_PAGES = 4
_HEADERS = {"User-Agent": "StockValuationPredictor/1.0 (educational)"}
_TIMEOUT = 12

THIN_LIQUIDITY = 10_000.0          # below this, the flag ships with the row

# Circuit breaker — same discipline as the options and futures feeds: one
# failure trips a cooldown so a shared-IP deployment stops hammering the API.
_COOLDOWN_UNTIL = 0.0
_COOLDOWN_REASON = ""
_RATE_LIMIT_COOLDOWN = 900.0
_ERROR_COOLDOWN = 180.0


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "rate" in s or "too many" in s


def _trip(exc: Exception) -> None:
    global _COOLDOWN_UNTIL, _COOLDOWN_REASON
    wait = _RATE_LIMIT_COOLDOWN if _is_rate_limit(exc) else _ERROR_COOLDOWN
    _COOLDOWN_UNTIL = time.time() + wait
    _COOLDOWN_REASON = f"{type(exc).__name__}: {exc}"[:160]


def cooldown_remaining() -> float:
    return max(0.0, _COOLDOWN_UNTIL - time.time())


@dataclass
class Market:
    question: str
    yes_prob: float                 # 0..1 — the price of Yes
    volume: float
    liquidity: float
    ends: str                       # human date, or "" when unstated
    url: str

    @property
    def is_thin(self) -> bool:
        return self.liquidity < THIN_LIQUIDITY


def _jlist(raw) -> list:
    """Gamma encodes list fields as JSON strings; accept either form."""
    if isinstance(raw, list):
        return raw
    try:
        v = json.loads(raw or "[]")
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _fnum(d: dict, *keys) -> float:
    for k in keys:
        try:
            v = d.get(k)
            if v is not None:
                return float(v)
        except Exception:
            continue
    return 0.0


def _end_date(raw) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%b %d, %Y")
    except Exception:
        return ""


def parse_market(d: dict) -> Optional[Market]:
    """One Gamma market dict → :class:`Market`, or ``None`` if unusable."""
    if not isinstance(d, dict):
        return None
    question = str(d.get("question") or "").strip()
    outcomes = [str(o).strip().lower() for o in _jlist(d.get("outcomes"))]
    prices = _jlist(d.get("outcomePrices"))
    if not question or len(outcomes) != 2 or len(prices) != 2:
        return None
    if "yes" not in outcomes:
        return None
    try:
        yes = float(prices[outcomes.index("yes")])
    except Exception:
        return None
    if not (0.0 <= yes <= 1.0):
        return None
    slug = str(d.get("slug") or "").strip()
    return Market(
        question=question,
        yes_prob=yes,
        volume=_fnum(d, "volumeNum", "volume"),
        liquidity=_fnum(d, "liquidityNum", "liquidity"),
        ends=_end_date(d.get("endDate")),
        url=f"https://polymarket.com/market/{slug}" if slug
            else "https://polymarket.com",
    )


def keywords_for(symbol: str, label: str = "") -> list[str]:
    """Search terms for one coin; a generic net when the coin is unmapped."""
    named = {
        "BTC": ["bitcoin", "btc"],
        "ETH": ["ethereum", " eth "],
        "SOL": ["solana"],
        "XRP": ["xrp", "ripple"],
        "DOGE": ["dogecoin", "doge"],
        "ADA": ["cardano"],
        "AVAX": ["avalanche"],
        "LINK": ["chainlink"],
    }
    sym = (symbol or "").upper().strip()
    if sym in named:
        return named[sym]
    words = [w.lower() for w in (label or "").replace("(", " ").replace(")", " ").split()
             if len(w) > 3]
    return (words or []) + ["crypto"]


def filter_markets(raw: list[dict], keywords: list[str],
                   max_n: int = 8) -> list[Market]:
    """Parse, keep matches on question *or* slug, largest volume first.

    Slugs are searched too because a question's wording and its slug often
    differ ("Will BTC…" vs "bitcoin-above-150k"); hyphens become spaces so
    word-ish keywords match either form.
    """
    out = []
    for d in raw or []:
        m = parse_market(d)
        if m is None:
            continue
        slug = str((d or {}).get("slug") or "").replace("-", " ") \
            if isinstance(d, dict) else ""
        hay = " " + m.question.lower() + " " + slug.lower() + " "
        if any(k.lower() in hay for k in keywords):
            out.append(m)
    out.sort(key=lambda m: -m.volume)
    return out[:max_n]


def _get_page(offset: int) -> list:
    """One page of active markets from Gamma. Raises on transport errors."""
    import requests

    r = requests.get(
        f"{_BASE}?active=true&closed=false&limit={_PAGE_LIMIT}"
        f"&offset={offset}&volume_num_min=1000",
        headers=_HEADERS, timeout=_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def fetch_markets(page_fn=None) -> Optional[list[dict]]:
    """
    Active markets from Gamma across several pages, or ``None``.

    Paginates rather than trusting any server-side ordering: the volume
    leaderboard is dominated by sports and politics, so "the top 150" can
    legitimately contain zero crypto questions — the original bug. A failure
    mid-scan keeps the pages already fetched (a partial scan of real data
    beats none) and still trips the breaker for the next caller; a failure
    on the first page returns ``None``.
    """
    if cooldown_remaining() > 0:
        return None
    fetch = page_fn or _get_page
    out: list[dict] = []
    for page in range(_MAX_PAGES):
        try:
            batch = fetch(page * _PAGE_LIMIT)
        except Exception as exc:
            _trip(exc)
            return out if out else None
        out.extend(batch)
        if len(batch) < _PAGE_LIMIT:
            break
    return out


@dataclass
class Scan:
    """What a coin scan actually looked at, so an empty result is checkable."""
    markets: list[Market]
    scanned: int


def crypto_markets(symbol: str, label: str = "",
                   max_n: int = 8) -> Optional[Scan]:
    """Top prediction markets for one coin, or ``None`` when unreachable."""
    raw = fetch_markets()
    if raw is None:
        return None
    return Scan(
        markets=filter_markets(raw, keywords_for(symbol, label), max_n=max_n),
        scanned=len(raw),
    )
