"""
SEC filing text retrieval and section extraction.
=================================================

Pulls the primary document of a 10-K or 10-Q from EDGAR, strips it to plain
text, and splits it into the numbered Items a reader actually cites — above
all **Item 1A Risk Factors** and **Item 7 MD&A**.

Section splitting is the part that earns its keep. A filing is one long HTML
blob; retrieving over it whole means a hit in the exhibit index looks the same
as a hit in Risk Factors. Cutting on Item boundaries first means every chunk
carries the section it came from, so an answer can cite "Item 1A" and be right.

Two shapes of the same heading have to be handled: the table-of-contents entry
near the top and the real heading further down. The last occurrence wins,
because the ToC always precedes the body.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from ..data import sec, storage

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"

# Item headings we index, in filing order. The regex tolerates the ways filers
# write them: "Item 1A.", "ITEM 1A —", "Item&nbsp;1A:".
_ITEMS_10K = [
    ("Item 1", r"item\s*1\s*[\.\-—:]?\s*business"),
    ("Item 1A", r"item\s*1a\s*[\.\-—:]?\s*risk\s*factors"),
    ("Item 3", r"item\s*3\s*[\.\-—:]?\s*legal\s*proceedings"),
    ("Item 7", r"item\s*7\s*[\.\-—:]?\s*management'?s?\s*discussion"),
    ("Item 7A", r"item\s*7a\s*[\.\-—:]?\s*quantitative\s*and\s*qualitative"),
    ("Item 8", r"item\s*8\s*[\.\-—:]?\s*financial\s*statements"),
]
_ITEMS_10Q = [
    ("Item 2", r"item\s*2\s*[\.\-—:]?\s*management'?s?\s*discussion"),
    ("Item 3", r"item\s*3\s*[\.\-—:]?\s*quantitative\s*and\s*qualitative"),
    ("Item 1A", r"item\s*1a\s*[\.\-—:]?\s*risk\s*factors"),
]

# Canonical labels so a 10-Q's MD&A (Item 2) and a 10-K's (Item 7) answer the
# same question without the caller special-casing form type.
CANONICAL = {
    "Item 1A": "Risk Factors",
    "Item 7": "MD&A",
    "Item 2": "MD&A",
    "Item 7A": "Market Risk",
    "Item 3": "Market Risk",
    "Item 1": "Business",
    "Item 8": "Financial Statements",
}


@dataclass
class FilingSection:
    item: str                  # "Item 1A"
    label: str                 # "Risk Factors"
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Filing:
    ticker: str
    cik: str
    form: str                  # "10-K" | "10-Q"
    accession: str
    filed: str                 # ISO date
    period: str
    url: str
    text: str = ""
    sections: list[FilingSection] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.form} filed {self.filed}"

    def section(self, item_or_label: str) -> Optional[FilingSection]:
        for s in self.sections:
            if item_or_label.lower() in (s.item.lower(), s.label.lower()):
                return s
        return None


def _html_to_text(html: str) -> str:
    """Strip an EDGAR document to readable text without a parser dependency."""
    # Drop script/style wholesale before tags, or their bodies survive.
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    # Tables and block ends become breaks so headings do not fuse with prose.
    html = re.sub(r"(?i)</(p|div|tr|td|h[1-6]|li|table)>", "\n", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"<[^>]+>", " ", html)
    html = (html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&#8217;", "'")
                .replace("&#8220;", '"').replace("&#8221;", '"').replace("&#151;", "—")
                .replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">"))
    html = re.sub(r"&#\d+;", " ", html)
    html = re.sub(r"[ \t\xa0]+", " ", html)
    html = re.sub(r"\n\s*\n\s*\n+", "\n\n", html)
    return html.strip()


def split_sections(text: str, form: str) -> list[FilingSection]:
    """
    Cut a filing into Items.

    The last match of each heading is used: the first is the table of contents,
    which would otherwise yield a "Risk Factors" section two lines long.
    """
    patterns = _ITEMS_10K if form.upper().startswith("10-K") else _ITEMS_10Q
    marks: list[tuple[int, str]] = []
    for item, pat in patterns:
        hits = list(re.finditer(pat, text, re.IGNORECASE))
        if not hits:
            continue
        # Prefer the last hit that leaves a plausible amount of body after it.
        chosen = None
        for m in reversed(hits):
            if len(text) - m.start() > 800:
                chosen = m
                break
        if chosen is None:
            chosen = hits[-1]
        marks.append((chosen.start(), item))

    if not marks:
        return []

    marks.sort()
    out: list[FilingSection] = []
    for i, (pos, item) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[pos:end].strip()
        if len(body.split()) < 40:      # a stray ToC line, not a section
            continue
        out.append(FilingSection(item=item, label=CANONICAL.get(item, item), text=body))
    return out


def list_filings(ticker: str, forms=("10-K", "10-Q"), limit: int = 6) -> list[Filing]:
    """Recent filings for a ticker, newest first, without downloading bodies."""
    cik = sec.get_cik(ticker)
    if not cik:
        return []

    key = f"submissions:{cik}"
    data = storage.cache_get(key)
    if data is None:
        try:
            r = requests.get(_SUBMISSIONS_URL.format(cik=cik),
                             headers={**sec._HEADERS, "Host": "data.sec.gov"}, timeout=20)
            r.raise_for_status()
            data = r.json()
            storage.cache_set(key, data, ttl=6 * 3600)
        except Exception:
            return []

    recent = (data.get("filings") or {}).get("recent") or {}
    rows = zip(recent.get("form", []), recent.get("accessionNumber", []),
               recent.get("filingDate", []), recent.get("reportDate", []),
               recent.get("primaryDocument", []))

    out: list[Filing] = []
    for form, acc, filed, period, doc in rows:
        if form not in forms or not doc:
            continue
        out.append(Filing(
            ticker=ticker.upper(), cik=cik, form=form, accession=acc,
            filed=filed, period=period or filed,
            url=_ARCHIVE.format(cik_int=int(cik), acc_nodash=acc.replace("-", ""), doc=doc),
        ))
        if len(out) >= limit:
            break
    return out


def fetch_text(filing: Filing, max_chars: int = 1_200_000) -> Filing:
    """Download and parse one filing, populating ``text`` and ``sections``."""
    if filing.text:
        return filing

    key = f"filingtext:{filing.accession}"
    cached = storage.cache_get(key)
    if cached:
        filing.text = cached
    else:
        try:
            r = requests.get(filing.url, headers=sec._HEADERS, timeout=45)
            r.raise_for_status()
            filing.text = _html_to_text(r.text)[:max_chars]
            storage.cache_set(key, filing.text, ttl=30 * 24 * 3600)
            time.sleep(0.12)          # SEC asks for <10 req/s; stay well under
        except Exception:
            filing.text = ""

    filing.sections = split_sections(filing.text, filing.form) if filing.text else []
    return filing


def latest_pair(ticker: str, form: str = "10-K") -> tuple[Optional[Filing], Optional[Filing]]:
    """The two most recent filings of one form — the comparison the app needs."""
    got = [f for f in list_filings(ticker, forms=(form,), limit=4)]
    latest = fetch_text(got[0]) if got else None
    prior = fetch_text(got[1]) if len(got) > 1 else None
    return latest, prior
