"""
Segment revenue — where the money actually comes from.
======================================================

**Why this is not built on the API everything else here uses.** The SEC
``companyfacts`` endpoint, which backs the whole of :mod:`svp.data.sec`, returns
only facts with no dimensional qualifier. Segment disclosures are dimensional by
definition — revenue *by reportable segment* is a value tagged along an axis —
so they are absent from that endpoint entirely. Asking companyfacts for segments
does not return a partial answer, it returns nothing, for every filer.

Two routes remain. Ingesting the SEC's Financial Statement Data Sets means
downloading a ~50 MB archive per quarter and standing up a database to query it:
correct, and far too heavy for a request-time lookup in a Streamlit app. The
other is the one taken here — every filing ships a ``FilingSummary.xml``
indexing the rendered statement tables ("R-files") that EDGAR's own viewer
displays, and the segment disclosure is one of them. It is a few HTTP requests
and the tables are already assembled.

**What that costs.** These are rendered tables built for humans, so the parsing
is heuristic, not a schema read. Filers name the disclosure differently, put the
periods in different columns, and scale the numbers in the header rather than
the cells. Everything here fails closed: an unrecognised layout returns no
breakdown rather than a confidently wrong one, and :attr:`SegmentBreakdown.
confidence` says which of those happened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import sec, storage

_SUMMARY_URL = ("https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                "{acc_nodash}/FilingSummary.xml")
_REPORT_URL = ("https://www.sec.gov/Archives/edgar/data/{cik_int}/"
               "{acc_nodash}/{doc}")

_TTL = 30 * 24 * 3600          # a filing's tables never change once filed

# Ordered best-first: a disclosure explicitly about segments beats a general
# revenue disaggregation, which in turn beats a bare "Segments" heading that
# might be the accounting-policy note rather than the numbers.
_REPORT_PATTERNS = (
    r"revenue.*by.*segment|segment.*revenue",
    r"disaggregation of revenue",
    r"reportable segment|operating segment",
    r"segment (?:information|reporting|data)",
)

# Header text such as "$ in Millions" is where the magnitude lives; the cells
# carry bare numbers. Missing this turns a $95bn segment into $95.
_SCALES = (
    (re.compile(r"in\s+billions", re.I), 1e9),
    (re.compile(r"in\s+millions", re.I), 1e6),
    (re.compile(r"in\s+thousands", re.I), 1e3),
)

_REVENUE_ROW = re.compile(
    r"^(total\s+)?(net\s+)?(revenue|sales|net sales|total revenue)s?\b", re.I)

# Rows that are structure, not data.
_SKIP_ROW = re.compile(
    r"^(total|consolidated|eliminations?|corporate|intersegment|reconcil\w*|"
    r"unallocated|other\s*\(|adjustments?)\b", re.I)


@dataclass
class SegmentBreakdown:
    ticker: str
    segments: dict = field(default_factory=dict)     # name -> value in currency
    period: str = ""
    scale_note: str = ""
    source_report: str = ""
    confidence: str = "none"        # "none" | "low" | "table-only" | "parsed"
    note: str = ""
    table_html: str = ""

    @property
    def total(self) -> Optional[float]:
        return sum(self.segments.values()) if self.segments else None

    def shares(self) -> dict:
        """Each segment as a percentage of the total."""
        t = self.total
        if not t:
            return {}
        return {k: v / t * 100.0 for k, v in self.segments.items()}

    @property
    def is_concentrated(self) -> bool:
        """One segment carrying more than 60% of revenue."""
        sh = self.shares()
        return bool(sh) and max(sh.values()) > 60.0


def _detect_scale(text: str) -> tuple[float, str]:
    for pattern, mult in _SCALES:
        if pattern.search(text or ""):
            return mult, pattern.pattern.replace(r"\s+", " ")
    return 1.0, ""


def _to_number(cell) -> Optional[float]:
    """
    Coerce a rendered financial cell to a number.

    Handles the conventions these tables use: thousands separators, currency
    symbols, footnote markers, and parentheses for negatives.
    """
    if cell is None:
        return None
    s = str(cell).strip()
    if not s or s in {"—", "-", "–", "N/A", "nan"}:
        return None

    # Footnote markers are rendered as [1] and carry no value. Stripped before
    # the digit filter, or "12 [1]" silently becomes 121.
    s = re.sub(r"\[\s*\d+\s*\]", "", s).strip()

    # Accounting negatives are parenthesised, and the currency symbol may sit
    # outside the bracket ("$(1,200)"), so anchoring on the first character
    # misses them.
    negative = "(" in s and ")" in s
    s = re.sub(r"[^\d.\-]", "", s)
    if not s or s in {"-", ".", "-."}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if negative else v


def _fetch(url: str, cache_key: str) -> str:
    cached = storage.cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        import requests

        r = requests.get(url, headers=sec._HEADERS, timeout=30)
        r.raise_for_status()
        storage.cache_set(cache_key, r.text, ttl=_TTL)
        return r.text
    except Exception:
        return ""


def _pick_report(summary_xml: str) -> tuple[str, str]:
    """
    Choose the R-file most likely to hold the segment numbers.

    Returns ``(filename, short_name)``, or empty strings. Patterns are tried in
    priority order rather than taking the first regex that hits anywhere, so a
    filing containing both a policy note and the actual table gets the table.
    """
    reports = re.findall(
        r"<Report[^>]*>(.*?)</Report>", summary_xml, re.S | re.I)
    parsed = []
    for block in reports:
        name = re.search(r"<ShortName>(.*?)</ShortName>", block, re.S | re.I)
        fname = re.search(r"<HtmlFileName>(.*?)</HtmlFileName>", block, re.S | re.I)
        if not fname:
            fname = re.search(r"<XmlFileName>(.*?)</XmlFileName>", block, re.S | re.I)
        if name and fname:
            parsed.append((name.group(1).strip(), fname.group(1).strip()))

    for pattern in _REPORT_PATTERNS:
        rx = re.compile(pattern, re.I)
        for short, fname in parsed:
            if rx.search(short) and fname.lower().endswith((".htm", ".html")):
                return fname, short
    return "", ""


_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&quot;": '"', "&lt;": "<", "&gt;": ">",
    "&#8217;": "'", "&#8220;": '"', "&#8221;": '"', "&#151;": "—",
    "&#8212;": "—", "&mdash;": "—", "&ndash;": "–",
}


def _cell_text(fragment: str) -> str:
    """Tag-strip one cell. Mirrors the parser-free approach in svp.rag.filings."""
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", fragment)
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, char in _ENTITIES.items():
        text = text.replace(ent, char)
    text = re.sub(r"&#\d+;", " ", text)
    return re.sub(r"[\s\xa0]+", " ", text).strip()


def _html_rows(html: str) -> list[list[str]]:
    """
    Rows and cells of the largest table in a rendered R-file.

    Deliberately not pandas.read_html: that needs lxml or html5lib, and this
    package strips EDGAR HTML with regex everywhere else precisely to avoid
    carrying a parser. R-files are machine-generated and well-formed, which is
    the narrow case where that trade is safe.
    """
    tables = re.findall(r"(?is)<table[^>]*>(.*?)</table>", html)
    if not tables:
        return []
    body = max(tables, key=len)

    rows: list[list[str]] = []
    for tr in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", body):
        cells = [_cell_text(c) for c in
                 re.findall(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>", tr)]
        if cells:
            rows.append(cells)
    return rows


def _parse_table(html: str) -> tuple[dict, str, str]:
    """
    Pull ``{segment: value}`` out of a rendered R-file table.

    Returns ``(segments, period, scale_note)``. The first cell of each row holds
    the label; the first column that parses as numbers holds the most recent
    period, which is the convention these tables follow.
    """
    rows = _html_rows(html)
    if len(rows) < 2:
        return {}, "", ""

    header = " ".join(rows[0])
    scale, scale_note = _detect_scale(header)
    if scale == 1.0:
        scale, scale_note = _detect_scale(html[:4000])

    period = ""
    for cell in rows[0][1:]:
        m = re.search(r"(\d{1,2}\s+Months\s+Ended\s+\S+\s+\d{1,2},\s*\d{4}|"
                      r"[A-Z][a-z]{2}\.?\s+\d{1,2},\s*\d{4})", cell)
        if m:
            period = m.group(1)
            break

    width = max(len(r) for r in rows)
    for ci in range(1, width):
        column = [(r[0], _to_number(r[ci])) for r in rows[1:] if len(r) > ci]
        if sum(v is not None for _, v in column) < 2:
            continue

        out: dict = {}
        for label, value in column:
            name = re.sub(r"\s+", " ", label).strip(" :[]")
            if (value is None or not name or _SKIP_ROW.match(name)
                    or len(name) > 80):
                continue
            out[name] = value * scale
        if len(out) >= 2:
            return out, period, scale_note
    return {}, period, scale_note


def get_segments(ticker: str) -> SegmentBreakdown:
    """
    Revenue by reportable segment for the most recent annual filing.

    Returns an empty breakdown with ``confidence="none"`` whenever the filing
    cannot be found, the disclosure cannot be located, or the table does not
    parse. That is the common case for some filers and is not an error — a
    single-segment company has nothing to disaggregate.
    """
    tkr = ticker.upper().strip()
    key = f"segments:{tkr}"
    cached = storage.cache_get(key)
    if cached is not None:
        return SegmentBreakdown(**cached)

    empty = SegmentBreakdown(ticker=tkr, note="No segment disclosure located.")

    from ..rag import filings as filings_mod

    try:
        recent = filings_mod.list_filings(tkr, forms=("10-K",), limit=1)
    except Exception:
        return empty
    if not recent:
        return empty

    f = recent[0]
    acc = f.accession.replace("-", "")
    try:
        cik_int = int(f.cik)
    except (TypeError, ValueError):
        return empty

    summary = _fetch(_SUMMARY_URL.format(cik_int=cik_int, acc_nodash=acc),
                     f"filingsummary:{f.accession}")
    if not summary:
        return empty

    doc, short = _pick_report(summary)
    if not doc:
        empty.note = ("This filing indexes no table named as a segment or "
                      "revenue-disaggregation disclosure.")
        return empty

    html = _fetch(_REPORT_URL.format(cik_int=cik_int, acc_nodash=acc, doc=doc),
                  f"segmenttable:{f.accession}:{doc}")
    if not html:
        return empty

    segments, period, scale_note = _parse_table(html)
    out = SegmentBreakdown(
        ticker=tkr,
        segments=segments,
        period=period or f.period,
        scale_note=scale_note,
        source_report=short,
        confidence="parsed" if segments else "table-only",
        note=("" if segments else
              "The disclosure was found but its layout did not parse into "
              "segment totals. The table is shown as filed."),
        table_html=("" if segments else html[:200_000]),
    )
    storage.cache_set(key, out.__dict__, ttl=_TTL)
    return out
