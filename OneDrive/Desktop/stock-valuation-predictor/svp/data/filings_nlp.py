"""
Fundamental-delta NLP.
======================

Flags rapid changes in corporate disclosure by computing the textual divergence
between consecutive SEC filings (10-K / 10-Q) — a low cosine similarity between
the latest filing's risk-factor language and the prior one's can surface rising
risk before the broader market digests it.

Uses TF-IDF cosine similarity (scikit-learn). Filing text is fetched best-effort
from SEC EDGAR; if the documents can't be retrieved, callers may pass text
directly (or use the bundled samples) so the feature still demonstrates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_HEADERS = {"User-Agent": "StockValuationApp contact@example.com"}
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"


@dataclass
class FilingDivergence:
    similarity: Optional[float]        # 1.0 = identical, lower = more change
    divergence: Optional[float]        # 1 - similarity
    added_terms: list = field(default_factory=list)
    dropped_terms: list = field(default_factory=list)
    source: str = "unavailable"

    @property
    def alert(self) -> Optional[bool]:
        """Material change when similarity drops below ~0.85."""
        if self.similarity is None:
            return None
        return self.similarity < 0.85


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)         # strip any HTML
    text = re.sub(r"&#?\w+;", " ", text)          # entities
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _fetch_two_latest_docs(cik: str, forms=("10-K", "10-Q")) -> Optional[tuple[str, str]]:
    """Fetch the two most recent primary documents for the given form types."""
    try:
        data = requests.get(_SUBMISSIONS.format(cik=cik), headers=_HEADERS, timeout=15).json()
        recent = data.get("filings", {}).get("recent", {})
        forms_list = recent.get("form", [])
        accns = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
        picks = [
            (accns[i].replace("-", ""), docs[i])
            for i, f in enumerate(forms_list)
            if f in forms and i < len(docs)
        ][:2]
        if len(picks) < 2:
            return None
        cik_int = str(int(cik))
        texts = []
        for accn, doc in picks:
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn}/{doc}"
            texts.append(_clean(requests.get(url, headers=_HEADERS, timeout=20).text)[:200000])
        return texts[0], texts[1]
    except Exception:
        return None


def compare_texts(latest: str, prior: str, top_n: int = 12) -> FilingDivergence:
    """Compute TF-IDF cosine similarity + the most distinctive added/dropped terms."""
    latest_c, prior_c = _clean(latest), _clean(prior)
    if not latest_c.strip() or not prior_c.strip():
        return FilingDivergence(None, None, source="empty")
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1, 2))
        tfidf = vec.fit_transform([latest_c, prior_c])
        sim = float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
        # Terms weighted much higher in latest vs prior (added) and vice-versa.
        arr = tfidf.toarray()
        names = vec.get_feature_names_out()
        diff = arr[0] - arr[1]
        added = [names[i] for i in diff.argsort()[::-1][:top_n] if diff[i] > 0]
        dropped = [names[i] for i in diff.argsort()[:top_n] if diff[i] < 0]
        return FilingDivergence(sim, 1 - sim, added, dropped, source="text")
    except Exception:
        return FilingDivergence(None, None, source="error")


def analyze_filings(cik: Optional[str], latest: Optional[str] = None,
                    prior: Optional[str] = None) -> FilingDivergence:
    """
    Compare the two most recent filings for ``cik`` (fetched from EDGAR), or the
    supplied ``latest``/``prior`` texts when provided/EDGAR is unreachable.
    """
    if latest and prior:
        return compare_texts(latest, prior)
    if cik:
        fetched = _fetch_two_latest_docs(cik)
        if fetched:
            d = compare_texts(*fetched)
            d.source = "sec-edgar"
            return d
    return FilingDivergence(None, None, source="unavailable")


# Illustrative samples (consecutive risk-factor excerpts) for offline demos.
SAMPLE_LATEST = (
    "Our business faces intense competition and supply chain disruptions. We now "
    "face heightened regulatory scrutiny and pending litigation that could materially "
    "affect results. Rising interest rates and macroeconomic uncertainty may reduce "
    "customer demand. A cybersecurity incident could impair operations."
)
SAMPLE_PRIOR = (
    "Our business faces intense competition and supply chain disruptions. Rising "
    "interest rates and macroeconomic uncertainty may reduce customer demand. A "
    "cybersecurity incident could impair operations."
)
