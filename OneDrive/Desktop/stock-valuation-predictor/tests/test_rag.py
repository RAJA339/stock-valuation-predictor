"""
Pytest suite for the filings RAG layer.

Runs entirely on synthetic filing text, so no EDGAR call is made. The document
below deliberately mimics the two shapes that break naive parsers: a table of
contents that repeats every Item heading, and a section that is largely
carried over between years with a small genuine insertion.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.rag import engine as E, filings as F, store as S   # noqa: E402

BOILERPLATE = "Our results depend on general economic conditions and consumer demand. "
NEW_RISK = (
    "Regulatory scrutiny of artificial intelligence systems may impose new compliance "
    "obligations and delay product launches in the European Union. "
)


def _filing(filed: str, accession: str, extra: str = "") -> F.Filing:
    text = f"""
    TABLE OF CONTENTS
    Item 1A. Risk Factors ..... 12
    Item 7. Management's Discussion and Analysis ..... 40

    Item 1. Business
    {'We design and sell consumer hardware and services worldwide. ' * 30}

    Item 1A. Risk Factors
    {BOILERPLATE * 25}
    We face intense competition in every market in which we operate. {extra}

    Item 7. Management's Discussion and Analysis of Financial Condition
    {'Gross margin was affected by product mix and foreign exchange. ' * 25}

    Item 8. Financial Statements and Supplementary Data
    {'See the consolidated financial statements. ' * 20}
    """
    f = F.Filing("AAPL", "0000320193", "10-K", accession, filed, filed, "https://example")
    f.text = text
    f.sections = F.split_sections(text, "10-K")
    return f


@pytest.fixture(scope="module")
def pair():
    latest = _filing("2026-02-01", "0001", extra=NEW_RISK)
    prior = _filing("2025-02-01", "0002")
    return latest, prior


@pytest.fixture(scope="module")
def indexes(pair):
    latest, prior = pair
    return S.build_index(latest), S.build_index(prior)


class TestSectionSplitting:
    def test_finds_the_expected_items(self, pair):
        items = {s.item for s in pair[0].sections}
        assert {"Item 1", "Item 1A", "Item 7"} <= items

    def test_body_wins_over_table_of_contents(self, pair):
        """The ToC line comes first; the real section must be the one kept."""
        risk = pair[0].section("Item 1A")
        assert risk is not None and risk.words > 100

    def test_canonical_labels_applied(self, pair):
        assert pair[0].section("Item 7").label == "MD&A"
        assert pair[0].section("Risk Factors").item == "Item 1A"

    def test_sections_do_not_overlap(self, pair):
        risk, mda = pair[0].section("Item 1A"), pair[0].section("Item 7")
        assert "Gross margin" not in risk.text
        assert "intense competition" not in mda.text

    def test_unparseable_document_returns_empty(self):
        assert F.split_sections("no headings here at all", "10-K") == []

    def test_html_stripping(self):
        out = F._html_to_text("<p>Hello <b>world</b></p><script>bad()</script>&amp;")
        assert "Hello" in out and "world" in out
        assert "bad()" not in out and "<" not in out


class TestChunkingAndRetrieval:
    def test_chunks_carry_their_section(self, indexes):
        assert all(c.item and c.label for c in indexes[0].chunks)

    def test_no_chunk_spans_two_sections(self, indexes):
        for c in indexes[0].chunks:
            assert c.item in {"Item 1", "Item 1A", "Item 7", "Item 8"}

    def test_citation_names_filing_and_item(self, indexes):
        cite = indexes[0].chunks[0].citation
        assert "10-K" in cite and "Item" in cite

    def test_retrieval_finds_the_right_section(self, indexes):
        hits = indexes[0].search("gross margin and product mix", k=3)
        assert hits and hits[0].chunk.label == "MD&A"

    def test_item_filter_restricts_results(self, indexes):
        hits = indexes[0].search("competition", k=5, item_filter="Risk Factors")
        assert hits and all(h.chunk.label == "Risk Factors" for h in hits)

    def test_scores_are_ordered(self, indexes):
        hits = indexes[0].search("economic conditions", k=4)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_empty_index_is_safe(self):
        assert S.FilingIndex([]).search("anything") == []


class TestNoveltyDetection:
    def test_finds_the_inserted_risk(self, indexes):
        hits = indexes[0].novel_chunks(indexes[1], item_filter="Risk Factors")
        assert hits, "the newly inserted risk factor was not detected"
        assert any("artificial intelligence" in h.chunk.text.lower() for h in hits)

    def test_carried_over_language_is_not_flagged(self, indexes):
        hits = indexes[0].novel_chunks(indexes[1], item_filter="Risk Factors")
        assert not any(h.chunk.text.strip().startswith(BOILERPLATE.strip()[:40])
                       for h in hits)

    def test_identical_filings_yield_no_novelty(self):
        a, b = _filing("2026-02-01", "1"), _filing("2025-02-01", "2")
        assert S.build_index(a).novel_chunks(S.build_index(b)) == []

    def test_novelty_scores_are_bounded(self, indexes):
        hits = indexes[0].novel_chunks(indexes[1])
        assert all(0.0 <= h.score <= 1.0 for h in hits)


class TestQueryEngine:
    def test_comparative_question_routes_to_diff(self, indexes):
        ans = E.answer("What newly added risk factors appear?", *indexes)
        assert ans.mode == "diff"

    def test_plain_question_routes_to_lookup(self, indexes):
        ans = E.answer("What does management say about margins?", *indexes)
        assert ans.mode == "lookup"

    def test_section_routing_from_wording(self):
        assert E.detect_section("what are the risk factors") == "Risk Factors"
        assert E.detect_section("discuss liquidity and revenue") == "MD&A"
        assert E.detect_section("hello there") is None

    def test_bullets_are_quoted_and_cited(self, indexes):
        ans = E.answer("What newly added risk factors appear?", *indexes, k=3)
        assert ans.bullets
        for b in ans.bullets:
            assert b.quote.strip()
            assert "10-K" in b.citation and "Item" in b.citation

    def test_quotes_are_verbatim_from_the_filing(self, pair, indexes):
        """Every quote must appear in the source text — no paraphrase."""
        ans = E.answer("What newly added risk factors appear?", *indexes, k=3)
        haystack = " ".join(pair[0].text.split())
        for b in ans.bullets:
            assert " ".join(b.quote.split()[:8]) in haystack

    def test_quotes_are_bounded_in_length(self, indexes):
        ans = E.answer("risk factors", *indexes, k=4)
        assert all(len(b.quote.split()) <= 70 for b in ans.bullets)

    def test_no_prior_filing_falls_back_to_lookup(self, indexes):
        ans = E.answer("What newly added risk factors appear?", indexes[0], None)
        assert ans.mode == "lookup"

    def test_unanswerable_question_reports_it(self, indexes):
        ans = E.answer("zqxjkv wombat telemetry", *indexes)
        assert ans.empty or ans.bullets
        if ans.empty:
            assert ans.note
