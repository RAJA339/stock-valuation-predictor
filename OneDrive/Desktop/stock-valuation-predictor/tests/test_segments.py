"""
Pytest suite for segment-revenue extraction.

Every case runs against fixtures shaped like real EDGAR R-files, because the
network is not available in CI and because the interesting cases are layout
variations rather than transport. What is *not* claimed here: that these
fixtures cover the layouts of every filer. They cover the conventions the
format uses — scale in the header, parentheses for negatives, totals and
eliminations mixed in with segments — and the failure mode is designed so that
an unrecognised layout yields no breakdown rather than a wrong one.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svp.data import segments as S            # noqa: E402


def table(rows, header=None):
    head = ("<tr>" + "".join(f"<th>{c}</th>" for c in header) + "</tr>"
            if header else "")
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                   for r in rows)
    return f"<table>{head}{body}</table>"


NVDA_HEADER = ["Revenue ($ in Millions)", "12 Months Ended Dec. 31, 2025",
               "12 Months Ended Dec. 31, 2024"]
NVDA_ROWS = [
    ["Data Center", "115,186", "47,525"],
    ["Gaming", "11,350", "10,447"],
    ["Professional Visualization", "1,878", "1,553"],
    ["Automotive", "1,694", "1,091"],
    ["Eliminations", "(250)", "(180)"],
    ["Total revenue", "130,497", "60,922"],
]


# ── Choosing the right report ────────────────────────────────────────────────
class TestReportSelection:
    SUMMARY = """<FilingSummary><MyReports>
      <Report><ShortName>Document and Entity Information</ShortName>
              <HtmlFileName>R1.htm</HtmlFileName></Report>
      <Report><ShortName>Segment Information (Policies)</ShortName>
              <HtmlFileName>R7.htm</HtmlFileName></Report>
      <Report><ShortName>Revenue by Reportable Segment (Details)</ShortName>
              <HtmlFileName>R42.htm</HtmlFileName></Report>
    </MyReports></FilingSummary>"""

    def test_prefers_the_numbers_over_the_policy_note(self):
        """
        A 10-K contains both an accounting-policy note about segments and the
        table of actual figures. Taking the first regex hit would get the note.
        """
        doc, short = S._pick_report(self.SUMMARY)
        assert doc == "R42.htm"
        assert "Revenue by Reportable Segment" in short

    def test_falls_back_to_a_disaggregation_disclosure(self):
        summary = """<FilingSummary><MyReports>
          <Report><ShortName>Disaggregation of Revenue</ShortName>
                  <HtmlFileName>R30.htm</HtmlFileName></Report>
        </MyReports></FilingSummary>"""
        assert S._pick_report(summary)[0] == "R30.htm"

    def test_no_segment_report_returns_empty(self):
        summary = """<FilingSummary><MyReports>
          <Report><ShortName>Balance Sheets</ShortName>
                  <HtmlFileName>R2.htm</HtmlFileName></Report>
        </MyReports></FilingSummary>"""
        assert S._pick_report(summary) == ("", "")

    def test_non_html_reports_are_skipped(self):
        summary = """<FilingSummary><MyReports>
          <Report><ShortName>Segment Information</ShortName>
                  <XmlFileName>R9.xml</XmlFileName></Report>
        </MyReports></FilingSummary>"""
        assert S._pick_report(summary) == ("", "")

    def test_malformed_xml_does_not_raise(self):
        assert S._pick_report("<FilingSummary><Report>broken") == ("", "")

    def test_empty_input(self):
        assert S._pick_report("") == ("", "")


# ── Number coercion ──────────────────────────────────────────────────────────
class TestNumberCoercion:
    @pytest.mark.parametrize("raw,want", [
        ("115,186", 115186.0), ("$ 1,878", 1878.0), ("(250)", -250.0),
        ("1.5", 1.5), ("$(1,200)", -1200.0), ("12 [1]", 12.0),
    ])
    def test_parses(self, raw, want):
        assert S._to_number(raw) == pytest.approx(want)

    @pytest.mark.parametrize("raw", ["", "  ", "—", "-", "N/A", None, "nan", "$"])
    def test_blanks_are_none(self, raw):
        assert S._to_number(raw) is None


# ── Scale detection ──────────────────────────────────────────────────────────
class TestScale:
    @pytest.mark.parametrize("text,mult", [
        ("$ in Millions", 1e6), ("$ in Thousands", 1e3),
        ("$ in Billions", 1e9), ("USD ($)", 1.0),
    ])
    def test_detects(self, text, mult):
        assert S._detect_scale(text)[0] == mult

    def test_scale_is_applied_to_values(self):
        """Missing this turns a $115bn segment into $115."""
        segs, _, _ = S._parse_table(table(NVDA_ROWS, NVDA_HEADER))
        assert segs["Data Center"] == pytest.approx(115_186 * 1e6)

    def test_scale_found_in_body_when_absent_from_the_header(self):
        html = ("<p>$ in Thousands</p>"
                + table([["Alpha", "100"], ["Beta", "200"]], ["Segment", "2025"]))
        segs, _, note = S._parse_table(html)
        assert segs["Alpha"] == pytest.approx(100_000)
        assert "thousand" in note.lower()


# ── Table parsing ────────────────────────────────────────────────────────────
class TestParseTable:
    def test_extracts_the_operating_segments(self):
        segs, _, _ = S._parse_table(table(NVDA_ROWS, NVDA_HEADER))
        assert set(segs) == {"Data Center", "Gaming",
                             "Professional Visualization", "Automotive"}

    def test_totals_and_eliminations_are_excluded(self):
        """
        Leaving them in would double the total and make every share half its
        real value — a wrong answer that looks entirely plausible.
        """
        segs, _, _ = S._parse_table(table(NVDA_ROWS, NVDA_HEADER))
        assert "Total revenue" not in segs
        assert "Eliminations" not in segs

    @pytest.mark.parametrize("label", [
        "Total", "Consolidated revenue", "Eliminations", "Corporate",
        "Intersegment revenue", "Unallocated", "Reconciling items",
        "Adjustments",
    ])
    def test_structural_rows_are_skipped(self, label):
        rows = [["Alpha", "100"], ["Beta", "200"], [label, "300"]]
        segs, _, _ = S._parse_table(table(rows, ["Segment", "2025"]))
        assert label not in segs

    def test_takes_the_most_recent_period_column(self):
        segs, _, _ = S._parse_table(table(NVDA_ROWS, NVDA_HEADER))
        assert segs["Gaming"] == pytest.approx(11_350 * 1e6)   # not 10,447

    def test_period_label_is_captured(self):
        _, period, _ = S._parse_table(table(NVDA_ROWS, NVDA_HEADER))
        assert "2025" in period

    def test_a_leading_non_numeric_column_is_skipped(self):
        """Some filers put a units or footnote column before the figures."""
        rows = [["Alpha", "", "100"], ["Beta", "", "200"]]
        segs, _, _ = S._parse_table(table(rows, ["Segment", "Note", "2025"]))
        assert segs == {"Alpha": 100.0, "Beta": 200.0}

    def test_entities_and_nested_tags_are_stripped(self):
        rows = [["<b>Data&nbsp;Center</b>", "100"], ["Gaming&amp;Other", "200"]]
        segs, _, _ = S._parse_table(table(rows, ["Segment", "2025"]))
        assert "Data Center" in segs
        assert "Gaming&Other" in segs

    def test_a_single_segment_is_not_a_breakdown(self):
        rows = [["Alpha", "100"], ["Total", "100"]]
        assert S._parse_table(table(rows, ["Segment", "2025"]))[0] == {}

    def test_largest_table_wins(self):
        small = table([["Ignore", "1"]], ["x", "y"])
        assert "Data Center" in S._parse_table(small + table(NVDA_ROWS, NVDA_HEADER))[0]

    @pytest.mark.parametrize("html", [
        "", "<html><body>no tables here</body></html>",
        "<table></table>", "<table><tr></tr></table>",
        "<table><tr><td>only one row</td></tr></table>",
    ])
    def test_unparseable_input_yields_nothing_rather_than_raising(self, html):
        assert S._parse_table(html)[0] == {}

    def test_text_only_table_yields_nothing(self):
        """A policy note is prose in a table. It must not become fake segments."""
        rows = [["The Company operates in one segment.", "See Note 12"]]
        assert S._parse_table(table(rows, ["Segment", "Detail"]))[0] == {}


# ── Breakdown behaviour ──────────────────────────────────────────────────────
class TestBreakdown:
    def _nvda(self):
        segs, period, note = S._parse_table(table(NVDA_ROWS, NVDA_HEADER))
        return S.SegmentBreakdown("NVDA", segments=segs, period=period,
                                  scale_note=note, confidence="parsed")

    def test_shares_sum_to_one_hundred(self):
        assert sum(self._nvda().shares().values()) == pytest.approx(100.0)

    def test_concentration_is_detected(self):
        assert self._nvda().is_concentrated

    def test_even_split_is_not_concentrated(self):
        b = S.SegmentBreakdown("X", segments={"A": 50.0, "B": 50.0})
        assert not b.is_concentrated

    def test_empty_breakdown_is_safe(self):
        b = S.SegmentBreakdown("X")
        assert b.total is None
        assert b.shares() == {}
        assert not b.is_concentrated

    def test_zero_total_does_not_divide_by_zero(self):
        b = S.SegmentBreakdown("X", segments={"A": 0.0, "B": 0.0})
        assert b.shares() == {}


# ── End to end, offline ──────────────────────────────────────────────────────
class TestGetSegments:
    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch):
        monkeypatch.setattr(S.storage, "cache_get", lambda k: None)
        monkeypatch.setattr(S.storage, "cache_set", lambda *a, **k: None)

    def _wire(self, monkeypatch, summary, table_html, filing=True):
        import svp.rag.filings as F

        class Filing:
            accession, cik, period = "0001-23-000045", "0001045810", "2025-01-26"

        monkeypatch.setattr(F, "list_filings",
                            lambda *a, **k: ([Filing()] if filing else []))
        monkeypatch.setattr(S, "_fetch",
                            lambda url, key: (summary if "FilingSummary" in url
                                              else table_html))

    def test_happy_path(self, monkeypatch):
        self._wire(monkeypatch, TestReportSelection.SUMMARY,
                   table(NVDA_ROWS, NVDA_HEADER))
        out = S.get_segments("NVDA")
        assert out.confidence == "parsed"
        assert out.source_report.startswith("Revenue by Reportable Segment")
        assert out.total == pytest.approx(130_108 * 1e6)

    def test_no_filing_returns_empty(self, monkeypatch):
        self._wire(monkeypatch, "", "", filing=False)
        assert S.get_segments("NVDA").confidence == "none"

    def test_no_segment_report_explains_itself(self, monkeypatch):
        self._wire(monkeypatch,
                   "<FilingSummary><MyReports></MyReports></FilingSummary>", "")
        out = S.get_segments("NVDA")
        assert out.confidence == "none"
        assert "no table" in out.note.lower()

    def test_unparseable_table_keeps_the_html_and_says_so(self, monkeypatch):
        """The reader still gets the disclosure, just not the numbers."""
        self._wire(monkeypatch, TestReportSelection.SUMMARY,
                   "<table><tr><td>prose only</td></tr></table>")
        out = S.get_segments("NVDA")
        assert out.confidence == "table-only"
        assert out.segments == {}
        assert out.table_html

    def test_a_broken_filing_index_does_not_raise(self, monkeypatch):
        import svp.rag.filings as F

        def boom(*a, **k):
            raise RuntimeError("EDGAR down")

        monkeypatch.setattr(F, "list_filings", boom)
        assert S.get_segments("NVDA").confidence == "none"
