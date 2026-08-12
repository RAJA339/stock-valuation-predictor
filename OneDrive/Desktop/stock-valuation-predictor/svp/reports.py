"""
One-click PDF equity research reports.
======================================

Renders a formatted PDF research summary — ticker breakdown, valuation range,
signal, feature impacts (SHAP), DCF and peer multiples — using **ReportLab**.
Returns the PDF as raw ``bytes`` suitable for a Streamlit ``st.download_button``.
If ReportLab is unavailable, a plain-text report is returned instead so the
download button still works.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Optional

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )

    _HAS_REPORTLAB = True
except Exception:  # pragma: no cover
    _HAS_REPORTLAB = False

# Palette (hex) reused from the app theme.
_NAVY = "#0D1B2A"
_TEAL = "#028090"
_GREEN = "#02C39A"
_PANEL = "#1E293B"


def _fmt(v, kind="num"):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    if kind == "pct":
        return f"{v*100:.1f}%"
    if kind == "usd":
        return f"${v:,.2f}"
    return f"{v:,.2f}"


def build_report(
    ticker: str,
    company: str,
    market_price: float,
    valuation: dict,
    signal_text: str,
    attribution: Optional[pd.DataFrame] = None,
    dcf: Optional[dict] = None,
    peers: Optional[pd.DataFrame] = None,
    macro: Optional[dict] = None,
) -> bytes:
    """
    Assemble the equity report.

    ``valuation`` expects keys: point, low, median, high (floats).
    Returns PDF bytes (or UTF-8 text bytes if ReportLab is missing).
    """
    if not _HAS_REPORTLAB:
        return _text_report(ticker, company, market_price, valuation, signal_text, dcf, peers)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], textColor=colors.HexColor(_TEAL), fontSize=20)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=colors.HexColor(_NAVY), fontSize=13)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=styles["BodyText"], fontSize=8, textColor=colors.grey)

    story = []
    story.append(Paragraph("Intrinsic Stock Valuation — Equity Research Summary", title))
    story.append(Paragraph(f"{company} ({ticker}) &nbsp;·&nbsp; {date.today():%B %d, %Y}", small))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(_TEAL)))
    story.append(Spacer(1, 10))

    # ── Valuation summary ────────────────────────────────────────────────────
    story.append(Paragraph("Valuation Summary", h2))
    val_rows = [
        ["Metric", "Value"],
        ["Current Market Price", _fmt(market_price, "usd")],
        ["ML Intrinsic Value (point)", _fmt(valuation.get("point"), "usd")],
        [
            "ML Intrinsic Range (p10–p90)",
            f"{_fmt(valuation.get('low'), 'usd')} – {_fmt(valuation.get('high'), 'usd')}",
        ],
        ["Valuation Signal", signal_text.replace("🟢", "").replace("🔴", "").replace("🟡", "").strip()],
    ]
    if dcf:
        val_rows.append(["DCF Intrinsic Value", _fmt(dcf.get("intrinsic_per_share"), "usd")])
    story.append(_table(val_rows, [2.7 * inch, 3.0 * inch]))
    story.append(Spacer(1, 12))

    # ── Feature impacts ──────────────────────────────────────────────────────
    if attribution is not None and not attribution.empty:
        story.append(Paragraph("Top Feature Impacts (SHAP)", h2))
        rows = [["Feature", "Value", "Contribution to Value"]]
        for _, r in attribution.head(8).iterrows():
            rows.append([str(r["feature"]), _fmt(r["value"]), f"{r['contribution']:+.2f}"])
        story.append(_table(rows, [2.6 * inch, 1.5 * inch, 1.6 * inch]))
        story.append(Spacer(1, 12))

    # ── Peer benchmarking ────────────────────────────────────────────────────
    if peers is not None and not peers.empty:
        story.append(Paragraph("Peer Group Benchmarking", h2))
        cols = ["Ticker", "EV/EBITDA", "P/E", "Debt/Equity", "Profit Margin"]
        rows = [cols]
        for _, r in peers[cols].head(6).iterrows():
            rows.append(
                [
                    str(r["Ticker"]),
                    _fmt(r["EV/EBITDA"]),
                    _fmt(r["P/E"]),
                    _fmt(r["Debt/Equity"]),
                    _fmt(r["Profit Margin"], "pct"),
                ]
            )
        story.append(_table(rows, [1.1 * inch] + [1.15 * inch] * 4))
        story.append(Spacer(1, 12))

    # ── Macro ────────────────────────────────────────────────────────────────
    if macro:
        story.append(Paragraph("Macro Backdrop", h2))
        rows = [
            ["Indicator", "Level"],
            ["CPI", _fmt(macro.get("cpi"))],
            ["Fed Funds Rate", _fmt(macro.get("fed_funds"), "pct") if macro.get("fed_funds", 0) < 1 else f"{macro.get('fed_funds'):.2f}%"],
            ["10y–2y Yield Curve", f"{macro.get('yield_curve', 0):+.2f}"],
        ]
        story.append(_table(rows, [2.7 * inch, 3.0 * inch]))
        story.append(Spacer(1, 12))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(
        Paragraph(
            "Generated by the Intrinsic Stock Valuation Predictor · Webster University — "
            "MS Business Analytics. For educational purposes only. Not financial advice.",
            small,
        )
    )

    doc.build(story)
    return buf.getvalue()


def _table(rows, col_widths):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_NAVY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F7")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def _text_report(ticker, company, market_price, valuation, signal_text, dcf, peers) -> bytes:
    """Plain-text fallback when ReportLab isn't installed."""
    lines = [
        "INTRINSIC STOCK VALUATION — EQUITY RESEARCH SUMMARY",
        "=" * 55,
        f"{company} ({ticker})    {date.today():%Y-%m-%d}",
        "",
        f"Current Market Price : {_fmt(market_price, 'usd')}",
        f"ML Intrinsic (point) : {_fmt(valuation.get('point'), 'usd')}",
        f"ML Intrinsic Range   : {_fmt(valuation.get('low'), 'usd')} - {_fmt(valuation.get('high'), 'usd')}",
        f"Signal               : {signal_text}",
    ]
    if dcf:
        lines.append(f"DCF Intrinsic Value  : {_fmt(dcf.get('intrinsic_per_share'), 'usd')}")
    if peers is not None and not peers.empty:
        lines += ["", "PEER BENCHMARKING", "-" * 55, peers.to_string(index=False)]
    lines += ["", "For educational purposes only. Not financial advice."]
    return "\n".join(lines).encode("utf-8")


def has_reportlab() -> bool:
    return _HAS_REPORTLAB
