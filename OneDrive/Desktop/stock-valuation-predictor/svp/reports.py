"""
One-click PDF equity research reports.
======================================

Renders a formatted PDF research summary covering **every analysis tab** in the
app — valuation range and signal, SHAP feature impacts, DCF, peer multiples,
backtest performance, market regime and technical timing, forensic guardrails,
insider flow, position sizing, screener rank, filing divergence and the options
/ futures desk — using **ReportLab**.

Every section is optional: pass only what you have and the rest is skipped, so
the report degrades cleanly when a data source is unavailable. Returns the PDF
as raw ``bytes`` suitable for a Streamlit ``st.download_button``. If ReportLab
is unavailable, an equivalent plain-text report is returned instead so the
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
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas as _canvas
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        Image, PageBreak, KeepTogether,
    )

    _HAS_REPORTLAB = True
except Exception:  # pragma: no cover
    _HAS_REPORTLAB = False

# ── Print palette ────────────────────────────────────────────────────────────
# The app renders on obsidian; paper is the opposite problem. These are the
# same four hues re-valued for white stock, where a light emerald or champagne
# would be illegible: ink near-black, accents dark enough to hold on white.
_INK        = "#14181C"   # body text
_INK_SOFT   = "#4A5158"   # secondary text
_RULE       = "#D6D2CA"   # hairlines
_BAND       = "#F6F4F0"   # alternating row (warm, not blue-grey)
_HEAD_BG    = "#121417"   # table header — obsidian
_HEAD_FG    = "#F4F2ED"   # table header type — pearl
_EMERALD    = "#0E6B50"   # positive / brand rule
_GARNET     = "#9E3B33"   # negative — darkened for print
_CHAMPAGNE  = "#8A6D2F"   # accent type on white

# Legacy aliases retained for any caller still referencing them.
_NAVY, _TEAL, _GREEN, _PANEL = _HEAD_BG, _EMERALD, _EMERALD, _INK

_FONT = "Helvetica"
_FONT_B = "Helvetica-Bold"


def _fmt(v, kind="num"):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if not isinstance(v, (int, float)):
        return str(v)
    if kind == "pct":
        return f"{v*100:.1f}%"
    if kind == "usd":
        return f"${v:,.2f}"
    if kind == "big":
        return _human(v)
    return f"{v:,.2f}"


def _human(v: float) -> str:
    """Compact magnitude for balance-sheet scale numbers (1.23B, 456.7M)."""
    a = abs(v)
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= cutoff:
            return f"{v / cutoff:,.2f}{suffix}"
    return f"{v:,.2f}"


def _yn(v) -> str:
    """Render an optional boolean, distinguishing False from missing."""
    if v is None:
        return "N/A"
    return "Yes" if bool(v) else "No"


def _attr(obj, name, default=None):
    """Read an attribute off a result object, or a key off a dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _call(obj, name, default=None):
    """Invoke a zero-argument method if present (e.g. guardrail_triggered())."""
    fn = _attr(obj, name)
    if callable(fn):
        try:
            return fn()
        except Exception:
            return default
    return fn if fn is not None else default


# Human-friendly labels for the model-input feature table. Kept local rather
# than imported from svp.features to avoid a circular import at module load.
_LABELS = {
    "pe_ratio": "P/E Ratio",
    "roe": "Return on Equity",
    "roa": "Return on Assets",
    "debt_to_equity": "Debt / Equity",
    "fcf_yield": "FCF Yield",
    "profit_margin": "Profit Margin",
    "asset_turnover": "Asset Turnover",
    "revenue_yoy": "Revenue Growth (YoY)",
    "revenue_qoq": "Revenue Growth (QoQ)",
    "net_income_yoy": "Net Income Growth (YoY)",
    "fcf_yoy": "FCF Growth (YoY)",
    "sentiment": "Earnings Sentiment",
    "cpi": "CPI",
    "fed_funds": "Fed Funds Rate",
    "yield_curve": "Yield Curve (10y-2y)",
    "market_price": "Market Price",
}


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
    features: Optional[dict] = None,
    fundamentals: Optional[dict] = None,
    backtest: Optional[list] = None,
    regime=None,
    technical=None,
    sizing=None,
    quality=None,
    insider=None,
    filings=None,
    screener: Optional[pd.DataFrame] = None,
    options: Optional[dict] = None,
    excess_return: Optional[float] = None,
    chart_png: Optional[bytes] = None,
    chart_interval: Optional[str] = None,
    indicators=None,
    indicator_accuracy: Optional[list] = None,
    accuracy_horizon: Optional[int] = None,
) -> bytes:
    """
    Assemble the full equity report, one section per analysis tab.

    ``valuation`` expects keys: point, low, median, high (floats). Every other
    argument is optional — a section is emitted only when its data is supplied,
    so callers can build a partial report without special-casing.

    The typed arguments accept the app's own result objects:
    ``regime`` a ``RegimeResult``, ``technical`` a ``TechnicalSignals``,
    ``sizing`` a ``SizingResult``, ``quality`` a ``QualityScores``,
    ``insider`` an ``InsiderSignal``, ``filings`` a ``FilingDivergence`` and
    ``backtest`` a list of ``HorizonResult``. ``options`` is a plain dict from
    the derivatives tab.

    Returns PDF bytes (or UTF-8 text bytes if ReportLab is missing).
    """
    if not _HAS_REPORTLAB:
        return _text_report(
            ticker, company, market_price, valuation, signal_text, dcf, peers,
            features, fundamentals, backtest, regime, technical, sizing,
            quality, insider, filings, screener, options, excess_return, macro,
            indicators, indicator_accuracy, chart_interval, accuracy_horizon,
        )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.78 * inch, bottomMargin=0.85 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        title=f"{ticker} — Equity Research", author="Intrinsic Stock Valuation Predictor",
        subject=f"Equity research summary for {company}",
    )
    avail = doc.width

    title = ParagraphStyle("t", fontName=_FONT_B, fontSize=21, leading=24,
                           textColor=colors.HexColor(_INK), spaceAfter=0)
    kicker = ParagraphStyle("k", fontName=_FONT_B, fontSize=8, leading=10,
                            textColor=colors.HexColor(_EMERALD), spaceAfter=4)
    sub = ParagraphStyle("sub", fontName=_FONT, fontSize=9.5, leading=12.5,
                         textColor=colors.HexColor(_INK_SOFT))
    h2 = ParagraphStyle("h2", fontName=_FONT_B, fontSize=10.5, leading=13,
                        textColor=colors.HexColor(_INK),
                        spaceBefore=15, spaceAfter=7)
    body = ParagraphStyle("b", fontName=_FONT, fontSize=9, leading=12.6,
                          textColor=colors.HexColor(_INK), spaceAfter=4)
    small = ParagraphStyle("s", fontName=_FONT, fontSize=7.6, leading=10,
                           textColor=colors.HexColor(_INK_SOFT))

    # Section numbering, so a reader can cite "see §4".
    counter = {"n": 0}

    def section(label: str):
        counter["n"] += 1
        return Paragraph(f'<font color="{_EMERALD}">{counter["n"]}.</font>&nbsp;&nbsp;'
                         f'{label.upper()}', h2)

    story = []

    # ── Masthead ─────────────────────────────────────────────────────────────
    story.append(Paragraph("EQUITY RESEARCH — INTRINSIC VALUATION", kicker))
    story.append(Paragraph(f"{company} <font color='{_INK_SOFT}'>({ticker})</font>", title))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        f"{date.today():%d %B %Y}&nbsp;&nbsp;·&nbsp;&nbsp;"
        f"Last price {_fmt(market_price, 'usd')}&nbsp;&nbsp;·&nbsp;&nbsp;"
        f"Model-derived, machine-generated", sub))
    story.append(Spacer(1, 7))
    story.append(_rule(_EMERALD, 1.6, 4))
    story.append(_rule(_RULE, 0.5, 12))

    # ── At a glance ──────────────────────────────────────────────────────────
    _sig = signal_text.replace("🟢", "").replace("🔴", "").replace("🟡", "").strip()
    _upside = ((valuation.get("point") or 0) / market_price - 1) if market_price else None
    glance = [
        ["Market Price", "Intrinsic Value", "Implied Upside", "Signal"],
        [_fmt(market_price, "usd"), _fmt(valuation.get("point"), "usd"),
         _fmt(_upside, "pct") if _upside is not None else "N/A", _sig],
    ]
    gt = Table(glance, colWidths=[avail / 4.0] * 4, hAlign="LEFT")
    gt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), _FONT_B),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(_INK_SOFT)),
        ("FONTNAME", (0, 1), (-1, 1), _FONT_B),
        ("FONTSIZE", (0, 1), (-1, 1), 15),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor(_INK)),
        ("TEXTCOLOR", (2, 1), (2, 1),
         colors.HexColor(_EMERALD if (_upside or 0) >= 0 else _GARNET)),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 1), (-1, 1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("LINEAFTER", (0, 0), (-2, -1), 0.5, colors.HexColor(_RULE)),
        ("RIGHTPADDING", (0, 0), (-2, -1), 12),
        ("LEFTPADDING", (1, 0), (-1, -1), 12),
    ]))
    story.append(gt)
    story.append(Spacer(1, 6))
    story.append(_rule(_RULE, 0.5, 10))

    # ── Valuation summary ────────────────────────────────────────────────────
    story.append(section("Valuation Summary"))
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
    if excess_return is not None:
        val_rows.append(["Predicted Excess Return vs Benchmark", _fmt(excess_return, "pct")])
    story.append(_table(val_rows, [2.7 * inch, 3.0 * inch]))
    story.append(Spacer(1, 12))

    # ── Fundamentals / extracted features ────────────────────────────────────
    if fundamentals:
        story.append(section("Company Fundamentals"))
        rows = [["Item", "Value"]]
        for label, key, kind in (
            ("Sector", "sector", "raw"),
            ("Revenue", "revenue", "big"),
            ("Net Income", "net_income", "big"),
            ("Total Assets", "total_assets", "big"),
            ("Shareholders' Equity", "equity", "big"),
            ("Operating Cash Flow", "op_cash_flow", "big"),
            ("Free Cash Flow", "free_cash_flow", "big"),
            ("Long-Term Debt", "long_term_debt", "big"),
            ("Shares Outstanding", "shares", "big"),
            ("Market Cap", "market_cap", "big"),
            ("EPS", "eps", "usd"),
        ):
            v = fundamentals.get(key)
            if v is None:
                continue
            rows.append([label, str(v) if kind == "raw" else _fmt(v, kind)])
        if len(rows) > 1:
            story.append(_table(rows, [2.7 * inch, 3.0 * inch]))
            story.append(Spacer(1, 12))

    if features:
        story.append(section("Model Input Features"))
        rows = [["Feature", "Value"]]
        pct_like = {
            "roe", "roa", "fcf_yield", "profit_margin", "revenue_yoy", "revenue_qoq",
            "net_income_yoy", "fcf_yoy",
        }
        for k, v in features.items():
            if k.startswith("_"):
                continue
            rows.append([_LABELS.get(k, k), _fmt(v, "pct" if k in pct_like else "num")])
        if len(rows) > 1:
            story.append(_table(rows, [2.7 * inch, 3.0 * inch]))
            story.append(Spacer(1, 12))

    # ── Feature impacts ──────────────────────────────────────────────────────
    if attribution is not None and not attribution.empty:
        story.append(section("Top Feature Impacts (SHAP)"))
        rows = [["Feature", "Value", "Contribution to Value"]]
        for _, r in attribution.head(8).iterrows():
            rows.append([str(r["feature"]), _fmt(r["value"]), f"{r['contribution']:+.2f}"])
        story.append(_table(rows, [2.6 * inch, 1.5 * inch, 1.6 * inch]))
        story.append(Spacer(1, 12))

    # ── Peer benchmarking ────────────────────────────────────────────────────
    if peers is not None and not peers.empty:
        story.append(section("Peer Group Benchmarking"))
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

    # ── Technical chart + indicators ─────────────────────────────────────────
    if chart_png or indicators is not None or indicator_accuracy:
        story.append(PageBreak())
        story.append(section(
            f"Technical Analysis{f' — {chart_interval} bars' if chart_interval else ''}"))

        if chart_png:
            try:
                img = Image(io.BytesIO(chart_png))
                # Scale to the text width, preserving aspect ratio.
                avail = 6.9 * inch
                ratio = img.imageHeight / float(img.imageWidth)
                img.drawWidth, img.drawHeight = avail, avail * ratio
                story.append(img)
                story.append(Spacer(1, 8))
            except Exception:
                pass

        if indicators is not None and getattr(indicators, "signals", None):
            story.append(Paragraph(
                f"<b>Consensus: {indicators.consensus}</b> — "
                f"{indicators.bullish} bullish, {indicators.neutral} neutral, "
                f"{indicators.bearish} bearish across {len(indicators.signals)} indicators "
                f"(net score {indicators.net_score:+.2f}).", body))
            story.append(Spacer(1, 6))
            rows = [["Indicator", "Category", "Value", "Signal", "Reading"]]
            for s in indicators.signals:
                rows.append([s.name, s.category, s.display, s.call, s.note[:52]])
            story.append(_table(rows, [1.45 * inch, 0.8 * inch, 0.8 * inch,
                                       0.7 * inch, 2.95 * inch]))
            story.append(Spacer(1, 12))

        if indicator_accuracy:
            story.append(section("Measured Signal Accuracy"))
            story.append(Paragraph(
                f"Each indicator's call is recomputed at every bar and scored against the "
                f"actual move {accuracy_horizon or 6} bars later. These are realised hit rates "
                f"on this ticker and interval, not vendor claims. <b>Verdict follows the 95% "
                f"confidence interval, not the point estimate</b> — a 56% hit rate on 200 "
                f"signals spans roughly 49–63%, which includes a coin flip. Costs are not "
                f"modelled: spread and commission can turn a directional edge into a loss.",
                small))
            story.append(Spacer(1, 6))
            rows = [["Indicator", "Signals", "Hit Rate", "95% CI", "Avg Fwd", "Verdict"]]
            for a in indicator_accuracy:
                nan = a.hit_rate != a.hit_rate
                rows.append([
                    a.name, str(a.n_signals),
                    "—" if nan else f"{a.hit_rate * 100:.1f}%",
                    "—" if nan else f"{a.ci_low * 100:.0f}–{a.ci_high * 100:.0f}%",
                    "—" if a.avg_forward_return != a.avg_forward_return
                        else f"{a.avg_forward_return * 100:+.3f}%",
                    a.verdict,
                ])
            story.append(_table(rows, [1.6 * inch, 0.65 * inch, 0.75 * inch,
                                       0.85 * inch, 0.8 * inch, 1.45 * inch]))
            story.append(Spacer(1, 12))

    # ── Backtesting ──────────────────────────────────────────────────────────
    if backtest:
        story.append(section("Signal Backtest"))
        rows = [["Horizon", "Signals", "Hit Rate", "Avg Fwd Return", "Strategy", "Buy & Hold"]]
        for hr in backtest:
            rows.append([
                f"{_attr(hr, 'horizon_years'):.0f}y",
                str(_attr(hr, "n_signals")),
                _fmt(_attr(hr, "hit_rate"), "pct"),
                _fmt(_attr(hr, "avg_forward_return"), "pct"),
                _fmt(_attr(hr, "strategy_return"), "pct"),
                _fmt(_attr(hr, "buy_hold_return"), "pct"),
            ])
        story.append(_table(rows, [0.75 * inch, 0.75 * inch, 0.9 * inch, 1.2 * inch, 1.0 * inch, 1.1 * inch]))
        story.append(Spacer(1, 12))

    # ── Execution & timing ───────────────────────────────────────────────────
    if regime is not None or technical is not None:
        story.append(section("Execution &amp; Timing"))
        rows = [["Signal", "Reading"]]
        if regime is not None:
            rows += [
                ["Market Regime", str(_attr(regime, "regime"))],
                ["Regime Confidence", _fmt(_attr(regime, "confidence"), "pct")],
                ["Signal Scaler", _fmt(_attr(regime, "signal_scaler"))],
                ["VIX", _fmt(_attr(regime, "vix"))],
                ["Regime Source", str(_attr(regime, "source"))],
            ]
            desc = _attr(regime, "description")
            if desc:
                rows.append(["Regime Note", str(desc)])
        if technical is not None:
            rows += [
                ["Price vs 200-day SMA", _fmt(_attr(technical, "sma200"), "usd")],
                ["Trend Up", _yn(_attr(technical, "trend_up"))],
                ["RSI", _fmt(_attr(technical, "rsi"))],
                ["RSI Bullish Divergence", _yn(_attr(technical, "rsi_bullish_divergence"))],
                ["Volume Point of Control", _fmt(_attr(technical, "volume_poc"), "usd")],
                ["Near Volume Support", _yn(_attr(technical, "near_volume_support"))],
                ["ATR", _fmt(_attr(technical, "atr"), "usd")],
                ["Entry Confirmed", _yn(_attr(technical, "confirmed"))],
            ]
        story.append(_table(rows, [2.7 * inch, 3.0 * inch]))
        story.append(Spacer(1, 12))

    # ── Position sizing ──────────────────────────────────────────────────────
    if sizing is not None:
        story.append(section("Position Sizing &amp; Risk"))
        rows = [
            ["Metric", "Value"],
            ["Win Probability (Monte-Carlo)", _fmt(_attr(sizing, "win_probability"), "pct")],
            ["Payoff Ratio", _fmt(_attr(sizing, "payoff_ratio"))],
            ["Full Kelly", _fmt(_attr(sizing, "kelly_fraction"), "pct")],
            ["Recommended Weight (fractional Kelly)", _fmt(_attr(sizing, "fractional_kelly"), "pct")],
            ["Volatility-Parity Weight", _fmt(_attr(sizing, "vol_parity_weight"), "pct")],
            ["Stop Loss", _fmt(_attr(sizing, "stop_loss"), "usd")],
            ["Take Profit", _fmt(_attr(sizing, "take_profit"), "usd")],
            ["Risk per Share", _fmt(_attr(sizing, "risk_per_share"), "usd")],
            ["Reward / Risk Ratio", _fmt(_attr(sizing, "reward_risk_ratio"))],
        ]
        story.append(_table(rows, [2.7 * inch, 3.0 * inch]))
        story.append(Spacer(1, 12))

    # ── Forensic guardrails ──────────────────────────────────────────────────
    if quality is not None or insider is not None:
        story.append(section("Guardrails — Quality, Distress &amp; Insider Flow"))
        rows = [["Check", "Result"]]
        if quality is not None:
            piotroski = _attr(quality, "piotroski")
            rows += [
                ["Piotroski F-Score", f"{piotroski} / 9" if piotroski is not None else "N/A"],
                ["Altman Z-Score", _fmt(_attr(quality, "altman_z"))],
                ["Altman Zone", str(_attr(quality, "altman_zone") or "N/A")],
                ["Beneish M-Score", _fmt(_attr(quality, "beneish_m"))],
                ["Beneish Flag", str(_attr(quality, "beneish_flag") or "N/A")],
                ["Guardrail Triggered", _yn(_call(quality, "guardrail_triggered"))],
            ]
        if insider is not None:
            rows += [
                ["Insider Net Direction", str(_attr(insider, "net_direction") or "N/A")],
                ["Insider Net Shares", _fmt(_attr(insider, "net_shares"), "big")],
                ["Insider Buys / Sells",
                 f"{_attr(insider, 'buy_count')} / {_attr(insider, 'sell_count')}"],
                ["Lookback Window", f"{_attr(insider, 'window_days')} days"],
                ["Short % of Float", _fmt(_attr(insider, "short_percent_float"), "pct")],
            ]
        story.append(_table(rows, [2.7 * inch, 3.0 * inch]))

        notes = _attr(quality, "notes") if quality is not None else None
        if notes:
            story.append(Spacer(1, 4))
            story.append(Paragraph("Notes: " + "; ".join(str(n) for n in notes), small))
        story.append(Spacer(1, 12))

    # ── Screener ─────────────────────────────────────────────────────────────
    if screener is not None and not screener.empty:
        story.append(section("Margin-of-Safety Screener"))
        cols = [c for c in screener.columns if c.lower() != "index"][:5]
        rows = [[str(c) for c in cols]]
        for _, r in screener.head(10).iterrows():
            rows.append([_fmt(r[c]) if isinstance(r[c], (int, float)) else str(r[c]) for c in cols])
        story.append(_table(rows, [5.7 * inch / len(cols)] * len(cols)))
        story.append(Spacer(1, 12))

    # ── Filing divergence ────────────────────────────────────────────────────
    if filings is not None and _attr(filings, "similarity") is not None:
        story.append(section("Filing Divergence (10-K / 10-Q)"))
        rows = [
            ["Metric", "Value"],
            ["Cosine Similarity vs Prior Filing", _fmt(_attr(filings, "similarity"))],
            ["Divergence", _fmt(_attr(filings, "divergence"))],
            ["Material Change Alert", _yn(_attr(filings, "alert"))],
        ]
        story.append(_table(rows, [2.7 * inch, 3.0 * inch]))
        added = _attr(filings, "added_terms") or []
        dropped = _attr(filings, "dropped_terms") or []
        if added:
            story.append(Spacer(1, 4))
            story.append(Paragraph("<b>New emphasis:</b> " + ", ".join(map(str, added[:15])), body))
        if dropped:
            story.append(Paragraph("<b>De-emphasized:</b> " + ", ".join(map(str, dropped[:15])), body))
        story.append(Spacer(1, 12))

    # ── Options & futures ────────────────────────────────────────────────────
    if options:
        story.append(section("Options &amp; Futures"))
        rows = [["Metric", "Value"]]
        for label, key, kind in (
            ("Expiry", "expiry", "raw"),
            ("Days to Expiry", "days_to_expiry", "num"),
            ("Chain Source", "source", "raw"),
            ("Spot", "spot", "usd"),
            ("Contract Type", "kind", "raw"),
            ("Strike", "strike", "usd"),
            ("Volatility Input", "sigma", "pct"),
            ("Black-Scholes Value", "bs_price", "usd"),
            ("Binomial (American)", "binomial_price", "usd"),
            ("Monte-Carlo Value", "mc_price", "usd"),
            ("Delta", "delta", "num"),
            ("Gamma", "gamma", "num"),
            ("Theta (per day)", "theta", "num"),
            ("Vega (per vol pt)", "vega", "num"),
            ("Rho (per 1% rate)", "rho", "num"),
            ("Projected S_T", "projected_st", "usd"),
            ("Edge Method", "edge_method", "raw"),
            ("Gap Closed by Expiry", "convergence", "pct"),
            ("Terminal Volatility", "terminal_vol", "pct"),
            ("Best Contract Strike", "best_strike", "usd"),
            ("Best Contract Premium", "best_premium", "usd"),
            ("Best Contract P(ITM)", "best_prob_itm", "pct"),
            ("Expected Edge", "best_edge", "usd"),
            ("View Premium vs Risk-Neutral", "best_view_premium", "usd"),
            ("Verdict", "best_verdict", "raw"),
            ("Futures Fair Value", "futures_fair_value", "usd"),
            ("Futures Basis", "futures_basis", "usd"),
            ("Net Carry", "futures_carry", "pct"),
        ):
            v = options.get(key)
            if v is None:
                continue
            rows.append([label, str(v) if kind == "raw" else _fmt(v, kind)])
        if len(rows) > 1:
            story.append(_table(rows, [2.7 * inch, 3.0 * inch]))
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                "Model edge is a point-estimate payoff at the projected target, not a "
                "risk-adjusted expectation: it ignores the full terminal distribution and "
                "assumes the target is reached exactly at expiry. Treat it as a screen.",
                small,
            ))
            story.append(Spacer(1, 12))

    # ── Macro ────────────────────────────────────────────────────────────────
    if macro:
        story.append(section("Macro Backdrop"))
        rows = [
            ["Indicator", "Level"],
            ["CPI", _fmt(macro.get("cpi"))],
            ["Fed Funds Rate", _fmt(macro.get("fed_funds"), "pct") if macro.get("fed_funds", 0) < 1 else f"{macro.get('fed_funds'):.2f}%"],
            ["10y–2y Yield Curve", f"{macro.get('yield_curve', 0):+.2f}"],
        ]
        story.append(_table(rows, [2.7 * inch, 3.0 * inch]))
        story.append(Spacer(1, 12))

    # ── Basis & disclaimer ───────────────────────────────────────────────────
    story.append(Spacer(1, 14))
    story.append(_rule(_RULE, 0.5, 8))
    story.append(section("Basis of Preparation & Disclaimer"))
    story.append(Paragraph(
        "<b>Basis.</b> Fundamentals are taken from SEC EDGAR XBRL company facts "
        "(10-K, 20-F, 40-F and 10-Q as filed). Prices, volumes and option chains come "
        "from public market data feeds and may be delayed. Macro series are sourced "
        "from FRED and BLS. The intrinsic value is produced by a gradient-boosted "
        "regression trained on synthetic data, reported as a p10–p90 range rather "
        "than a point, and is a model output — not a price target.", body))
    story.append(Paragraph(
        "<b>Limitations.</b> Where a filing does not tag a concept this parser "
        "recognises, the corresponding input is absent and any ratio built on it is "
        "omitted. Backtested and measured-accuracy figures are in-sample statistics on "
        "the data shown, carry sampling error, and exclude commission, spread, slippage "
        "and financing. Past performance does not indicate future results.", body))
    story.append(Paragraph(
        "<b>Disclaimer.</b> This document is generated automatically for educational "
        "and academic purposes. It is not investment advice, not a recommendation or "
        "solicitation to buy or sell any security, and has not been prepared in "
        "accordance with regulatory requirements for investment research. No "
        "representation is made as to its accuracy or completeness. Recipients should "
        "conduct their own analysis and consult a qualified financial adviser.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Generated by the Intrinsic Stock Valuation Predictor · Webster University — "
        f"MS Business Analytics · {date.today():%d %b %Y}", small))

    meta = {"ticker": ticker, "company": company, "date": f"{date.today():%d %b %Y}"}

    def _mk_canvas(*args, **kwargs):
        return _NumberedCanvas(*args, meta=meta, **kwargs)

    doc.build(story, canvasmaker=_mk_canvas)
    return buf.getvalue()


# Cell styles are built once and reused — ParagraphStyle construction is not
# free and a full report renders a few hundred cells.
def _cell_styles():
    return {
        "head": ParagraphStyle("th", fontName=_FONT_B, fontSize=7.6, leading=9.5,
                               textColor=colors.HexColor(_HEAD_FG),
                               spaceBefore=0, spaceAfter=0),
        "body": ParagraphStyle("td", fontName=_FONT, fontSize=8.2, leading=10.6,
                               textColor=colors.HexColor(_INK),
                               spaceBefore=0, spaceAfter=0),
        "label": ParagraphStyle("tdl", fontName=_FONT_B, fontSize=8.2, leading=10.6,
                                textColor=colors.HexColor(_INK),
                                spaceBefore=0, spaceAfter=0),
        "num": ParagraphStyle("tdn", fontName=_FONT, fontSize=8.2, leading=10.6,
                              alignment=TA_RIGHT, textColor=colors.HexColor(_INK),
                              spaceBefore=0, spaceAfter=0),
        "head_num": ParagraphStyle("thn", fontName=_FONT_B, fontSize=7.6, leading=9.5,
                                   alignment=TA_RIGHT, textColor=colors.HexColor(_HEAD_FG),
                                   spaceBefore=0, spaceAfter=0),
    }


_CELL = None


def _numeric(text: str) -> bool:
    """Does this cell hold a figure? Numeric columns are right-aligned."""
    t = str(text).strip().lstrip("+-$").rstrip("%BKMT").replace(",", "").replace("/", "")
    if not t:
        return False
    try:
        float(t.split()[0]) if t.split() else float(t)
        return True
    except ValueError:
        return False


def _table(rows, col_widths, first_col_label: bool = True):
    """
    Build a table whose cells wrap.

    Passing bare strings to ReportLab is what caused text to run past the cell
    edge: a plain string is drawn on one line with no knowledge of the column
    width. Wrapping every cell in a Paragraph lets the layout engine break the
    line, which is why long values like the regime note now stay inside the
    rule.
    """
    global _CELL
    if _CELL is None:
        _CELL = _cell_styles()

    # Decide each column's alignment from its body cells first, so the header
    # can match it. A right-aligned figure under a left-aligned label reads as
    # a misalignment rather than a deliberate choice.
    n_cols = max(len(r) for r in rows)
    body_rows = rows[1:]
    right_col = []
    for c in range(n_cols):
        vals = [str(r[c]) for r in body_rows if c < len(r) and r[c] not in (None, "")]
        numeric = [v for v in vals if _numeric(v)]
        right_col.append(bool(vals) and len(numeric) >= max(1, len(vals) * 0.6)
                         and not (c == 0 and first_col_label))

    data = []
    for r, row in enumerate(rows):
        out = []
        for c, val in enumerate(row):
            text = "" if val is None else str(val)
            if r == 0:
                style = _CELL["head_num"] if right_col[c] else _CELL["head"]
            elif c == 0 and first_col_label:
                style = _CELL["label"]
            elif right_col[c]:
                style = _CELL["num"]
            else:
                style = _CELL["body"]
            out.append(Paragraph(text.replace("&", "&amp;"), style))
        data.append(out)

    t = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        # Header: obsidian band, no vertical rules — the eye follows rows.
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_HEAD_BG)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, colors.HexColor(_EMERALD)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(_BAND)]),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor(_RULE)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.HexColor(_RULE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def _rule(colour=_RULE, thickness=0.6, space_after=8):
    return HRFlowable(width="100%", thickness=thickness,
                      color=colors.HexColor(colour), spaceAfter=space_after)


class _NumberedCanvas(_canvas.Canvas):
    """
    Two-pass canvas so the footer can print "Page 2 of 7".

    Pages are buffered on the first pass and the total is only known once the
    document is complete, which is why the count is stamped on the second.
    """

    def __init__(self, *args, **kwargs):
        self._meta = kwargs.pop("meta", {})
        super().__init__(*args, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._furniture(total)
            super().showPage()
        super().save()

    def _furniture(self, total):
        w, h = letter
        m = 0.75 * inch
        meta = self._meta

        # Running header — suppressed on page 1, which carries the masthead.
        if self._pageNumber > 1:
            self.setFont(_FONT_B, 7.5)
            self.setFillColor(colors.HexColor(_INK))
            self.drawString(m, h - 0.5 * inch, meta.get("ticker", ""))
            self.setFont(_FONT, 7.5)
            self.setFillColor(colors.HexColor(_INK_SOFT))
            self.drawString(m + 0.42 * inch, h - 0.5 * inch, meta.get("company", ""))
            self.drawRightString(w - m, h - 0.5 * inch, meta.get("date", ""))
            self.setStrokeColor(colors.HexColor(_RULE))
            self.setLineWidth(0.5)
            self.line(m, h - 0.56 * inch, w - m, h - 0.56 * inch)

        # Footer rule + page count + standing disclaimer.
        self.setStrokeColor(colors.HexColor(_RULE))
        self.setLineWidth(0.5)
        self.line(m, 0.62 * inch, w - m, 0.62 * inch)
        self.setFont(_FONT, 7)
        self.setFillColor(colors.HexColor(_INK_SOFT))
        self.drawString(m, 0.46 * inch,
                        "Educational use only. Not investment advice.")
        self.drawRightString(w - m, 0.46 * inch,
                             f"Page {self._pageNumber} of {total}")


def _text_report(
    ticker, company, market_price, valuation, signal_text, dcf, peers,
    features=None, fundamentals=None, backtest=None, regime=None, technical=None,
    sizing=None, quality=None, insider=None, filings=None, screener=None,
    options=None, excess_return=None, macro=None,
    indicators=None, indicator_accuracy=None, chart_interval=None,
    accuracy_horizon=None,
) -> bytes:
    """Plain-text fallback when ReportLab isn't installed — same coverage as the PDF."""
    W = 62
    lines = [
        "INTRINSIC STOCK VALUATION — EQUITY RESEARCH SUMMARY",
        "=" * W,
        f"{company} ({ticker})    {date.today():%Y-%m-%d}",
        "",
        "VALUATION",
        "-" * W,
        f"Current Market Price : {_fmt(market_price, 'usd')}",
        f"ML Intrinsic (point) : {_fmt(valuation.get('point'), 'usd')}",
        f"ML Intrinsic Range   : {_fmt(valuation.get('low'), 'usd')} - {_fmt(valuation.get('high'), 'usd')}",
        f"Signal               : {signal_text}",
    ]
    if dcf:
        lines.append(f"DCF Intrinsic Value  : {_fmt(dcf.get('intrinsic_per_share'), 'usd')}")
    if excess_return is not None:
        lines.append(f"Excess Return vs Bmk : {_fmt(excess_return, 'pct')}")

    def section(title, pairs):
        rows = [(k, v) for k, v in pairs if v is not None]
        if not rows:
            return
        lines.extend(["", title, "-" * W])
        width = max(len(k) for k, _ in rows)
        lines.extend(f"{k:<{width}} : {v}" for k, v in rows)

    if fundamentals:
        section("FUNDAMENTALS", [
            ("Sector", fundamentals.get("sector")),
            ("Revenue", _fmt(fundamentals.get("revenue"), "big")),
            ("Net Income", _fmt(fundamentals.get("net_income"), "big")),
            ("Free Cash Flow", _fmt(fundamentals.get("free_cash_flow"), "big")),
            ("Market Cap", _fmt(fundamentals.get("market_cap"), "big")),
            ("EPS", _fmt(fundamentals.get("eps"), "usd")),
        ])

    if features:
        section("MODEL INPUT FEATURES",
                [(_LABELS.get(k, k), _fmt(v)) for k, v in features.items() if not k.startswith("_")])

    if backtest:
        lines.extend(["", "SIGNAL BACKTEST", "-" * W])
        for hr in backtest:
            lines.append(
                f"{_attr(hr, 'horizon_years'):.0f}y  signals={_attr(hr, 'n_signals')}  "
                f"hit={_fmt(_attr(hr, 'hit_rate'), 'pct')}  "
                f"strategy={_fmt(_attr(hr, 'strategy_return'), 'pct')}  "
                f"buy&hold={_fmt(_attr(hr, 'buy_hold_return'), 'pct')}"
            )

    if regime is not None or technical is not None:
        section("EXECUTION & TIMING", [
            ("Market Regime", _attr(regime, "regime")),
            ("Regime Confidence", _fmt(_attr(regime, "confidence"), "pct") if regime is not None else None),
            ("Signal Scaler", _fmt(_attr(regime, "signal_scaler")) if regime is not None else None),
            ("VIX", _fmt(_attr(regime, "vix")) if regime is not None else None),
            ("200-day SMA", _fmt(_attr(technical, "sma200"), "usd") if technical is not None else None),
            ("Trend Up", _yn(_attr(technical, "trend_up")) if technical is not None else None),
            ("RSI", _fmt(_attr(technical, "rsi")) if technical is not None else None),
            ("ATR", _fmt(_attr(technical, "atr"), "usd") if technical is not None else None),
            ("Entry Confirmed", _yn(_attr(technical, "confirmed")) if technical is not None else None),
        ])

    if sizing is not None:
        section("POSITION SIZING & RISK", [
            ("Win Probability", _fmt(_attr(sizing, "win_probability"), "pct")),
            ("Payoff Ratio", _fmt(_attr(sizing, "payoff_ratio"))),
            ("Full Kelly", _fmt(_attr(sizing, "kelly_fraction"), "pct")),
            ("Recommended Weight", _fmt(_attr(sizing, "fractional_kelly"), "pct")),
            ("Stop Loss", _fmt(_attr(sizing, "stop_loss"), "usd")),
            ("Take Profit", _fmt(_attr(sizing, "take_profit"), "usd")),
            ("Reward / Risk", _fmt(_attr(sizing, "reward_risk_ratio"))),
        ])

    if quality is not None or insider is not None:
        piotroski = _attr(quality, "piotroski")
        section("GUARDRAILS", [
            ("Piotroski F-Score", f"{piotroski} / 9" if piotroski is not None else None),
            ("Altman Z-Score", _fmt(_attr(quality, "altman_z")) if quality is not None else None),
            ("Altman Zone", _attr(quality, "altman_zone")),
            ("Beneish M-Score", _fmt(_attr(quality, "beneish_m")) if quality is not None else None),
            ("Beneish Flag", _attr(quality, "beneish_flag")),
            ("Guardrail Triggered", _yn(_call(quality, "guardrail_triggered")) if quality is not None else None),
            ("Insider Direction", _attr(insider, "net_direction")),
            ("Short % of Float", _fmt(_attr(insider, "short_percent_float"), "pct") if insider is not None else None),
        ])

    if filings is not None and _attr(filings, "similarity") is not None:
        section("FILING DIVERGENCE", [
            ("Cosine Similarity", _fmt(_attr(filings, "similarity"))),
            ("Divergence", _fmt(_attr(filings, "divergence"))),
            ("Material Change", _yn(_attr(filings, "alert"))),
            ("New emphasis", ", ".join(map(str, (_attr(filings, "added_terms") or [])[:10])) or None),
            ("De-emphasized", ", ".join(map(str, (_attr(filings, "dropped_terms") or [])[:10])) or None),
        ])

    if options:
        section("OPTIONS & FUTURES", [
            ("Expiry", options.get("expiry")),
            ("Chain Source", options.get("source")),
            ("Spot", _fmt(options.get("spot"), "usd")),
            ("Strike", _fmt(options.get("strike"), "usd")),
            ("Black-Scholes", _fmt(options.get("bs_price"), "usd")),
            ("Binomial (American)", _fmt(options.get("binomial_price"), "usd")),
            ("Monte-Carlo", _fmt(options.get("mc_price"), "usd")),
            ("Delta", _fmt(options.get("delta"))),
            ("Gamma", _fmt(options.get("gamma"))),
            ("Theta (per day)", _fmt(options.get("theta"))),
            ("Vega", _fmt(options.get("vega"))),
            ("Rho", _fmt(options.get("rho"))),
            ("Projected S_T", _fmt(options.get("projected_st"), "usd")),
            ("Best Strike", _fmt(options.get("best_strike"), "usd")),
            ("Model Edge", _fmt(options.get("best_edge"), "usd")),
            ("Verdict", options.get("best_verdict")),
            ("Futures Fair Value", _fmt(options.get("futures_fair_value"), "usd")),
            ("Futures Basis", _fmt(options.get("futures_basis"), "usd")),
        ])

    if indicators is not None and getattr(indicators, "signals", None):
        lines.extend(["", f"TECHNICAL INDICATORS ({chart_interval or 'intraday'})", "-" * W,
                      f"Consensus: {indicators.consensus}  "
                      f"({indicators.bullish} bull / {indicators.neutral} neutral / "
                      f"{indicators.bearish} bear, net {indicators.net_score:+.2f})"])
        width = max(len(s.name) for s in indicators.signals)
        for s_ in indicators.signals:
            lines.append(f"{s_.name:<{width}} : {s_.display:>10}  {s_.call:<8} {s_.note}")

    if indicator_accuracy:
        lines.extend(["", f"MEASURED SIGNAL ACCURACY ({accuracy_horizon or 6} bars forward)",
                      "-" * W,
                      "Verdict follows the 95% CI, not the point estimate. Costs not modelled."])
        width = max(len(a.name) for a in indicator_accuracy)
        for a in indicator_accuracy:
            nan = a.hit_rate != a.hit_rate
            rate = "n/a" if nan else f"{a.hit_rate * 100:5.1f}%"
            ci = "" if nan else f" [{a.ci_low * 100:.0f}-{a.ci_high * 100:.0f}%]"
            lines.append(f"{a.name:<{width}} : {rate}{ci:<14} n={a.n_signals:<5} {a.verdict}")

    if macro:
        section("MACRO BACKDROP", [
            ("CPI", _fmt(macro.get("cpi"))),
            ("Fed Funds Rate", _fmt(macro.get("fed_funds"))),
            ("10y-2y Yield Curve", _fmt(macro.get("yield_curve"))),
        ])

    if peers is not None and not peers.empty:
        lines += ["", "PEER BENCHMARKING", "-" * W, peers.to_string(index=False)]

    if screener is not None and not screener.empty:
        lines += ["", "MARGIN-OF-SAFETY SCREENER", "-" * W, screener.head(10).to_string(index=False)]

    lines += ["", "For educational purposes only. Not financial advice."]
    return "\n".join(lines).encode("utf-8")


def has_reportlab() -> bool:
    return _HAS_REPORTLAB
