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
        )

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
    if excess_return is not None:
        val_rows.append(["Predicted Excess Return vs Benchmark", _fmt(excess_return, "pct")])
    story.append(_table(val_rows, [2.7 * inch, 3.0 * inch]))
    story.append(Spacer(1, 12))

    # ── Fundamentals / extracted features ────────────────────────────────────
    if fundamentals:
        story.append(Paragraph("Company Fundamentals", h2))
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
        story.append(Paragraph("Model Input Features", h2))
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

    # ── Backtesting ──────────────────────────────────────────────────────────
    if backtest:
        story.append(Paragraph("Signal Backtest", h2))
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
        story.append(Paragraph("Execution &amp; Timing", h2))
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
        story.append(Paragraph("Position Sizing &amp; Risk", h2))
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
        story.append(Paragraph("Guardrails — Quality, Distress &amp; Insider Flow", h2))
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
        story.append(Paragraph("Margin-of-Safety Screener", h2))
        cols = [c for c in screener.columns if c.lower() != "index"][:5]
        rows = [[str(c) for c in cols]]
        for _, r in screener.head(10).iterrows():
            rows.append([_fmt(r[c]) if isinstance(r[c], (int, float)) else str(r[c]) for c in cols])
        story.append(_table(rows, [5.7 * inch / len(cols)] * len(cols)))
        story.append(Spacer(1, 12))

    # ── Filing divergence ────────────────────────────────────────────────────
    if filings is not None and _attr(filings, "similarity") is not None:
        story.append(Paragraph("Filing Divergence (10-K / 10-Q)", h2))
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
        story.append(Paragraph("Options &amp; Futures", h2))
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
            ("Best Contract Strike", "best_strike", "usd"),
            ("Best Contract Premium", "best_premium", "usd"),
            ("Model Edge", "best_edge", "usd"),
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


def _text_report(
    ticker, company, market_price, valuation, signal_text, dcf, peers,
    features=None, fundamentals=None, backtest=None, regime=None, technical=None,
    sizing=None, quality=None, insider=None, filings=None, screener=None,
    options=None, excess_return=None, macro=None,
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
