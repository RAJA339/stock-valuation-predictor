"""
Alternative "quant terminal" theme.
===================================

This implements the Onyx / Slate / Emerald / Cyan palette with Inter, Outfit
and JetBrains Mono, as an **opt-in** alternative to ``svp.theme``.

It is deliberately not the default. The app currently ships the obsidian /
pearl / emerald / champagne palette on a 60-30-10 split, and ``svp.charts``
imports its colours from ``svp.theme`` so the charts and the PDF stay in step
with the UI. Swapping the default without also re-pointing the chart and print
palettes would leave three surfaces disagreeing about what "up" looks like.

To adopt it, call ``apply_custom_theme()`` instead of injecting ``theme.CSS``
in ``app.py``, and re-point ``svp/charts.py`` and the print palette in
``svp/reports.py`` at these values.
"""

from __future__ import annotations

# ── Palette ──────────────────────────────────────────────────────────────────
ONYX = "#0B0F19"          # main background
SLATE = "#111827"         # sidebar
CARD = "#161E2E"          # glass card fill
HAIRLINE = "rgba(255, 255, 255, 0.08)"

EMERALD = "#00E676"       # primary accent / undervalued
CYAN = "#00F0FF"          # secondary accent / metrics
AMBER = "#FFC53D"         # hold
CRIMSON = "#FF5370"       # overvalued

TEXT = "#E8EDF5"
MUTED = "#8A94A6"

FONT_BODY = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"
FONT_HEAD = "'Outfit', 'Inter', sans-serif"
FONT_MONO = "'JetBrains Mono', 'SFMono-Regular', Consolas, monospace"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700\
&family=Outfit:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, .stApp {{ font-family: {FONT_BODY}; }}
.stApp {{ background: {ONYX}; color: {TEXT}; }}
.main .block-container {{ padding-top: 2rem; max-width: 1500px; }}

h1, h2, h3, h4, .section-header {{
    font-family: {FONT_HEAD}; letter-spacing: -.01em; color: {TEXT};
}}

/* Every figure in the app is monospaced so digits align column to column. */
.metric-value, .metric-sub, .quote-price, .quote-change, .mono,
[data-testid="stMetricValue"], [data-testid="stMetricDelta"],
.stDataFrame, code, pre {{
    font-family: {FONT_MONO}; font-variant-numeric: tabular-nums;
}}

/* ── Glass cards ─────────────────────────────────────────────────────────── */
.metric-card, .glass-card {{
    background: {CARD};
    border: 1px solid {HAIRLINE};
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
    backdrop-filter: blur(6px);
    transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}}
.metric-card:hover, .glass-card:hover {{
    border-color: rgba(0, 240, 255, 0.35);
    box-shadow: 0 0 0 1px rgba(0, 240, 255, 0.12), 0 8px 28px rgba(0, 240, 255, 0.07);
    transform: translateY(-1px);
}}
.metric-label {{
    font-family: {FONT_BODY}; font-size: .70rem; color: {MUTED};
    text-transform: uppercase; letter-spacing: .08em; font-weight: 600;
    margin-bottom: 6px;
}}
.metric-value {{ font-size: 1.7rem; font-weight: 700; color: {CYAN}; line-height: 1.15; }}
.metric-sub {{ font-size: 1rem; font-weight: 500; color: {TEXT}; }}
.metric-note {{ font-family: {FONT_BODY}; font-size: .68rem; color: {MUTED}; margin-top: 7px; }}

/* ── Native metrics ──────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {CARD}; border: 1px solid {HAIRLINE};
    border-radius: 12px; padding: 14px 16px;
}}
[data-testid="stMetricLabel"] p {{
    font-family: {FONT_BODY} !important; font-size: .70rem !important;
    color: {MUTED} !important; text-transform: uppercase; letter-spacing: .08em;
}}
[data-testid="stMetricValue"] {{ color: {CYAN} !important; font-size: 1.5rem !important; }}

/* ── Status pills ────────────────────────────────────────────────────────── */
.badge {{
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-family: {FONT_BODY}; font-size: .76rem; font-weight: 600;
    letter-spacing: .02em; border: 1px solid transparent;
}}
.badge-under {{ background: rgba(0, 230, 118, .12); color: {EMERALD};
                border-color: rgba(0, 230, 118, .35); }}
.badge-over  {{ background: rgba(255, 83, 112, .12); color: {CRIMSON};
                border-color: rgba(255, 83, 112, .35); }}
.badge-hold  {{ background: rgba(255, 197, 61, .12); color: {AMBER};
                border-color: rgba(255, 197, 61, .35); }}
.badge-info  {{ background: rgba(0, 240, 255, .10); color: {CYAN};
                border-color: rgba(0, 240, 255, .30); }}

.signal-buy, .signal-under {{ color: {EMERALD}; font-family: {FONT_MONO};
                              font-size: 1.3rem; font-weight: 700; }}
.signal-over {{ color: {CRIMSON}; font-family: {FONT_MONO};
                font-size: 1.3rem; font-weight: 700; }}
.signal-hold {{ color: {AMBER}; font-family: {FONT_MONO};
                font-size: 1.3rem; font-weight: 700; }}

/* ── Chrome ──────────────────────────────────────────────────────────────── */
div[data-testid="stSidebar"] {{ background: {SLATE}; border-right: 1px solid {HAIRLINE}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {HAIRLINE}; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent; color: {MUTED}; border-radius: 0;
    font-size: .82rem; font-weight: 600; padding: 9px 14px;
}}
.stTabs [aria-selected="true"] {{
    color: {CYAN} !important; box-shadow: inset 0 -2px 0 0 {CYAN};
}}
.stTabs [data-baseweb="tab-highlight"] {{ background: transparent; }}

.stButton > button {{
    font-family: {FONT_BODY}; border-radius: 10px; font-weight: 600;
    background: {CARD}; color: {TEXT}; border: 1px solid {HAIRLINE};
}}
.stButton > button:hover {{ border-color: {CYAN}; color: {CYAN}; }}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {EMERALD}, {CYAN});
    color: {ONYX}; border: none; font-weight: 700;
}}
.stTextInput input, .stNumberInput input, .stTextArea textarea,
div[data-baseweb="select"] > div {{
    background: {CARD} !important; border: 1px solid {HAIRLINE} !important;
    color: {TEXT} !important; border-radius: 10px !important;
}}
.stDataFrame {{ border: 1px solid {HAIRLINE}; border-radius: 12px; }}
hr {{ border-color: {HAIRLINE}; }}
#MainMenu, footer {{ visibility: hidden; }}
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: {ONYX}; }}
::-webkit-scrollbar-thumb {{ background: #223047; border-radius: 5px; }}
::-webkit-scrollbar-thumb:hover {{ background: #2E4260; }}
</style>
"""


def apply_custom_theme() -> None:
    """Inject the quant-terminal stylesheet. Call once, at the top of the app."""
    import streamlit as st

    st.markdown(_CSS, unsafe_allow_html=True)


def badge(text: str, kind: str = "info") -> str:
    """
    Pill-style status badge.

    ``kind`` is one of ``under`` / ``over`` / ``hold`` / ``info``; pass the
    meaning rather than a colour so the palette stays swappable.

        st.markdown(badge("🟢 Undervalued", "under"), unsafe_allow_html=True)
    """
    cls = {"under": "badge-under", "over": "badge-over",
           "hold": "badge-hold"}.get(kind, "badge-info")
    return f'<span class="badge {cls}">{text}</span>'


def card(label: str, value: str, note: str = "") -> str:
    """Glass card with a monospaced value and an optional footnote."""
    tail = f'<div class="metric-note">{note}</div>' if note else ""
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>{tail}</div>')
