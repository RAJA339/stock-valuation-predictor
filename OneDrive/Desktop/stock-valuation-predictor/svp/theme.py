"""
Shared palette, matplotlib styling and Streamlit CSS.
=====================================================

The look follows US retail brokerage terminals — Robinhood, Webull,
Thinkorswim, TradingView — rather than a generic dashboard:

  * near-black ground (#0B0E11) with slightly raised cards, not flat panels
  * a single accent, with green/red reserved for direction so colour always
    means P&L rather than decoration
  * tabular numerals everywhere a price or percentage appears, so digits line
    up column-to-column and don't jitter as values update
  * dense type — brokerages fit a lot on screen; padding is tight by design

``BULL`` and ``BEAR`` are the only semantically loaded colours. Nothing else in
the palette should be used to imply direction.
"""

from __future__ import annotations

# ── Ground ────────────────────────────────────────────────────────────────────
BG          = "#0B0E11"   # app background — near black, like a trading terminal
PANEL       = "#151A21"   # card surface
PANEL_HI    = "#1C232C"   # hovered / raised surface
SIDEBAR     = "#0E1217"
BORDER      = "#232B36"
GRID        = "#1E252F"

# ── Type ──────────────────────────────────────────────────────────────────────
TEXT        = "#E6EDF3"
MUTED       = "#8B949E"
FAINT       = "#5A626C"

# ── Direction (the only colours that carry meaning) ───────────────────────────
BULL        = "#00C805"   # Robinhood green
BEAR        = "#FF3B30"
BULL_DIM    = "#0B2E12"
BEAR_DIM    = "#33110F"

# ── Accent ────────────────────────────────────────────────────────────────────
ACCENT      = "#4C8DFF"
ACCENT_DIM  = "#132A4D"
AMBER       = "#F0B429"
VIOLET      = "#B392F0"

# Backwards-compatible aliases — older modules import these names.
TEAL, GREEN, YELLOW, RED, BLUE = ACCENT, BULL, AMBER, BEAR, ACCENT

# Ordered categorical sequence for multi-series charts.
SERIES = [ACCENT, BULL, AMBER, VIOLET, "#4CB5AE", "#F19A3E", BEAR, "#7D8590"]

CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    .main .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}

    /* Prices and percentages must not jitter as digits change. */
    .metric-value, .metric-sub, .quote-price, .quote-change,
    .stMetric, [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
    .stDataFrame, code {{ font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; }}

    /* ── Cards ──────────────────────────────────────────────────────────── */
    .metric-card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: border-color .15s ease, background .15s ease;
    }}
    .metric-card:hover {{ background: {PANEL_HI}; border-color: #2E3742; }}
    .metric-label {{
        font-size: .70rem; color: {MUTED}; text-transform: uppercase;
        letter-spacing: .07em; font-weight: 600; margin-bottom: 6px;
    }}
    .metric-value {{ font-size: 1.75rem; font-weight: 700; color: {TEXT}; line-height: 1.15; }}
    .metric-sub   {{ font-size: 1.02rem; font-weight: 600; color: {TEXT}; line-height: 1.3; }}

    /* ── Directional text ───────────────────────────────────────────────── */
    .signal-buy, .signal-under {{ color: {BULL}; font-size: 1.35rem; font-weight: 700; }}
    .signal-over  {{ color: {BEAR}; font-size: 1.35rem; font-weight: 700; }}
    .signal-hold  {{ color: {AMBER}; font-size: 1.35rem; font-weight: 700; }}

    /* ── Section headers ────────────────────────────────────────────────── */
    .section-header {{
        font-size: .95rem; font-weight: 700; color: {TEXT};
        text-transform: uppercase; letter-spacing: .06em;
        border-bottom: 1px solid {BORDER};
        padding-bottom: 8px; margin: 6px 0 14px 0;
    }}

    /* ── Pills ──────────────────────────────────────────────────────────── */
    .pill {{
        display: inline-block; padding: 2px 9px; border-radius: 999px;
        font-size: .68rem; font-weight: 700; margin-left: 6px;
        letter-spacing: .04em; vertical-align: middle;
    }}
    .pill-live    {{ background: {BULL_DIM}; color: {BULL}; border: 1px solid #14431C; }}
    .pill-offline {{ background: {BEAR_DIM}; color: #FF8A82; border: 1px solid #4A1A16; }}
    .pill-accent  {{ background: {ACCENT_DIM}; color: {ACCENT}; border: 1px solid #1D3A66; }}

    /* ── Tabs — brokerage-style underline, not boxes ────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px; background: transparent; border-bottom: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; border: none; border-radius: 0;
        color: {MUTED}; font-size: .82rem; font-weight: 600;
        padding: 9px 14px; margin: 0;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {TEXT}; background: {PANEL}; }}
    .stTabs [aria-selected="true"] {{
        color: {TEXT} !important; background: transparent !important;
        box-shadow: inset 0 -2px 0 0 {ACCENT};
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: transparent; }}

    /* ── Sidebar ────────────────────────────────────────────────────────── */
    div[data-testid="stSidebar"] {{
        background-color: {SIDEBAR}; border-right: 1px solid {BORDER};
    }}
    div[data-testid="stSidebar"] .stMarkdown p {{ font-size: .86rem; }}

    /* ── Inputs ─────────────────────────────────────────────────────────── */
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    div[data-baseweb="select"] > div {{
        background-color: {PANEL} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT} !important; border-radius: 8px !important;
    }}
    .stTextInput input:focus, .stNumberInput input:focus {{
        border-color: {ACCENT} !important; box-shadow: 0 0 0 2px {ACCENT_DIM} !important;
    }}

    /* ── Buttons ────────────────────────────────────────────────────────── */
    .stButton > button {{
        border-radius: 8px; font-weight: 600; font-size: .84rem;
        border: 1px solid {BORDER}; background: {PANEL}; color: {TEXT};
        transition: all .15s ease;
    }}
    .stButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; background: {PANEL_HI}; }}
    .stButton > button[kind="primary"] {{
        background: {ACCENT}; border-color: {ACCENT}; color: #06090D;
    }}
    .stButton > button[kind="primary"]:hover {{ background: #6BA1FF; color: #06090D; }}
    .stDownloadButton > button {{
        background: {BULL}; border: none; color: #06110A; font-weight: 700; border-radius: 8px;
    }}

    /* ── Tables ─────────────────────────────────────────────────────────── */
    .stDataFrame {{ border: 1px solid {BORDER}; border-radius: 10px; }}
    .stDataFrame [role="columnheader"] {{
        background: {PANEL} !important; color: {MUTED} !important;
        font-size: .72rem !important; text-transform: uppercase; letter-spacing: .05em;
    }}

    /* ── Native metrics ─────────────────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid {BORDER};
        border-radius: 10px; padding: 12px 14px;
    }}
    [data-testid="stMetricLabel"] p {{
        font-size: .70rem !important; color: {MUTED} !important;
        text-transform: uppercase; letter-spacing: .06em; font-weight: 600;
    }}
    [data-testid="stMetricValue"] {{ font-size: 1.5rem !important; font-weight: 700; }}

    /* ── Alerts — flat, with a leading rule ─────────────────────────────── */
    div[data-testid="stAlert"] {{
        border-radius: 8px; border: 1px solid {BORDER};
        background: {PANEL}; padding: 12px 14px;
    }}

    /* ── Expander / divider / misc ──────────────────────────────────────── */
    .streamlit-expanderHeader {{
        background: {PANEL}; border: 1px solid {BORDER};
        border-radius: 8px; font-size: .84rem; font-weight: 600;
    }}
    hr {{ border-color: {BORDER}; margin: 1.1rem 0; }}
    #MainMenu, footer {{ visibility: hidden; }}
    ::-webkit-scrollbar {{ width: 9px; height: 9px; }}
    ::-webkit-scrollbar-track {{ background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: #2A323D; border-radius: 5px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #3A434F; }}

    /* ── Ticker header ──────────────────────────────────────────────────── */
    .ticker-head {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
                   margin-bottom: 2px; }}
    .ticker-sym  {{ font-size: 1.9rem; font-weight: 700; color: {TEXT}; letter-spacing: -.01em; }}
    .ticker-name {{ font-size: .92rem; color: {MUTED}; }}
    .quote-price {{ font-size: 2.3rem; font-weight: 700; color: {TEXT}; line-height: 1.1; }}
    .quote-change {{ font-size: 1.0rem; font-weight: 600; }}
    .quote-up   {{ color: {BULL}; }}
    .quote-down {{ color: {BEAR}; }}
</style>
"""


def style_axes(fig, ax) -> None:
    """Apply the terminal styling to a matplotlib figure/axis."""
    fig.patch.set_facecolor(PANEL)
    if not isinstance(ax, (list, tuple)):
        ax = [ax]
    for a in ax:
        a.set_facecolor(PANEL)
        a.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        a.tick_params(colors=MUTED, labelsize=8)
        for spine in a.spines.values():
            spine.set_color(BORDER)
        a.title.set_color(TEXT)
        a.xaxis.label.set_color(MUTED)
        a.yaxis.label.set_color(MUTED)


def source_pill(is_live: bool, live_label: str = "LIVE", offline_label: str = "OFFLINE") -> str:
    """Small status pill showing whether a data source is live."""
    if is_live:
        return f'<span class="pill pill-live">● {live_label}</span>'
    return f'<span class="pill pill-offline">● {offline_label}</span>'


def quote_header(symbol: str, name: str, price: float, change_pct: float | None,
                 pills: str = "") -> str:
    """
    The price block every brokerage app opens with: symbol, name, big price,
    and the change coloured by direction.
    """
    up = (change_pct or 0) >= 0
    arrow = "▲" if up else "▼"
    cls = "quote-up" if up else "quote-down"
    chg = (f'<span class="quote-change {cls}">{arrow} {abs(change_pct or 0) * 100:.2f}%</span>'
           if change_pct is not None else "")
    return (
        f'<div class="ticker-head"><span class="ticker-sym">{symbol}</span>'
        f'<span class="ticker-name">{name}</span>{pills}</div>'
        f'<div class="ticker-head"><span class="quote-price">${price:,.2f}</span>{chg}</div>'
    )
