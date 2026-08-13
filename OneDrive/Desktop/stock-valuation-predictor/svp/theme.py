"""
Shared palette, matplotlib styling and Streamlit CSS.
=====================================================

Four colours, allocated on a strict 60 / 30 / 10 rule.

  60%  OBSIDIAN BLACK  — every surface: page ground, cards, sidebar, table
                         bodies, inputs, buttons at rest. Four values of the
                         one hue, so depth comes from lightness rather than
                         from introducing colours.
  30%  PEARL WHITE     — all typography, borders, grid lines, axis labels,
                         chart overlays. This is what the eye actually reads.
  10%  DEEP EMERALD +  — spent only where it must be: emerald carries "up"
       CHAMPAGNE BEIGE   and positive value, champagne marks the single
                         active or selected element.

**One derived colour.** The four given colours contain nothing that can mean
"down", and a price chart without a down colour is unreadable — a red candle
has to be red. ``GARNET`` is therefore derived at the same saturation and
value as the emerald so it sits inside the family rather than shouting over
it. It is used for direction only, never for decoration.

Deep emerald at full depth disappears against obsidian, so ``EMERALD_BRIGHT``
is used for type and candles while ``EMERALD`` fills larger areas where the
darker value reads correctly.

Nothing outside these groups introduces a hue. ``svp.charts`` imports directly
from here, so the charts cannot drift from the UI.
"""

from __future__ import annotations

# ── 60% — OBSIDIAN BLACK: every surface ──────────────────────────────────────
OBSIDIAN_DEEP = "#0A0B0D"   # page ground
OBSIDIAN      = "#121417"   # card / panel surface
OBSIDIAN_HI   = "#1B1F23"   # hover / raised
OBSIDIAN_LINE = "#272C31"   # borders, dividers

BG       = OBSIDIAN_DEEP
PANEL    = OBSIDIAN
PANEL_HI = OBSIDIAN_HI
SIDEBAR  = "#0E1013"
BORDER   = OBSIDIAN_LINE
GRID     = "#20242A"

# ── 30% — PEARL WHITE: everything you read ───────────────────────────────────
PEARL       = "#F4F2ED"     # primary type
PEARL_DIM   = "#CBC7BF"     # secondary type, chart lines
PEARL_MUTED = "#948F86"     # labels
PEARL_FAINT = "#5F5B55"

TEXT  = PEARL
MUTED = PEARL_MUTED
FAINT = PEARL_FAINT

# ── 10% — DEEP EMERALD + CHAMPAGNE BEIGE ─────────────────────────────────────
EMERALD        = "#0E6B50"  # deep emerald — fills and large areas
EMERALD_BRIGHT = "#1FA87C"  # legible on obsidian: type, candles, lines
EMERALD_DIM    = "#0C2A22"

CHAMPAGNE      = "#E3D2AE"  # the single active / selected element
CHAMPAGNE_DIM  = "#2E2718"

GARNET       = "#B4483F"    # derived — direction only, see module docstring
GARNET_BRIGHT = "#D45F53"
GARNET_DIM   = "#2E1917"

BULL, BEAR = EMERALD_BRIGHT, GARNET_BRIGHT
BULL_DIM, BEAR_DIM = EMERALD_DIM, GARNET_DIM
AMBER, AMBER_DIM = CHAMPAGNE, CHAMPAGNE_DIM

# Aliases for modules importing the older names. ACCENT resolves to pearl, not
# a fifth hue, so the 30% band absorbs generic emphasis and the 10% band stays
# reserved for direction and selection.
ACCENT     = PEARL_DIM
ACCENT_DIM = OBSIDIAN_HI
VIOLET     = CHAMPAGNE
TEAL, GREEN, YELLOW, RED, BLUE = PEARL_DIM, BULL, CHAMPAGNE, BEAR, PEARL_DIM

# Categorical sequence: pearl values first, the 10% accents last and sparing.
SERIES = [PEARL_DIM, PEARL_MUTED, "#A8A29A", PEARL_FAINT, CHAMPAGNE,
          EMERALD_BRIGHT, GARNET_BRIGHT, EMERALD]

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
    .metric-card:hover {{ background: {PANEL_HI}; border-color: {OBSIDIAN_HI}; }}
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
    .pill-live    {{ background: {BULL_DIM}; color: {BULL}; border: 1px solid {BULL_DIM}; }}
    .pill-offline {{ background: {BEAR_DIM}; color: {BEAR}; border: 1px solid {BEAR_DIM}; }}
    .pill-accent  {{ background: {AMBER_DIM}; color: {AMBER}; border: 1px solid {AMBER_DIM}; }}

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
        box-shadow: inset 0 -2px 0 0 {AMBER};
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
        background: {AMBER}; border-color: {AMBER}; color: {OBSIDIAN_DEEP};
    }}
    .stButton > button[kind="primary"]:hover {{ background: {AMBER}; color: {OBSIDIAN_DEEP}; }}
    .stDownloadButton > button {{
        background: {BULL}; border: none; color: {OBSIDIAN_DEEP}; font-weight: 700; border-radius: 8px;
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
    ::-webkit-scrollbar-thumb {{ background: {OBSIDIAN_HI}; border-radius: 5px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {PEARL_FAINT}; }}

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
