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

    /* ── Type scale ─────────────────────────────────────────────────────────
       Seven steps on a ~1.2 ratio, replacing the seventeen ad-hoc sizes this
       sheet accumulated (.68 .70 .72 .78 .82 .84 .86 .92 .95 1.0 1.02 1.35 1.5
       1.75 1.9 2.3rem). Sizes that sit a hair apart read as accidents; a fixed
       ratio is most of what separates a designed interface from a styled one.
       Every rule below sizes itself from this scale — add a step here rather
       than a one-off value at a call site. */
    :root {{
        --fs-1: .70rem;    /* micro labels, eyebrows, pills   */
        --fs-2: .82rem;    /* captions, sidebar body, buttons */
        --fs-3: .95rem;    /* section heads, dense body       */
        --fs-4: 1.15rem;   /* secondary values                */
        --fs-5: 1.45rem;   /* signals, st.metric values       */
        --fs-6: 1.80rem;   /* card values, ticker symbol      */
        --fs-7: 2.30rem;   /* quote price                     */
    }}

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
        font-size: var(--fs-1); color: {MUTED}; text-transform: uppercase;
        letter-spacing: .07em; font-weight: 600; margin-bottom: 6px;
    }}
    .metric-value {{ font-size: var(--fs-6); font-weight: 700; color: {TEXT}; line-height: 1.15; }}
    .metric-sub   {{ font-size: var(--fs-4); font-weight: 600; color: {TEXT}; line-height: 1.3; }}

    /* ── Directional text ───────────────────────────────────────────────── */
    .signal-buy, .signal-under {{ color: {BULL}; font-size: var(--fs-5); font-weight: 700; }}
    .signal-over  {{ color: {BEAR}; font-size: var(--fs-5); font-weight: 700; }}
    .signal-hold  {{ color: {AMBER}; font-size: var(--fs-5); font-weight: 700; }}

    /* ── Delta pills — coloured strictly by sign ────────────────────────── */
    .delta-pos {{
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        background: {BULL_DIM}; color: {BULL};
        font-size: var(--fs-2); font-weight: 700; margin-top: 6px;
    }}
    .delta-neg {{
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        background: {BEAR_DIM}; color: {BEAR};
        font-size: var(--fs-2); font-weight: 700; margin-top: 6px;
    }}
    .metric-note {{
        font-size: var(--fs-1); color: {MUTED}; margin-top: 7px;
        letter-spacing: .02em; line-height: 1.35;
    }}

    /* ── Verdict bar — the above-the-fold answer ────────────────────────────
       The market marker is champagne: this is the single "you are here"
       element on the page, which is exactly what the 10% selection colour is
       reserved for. The range track stays pearl so the accent has something
       quiet to sit against. */
    .verdict {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-left: 3px solid {EMERALD};
        border-radius: 10px;
        padding: 14px 18px 16px;
        margin: 4px 0 14px;
    }}
    .verdict-line {{ font-size: var(--fs-4); color: {TEXT}; line-height: 1.45; }}
    .verdict-tail {{ color: {MUTED}; font-size: var(--fs-3); }}
    .verdict-under  {{ color: {BULL}; font-weight: 700; }}
    .verdict-over   {{ color: {BEAR}; font-weight: 700; }}
    .verdict-inside {{ color: {AMBER}; font-weight: 700; }}
    .verdict-track {{
        position: relative; height: 6px; border-radius: 3px;
        background: linear-gradient(90deg, {PEARL_FAINT}, {PEARL_MUTED}, {PEARL_FAINT});
        margin: 14px 0 6px;
    }}
    .verdict-point {{
        position: absolute; top: -3px; width: 2px; height: 12px;
        background: {PEARL}; transform: translateX(-1px);
    }}
    .verdict-mark {{
        position: absolute; top: -5px; width: 3px; height: 16px;
        background: {CHAMPAGNE}; border-radius: 2px;
        transform: translateX(-1.5px);
        box-shadow: 0 0 0 3px {CHAMPAGNE_DIM};
    }}
    .verdict-scale {{
        display: flex; justify-content: space-between;
        font-size: var(--fs-1); color: {MUTED};
        font-variant-numeric: tabular-nums; letter-spacing: .03em;
    }}

    /* ── Section headers ─────────────────────────────────────────────────────
       An eyebrow names the section's role, the title names the section, a rule
       separates it from the content. No terminal a professional uses puts a
       pictograph in a heading — hierarchy comes from size, weight and tracking,
       which is what this pair does. .section-header is the older single-line
       form, kept so any call site not yet converted still renders. */
    .section-head {{
        border-bottom: 1px solid {BORDER};
        padding-bottom: 8px; margin: 14px 0 14px 0;
    }}
    .section-eyebrow {{
        font-size: var(--fs-1); font-weight: 600; color: {MUTED};
        text-transform: uppercase; letter-spacing: .15em;
        margin-bottom: 2px;
    }}
    .section-title {{
        font-size: var(--fs-4); font-weight: 650; color: {TEXT};
        letter-spacing: -.01em; line-height: 1.25;
    }}
    .section-header {{
        font-size: var(--fs-3); font-weight: 700; color: {TEXT};
        text-transform: uppercase; letter-spacing: .06em;
        border-bottom: 1px solid {BORDER};
        padding-bottom: 8px; margin: 6px 0 14px 0;
    }}

    /* ── Pills ──────────────────────────────────────────────────────────── */
    .pill {{
        display: inline-block; padding: 2px 9px; border-radius: 999px;
        font-size: var(--fs-1); font-weight: 700; margin-left: 6px;
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
        color: {MUTED}; font-size: var(--fs-2); font-weight: 600;
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
    div[data-testid="stSidebar"] .stMarkdown p {{ font-size: var(--fs-2); }}

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
        border-radius: 8px; font-weight: 600; font-size: var(--fs-2);
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
        font-size: var(--fs-1) !important; text-transform: uppercase; letter-spacing: .05em;
    }}

    /* ── Native metrics ─────────────────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid {BORDER};
        border-radius: 10px; padding: 12px 14px;
    }}
    [data-testid="stMetricLabel"] p {{
        font-size: var(--fs-1) !important; color: {MUTED} !important;
        text-transform: uppercase; letter-spacing: .06em; font-weight: 600;
    }}
    [data-testid="stMetricValue"] {{ font-size: var(--fs-5) !important; font-weight: 700; }}

    /* ── Alerts — flat, with a leading rule ─────────────────────────────── */
    div[data-testid="stAlert"] {{
        border-radius: 8px; border: 1px solid {BORDER};
        background: {PANEL}; padding: 12px 14px;
    }}

    /* ── Expander / divider / misc ──────────────────────────────────────── */
    .streamlit-expanderHeader {{
        background: {PANEL}; border: 1px solid {BORDER};
        border-radius: 8px; font-size: var(--fs-2); font-weight: 600;
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
    .ticker-sym  {{ font-size: var(--fs-6); font-weight: 700; color: {TEXT}; letter-spacing: -.01em; }}
    .ticker-name {{ font-size: var(--fs-3); color: {MUTED}; }}
    .quote-price {{ font-size: var(--fs-7); font-weight: 700; color: {TEXT}; line-height: 1.1; }}
    .quote-change {{ font-size: var(--fs-3); font-weight: 600; }}
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


def verdict_bar(price: float, low: float, point: float, high: float) -> str:
    """
    The answer to the only question a visitor actually arrives with.

    One sentence and one bar, above the fold, before any tab. The bar plots the
    market price inside the model's p10–p90 range, because "12% undervalued" is
    a different claim depending on whether the range is $10 wide or $200 — and
    a point estimate quoted without its range is the thing this project exists
    to argue against.
    """
    span = max(high - low, 1e-9)
    pos = min(max((price - low) / span, 0.0), 1.0) * 100.0
    pt = min(max((point - low) / span, 0.0), 1.0) * 100.0
    gap = (point - price) / price * 100.0 if price else 0.0

    if price < low:
        word, cls = "below", "verdict-under"
        tail = "the market is cheaper than the model's tenth percentile"
    elif price > high:
        word, cls = "above", "verdict-over"
        tail = "the market is richer than the model's ninetieth percentile"
    else:
        word, cls = "inside", "verdict-inside"
        tail = "the market sits within the model's range, so the signal is weak"

    return (
        f'<div class="verdict">'
        f'<div class="verdict-line">'
        f'Trading <span class="{cls}">{word}</span> the estimated range — '
        f'<span class="{cls}">{gap:+.1f}%</span> to the point estimate. '
        f'<span class="verdict-tail">{tail}.</span>'
        f'</div>'
        f'<div class="verdict-track">'
        f'<div class="verdict-point" style="left:{pt:.2f}%"></div>'
        f'<div class="verdict-mark" style="left:{pos:.2f}%"></div>'
        f'</div>'
        f'<div class="verdict-scale">'
        f'<span>p10 ${low:,.2f}</span>'
        f'<span>fair ${point:,.2f}</span>'
        f'<span>p90 ${high:,.2f}</span>'
        f'</div></div>'
    )


def section(title: str, eyebrow: str = "") -> str:
    """
    Section heading markup: a role eyebrow above the title, then a rule.

    ``eyebrow`` says what kind of thing the section is ("VALIDATION",
    "DERIVATIVES") and repeats across sections that share a purpose, so a reader
    scanning a long tab can tell analysis from evidence without reading. Pass an
    empty eyebrow for a subsection that continues the one above it.
    """
    eb = f'<div class="section-eyebrow">{eyebrow}</div>' if eyebrow else ""
    return (f'<div class="section-head">{eb}'
            f'<div class="section-title">{title}</div></div>')


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
