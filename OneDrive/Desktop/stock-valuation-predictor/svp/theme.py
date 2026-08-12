"""Shared colour palette, matplotlib styling helpers and Streamlit CSS."""

from __future__ import annotations

# ── Palette (matches .streamlit/config.toml) ─────────────────────────────────
BG          = "#0D1B2A"   # app background
PANEL       = "#1E293B"   # card / panel background
SIDEBAR     = "#1E2761"
TEXT        = "#CADCFC"
MUTED       = "#94A3B8"
GRID        = "#334155"

TEAL        = "#028090"   # primary
GREEN       = "#02C39A"   # positive / undervalued
YELLOW      = "#F9C846"   # neutral / hold
RED         = "#F96167"   # negative / overvalued
BLUE        = "#2A4494"

# Ordered categorical sequence for multi-series charts.
SERIES = [TEAL, GREEN, YELLOW, "#8E7DBE", "#F19A3E", "#4CB5AE", RED, BLUE]

CSS = """
<style>
    .main { background-color: #0D1B2A; }
    .stApp { background-color: #0D1B2A; color: #CADCFC; }
    .metric-card {
        background: #1E293B;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #028090;
        margin-bottom: 15px;
    }
    .metric-value { font-size: 2.4rem; font-weight: bold; color: #02C39A; }
    .metric-sub   { font-size: 1.1rem; font-weight: 600; color: #CADCFC; }
    .metric-label { font-size: 0.85rem; color: #94A3B8; }
    .signal-buy  { color: #02C39A; font-size: 1.3rem; font-weight: bold; }
    .signal-hold { color: #F9C846; font-size: 1.3rem; font-weight: bold; }
    .signal-over { color: #F96167; font-size: 1.3rem; font-weight: bold; }
    .section-header {
        font-size: 1.2rem; font-weight: bold;
        color: #CADCFC; border-bottom: 2px solid #028090;
        padding-bottom: 6px; margin-bottom: 12px;
    }
    .pill {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 600; margin-left: 6px;
    }
    .pill-live { background: #064e3b; color: #6ee7b7; }
    .pill-offline { background: #4c1d24; color: #fca5a5; }
    div[data-testid="stSidebar"] { background-color: #1E2761; }
</style>
"""


def style_axes(fig, ax) -> None:
    """Apply the dark dashboard styling to a matplotlib figure/axis."""
    fig.patch.set_facecolor(PANEL)
    if not isinstance(ax, (list, tuple)):
        ax = [ax]
    for a in ax:
        a.set_facecolor(PANEL)
        a.tick_params(colors=TEXT, labelsize=8.5)
        a.xaxis.label.set_color(MUTED)
        a.yaxis.label.set_color(MUTED)
        a.title.set_color(TEXT)
        for spine in a.spines.values():
            spine.set_edgecolor(GRID)


def source_pill(is_live: bool, live_label: str = "LIVE", offline_label: str = "OFFLINE") -> str:
    """Return an HTML pill indicating whether data is live or a fallback."""
    if is_live:
        return f'<span class="pill pill-live">● {live_label}</span>'
    return f'<span class="pill pill-offline">● {offline_label}</span>'
