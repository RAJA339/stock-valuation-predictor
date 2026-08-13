"""
Brokerage-style candlestick charts.
===================================

Renders the same view twice: on screen in the app, and as a PNG embedded in the
PDF. The TradingView widget in the app is an iframe and cannot be captured, so
the report needs a native chart — this is it.

Layout follows what Robinhood / Webull / Thinkorswim put on a default intraday
screen: price with EMAs, VWAP and Bollinger band, a volume histogram coloured
by candle direction, then RSI and MACD panels beneath.
"""

from __future__ import annotations

import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Brokerage palette — green/red candles on a dark ground.
from . import theme

# One palette across the app: charts must not drift from the UI.
UP = theme.BULL
DOWN = theme.BEAR
BG = theme.BG
PANEL = theme.PANEL
GRID = theme.GRID
TEXT = theme.TEXT
MUTED = theme.MUTED
ACCENT = theme.ACCENT
AMBER = theme.AMBER


def _style(ax):
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.label.set_color(MUTED)


def _candles(ax, d: pd.DataFrame, width_frac: float = 0.7):
    """Draw OHLC candles with wicks, sized off the median bar spacing."""
    x = mdates.date2num(d.index.to_pydatetime()) if isinstance(d.index, pd.DatetimeIndex) \
        else np.arange(len(d), dtype=float)
    if len(x) > 1:
        w = float(np.median(np.diff(x))) * width_frac
    else:
        w = 0.01

    up = d["Close"] >= d["Open"]
    colors = np.where(up, UP, DOWN)

    ax.vlines(x, d["Low"], d["High"], color=colors, lw=0.7, alpha=0.9)
    bottoms = np.minimum(d["Open"], d["Close"])
    heights = (d["Close"] - d["Open"]).abs().clip(lower=1e-9)
    ax.bar(x, heights, bottom=bottoms, width=w, color=colors,
           edgecolor=colors, linewidth=0.4, zorder=3)
    return x


def render(
    d: pd.DataFrame,
    ticker: str,
    interval: str,
    max_bars: int = 160,
    show_bollinger: bool = True,
) -> bytes:
    """Render the chart to PNG bytes. ``d`` must carry the indicator columns."""
    if d is None or d.empty:
        return b""

    d = d.tail(max_bars)
    fig = plt.figure(figsize=(11, 7.4), facecolor=BG)
    gs = fig.add_gridspec(4, 1, height_ratios=[3.2, 0.9, 1.0, 1.0], hspace=0.08)

    ax_p = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax_p)
    ax_r = fig.add_subplot(gs[2], sharex=ax_p)
    ax_m = fig.add_subplot(gs[3], sharex=ax_p)
    for ax in (ax_p, ax_v, ax_r, ax_m):
        _style(ax)

    x = _candles(ax_p, d)

    if show_bollinger and "BB_UP" in d and d["BB_UP"].notna().any():
        ax_p.fill_between(x, d["BB_LOW"], d["BB_UP"], color=ACCENT, alpha=0.07, zorder=1)
        ax_p.plot(x, d["BB_UP"], color=ACCENT, lw=0.6, alpha=0.45)
        ax_p.plot(x, d["BB_LOW"], color=ACCENT, lw=0.6, alpha=0.45)

    for col, colour, label in (("EMA9", "#FFFFFF", "EMA 9"),
                               ("EMA20", AMBER, "EMA 20"),
                               ("EMA50", ACCENT, "EMA 50")):
        if col in d and d[col].notna().any():
            ax_p.plot(x, d[col], color=colour, lw=1.0, label=label, alpha=0.9)

    if "VWAP" in d and d["VWAP"].notna().any():
        # VWAP is session-anchored, so break the line at each day boundary —
        # otherwise the reset draws as a vertical jump across the panel.
        vw = d["VWAP"].copy()
        if isinstance(d.index, pd.DatetimeIndex):
            new_session = d.index.normalize().to_series().diff().ne(pd.Timedelta(0)).to_numpy(copy=True)
            new_session[0] = False
            vw = vw.mask(pd.Series(new_session, index=d.index))
        ax_p.plot(x, vw, color=theme.VIOLET, lw=1.2, ls="--", label="VWAP", alpha=0.9)

    last = float(d["Close"].iloc[-1])
    first = float(d["Close"].iloc[0])
    chg = (last - first) / first * 100 if first else 0.0
    tint = UP if chg >= 0 else DOWN
    ax_p.axhline(last, color=tint, lw=0.7, ls=":", alpha=0.8)
    ax_p.annotate(f" {last:,.2f} ", xy=(1.0, last), xycoords=("axes fraction", "data"),
                  color="#000000", fontsize=8, fontweight="bold", va="center",
                  bbox=dict(boxstyle="round,pad=0.25", fc=tint, ec="none"))

    ax_p.set_title(f"{ticker}   ·   {interval}   ·   {last:,.2f}  ({chg:+.2f}%)",
                   color=TEXT, fontsize=12, fontweight="bold", loc="left", pad=10)
    leg = ax_p.legend(loc="upper left", fontsize=7, framealpha=0.0, ncol=4)
    for t in leg.get_texts():
        t.set_color(MUTED)

    # Volume
    vcolors = np.where(d["Close"] >= d["Open"], UP, DOWN)
    ax_v.bar(x, d["Volume"], color=vcolors, width=float(np.median(np.diff(x))) * 0.7 if len(x) > 1 else 0.01,
             alpha=0.55)
    ax_v.set_ylabel("Vol", fontsize=7)
    ax_v.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else (f"{v/1e3:.0f}K" if v >= 1e3 else f"{v:.0f}")))

    # RSI
    if "RSI" in d:
        ax_r.plot(x, d["RSI"], color=AMBER, lw=1.1)
        ax_r.axhline(70, color=DOWN, lw=0.6, ls="--", alpha=0.7)
        ax_r.axhline(30, color=UP, lw=0.6, ls="--", alpha=0.7)
        ax_r.fill_between(x, 30, 70, color=MUTED, alpha=0.06)
        ax_r.set_ylim(0, 100)
        ax_r.set_ylabel("RSI", fontsize=7)

    # MACD
    if "MACD" in d:
        hist = d["MACD_HIST"]
        ax_m.bar(x, hist, color=np.where(hist >= 0, UP, DOWN),
                 width=float(np.median(np.diff(x))) * 0.7 if len(x) > 1 else 0.01, alpha=0.6)
        ax_m.plot(x, d["MACD"], color=ACCENT, lw=1.0, label="MACD")
        ax_m.plot(x, d["MACD_SIG"], color=AMBER, lw=1.0, label="Signal")
        ax_m.axhline(0, color=MUTED, lw=0.6)
        ax_m.set_ylabel("MACD", fontsize=7)

    for ax in (ax_p, ax_v, ax_r):
        plt.setp(ax.get_xticklabels(), visible=False)

    if isinstance(d.index, pd.DatetimeIndex):
        ax_m.xaxis_date()
        span_hours = (d.index[-1] - d.index[0]).total_seconds() / 3600
        fmt = "%H:%M" if span_hours <= 30 else "%m-%d %H:%M"
        ax_m.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
        fig.autofmt_xdate(rotation=0, ha="center")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=135, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def signal_strip(iset, width: int = 900) -> bytes:
    """A compact bull/bear/neutral tally bar for the report."""
    if not getattr(iset, "signals", None):
        return b""
    fig, ax = plt.subplots(figsize=(width / 100, 0.85), facecolor=BG)
    ax.set_facecolor(BG)
    total = max(len(iset.signals), 1)
    parts = [("Bullish", iset.bullish, UP), ("Neutral", iset.neutral, MUTED),
             ("Bearish", iset.bearish, DOWN)]
    left = 0.0
    for label, count, colour in parts:
        frac = count / total
        if frac <= 0:
            continue
        ax.barh([0], [frac], left=[left], color=colour, height=0.55)
        if frac > 0.08:
            ax.text(left + frac / 2, 0, f"{label} {count}", ha="center", va="center",
                    color="#0D1117" if colour != MUTED else TEXT, fontsize=9, fontweight="bold")
        left += frac
    ax.set_xlim(0, 1)
    ax.axis("off")
    ax.set_title(f"Indicator consensus: {iset.consensus}   "
                 f"(net score {iset.net_score:+.2f})",
                 color=TEXT, fontsize=10, fontweight="bold", loc="left", pad=6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=135, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
