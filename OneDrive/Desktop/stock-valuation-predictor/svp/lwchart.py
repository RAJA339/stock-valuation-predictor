"""
Lightweight-Charts workspace builder.
=====================================

Emits a self-contained HTML block that renders TradingView's open-source
Lightweight Charts library — the same engine tradingview.com uses for its
embedded charts — themed to this app's obsidian palette.

Design constraints, in order:

1. **Pure function of its inputs.** Everything here turns dataframes into a
   JSON payload and an HTML string; no network, no Streamlit, no state. That
   keeps it testable the way the analytics modules are testable, and keeps the
   rendering layer swappable (the brief's "isolate the component so it can
   later be upgraded").

2. **Honest with missing data.** An empty or too-short frame returns ``None``
   and the caller explains why — the chart never interpolates, back-fills, or
   invents bars. NaN rows are dropped, not smoothed over.

3. **Valuation on the price axis.** The signature overlay draws the model's
   bear / base / bull fair values and the margin-of-safety threshold as
   labelled price lines on the *price* chart, so "is the price below the
   estimate?" is answered by looking, not by reading a table. The labels say
   "est." because these are scenario outputs, not predictions.

The library is loaded from unpkg (pinned version). The existing TradingView
widget embeds already load from CDNs on the deployed app, so this introduces
no new class of dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from . import theme

_LWC_VERSION = "4.2.0"
_CDN = f"https://unpkg.com/lightweight-charts@{_LWC_VERSION}/dist/lightweight-charts.standalone.production.js"

#: Fewer bars than this and a candlestick chart is a scatter of noise; the
#: caller should say "not enough history" instead of rendering one.
MIN_BARS = 20


@dataclass
class ChartHTML:
    """A rendered workspace: the HTML block and the pixel height to give it."""

    html: str
    height: int


@dataclass
class FairValueZones:
    """The valuation band, expressed as price levels for the overlay.

    ``bear``/``base``/``bull`` are the model's p10 / point / p90 estimates.
    ``mos`` is the margin-of-safety threshold below the base estimate that the
    user dialled in. Any of them may be ``None`` — absent levels are simply
    not drawn.
    """

    bear: Optional[float] = None
    base: Optional[float] = None
    bull: Optional[float] = None
    mos: Optional[float] = None

    def lines(self) -> list[dict]:
        out = []
        if self.bull is not None:
            out.append({"price": round(self.bull, 2), "color": theme.BULL,
                        "style": 2, "title": f"Bull est. {self.bull:,.0f}"})
        if self.base is not None:
            out.append({"price": round(self.base, 2), "color": theme.AMBER,
                        "style": 0, "title": f"Base est. {self.base:,.0f}"})
        if self.bear is not None:
            out.append({"price": round(self.bear, 2), "color": theme.BEAR,
                        "style": 2, "title": f"Bear est. {self.bear:,.0f}"})
        if self.mos is not None:
            out.append({"price": round(self.mos, 2), "color": theme.EMERALD_BRIGHT,
                        "style": 1, "title": f"Buy-zone < {self.mos:,.0f}"})
        return out


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """OHLC rows with a datetime index, NaN price rows dropped, ascending."""
    need = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    if len(need) < 4 or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    out = df.dropna(subset=need).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _t(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _candle_data(df: pd.DataFrame) -> list[dict]:
    return [
        {"time": _t(ts), "open": round(float(r.Open), 4), "high": round(float(r.High), 4),
         "low": round(float(r.Low), 4), "close": round(float(r.Close), 4)}
        for ts, r in df.iterrows()
    ]


def _line_data(s: pd.Series) -> list[dict]:
    out = []
    for ts, v in s.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        out.append({"time": _t(ts), "value": round(float(v), 4)})
    return out


def _volume_data(df: pd.DataFrame) -> list[dict]:
    if "Volume" not in df.columns:
        return []
    out = []
    for ts, r in df.iterrows():
        v = r.get("Volume")
        if v is None or (isinstance(v, float) and math.isnan(v)) or v <= 0:
            continue
        up = float(r.Close) >= float(r.Open)
        out.append({"time": _t(ts), "value": float(v),
                    "color": theme.EMERALD_DIM if up else theme.GARNET_DIM})
    return out


# Overlay line colours — pearl/champagne family so the price series keeps the
# only saturated colours on the chart.
_OVERLAY_COLORS = ["#9BA3AE", "#C9A96A", "#6B7683", "#8A93A0", "#5C6570"]


def build_chart_html(
    df: pd.DataFrame,
    *,
    kind: str = "candles",                 # candles | line | area
    overlays: Optional[dict[str, pd.Series]] = None,
    rsi: Optional[pd.Series] = None,
    zones: Optional[FairValueZones] = None,
    levels: Optional[list[dict]] = None,
    log_scale: bool = False,
    height: int = 480,
) -> Optional[ChartHTML]:
    """
    Build one chart workspace, or ``None`` if the history is missing or too
    short to chart honestly.
    """
    data = _clean(df)
    if len(data) < MIN_BARS:
        return None

    candles = _candle_data(data)
    closes = _line_data(data["Close"])
    volume = _volume_data(data)
    overlay_series = [
        {"name": name, "color": _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)],
         "data": _line_data(s.reindex(data.index))}
        for i, (name, s) in enumerate((overlays or {}).items())
    ]
    overlay_series = [o for o in overlay_series if len(o["data"]) >= 2]
    rsi_data = _line_data(rsi.reindex(data.index)) if rsi is not None else []
    # Valuation zones and any caller-supplied levels (volume-profile POC and
    # value area, say) share one price-line list — they are the same kind of
    # object on the price axis, and the chart should not care which is which.
    price_lines = zones.lines() if zones is not None else []
    price_lines += [dict(ln) for ln in (levels or [])]

    payload = json.dumps({
        "kind": kind,
        "candles": candles,
        "closes": closes,
        "volume": volume,
        "overlays": overlay_series,
        "rsi": rsi_data,
        "priceLines": price_lines,
        "logScale": bool(log_scale),
        "mainHeight": int(height),
        "rsiHeight": 110,
        "colors": {
            "bg": theme.BG, "text": theme.MUTED, "grid": theme.GRID,
            "bull": theme.BULL, "bear": theme.BEAR,
            "line": theme.EMERALD_BRIGHT, "areaTop": "rgba(31,168,124,0.25)",
            "areaBottom": "rgba(31,168,124,0.02)",
            "rsi": "#C9A96A", "border": theme.BORDER,
        },
    })

    total_height = height + (140 if rsi_data else 0) + 8
    # The JS below is deliberately plain: create chart, add series, set data,
    # add price lines, optionally sync an RSI pane underneath.
    html = f"""
<div id="svp-lwc" style="width:100%;background:{theme.BG};
     border:1px solid {theme.BORDER};border-radius:8px;overflow:hidden;"></div>
<script src="{_CDN}"></script>
<script>
(function() {{
  const P = {payload};
  const root = document.getElementById("svp-lwc");
  if (!window.LightweightCharts) {{
    root.innerHTML = "<div style='color:{theme.MUTED};font:13px Inter,sans-serif;" +
      "padding:24px'>Chart library failed to load (network blocked). " +
      "The data itself is unaffected.</div>";
    return;
  }}
  const base = {{
    layout: {{ background: {{ type: "solid", color: P.colors.bg }},
              textColor: P.colors.text, fontSize: 11,
              fontFamily: "Inter, -apple-system, sans-serif" }},
    grid: {{ vertLines: {{ color: P.colors.grid }}, horzLines: {{ color: P.colors.grid }} }},
    crosshair: {{ mode: 0 }},
    timeScale: {{ borderColor: P.colors.border, rightOffset: 4 }},
    rightPriceScale: {{ borderColor: P.colors.border,
                        mode: P.logScale ? 1 : 0 }},
  }};

  const mainDiv = document.createElement("div");
  root.appendChild(mainDiv);
  const chart = LightweightCharts.createChart(mainDiv,
      Object.assign({{ width: root.clientWidth, height: P.mainHeight }}, base));

  let main;
  if (P.kind === "candles") {{
    main = chart.addCandlestickSeries({{
      upColor: P.colors.bull, downColor: P.colors.bear,
      wickUpColor: P.colors.bull, wickDownColor: P.colors.bear,
      borderVisible: false }});
    main.setData(P.candles);
  }} else if (P.kind === "area") {{
    main = chart.addAreaSeries({{ lineColor: P.colors.line, lineWidth: 2,
      topColor: P.colors.areaTop, bottomColor: P.colors.areaBottom }});
    main.setData(P.closes);
  }} else {{
    main = chart.addLineSeries({{ color: P.colors.line, lineWidth: 2 }});
    main.setData(P.closes);
  }}

  for (const pl of P.priceLines) {{
    main.createPriceLine({{ price: pl.price, color: pl.color,
      lineWidth: 1, lineStyle: pl.style, axisLabelVisible: true,
      title: pl.title }});
  }}

  for (const ov of P.overlays) {{
    const s = chart.addLineSeries({{ color: ov.color, lineWidth: 1,
      priceLineVisible: false, lastValueVisible: false,
      crosshairMarkerVisible: false }});
    s.setData(ov.data);
  }}

  if (P.volume.length) {{
    const vol = chart.addHistogramSeries({{ priceFormat: {{ type: "volume" }},
      priceScaleId: "vol" }});
    chart.priceScale("vol").applyOptions({{
      scaleMargins: {{ top: 0.82, bottom: 0 }} }});
    vol.setData(P.volume);
  }}

  let rsiChart = null;
  if (P.rsi.length) {{
    const rsiDiv = document.createElement("div");
    rsiDiv.style.borderTop = "1px solid " + P.colors.border;
    root.appendChild(rsiDiv);
    rsiChart = LightweightCharts.createChart(rsiDiv,
        Object.assign({{ width: root.clientWidth, height: P.rsiHeight }}, base,
                      {{ rightPriceScale: {{ borderColor: P.colors.border }} }}));
    const r = rsiChart.addLineSeries({{ color: P.colors.rsi, lineWidth: 1 }});
    r.setData(P.rsi);
    for (const lvl of [30, 70]) {{
      r.createPriceLine({{ price: lvl, color: P.colors.grid, lineWidth: 1,
        lineStyle: 3, axisLabelVisible: false, title: "" }});
    }}
    // Keep the two panes on the same time window in both directions.
    const link = (a, b) => a.timeScale().subscribeVisibleLogicalRangeChange(
      range => range && b.timeScale().setVisibleLogicalRange(range));
    link(chart, rsiChart); link(rsiChart, chart);
  }}

  chart.timeScale().fitContent();
  new ResizeObserver(() => {{
    const w = root.clientWidth;
    chart.applyOptions({{ width: w }});
    if (rsiChart) rsiChart.applyOptions({{ width: w }});
  }}).observe(root);
}})();
</script>
"""
    return ChartHTML(html=html, height=total_height)
