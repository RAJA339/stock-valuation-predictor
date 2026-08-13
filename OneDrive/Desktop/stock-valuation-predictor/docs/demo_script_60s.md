# Intrinsic — 60-Second Demo Script

**Format:** screen capture, 1920×1080, 30 fps. Dark UI throughout.
**Voice:** measured and technical, not hyped. Roughly 150 words per minute —
the script below is ~145 words, which leaves air to breathe.
**Music:** minimal pulse, ducked 8 dB under the voice, out by 0:56.

A note on claims: the voiceover never says the model predicts the market or
that any signal is accurate. It says what the app *shows*. That is both honest
and, for a finance tool, the more credible pitch — anyone in the audience who
knows the domain will discount a demo that overclaims.

---

| Time | Visual | Voiceover |
|---|---|---|
| **0:00–0:04** | Cold open on the logo against onyx. Delta strokes draw on, bell curve fades up behind. Cut to the app, ticker field empty, cursor blinking. | *"Every valuation model gives you a number. Almost none of them show their work."* |
| **0:04–0:11** | Type `NVDA`. Click **Analyze**. Spinner. Quote header snaps in: large price, day change in emerald, `REGIME: Calm` pill. | *"Intrinsic pulls SEC filings, live market data and macro series — then values the company from its fundamentals."* |
| **0:11–0:19** | Headline row fills: Market Price, Intrinsic Value, p10–p90 range, signal. Slow push on the range card. | *"Not a point estimate — a quantile range. Gradient-boosted trees at the tenth, fiftieth and ninetieth percentile, so you see the uncertainty, not just the answer."* |
| **0:19–0:26** | Charts tab. Cycle 5m → 15m → 1h. Candles redraw, EMAs and VWAP track. Volume and RSI panels scroll into frame. | *"Live charting at five, ten, fifteen, thirty minutes and hourly — with the indicator panel every desk expects."* |
| **0:26–0:33** | Scroll to the **Measured Signal Accuracy** table. Hold on a row reading *"No better than chance"*. Cursor rests there. | *"And here's what most tools won't show you: each indicator's measured hit rate, with a confidence interval. When a signal is no better than a coin flip, it says so."* |
| **0:33–0:40** | Paste an earnings-call transcript into the sidebar. Sentiment score animates. Cut to Explainability tab — SHAP waterfall builds bar by bar. | *"FinBERT scores the earnings call. SHAP shows exactly which features moved the valuation, and by how much."* |
| **0:40–0:47** | Guardrails tab: Piotroski, Altman Z, Beneish M land as cards. One flips to a red guardrail warning. | *"Forensic scores run underneath — Piotroski, Altman, Beneish — so a cheap stock that's cheap for a reason gets flagged before you act on it."* |
| **0:47–0:54** | Filings RAG tab. Type *"What newly added risk factors appear in the latest 10-K?"* Bulleted quotes appear, each with an Item citation. | *"Ask the filings directly. Answers come back quoted verbatim, cited to the Item — checkable against EDGAR."* |
| **0:54–0:58** | Click **Generate Report**. PDF preview flips through pages: masthead, chart, tables. | *"One click for the full research note."* |
| **0:58–1:00** | Cut to logo on onyx. Wordmark: **Intrinsic**. Small line beneath: *Educational use only. Not investment advice.* | *"Intrinsic. Show your work."* |

---

## Production notes

- **Record the accuracy table with a real ticker.** If most rows read
  "No better than chance", keep it. That beat is the differentiator — cutting
  to a cherry-picked winner undoes the point the line is making.
- **The 0:26–0:33 hold is the longest static shot on purpose.** It is the one
  claim competitors cannot copy without doing the measurement.
- Capture at 30 fps and slow the chart redraws to ~0.7× in post; Streamlit
  reruns faster than the eye can follow.
- Keep the disclaimer card on screen for the full final two seconds. For a
  finance tool it reads as confidence, not as a caveat.
- If the live feed is rate-limited on the day, record the Charts tab from a
  local run — the synthetic-data banner would otherwise appear on camera.
