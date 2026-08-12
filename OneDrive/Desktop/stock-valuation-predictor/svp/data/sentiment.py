"""
Earnings-call sentiment pipeline.
=================================

Scores the tone of quarterly earnings-call transcripts. When ``transformers`` +
``torch`` are available the model of record is **FinBERT**
(``ProsusAI/finbert``); otherwise a fast, dependency-free finance lexicon
fallback (a compact Loughran-McDonald-style word list) produces a comparable
positive/neutral/negative score so the feature is always populated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Lazy / optional heavyweight model.
_finbert = None
_finbert_failed = False

# Compact finance sentiment lexicon (Loughran-McDonald inspired).
_POSITIVE = {
    "growth", "record", "strong", "beat", "exceeded", "outperform", "profit",
    "profitable", "gain", "gains", "increase", "increased", "improve", "improved",
    "improvement", "expansion", "robust", "momentum", "upside", "confident",
    "accelerate", "accelerating", "surpassed", "opportunity", "favorable",
    "efficiency", "margin", "raised", "upgraded", "resilient", "healthy",
}
_NEGATIVE = {
    "decline", "declined", "decrease", "decreased", "loss", "losses", "weak",
    "weakness", "miss", "missed", "shortfall", "headwind", "headwinds",
    "pressure", "challenging", "slowdown", "downgrade", "downgraded", "risk",
    "risks", "uncertain", "uncertainty", "litigation", "impairment", "restructuring",
    "layoff", "layoffs", "default", "concern", "concerns", "disappointing", "cautious",
}


@dataclass
class SentimentResult:
    score: float          # -1 (bearish) … +1 (bullish)
    label: str            # "positive" | "neutral" | "negative"
    positive: float       # class probabilities (0..1)
    neutral: float
    negative: float
    source: str           # "finbert" | "lexicon"

    @property
    def is_live(self) -> bool:
        return self.source == "finbert"


def _load_finbert():
    global _finbert, _finbert_failed
    if _finbert is not None or _finbert_failed:
        return _finbert
    try:
        from transformers import pipeline  # type: ignore

        _finbert = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
    except Exception:
        _finbert_failed = True
        _finbert = None
    return _finbert


def _finbert_score(text: str):
    clf = _load_finbert()
    if clf is None:
        return None
    try:
        # Truncate to the model's window; average over chunks for long transcripts.
        chunks = [text[i : i + 1500] for i in range(0, min(len(text), 9000), 1500)] or [text]
        agg = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for ch in chunks:
            for d in clf(ch, truncation=True)[0]:
                agg[d["label"].lower()] += d["score"]
        n = len(chunks)
        pos, neu, neg = agg["positive"] / n, agg["neutral"] / n, agg["negative"] / n
        total = pos + neu + neg or 1.0
        pos, neu, neg = pos / total, neu / total, neg / total
        label = max((("positive", pos), ("neutral", neu), ("negative", neg)), key=lambda x: x[1])[0]
        return SentimentResult(pos - neg, label, pos, neu, neg, "finbert")
    except Exception:
        return None


def _lexicon_score(text: str) -> SentimentResult:
    tokens = [t.strip(".,!?:;()[]\"'").lower() for t in text.split()]
    pos = sum(t in _POSITIVE for t in tokens)
    neg = sum(t in _NEGATIVE for t in tokens)
    hits = pos + neg
    if hits == 0:
        return SentimentResult(0.0, "neutral", 0.0, 1.0, 0.0, "lexicon")
    score = (pos - neg) / hits
    # Map net score → pseudo-probabilities.
    p_pos = max(0.0, score)
    p_neg = max(0.0, -score)
    p_neu = 1.0 - (p_pos + p_neg)
    label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
    return SentimentResult(score, label, p_pos, p_neu, p_neg, "lexicon")


def score_transcript(text: Optional[str]) -> SentimentResult:
    """Return a :class:`SentimentResult` for an earnings-call transcript."""
    if not text or not text.strip():
        return SentimentResult(0.0, "neutral", 0.0, 1.0, 0.0, "lexicon")
    return _finbert_score(text) or _lexicon_score(text)


SAMPLE_TRANSCRIPT = (
    "Thank you all for joining. This quarter we delivered record revenue with strong "
    "double-digit growth and expanding margins. Demand momentum remained robust across "
    "segments and we raised our full-year guidance. While we noted some macro headwinds "
    "and modest pricing pressure in one region, our balance sheet is healthy and free cash "
    "flow improved meaningfully. We remain confident in the accelerating opportunity ahead."
)
