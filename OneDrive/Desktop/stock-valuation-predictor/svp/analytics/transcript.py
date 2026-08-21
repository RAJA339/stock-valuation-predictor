"""
Executive language on earnings calls — hedging, evasion, and commitment.
========================================================================

FinBERT (in :mod:`svp.data.sentiment`) answers "does this read positive or
negative". It does not answer the question that separates a confident
management team from an evasive one, which is *how much of what they said was
actually committed to*. "Revenue will grow 12% next quarter" and "revenue
could potentially see some improvement, subject to headwinds" can score the
same polarity while carrying completely different information.

This module measures that second axis directly and countably:

- **Hedges** — modal verbs and vague quantifiers that decline to commit
  ("might", "could", "approximately", "somewhat").
- **Headwind language** — the euphemisms used to name difficulty without
  owning it ("challenging environment", "transitory", "macro backdrop").
- **Commitments** — the opposite register: definite, dated, quantified
  ("will", "committed to", "on track to", plus any hard numeric guidance).

The **hedging index** is hedges as a share of hedges plus commitments, so it
is a ratio a reader can interpret without knowing the transcript's length: at
0.5 the speaker qualifies as often as they commit.

Deliberately mechanical. These are word counts, not a judgement about
honesty, and a high index is a prompt to read the transcript rather than a
finding about the company. The counts and the matched phrases are both
returned so any number here can be audited against the text that produced it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

#: Qualifiers that decline to commit. Multi-word entries are matched as
#: phrases; single words on word boundaries so "could" never matches "couldn't
#: be clearer" mid-word or "approximate" inside "approximately" twice.
HEDGE_TERMS = (
    "might", "may", "could", "possibly", "perhaps", "potentially",
    "approximately", "roughly", "around", "somewhat", "relatively",
    "hopefully", "we think", "we believe", "we feel", "should be",
    "in the range of", "or so", "to some extent", "generally",
    "for the most part", "at this point", "too early to", "hard to say",
    "difficult to predict", "not prepared to", "won't comment",
    "we'll see", "time will tell", "subject to",
)

#: Naming difficulty without owning it.
HEADWIND_TERMS = (
    "headwind", "headwinds", "challenging", "choppy", "soft demand",
    "macro backdrop", "macro environment", "uncertainty", "uncertain",
    "transitory", "one-time", "one-off", "non-recurring", "pressure",
    "pushed out", "elongated", "digestion", "pause", "slowdown",
    "cautious", "prudent", "reset expectations",
)

#: Definite, dated, quantified — the register of a team willing to be held to it.
COMMITMENT_TERMS = (
    "will", "we are", "we're", "committed to", "on track", "on track to",
    "confident", "confident that", "expect to deliver", "guidance of",
    "we guide", "reaffirm", "reiterate", "raising guidance", "raised our",
    "increasing our", "delivered", "achieved", "exceeded", "closed",
    "signed", "shipped", "launched",
)

#: Hard guidance: a number with a unit is a commitment whatever verb precedes it.
_NUMERIC_GUIDANCE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|bps|basis points|million|billion|"
    r"bn|mm|dollars)\b", re.I)

MIN_WORDS = 40          # below this, ratios are noise


@dataclass
class LanguageProfile:
    """Countable features of how management spoke."""
    words: int
    hedges: int
    headwinds: int
    commitments: int
    numeric_guidance: int
    hedging_index: float             # hedges / (hedges + commitments)
    hedges_per_1k: float
    headwinds_per_1k: float
    matched_hedges: list = field(default_factory=list)
    matched_headwinds: list = field(default_factory=list)
    matched_commitments: list = field(default_factory=list)

    @property
    def tone(self) -> str:
        """Where this sits on the qualify-versus-commit axis."""
        if self.hedging_index >= 0.65:
            return "Heavily hedged"
        if self.hedging_index >= 0.45:
            return "Mixed"
        return "Committed"

    def reading(self) -> str:
        base = (
            f"Across {self.words:,} words: {self.hedges} hedging phrases "
            f"({self.hedges_per_1k:.1f} per thousand words), "
            f"{self.commitments} commitments and {self.numeric_guidance} "
            f"quantified figures. The hedging index is "
            f"{self.hedging_index:.2f} — **{self.tone.lower()}**."
        )
        if self.hedging_index >= 0.65:
            base += (" Management qualified far more often than they committed. "
                     "That is a property of the language, not evidence about "
                     "the business — read the passages below and judge them.")
        elif self.hedging_index < 0.45:
            base += (" Management committed more often than they qualified, "
                     "which is the register of a team willing to be measured "
                     "against what they said.")
        if self.headwinds >= 8:
            base += (f" {self.headwinds} headwind-style phrases appear, "
                     "concentrated enough to be worth reading in context.")
        return base


def _count(text_low: str, terms) -> tuple[int, list]:
    """Count phrase occurrences on word boundaries; return count and samples."""
    total, matched = 0, []
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        found = re.findall(pattern, text_low)
        if found:
            total += len(found)
            matched.append((term, len(found)))
    matched.sort(key=lambda kv: -kv[1])
    return total, matched[:12]


def analyse(text: str) -> Optional[LanguageProfile]:
    """
    Measure hedging, headwind and commitment language in a transcript.

    Returns ``None`` for text too short for the ratios to mean anything —
    a two-sentence excerpt with one "might" in it is not 50% hedged.
    """
    if not text or not text.strip():
        return None
    low = " " + re.sub(r"\s+", " ", text.lower()) + " "
    words = len(low.split())
    if words < MIN_WORDS:
        return None

    hedges, m_hedge = _count(low, HEDGE_TERMS)
    headwinds, m_head = _count(low, HEADWIND_TERMS)
    commits, m_commit = _count(low, COMMITMENT_TERMS)
    numeric = len(_NUMERIC_GUIDANCE.findall(low))

    # Quantified guidance counts as commitment: a number is something you can
    # be held to regardless of the verb that introduced it.
    commit_total = commits + numeric
    denom = hedges + commit_total
    index = (hedges / denom) if denom else 0.0

    return LanguageProfile(
        words=words, hedges=hedges, headwinds=headwinds,
        commitments=commit_total, numeric_guidance=numeric,
        hedging_index=float(index),
        hedges_per_1k=float(hedges / words * 1000),
        headwinds_per_1k=float(headwinds / words * 1000),
        matched_hedges=m_hedge, matched_headwinds=m_head,
        matched_commitments=m_commit,
    )


def drift(current: LanguageProfile, prior: LanguageProfile) -> str:
    """
    Compare two calls: is management hedging more than they used to?

    The level of hedging varies by CFO, industry and legal counsel; the
    *change* between consecutive calls from the same company is the part that
    carries information, which is why this is offered separately.
    """
    delta = current.hedging_index - prior.hedging_index
    if abs(delta) < 0.05:
        return (f"Hedging index {current.hedging_index:.2f} versus "
                f"{prior.hedging_index:.2f} — essentially unchanged between "
                "the two transcripts.")
    direction = "more" if delta > 0 else "less"
    return (f"Hedging index moved from {prior.hedging_index:.2f} to "
            f"{current.hedging_index:.2f} ({delta:+.2f}) — management "
            f"qualified {direction} than in the earlier call. A rise is worth "
            "pairing with what changed in the numbers; it is a change in "
            "language, and language alone.")
