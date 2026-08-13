"""
Query engine over indexed filings.
==================================

Answers are **extractive by default**: bullets are quoted verbatim from the
filing, each carrying its Item and filing date. That is a deliberate choice
rather than a limitation. An abstractive summary of a legal document invites
paraphrase that a reader may take as the filer's own words, and a research
note that quotes accurately is worth more than one that reads smoothly. Every
bullet here can be checked against EDGAR.

When an LLM key is configured the same retrieved passages are handed to a
model for synthesis, and the quotes are still shown beneath it.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from .store import FilingIndex, Hit, has_llm_key

# Question shapes that mean "compare the two filings" rather than "look up".
_DIFF_HINTS = re.compile(
    r"\b(new|newly|added|change[sd]?|different|removed|dropped|no longer|"
    r"this year|latest|versus|vs\.?|compared)\b", re.IGNORECASE)

_SECTION_HINTS = [
    (re.compile(r"\brisk|risk factor|threat|litigation|regulatory\b", re.I), "Risk Factors"),
    (re.compile(r"\bmd&a|discussion|analysis|liquidity|margin|revenue|outlook|guidance\b", re.I), "MD&A"),
    (re.compile(r"\bmarket risk|interest rate|currency|hedg\w+\b", re.I), "Market Risk"),
    (re.compile(r"\bbusiness|segment|product|competition\b", re.I), "Business"),
]


@dataclass
class Bullet:
    quote: str
    citation: str
    item: str
    score: float

    def as_markdown(self) -> str:
        return f'- "{self.quote}"\n  <br/><sub>— {self.citation}</sub>'


@dataclass
class Answer:
    question: str
    mode: str                    # "lookup" | "diff"
    section: Optional[str]
    bullets: list[Bullet] = field(default_factory=list)
    synthesis: str = ""          # populated only when an LLM answered
    note: str = ""

    @property
    def empty(self) -> bool:
        return not self.bullets


def detect_section(question: str) -> Optional[str]:
    """Route a question to an Item so retrieval is not diluted by the whole filing."""
    for pat, label in _SECTION_HINTS:
        if pat.search(question):
            return label
    return None


def _trim(text: str, max_words: int = 55) -> str:
    """Cut a passage to a quotable length on a sentence edge where possible."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, n = [], 0
    for s in sentences:
        w = len(s.split())
        if out and n + w > max_words:
            break
        out.append(s)
        n += w
    quote = " ".join(out) if out else " ".join(text.split()[:max_words])
    return quote if len(quote.split()) <= max_words else \
        textwrap.shorten(quote, width=max_words * 7, placeholder=" …")


def answer(
    question: str,
    latest_index: FilingIndex,
    prior_index: Optional[FilingIndex] = None,
    k: int = 6,
    section: Optional[str] = None,
    force_mode: Optional[str] = None,
) -> Answer:
    """
    Answer a question over the indexed filings.

    Comparative questions ("what is new…") are routed to novelty scoring
    against the prior filing; everything else is straight retrieval.
    """
    section = section or detect_section(question)
    wants_diff = bool(_DIFF_HINTS.search(question))
    mode = force_mode or ("diff" if (wants_diff and prior_index is not None) else "lookup")

    ans = Answer(question=question, mode=mode, section=section)

    if mode == "diff":
        hits: list[Hit] = latest_index.novel_chunks(prior_index, item_filter=section, k=k * 3)
        # Re-rank novel passages by relevance to the question itself, so
        # "new risks about supply chain" does not return every new paragraph.
        if hits:
            rel = {id(h.chunk): 0.0 for h in hits}
            for h in latest_index.search(question, k=len(latest_index.chunks),
                                         item_filter=section):
                rel[id(h.chunk)] = h.score
            hits.sort(key=lambda h: (0.65 * h.score + 0.35 * rel.get(id(h.chunk), 0.0)),
                      reverse=True)
        hits = hits[:k]
        if not hits:
            ans.note = ("No passage in the latest filing was materially different "
                        "from the prior one for this question — the language appears "
                        "carried over.")
    else:
        hits = latest_index.search(question, k=k, item_filter=section)
        if not hits:
            hits = latest_index.search(question, k=k)      # widen if the Item was wrong
            ans.section = None
        if not hits:
            ans.note = "Nothing in the indexed sections matched this question."

    ans.bullets = [
        Bullet(quote=_trim(h.chunk.text), citation=h.chunk.citation,
               item=h.chunk.item, score=h.score)
        for h in hits
    ]

    if ans.bullets and has_llm_key():
        ans.synthesis = _synthesise(question, ans.bullets)
    return ans


def _synthesise(question: str, bullets: list[Bullet]) -> str:
    """
    Optional LLM pass over the retrieved passages.

    Deliberately narrow: the model sees only the retrieved quotes and is asked
    to summarise them, so it cannot introduce facts absent from the filing.
    Returns "" on any failure — the extractive bullets are the real answer and
    must never depend on this.
    """
    try:
        from llama_index.core import Settings
        from llama_index.core.llms import ChatMessage
    except Exception:
        return ""

    context = "\n\n".join(f"[{b.citation}] {b.quote}" for b in bullets)
    prompt = (
        "You are summarising verbatim excerpts from an SEC filing. Answer the "
        "question in at most four short bullets. Use only the excerpts below. "
        "Cite the bracketed source after each bullet. If the excerpts do not "
        "answer the question, say so.\n\n"
        f"Question: {question}\n\nExcerpts:\n{context}"
    )
    try:
        llm = Settings.llm
        if llm is None:
            return ""
        return str(llm.chat([ChatMessage(role="user", content=prompt)])).strip()
    except Exception:
        return ""


SUGGESTED_QUESTIONS = [
    "What newly added risk factors appear in the latest 10-K?",
    "What risk factors were removed or softened versus last year?",
    "What does management say about margins and pricing?",
    "How is liquidity and capital allocation described?",
    "What changed in the discussion of competition?",
    "What supply-chain or concentration risks are disclosed?",
]
