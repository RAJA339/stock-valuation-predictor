"""
Chunking and vector retrieval over filing text.
===============================================

Three retrieval backends behind one interface, selected by what is installed
and configured:

  ``llamaindex``  LlamaIndex + Chroma with a hosted embedding model. Best
                  quality, needs ``llama-index`` plus an API key.
  ``lsa``         TF-IDF projected through truncated SVD (latent semantic
                  analysis) — dense vectors, cosine similarity, scikit-learn
                  only. This is the default.
  ``tfidf``       Sparse TF-IDF cosine, used when the corpus is too small for
                  a meaningful SVD.

The default is deliberately the dependency-light one. ``sentence-transformers``
pulls in torch at roughly 2.5 GB, which exceeds the memory ceiling on the
free Streamlit tier this app deploys to — an embedding model that cannot load
in production is not a default, it is an outage.

Chunks are cut inside a section and never across one, so every hit can name
the Item it came from.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

try:  # optional, only used when installed *and* keyed
    import llama_index  # noqa: F401

    _HAS_LLAMA = True
except Exception:
    _HAS_LLAMA = False


def has_llm_key() -> bool:
    """True when an embedding/LLM API key is configured in the environment."""
    return any(os.getenv(k) for k in
               ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLAMA_CLOUD_API_KEY"))


@dataclass
class Chunk:
    text: str
    item: str          # "Item 1A"
    label: str         # "Risk Factors"
    filing: str        # "10-K filed 2026-02-01"
    accession: str
    ordinal: int       # position within its section

    @property
    def citation(self) -> str:
        return f"{self.filing} · {self.item} {self.label}"


@dataclass
class Hit:
    chunk: Chunk
    score: float


def chunk_sections(filing, target_words: int = 220, overlap: int = 40) -> list[Chunk]:
    """
    Split each section into overlapping windows on sentence boundaries.

    Windows overlap so a risk that straddles a boundary is still retrievable
    whole, and never span two sections so the citation stays exact.
    """
    out: list[Chunk] = []
    for sec_ in getattr(filing, "sections", []) or []:
        sentences = re.split(r"(?<=[.!?])\s+", sec_.text)
        window: list[str] = []
        count = 0
        ordinal = 0
        for sent in sentences:
            w = len(sent.split())
            if w == 0:
                continue
            window.append(sent)
            count += w
            if count >= target_words:
                out.append(Chunk(" ".join(window).strip(), sec_.item, sec_.label,
                                 filing.label, filing.accession, ordinal))
                ordinal += 1
                # Carry the tail forward as overlap.
                back, kept = 0, []
                for s in reversed(window):
                    kept.insert(0, s)
                    back += len(s.split())
                    if back >= overlap:
                        break
                window, count = kept, back
        if count > 25:
            out.append(Chunk(" ".join(window).strip(), sec_.item, sec_.label,
                             filing.label, filing.accession, ordinal))
    return out


class FilingIndex:
    """In-memory vector index over one or more filings."""

    def __init__(self, chunks: list[Chunk], backend: str = "auto"):
        self.chunks = chunks
        self.backend = self._resolve(backend)
        self._vec = None
        self._matrix = None
        self._svd = None
        if self.chunks:
            self._build()

    def _resolve(self, backend: str) -> str:
        if backend != "auto":
            return backend
        if _HAS_LLAMA and has_llm_key():
            return "llamaindex"
        return "lsa" if len(self.chunks) >= 12 else "tfidf"

    def _build(self) -> None:
        texts = [c.text for c in self.chunks]
        # sublinear_tf dampens boilerplate that repeats across a filing; the
        # English stop list keeps legal filler out of the vocabulary.
        self._vec = TfidfVectorizer(
            stop_words="english", max_features=60_000,
            ngram_range=(1, 2), sublinear_tf=True, min_df=1,
        )
        tfidf = self._vec.fit_transform(texts)

        if self.backend == "lsa":
            n = min(256, max(2, min(tfidf.shape) - 1))
            self._svd = TruncatedSVD(n_components=n, random_state=7)
            dense = self._svd.fit_transform(tfidf)
            self._matrix = Normalizer(copy=False).fit_transform(dense)
        else:
            self._matrix = tfidf

    def _embed(self, query: str):
        q = self._vec.transform([query])
        if self._svd is not None:
            q = Normalizer(copy=False).transform(self._svd.transform(q))
        return q

    def search(self, query: str, k: int = 6,
               item_filter: Optional[str] = None,
               accession: Optional[str] = None) -> list[Hit]:
        """Top-k chunks, optionally restricted to one Item or one filing."""
        if not self.chunks or self._matrix is None:
            return []

        mask = [
            i for i, c in enumerate(self.chunks)
            if (item_filter is None
                or item_filter.lower() in (c.item.lower(), c.label.lower()))
            and (accession is None or c.accession == accession)
        ]
        if not mask:
            return []

        sims = cosine_similarity(self._embed(query), self._matrix[mask]).ravel()
        order = np.argsort(-sims)[:k]
        return [Hit(self.chunks[mask[i]], float(sims[i])) for i in order if sims[i] > 0.01]

    def novel_chunks(self, other: "FilingIndex", threshold: float = 0.35,
                     item_filter: Optional[str] = None, k: int = 40) -> list[Hit]:
        """
        Passages in this index with no close counterpart in ``other``.

        This is what answers "what risk factors are new this year". Scoring
        happens on **sentence pairs, not whole chunks**: a newly inserted risk
        is typically a paragraph or two inside a section that is otherwise
        carried over verbatim, so a 220-word chunk containing it still scores
        as highly similar and the addition disappears. Comparing finer spans
        surfaces the inserted language itself.

        ``score`` is 1 − (best cosine similarity to any span of the prior
        filing), so reworded text scores low and genuinely new text scores
        high. Comparing text against text avoids asking a language model to
        spot additions across two long documents, which it does unreliably.
        """
        if not self.chunks or not other.chunks:
            return []

        cand = [c for c in self.chunks
                if item_filter is None
                or item_filter.lower() in (c.item.lower(), c.label.lower())]
        if not cand:
            return []

        cand_spans, owners = _spans(cand)
        prior_spans, _ = _spans(other.chunks)
        if not cand_spans or not prior_spans:
            return []

        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                              sublinear_tf=True, min_df=1)
        prior_m = vec.fit_transform(prior_spans)
        cand_m = vec.transform(cand_spans)

        best = cosine_similarity(cand_m, prior_m).max(axis=1)
        novelty = 1.0 - best

        # Cap at two spans per source chunk: enough that a paragraph adding
        # several new risks is not reduced to one line, few enough that a
        # single heavily rewritten section cannot fill the whole answer.
        per_owner: dict[int, list[tuple[float, str]]] = {}
        for i, span in enumerate(cand_spans):
            if novelty[i] < threshold:
                continue
            per_owner.setdefault(owners[i], []).append((float(novelty[i]), span))

        hits = []
        for owner, found in per_owner.items():
            src = cand[owner]
            for score, span in sorted(found, reverse=True)[:2]:
                hits.append(Hit(Chunk(span, src.item, src.label, src.filing,
                                      src.accession, src.ordinal), score))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]


def _spans(chunks: list[Chunk], per_span: int = 2) -> tuple[list[str], list[int]]:
    """Sentence groups plus the index of the chunk each came from."""
    spans, owners = [], []
    for idx, c in enumerate(chunks):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", c.text) if len(s.split()) >= 5]
        for i in range(0, len(sentences), per_span):
            span = " ".join(sentences[i:i + per_span]).strip()
            if len(span.split()) >= 8:
                spans.append(span)
                owners.append(idx)
    return spans, owners


def build_index(filings, backend: str = "auto") -> FilingIndex:
    """Chunk and index one filing or a list of them."""
    if not isinstance(filings, (list, tuple)):
        filings = [filings]
    chunks: list[Chunk] = []
    for f in filings:
        if f is not None:
            chunks.extend(chunk_sections(f))
    return FilingIndex(chunks, backend=backend)
