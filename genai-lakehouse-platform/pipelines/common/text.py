"""Pure-Python text handling for the RAG path.

Everything here is deliberately free of Spark so it can be unit tested at
speed and reused from a serving-side application that must chunk a query
document exactly the way the index was built. A chunker that behaves
differently at write time and read time is the most common cause of "the
retrieval quality dropped and nobody changed anything".
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from typing import NamedTuple

# Ordered from strongest to weakest semantic boundary. The splitter walks down
# this list, only falling back to a weaker separator when a piece is still too
# large -- so a chunk breaks at a paragraph if it can, at a sentence if it
# must, and mid-word only as a last resort.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")

_WHITESPACE_RE = re.compile(r"[ \t\x0b\x0c\r]+")
_NEWLINES_RE = re.compile(r"\n{3,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")

# Boilerplate that survives PDF/HTML extraction and pollutes embeddings.
_BOILERPLATE_RE = re.compile(
    r"^\s*(page\s+\d+\s+of\s+\d+|confidential(?:\s+and\s+proprietary)?|"
    r"all\s+rights\s+reserved|\[?\s*table\s+of\s+contents\s*\]?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class Chunk(NamedTuple):
    """One retrievable unit of text."""

    index: int
    text: str
    char_count: int
    token_estimate: int


def clean_text(raw: str | None) -> str:
    """Normalise extracted document text without destroying its structure.

    Paragraph breaks are preserved (they are the chunker's strongest boundary
    signal); everything else collapses.
    """
    if not raw:
        return ""
    # NFKC folds ligatures and full-width forms that PDF extraction emits.
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _BOILERPLATE_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~4 characters per token for English prose.

    Good enough to enforce a model's context budget and to spot a chunk that
    is wildly out of band. Swap for a real tokenizer if you need exactness.
    """
    return max(1, round(len(text) / 4)) if text else 0


def _split_on_separator(text: str, separator: str) -> list[str]:
    """Split while keeping the separator attached to the left-hand piece."""
    if separator == " ":
        parts = text.split(" ")
        return [p + " " for p in parts[:-1]] + parts[-1:]
    pieces = text.split(separator)
    out = [p + separator for p in pieces[:-1]]
    if pieces[-1]:
        out.append(pieces[-1])
    return out


def _hard_split(text: str, limit: int) -> list[str]:
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _recursive_split(text: str, limit: int, separators: Iterable[str]) -> list[str]:
    """Break ``text`` into pieces no longer than ``limit`` characters."""
    if len(text) <= limit:
        return [text] if text else []

    seps = list(separators)
    if not seps:
        return _hard_split(text, limit)

    separator, rest = seps[0], seps[1:]
    if separator not in text:
        return _recursive_split(text, limit, rest)

    out: list[str] = []
    for piece in _split_on_separator(text, separator):
        if not piece:
            continue
        if len(piece) <= limit:
            out.append(piece)
        else:
            out.extend(_recursive_split(piece, limit, rest))
    return out


def chunk_text(
    text: str | None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    min_chunk_chars: int = 50,
) -> list[Chunk]:
    """Split ``text`` into overlapping, semantically-aligned chunks.

    ``chunk_overlap`` carries the tail of each chunk into the next one so a
    fact that straddles a boundary is still retrievable from at least one
    chunk.

    The final chunk is exempt from ``min_chunk_chars`` only when it is the
    single chunk produced -- a short document should still be indexed, but a
    50-character orphan at the end of a long one is noise.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned = clean_text(text)
    if not cleaned:
        return []

    pieces = _recursive_split(cleaned, chunk_size, _SEPARATORS)

    # Re-assemble the small pieces greedily up to chunk_size, then step back by
    # chunk_overlap so consecutive chunks share context.
    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        if buffer and len(buffer) + len(piece) > chunk_size:
            chunks.append(buffer.strip())
            buffer = (buffer[-chunk_overlap:] if chunk_overlap else "") + piece
            # An overlap tail plus an oversized piece can exceed chunk_size;
            # flush it rather than letting the buffer grow unbounded.
            while len(buffer) > chunk_size:
                chunks.append(buffer[:chunk_size].strip())
                buffer = buffer[chunk_size - chunk_overlap :]
        else:
            buffer += piece
    if buffer.strip():
        chunks.append(buffer.strip())

    kept = [c for c in chunks if len(c) >= min_chunk_chars]
    if not kept and chunks:  # short document: keep it rather than lose it
        kept = [max(chunks, key=len)]

    return [
        Chunk(index=i, text=c, char_count=len(c), token_estimate=estimate_tokens(c))
        for i, c in enumerate(kept)
    ]


def content_hash(*parts: str | None) -> str:
    """Stable SHA-256 over the given parts, used for ids and change detection.

    Parts are length-prefixed so ``("ab", "c")`` and ``("a", "bc")`` cannot
    collide -- a real risk when hashing a natural key made of user-supplied
    strings. NULL is prefixed with ``-1``, which no real length can take, so a
    missing value and an empty one are distinguishable: "we have no address"
    and "the address is blank" are different facts about a record.
    """
    digest = hashlib.sha256()
    for part in parts:
        if part is None:
            digest.update(b"-1\x1f\x1e")
            continue
        value = str(part)
        digest.update(str(len(value)).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(value.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def detect_language(text: str | None) -> str:
    """Very cheap script detection: ``en``, ``cjk``, ``cyrillic`` or ``unknown``.

    A placeholder for a real language id model; the point is that the column
    exists in Silver so multilingual corpora can be filtered at retrieval time.
    """
    if not text:
        return "unknown"
    sample = text[:2000]
    counts = {"cjk": 0, "cyrillic": 0, "latin": 0}
    for ch in sample:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:
            counts["cjk"] += 1
        elif 0x0400 <= code <= 0x04FF:
            counts["cyrillic"] += 1
        elif ch.isalpha() and code < 0x0250:
            counts["latin"] += 1
    if not any(counts.values()):
        return "unknown"
    dominant = max(counts, key=counts.get)
    return "en" if dominant == "latin" else dominant
