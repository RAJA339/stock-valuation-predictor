"""Chunking, cleaning and hashing. Pure Python, no Spark."""

from __future__ import annotations

import pytest

from pipelines.common.text import (
    chunk_text,
    clean_text,
    content_hash,
    detect_language,
    estimate_tokens,
)


class TestCleanText:
    def test_collapses_horizontal_whitespace_but_keeps_paragraphs(self):
        assert clean_text("a   b\n\n\n\nc") == "a b\n\nc"

    def test_strips_control_characters(self):
        assert "\x00" not in clean_text("before\x00after")

    def test_removes_extraction_boilerplate(self):
        cleaned = clean_text("Real content.\nPage 3 of 12\nMore content.")
        assert "Page 3 of 12" not in cleaned
        assert "Real content." in cleaned
        assert "More content." in cleaned

    def test_normalises_unicode_ligatures(self):
        # NFKC folds the ﬁ ligature that PDF extraction loves to emit.
        assert clean_text("ﬁnancial") == "financial"

    @pytest.mark.parametrize("value", [None, "", "   \n\t  "])
    def test_empty_inputs_return_empty_string(self, value):
        assert clean_text(value) == ""


class TestChunkText:
    def test_short_document_is_a_single_chunk(self):
        chunks = chunk_text("A short but perfectly serviceable sentence about markets.", 1200, 200)
        assert len(chunks) == 1
        assert chunks[0].index == 0

    def test_empty_document_produces_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text(None) == []

    def test_every_chunk_respects_the_size_limit(self):
        text = " ".join(f"word{i}" for i in range(4000))
        for chunk in chunk_text(text, chunk_size=500, chunk_overlap=50):
            assert chunk.char_count <= 500

    def test_chunks_are_indexed_contiguously_from_zero(self):
        text = "\n\n".join(f"Paragraph number {i}. " * 20 for i in range(10))
        indices = [c.index for c in chunk_text(text, 400, 50)]
        assert indices == list(range(len(indices)))

    def test_overlap_carries_context_between_chunks(self):
        text = " ".join(f"token{i}" for i in range(600))
        chunks = chunk_text(text, chunk_size=400, chunk_overlap=120)
        assert len(chunks) > 1
        # The tail of chunk N must reappear at the head of chunk N+1, else a
        # fact spanning the boundary is retrievable from neither.
        first_tail = chunks[0].text[-60:]
        assert any(word in chunks[1].text for word in first_tail.split()[:3])

    def test_zero_overlap_is_allowed(self):
        chunks = chunk_text("x " * 2000, chunk_size=300, chunk_overlap=0)
        assert len(chunks) > 1

    def test_prefers_paragraph_boundaries(self):
        # Two paragraphs that comfortably fit one chunk each should not be
        # split mid-sentence.
        para_a = "Alpha sentence about revenue growth. " * 5
        para_b = "Beta sentence about margin compression. " * 5
        chunks = chunk_text(f"{para_a}\n\n{para_b}", chunk_size=220, chunk_overlap=0)
        assert all(c.text.strip() for c in chunks)
        assert any("Alpha" in c.text for c in chunks)
        assert any("Beta" in c.text for c in chunks)

    def test_is_deterministic(self):
        text = "Deterministic content. " * 200
        assert [c.text for c in chunk_text(text, 300, 50)] == [
            c.text for c in chunk_text(text, 300, 50)
        ]

    def test_no_content_is_lost_for_a_document_without_separators(self):
        text = "x" * 1000
        chunks = chunk_text(text, chunk_size=300, chunk_overlap=0)
        assert "".join(c.text for c in chunks) == text

    @pytest.mark.parametrize(
        "size,overlap",
        [(0, 0), (-1, 0), (100, -5), (100, 100), (100, 200)],
    )
    def test_invalid_parameters_are_rejected(self, size, overlap):
        with pytest.raises(ValueError):
            chunk_text("some text", size, overlap)

    def test_token_estimate_tracks_char_count(self):
        chunks = chunk_text("word " * 500, 400, 40)
        for chunk in chunks:
            assert chunk.token_estimate == max(1, round(chunk.char_count / 4))


class TestContentHash:
    def test_is_stable_across_calls(self):
        assert content_hash("a", "b") == content_hash("a", "b")

    def test_distinguishes_boundary_placement(self):
        # Without length prefixing, ("ab","c") and ("a","bc") would collide --
        # a real risk when hashing a natural key made of user strings.
        assert content_hash("ab", "c") != content_hash("a", "bc")

    def test_none_and_empty_string_are_distinct(self):
        assert content_hash(None) != content_hash("")

    def test_order_matters(self):
        assert content_hash("a", "b") != content_hash("b", "a")


class TestMisc:
    def test_estimate_tokens_of_empty_is_zero(self):
        assert estimate_tokens("") == 0

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The quarterly report was released today.", "en"),
            ("四半期報告書が本日公表されました", "cjk"),
            ("Квартальный отчет опубликован", "cyrillic"),
            ("12345 !!! ???", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_detect_language(self, text, expected):
        assert detect_language(text) == expected
