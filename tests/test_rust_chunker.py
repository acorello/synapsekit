"""Tests for the Rust text chunker — skipped if Rust extension is not built."""

from __future__ import annotations

import pytest

try:
    from synapsekit._rust_core import character_split, recursive_split

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust extension not built")


class TestRecursiveSplit:
    def test_empty(self):
        assert recursive_split("", 100, 0, ["\n\n"]) == []

    def test_short_text(self):
        assert recursive_split("hello", 100, 0, ["\n\n"]) == ["hello"]

    def test_paragraph_split(self):
        text = "para one\n\npara two\n\npara three"
        chunks = recursive_split(text, 20, 0, ["\n\n", "\n", " "])
        assert len(chunks) >= 2
        # All text should be present
        joined = "".join(chunks)
        assert "para one" in joined
        assert "para three" in joined

    def test_hard_split(self):
        text = "a" * 100
        chunks = recursive_split(text, 30, 0, ["\n\n"])
        assert all(len(c) <= 30 for c in chunks)


class TestCharacterSplit:
    def test_empty(self):
        assert character_split("", "\n", 100, 0) == []

    def test_short_text(self):
        assert character_split("hello", "\n", 100, 0) == ["hello"]

    def test_newline_split(self):
        text = "line1\nline2\nline3"
        chunks = character_split(text, "\n", 10, 0)
        assert len(chunks) >= 2
