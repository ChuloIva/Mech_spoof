"""Response-length heuristics."""

from __future__ import annotations

import re


_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")


def count_sentences(text: str) -> int:
    """Crude sentence counter: split on .!? boundaries."""
    if not text.strip():
        return 0
    matches = _SENTENCE_END.findall(text)
    # If the last sentence doesn't end with punctuation, still count it as one.
    if text.strip()[-1] not in ".!?":
        return max(1, len(matches) + 1)
    return max(1, len(matches))


def is_short(text: str, max_sentences: int = 2) -> bool:
    return count_sentences(text) <= max_sentences


def is_long(text: str, min_sentences: int = 5) -> bool:
    return count_sentences(text) >= min_sentences
