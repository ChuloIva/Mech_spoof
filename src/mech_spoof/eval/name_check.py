"""String-match helper for name-based conflict evaluation."""

from __future__ import annotations


def contains_name(text: str, name: str) -> bool:
    return name.lower() in text.lower()
