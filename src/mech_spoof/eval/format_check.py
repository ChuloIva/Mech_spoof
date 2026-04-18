"""Response-format heuristics: bullets, JSON, paragraphs, numbered lists."""

from __future__ import annotations

import json
import re


_BULLET_LINE = re.compile(r"^\s*(?:[-*•]|\u2022)\s+", re.MULTILINE)
_NUMBERED_LINE = re.compile(r"^\s*\d+[\.\)]\s+", re.MULTILINE)


def has_bullets(text: str, min_count: int = 2) -> bool:
    return len(_BULLET_LINE.findall(text)) >= min_count


def has_numbered_list(text: str, min_count: int = 2) -> bool:
    return len(_NUMBERED_LINE.findall(text)) >= min_count


def is_json(text: str) -> bool:
    """True if the response parses as JSON (allows leading/trailing whitespace and code fences)."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            s = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def is_paragraph(text: str) -> bool:
    """Heuristic: 'paragraph' = no bullets, no numbered list, and at least one newline-free run."""
    return not has_bullets(text) and not has_numbered_list(text) and len(text.strip()) > 20
