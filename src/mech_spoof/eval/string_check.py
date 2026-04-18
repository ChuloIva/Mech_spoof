"""Structured string predicates for factual / substring evaluation.

Recognized win specs:
  contains:<substr>
  absent:<substr>
  startswith:<prefix>
  not_startswith:<prefix>
  endswith:<suffix>
  not_endswith:<suffix>
"""

from __future__ import annotations


def evaluate_string_spec(text: str, spec: str) -> bool:
    if not spec:
        return False
    if ":" not in spec:
        return spec.lower() in text.lower()
    op, _, arg = spec.partition(":")
    op = op.strip().lower()
    arg = arg.strip()
    low_text = text.lower()
    low_arg = arg.lower()
    if op == "contains":
        return low_arg in low_text
    if op == "absent":
        return low_arg not in low_text
    if op == "startswith":
        return low_text.lstrip().startswith(low_arg)
    if op == "not_startswith":
        return not low_text.lstrip().startswith(low_arg)
    if op == "endswith":
        return low_text.rstrip().endswith(low_arg)
    if op == "not_endswith":
        return not low_text.rstrip().endswith(low_arg)
    return False
