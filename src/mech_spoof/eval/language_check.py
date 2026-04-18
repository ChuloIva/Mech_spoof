"""Language detection wrapper."""

from __future__ import annotations


_ISO_TO_NAME = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "ru": "russian",
    "ja": "japanese",
    "zh-cn": "chinese",
    "zh-tw": "chinese",
    "ar": "arabic",
    "ko": "korean",
}


def detect_language(text: str) -> str:
    """Return a lowercase language name or 'unknown'."""
    if not text or not text.strip():
        return "unknown"
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 42
        code = detect(text).lower()
    except Exception:
        return "unknown"
    return _ISO_TO_NAME.get(code, code)


def language_matches(text: str, expected: str) -> bool:
    return detect_language(text) == expected.lower()
