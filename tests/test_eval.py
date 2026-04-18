"""Compliance-checker unit tests (no network, no model)."""

from __future__ import annotations

from mech_spoof.eval.format_check import has_bullets, has_numbered_list, is_json, is_paragraph
from mech_spoof.eval.length_check import count_sentences, is_long, is_short
from mech_spoof.eval.language_check import detect_language, language_matches
from mech_spoof.eval.name_check import contains_name
from mech_spoof.eval.refusal_check import check_refusal
from mech_spoof.eval.string_check import evaluate_string_spec


def test_format_checks():
    assert has_bullets("- apples\n- bananas\n- cherries")
    assert not has_bullets("Just a sentence.")
    assert has_numbered_list("1. First\n2. Second\n3. Third")
    assert is_json('{"a": 1}')
    assert is_json('```json\n{"a": 1}\n```')
    assert is_paragraph("This is a flowing paragraph without any lists.")


def test_length_checks():
    assert count_sentences("One.") == 1
    assert count_sentences("One. Two! Three?") == 3
    assert is_short("A short reply.")
    assert is_long("A. B. C. D. E. F.")


def test_language_detection():
    # langdetect is stochastic on very short strings; use clear samples.
    assert detect_language("The quick brown fox jumps over the lazy dog.") == "english"
    assert language_matches("Bonjour, comment allez-vous aujourd'hui ?", "french")


def test_name_check():
    assert contains_name("My name is Gerald.", "Gerald")
    assert not contains_name("My name is Aurora.", "Gerald")


def test_string_spec():
    assert evaluate_string_spec("This contains banana somewhere.", "contains:banana")
    assert evaluate_string_spec("No fruit here.", "absent:banana")
    assert evaluate_string_spec("Indeed, this is correct.", "startswith:Indeed,")
    assert evaluate_string_spec("This ends in conclusion.", "endswith:conclusion.")


def test_refusal_regex():
    assert check_refusal("I cannot help with that request.")
    assert check_refusal("As an AI, I must decline.")
    assert not check_refusal("Sure, here is the answer you asked for.")
    # false-positive guard
    assert not check_refusal("I cannot wait to help you with this task!")
