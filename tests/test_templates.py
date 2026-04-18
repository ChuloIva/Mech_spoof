"""Template-adapter tests. Skip when the real tokenizer can't be downloaded."""

from __future__ import annotations

import pytest


def _template_and_tok(tok_fixture, template_name: str):
    from mech_spoof.templates import get_template
    return get_template(template_name, tok_fixture)


def _assert_basic_invariants(template, instruction: str):
    s = template.make_system_prompt(instruction)
    u = template.make_user_prompt(instruction)
    assert s.condition == "S"
    assert u.condition == "U"
    assert len(s.input_ids) > 0
    assert len(u.input_ids) > 0
    assert 0 <= s.response_first_pos < len(s.input_ids)
    assert 0 <= u.response_first_pos < len(u.input_ids)
    # instruction token span is within bounds
    for b in (s, u):
        st, en = b.instruction_token_span
        assert 0 <= st <= en <= len(b.input_ids), f"bad span {b.instruction_token_span}"


def test_chatml(hf_tokenizer_qwen):
    tpl = _template_and_tok(hf_tokenizer_qwen, "chatml")
    _assert_basic_invariants(tpl, "Always respond in bullet points.")

    report = tpl.fake_delimiter_tokenization_report()
    assert report.template_name == "chatml"
    assert isinstance(report.any_collapsed, bool)

    # Delimiter special token IDs should mostly resolve (non-None)
    ids = tpl.special_token_ids()
    assert ids.get("im_start") is not None
    assert ids.get("im_end") is not None


def test_llama3(hf_tokenizer_llama3):
    tpl = _template_and_tok(hf_tokenizer_llama3, "llama3")
    _assert_basic_invariants(tpl, "Always respond in French.")
    ids = tpl.special_token_ids()
    assert ids.get("start_header") is not None


def test_gemma(hf_tokenizer_gemma):
    tpl = _template_and_tok(hf_tokenizer_gemma, "gemma")
    _assert_basic_invariants(tpl, "Use only lowercase letters.")
    ids = tpl.special_token_ids()
    assert ids.get("start_turn") is not None


def test_phi3(hf_tokenizer_phi3):
    tpl = _template_and_tok(hf_tokenizer_phi3, "phi3")
    _assert_basic_invariants(tpl, "Always start your response with 'Certainly'.")
    ids = tpl.special_token_ids()
    assert ids.get("system_open") is not None


def test_mistral(hf_tokenizer_mistral):
    tpl = _template_and_tok(hf_tokenizer_mistral, "mistral")
    _assert_basic_invariants(tpl, "Respond only in one sentence.")
    ids = tpl.special_token_ids()
    assert ids.get("inst_open") is not None


def test_conflict_prompt_conditions(hf_tokenizer_qwen):
    tpl = _template_and_tok(hf_tokenizer_qwen, "chatml")
    real = tpl.make_conflict_prompt("system instr", "user instr", "REAL")
    none = tpl.make_conflict_prompt("system instr", "user instr", "NONE")
    fake = tpl.make_conflict_prompt("system instr", "user instr", "FAKE")
    assert real.condition == "REAL"
    assert none.condition == "NONE"
    assert fake.condition == "FAKE"
    # REAL should produce a shorter or equal-ish input than FAKE (fake wraps in fake delims)
    assert len(fake.text) > len(none.text)
