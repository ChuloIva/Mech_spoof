"""Shared fixtures.

Most unit tests run without a GPU. Integration tests load gpt2 (small, CPU-friendly) to smoke-test
the activation extraction + probe training pipeline. Tests that need a chat-template-aware
tokenizer use a mock-minimal tokenizer fixture.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def hf_tokenizer_qwen():
    """Real Qwen tokenizer (small download ~1MB for tokenizer-only)."""
    if os.environ.get("MECH_SPOOF_OFFLINE") == "1":
        pytest.skip("offline mode")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    except Exception as e:
        pytest.skip(f"could not load Qwen tokenizer: {e}")


@pytest.fixture(scope="session")
def hf_tokenizer_llama3():
    if os.environ.get("MECH_SPOOF_OFFLINE") == "1":
        pytest.skip("offline mode")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    except Exception as e:
        pytest.skip(f"could not load Llama-3 tokenizer (gated): {e}")


@pytest.fixture(scope="session")
def hf_tokenizer_gemma():
    if os.environ.get("MECH_SPOOF_OFFLINE") == "1":
        pytest.skip("offline mode")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    except Exception as e:
        pytest.skip(f"could not load Gemma tokenizer (gated): {e}")


@pytest.fixture(scope="session")
def hf_tokenizer_phi3():
    if os.environ.get("MECH_SPOOF_OFFLINE") == "1":
        pytest.skip("offline mode")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    except Exception as e:
        pytest.skip(f"could not load Phi-3.5 tokenizer: {e}")


@pytest.fixture(scope="session")
def hf_tokenizer_mistral():
    if os.environ.get("MECH_SPOOF_OFFLINE") == "1":
        pytest.skip("offline mode")
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    except Exception as e:
        pytest.skip(f"could not load Mistral tokenizer (gated): {e}")
