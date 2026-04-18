"""Activation extraction smoke test on gpt2 (small, CPU)."""

from __future__ import annotations

import os

import pytest


@pytest.mark.slow
def test_activation_extraction_gpt2():
    """End-to-end: load tiny gpt2, extract residual stream, probe on random synthetic labels runs."""
    if os.environ.get("MECH_SPOOF_OFFLINE") == "1":
        pytest.skip("offline mode")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    # Build a minimal LoadedModel-like object
    class _Loaded:
        hf_model = model
        tokenizer = tok
        device = "cpu"
        n_layers = model.config.n_layer
        d_model = model.config.n_embd

        def layer_module(self, i):
            return self.hf_model.transformer.h[i]

    loaded = _Loaded()

    # Minimal PromptBundle-compatible object
    class _Bundle:
        def __init__(self, ids):
            self.input_ids = ids
            self.response_first_pos = len(ids) - 1
            self.instruction_token_span = (0, len(ids))

    prompt_ids = tok.encode("Hello world, this is a test.")
    bundle = _Bundle(prompt_ids)

    from mech_spoof.activations import extract_at_positions, response_first_position
    acts = extract_at_positions(
        loaded, [bundle, bundle], response_first_position, progress=False
    )
    assert acts.shape == (2, loaded.n_layers, loaded.d_model)
