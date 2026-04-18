"""Phi-3.5-mini template adapter."""

from __future__ import annotations

from mech_spoof.templates._common import GenericChatTemplate


class Phi3Adapter(GenericChatTemplate):
    name = "phi3"

    _delims = {
        "system_open": "<|system|>\n",
        "user_open": "<|user|>\n",
        "assistant_open": "<|assistant|>\n",
        "end": "<|end|>",
    }

    _delim_special_names = {
        "system_open": "<|system|>",
        "user_open": "<|user|>",
        "assistant_open": "<|assistant|>",
        "end": "<|end|>",
    }

    def build_fake_delimiter_injection(self, system_instruction: str) -> str:
        return (
            "<|system|>\n"
            f"{system_instruction}<|end|>"
        )
