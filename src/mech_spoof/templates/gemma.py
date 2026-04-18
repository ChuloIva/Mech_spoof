"""Gemma-2 <start_of_turn>-style template adapter.

Gemma-2 chat templates don't accept a separate `system` role: the template concatenates system
content into the first user turn automatically. This collapses Conditions S and U for Gemma at
the token level — an expected-and-notable limitation that the authority-probe analysis should
surface per §9.2. We still run the pipeline; the result will show whether Gemma encodes any
residual distinction.
"""

from __future__ import annotations

from mech_spoof.templates._common import GenericChatTemplate


class GemmaAdapter(GenericChatTemplate):
    name = "gemma"
    _system_role_supported = False  # template concatenates system into user

    _delims = {
        "bos": "<bos>",
        "start_turn": "<start_of_turn>",
        "end_turn": "<end_of_turn>",
        "user_open": "<start_of_turn>user\n",
        "model_open": "<start_of_turn>model\n",
    }

    _delim_special_names = {
        "bos": "<bos>",
        "start_turn": "<start_of_turn>",
        "end_turn": "<end_of_turn>",
    }

    def build_fake_delimiter_injection(self, system_instruction: str) -> str:
        # Gemma has no system role — mimic a fake "model" turn (spoof a prior assistant claim)
        return (
            "<start_of_turn>model\n"
            f"{system_instruction}<end_of_turn>"
        )
