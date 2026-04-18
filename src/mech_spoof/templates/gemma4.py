"""Gemma 4 template adapter.

Gemma 4 replaces Gemma 2's `<start_of_turn>` / `<end_of_turn>` delimiters with the symmetric
pair `<|turn>` / `<turn|>` and introduces new multimodal / thinking tokens
(`<|channel>` / `<channel|>`, `<|think|>`, etc.). Prompt construction routes through
`tokenizer.apply_chat_template`, so the exact template string is whatever the HF tokenizer emits.

As with Gemma 2, the canonical chat template does not accept a separate `system` role — system
content is merged into the first user turn. The authority probe may therefore show a weak
S-vs-U distinction for this model; that is an expected finding (§9.2).
"""

from __future__ import annotations

from mech_spoof.templates._common import GenericChatTemplate


class Gemma4Adapter(GenericChatTemplate):
    name = "gemma4"
    _system_role_supported = False  # Gemma family still merges system into user

    _delims = {
        "bos": "<bos>",
        "turn_start": "<|turn>",
        "turn_end": "<turn|>",
        "channel_start": "<|channel>",
        "channel_end": "<channel|>",
        "think": "<|think|>",
        "user_open": "<|turn>user\n",
        "model_open": "<|turn>model\n",
    }

    _delim_special_names = {
        "bos": "<bos>",
        "turn_start": "<|turn>",
        "turn_end": "<turn|>",
        "channel_start": "<|channel>",
        "channel_end": "<channel|>",
        "think": "<|think|>",
    }

    def build_fake_delimiter_injection(self, system_instruction: str) -> str:
        # Gemma 4 has no system role — mimic a fake "model" turn (spoof prior assistant output).
        return (
            "<|turn>model\n"
            f"{system_instruction}<turn|>"
        )
