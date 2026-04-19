"""Shared implementation used by all concrete template adapters.

Concrete adapters override `name`, `_delims`, `_delim_special_names`, and
`build_fake_delimiter_injection`. Everything else is inherited.
"""

from __future__ import annotations

from typing import Literal

from mech_spoof.templates.base import FakeDelimReport, PromptBundle, TemplateAdapter


DEFAULT_FOLLOWUP = "Hello, how can you help me?"
DEFAULT_BASELINE_SYSTEM = "You are a helpful assistant."
ZWSP = "\u200b"  # zero-width space — breaks special-token tokenization


class GenericChatTemplate(TemplateAdapter):
    """Default implementation based on `tokenizer.apply_chat_template`.

    Subclasses must set:
        name: str
        _delims: dict[str, str]                 # literal text forms
        _delim_special_names: dict[str, str]    # map key -> special token string (for ID lookup)

    Subclasses must implement:
        build_fake_delimiter_injection(system_instruction) -> str
    """

    name: str = "generic"
    _delims: dict[str, str] = {}
    _delim_special_names: dict[str, str] = {}
    # Subclass may set True when the tokenizer is known to refuse a `system` role entry;
    # in that case make_system_prompt will merge system content into the first user turn manually.
    _system_role_supported: bool = True
    # Qwen3/Qwen3.5 accept `enable_thinking` as a chat-template kwarg; other tokenizers'
    # Jinja templates may warn or error on unknown kwargs, so opt in per adapter.
    _supports_enable_thinking: bool = False

    # ---------- Core prompt builders ----------

    def make_system_prompt(
        self, instruction: str, user_followup: str = DEFAULT_FOLLOWUP
    ) -> PromptBundle:
        if self._system_role_supported:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_followup},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{instruction}\n\n{user_followup}"},
            ]
        return self._build_bundle(messages, instruction, condition="S")

    def make_user_prompt(
        self, instruction: str, user_followup: str = DEFAULT_FOLLOWUP
    ) -> PromptBundle:
        content = f"{instruction}\n\n{user_followup}"
        messages = [{"role": "user", "content": content}]
        return self._build_bundle(messages, instruction, condition="U")

    # ---------- Matched-baseline variants (Exp 1 control) ----------
    # Both conditions share an identical baseline system prompt; only the *placement* of
    # the instruction-of-interest varies. Neutralizes the "is there a system block?" shortcut
    # that plain make_system_prompt vs make_user_prompt leaks at layer 0.

    def make_system_prompt_matched(
        self,
        instruction: str,
        baseline_system: str = DEFAULT_BASELINE_SYSTEM,
        user_followup: str = DEFAULT_FOLLOWUP,
    ) -> PromptBundle:
        system_content = f"{baseline_system}\n{instruction}"
        if self._system_role_supported:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_followup},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{system_content}\n\n{user_followup}"},
            ]
        return self._build_bundle(messages, instruction, condition="S")

    def make_user_prompt_matched(
        self,
        instruction: str,
        baseline_system: str = DEFAULT_BASELINE_SYSTEM,
        user_followup: str = DEFAULT_FOLLOWUP,
    ) -> PromptBundle:
        user_content = f"{instruction}\n\n{user_followup}"
        if self._system_role_supported:
            messages = [
                {"role": "system", "content": baseline_system},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{baseline_system}\n\n{user_content}"},
            ]
        return self._build_bundle(messages, instruction, condition="U")

    def make_conflict_prompt(
        self,
        system_instruction: str,
        user_instruction: str,
        condition: Literal["REAL", "NONE", "FAKE"],
    ) -> PromptBundle:
        if condition == "REAL":
            if self._system_role_supported:
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_instruction},
                ]
            else:
                messages = [
                    {"role": "user", "content": f"{system_instruction}\n\n{user_instruction}"},
                ]
            return self._build_bundle(messages, user_instruction, condition="REAL")

        if condition == "NONE":
            merged = f"{system_instruction}\n\n{user_instruction}"
            messages = [{"role": "user", "content": merged}]
            return self._build_bundle(messages, user_instruction, condition="NONE")

        if condition == "FAKE":
            fake = self.build_fake_delimiter_injection(system_instruction)
            combined = f"{fake}\n\n{user_instruction}"
            messages = [{"role": "user", "content": combined}]
            return self._build_bundle(messages, user_instruction, condition="FAKE")

        raise ValueError(f"Unknown condition: {condition}")

    # ---------- Shared delimiter lookups ----------

    def delimiter_strings(self) -> dict[str, str]:
        return dict(self._delims)

    def special_token_ids(self) -> dict[str, int | None]:
        out: dict[str, int | None] = {}
        for key, special_str in self._delim_special_names.items():
            tid = self.tok.convert_tokens_to_ids(special_str)
            unk = self.tok.unk_token_id
            out[key] = None if (tid is None or tid == unk) else int(tid)
        return out

    def _delimiter_special_ids_set(self) -> set[int]:
        return {v for v in self.special_token_ids().values() if v is not None}

    def find_delimiter_positions(self, input_ids) -> list[int]:
        ids = list(input_ids) if not isinstance(input_ids, list) else input_ids
        specials = self._delimiter_special_ids_set()
        return [i for i, t in enumerate(ids) if t in specials]

    def is_delimiter_token(self, token_str: str) -> bool:
        return token_str in self._delim_special_names.values()

    def classify_structural_role(self, pos: int, input_ids, tok=None) -> str:
        """Coarse label per position. Kept simple — used mainly for plotting annotations."""
        ids = list(input_ids) if not isinstance(input_ids, list) else input_ids
        if pos >= len(ids):
            return "oob"
        specials = self._delimiter_special_ids_set()
        t = ids[pos]
        if t in specials:
            # Try to resolve which delimiter
            decoded = (tok or self.tok).decode([t])
            if "system" in decoded:
                return "system_tag"
            if "user" in decoded:
                return "user_tag"
            if "assistant" in decoded or "model" in decoded:
                return "assistant_tag"
            if "end" in decoded or "eot" in decoded or "turn" in decoded:
                return "end_tag"
            return "delimiter"
        return "content"

    # ---------- Fake-delimiter tokenization report (§9.2) ----------

    def fake_delimiter_tokenization_report(self) -> FakeDelimReport:
        """Check: when a delimiter STRING appears inside user text, does the tokenizer collapse it
        to the real special-token ID? If yes, Condition FAKE ≈ Condition S at token level.
        """
        checks = []
        any_collapsed = False
        special_ids = self.special_token_ids()

        for key, delim_text in self._delims.items():
            ids_as_regular = self.tok(delim_text, add_special_tokens=False)["input_ids"]
            tokens_as_regular = self.tok.convert_ids_to_tokens(ids_as_regular)
            special_id = special_ids.get(key)
            collapsed = special_id is not None and special_id in ids_as_regular
            any_collapsed = any_collapsed or collapsed
            checks.append(
                {
                    "key": key,
                    "delim_text": delim_text,
                    "special_token_id": special_id,
                    "tokens_as_regular": tokens_as_regular,
                    "token_ids_as_regular": ids_as_regular,
                    "collapsed_to_special": collapsed,
                }
            )

        # Build a mangled variant of the fake injection that shouldn't collapse
        # (inserts zero-width space inside each delimiter opener).
        mangled = self.build_fake_delimiter_injection("__INSTR__").replace(
            "__INSTR__", "{instruction}"
        )
        for delim_text in sorted(self._delims.values(), key=len, reverse=True):
            if len(delim_text) < 3:
                continue
            mangled = mangled.replace(delim_text, delim_text[0] + ZWSP + delim_text[1:])

        return FakeDelimReport(
            template_name=self.name,
            checks=checks,
            any_collapsed=any_collapsed,
            mangled_variant=mangled,
        )

    # ---------- Internal ----------

    def _build_bundle(
        self,
        messages: list[dict],
        instruction_for_location: str,
        condition: Literal["S", "U", "REAL", "NONE", "FAKE"],
    ) -> PromptBundle:
        # Qwen3/Qwen3.5 hard switch to skip the <think> block and the informal
        # "Thinking Process:" chatter Qwen3.5-4B emits by default. Only opted-in adapters pass it.
        extra = {"enable_thinking": False} if self._supports_enable_thinking else {}
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra
        )
        ids = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **extra
        )
        # Newer tokenizers may return a BatchEncoding/dict when tokenize=True
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif isinstance(ids, dict) and "input_ids" in ids:
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):  # torch.Tensor
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        ids_list = [int(x) for x in ids]

        try:
            span = self._locate_substring_tokens(text, instruction_for_location, ids_list)
        except (ValueError, RuntimeError):
            # Instruction text didn't survive templating verbatim (rare — e.g. escaping).
            # Fall back to a degenerate span covering the full prompt minus the trailing gen prompt.
            span = (0, len(ids_list))

        return PromptBundle(
            text=text,
            input_ids=ids_list,
            instruction_text=instruction_for_location,
            instruction_token_span=span,
            response_first_pos=len(ids_list) - 1,
            condition=condition,
        )
