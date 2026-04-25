"""Abstract base for chat-template adapters.

Each model family has different role delimiters (ChatML, Llama-3 headers, [INST], etc.).
The adapter normalizes prompt construction, fake-delimiter injection, and position finding
so experiments can be written template-agnostically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class PromptBundle:
    """A built prompt with enough metadata to locate instruction + response positions."""

    text: str                                 # full prompt text (post-template)
    input_ids: list[int]                      # tokenized
    instruction_text: str                     # the user's instruction content
    instruction_token_span: tuple[int, int]   # inclusive start, exclusive end token idx
    response_first_pos: int                   # token idx whose residual predicts 1st response token
    condition: Literal["S", "U", "FAKE", "REAL", "NONE"]
    extras: dict[str, Any] | None = None


@dataclass
class FakeDelimReport:
    """Does the tokenizer collapse fake delimiter TEXT inside user input into real special tokens?

    If yes, Condition FAKE is indistinguishable from Condition S at the token level — an attack
    that "wins" may be trivially winning rather than exploiting authority representations.
    """

    template_name: str
    checks: list[dict]        # per-delimiter string: {delim, tokenizes_to_special_id, token_ids, tokens}
    any_collapsed: bool       # True if ANY delim collapses → flag
    mangled_variant: str      # a safe variant (e.g. with zero-width space) that provably won't collapse


class TemplateAdapter(ABC):
    """One adapter per chat-template family."""

    name: str

    def __init__(self, tokenizer):
        self.tok = tokenizer

    # ---------- Core prompt builders (§3.1, §3.2) ----------

    @abstractmethod
    def make_system_prompt(
        self,
        instruction: str,
        user_followup: str = "Hello, how can you help me?",
    ) -> PromptBundle:
        """Condition S: instruction placed in the REAL system field."""

    @abstractmethod
    def make_user_prompt(
        self,
        instruction: str,
        user_followup: str = "Hello, how can you help me?",
    ) -> PromptBundle:
        """Condition U: same instruction placed in the user field."""

    @abstractmethod
    def make_conflict_prompt(
        self,
        system_instruction: str,
        user_instruction: str,
        condition: Literal["REAL", "NONE", "FAKE", "NONE_REV"],
    ) -> PromptBundle:
        """Exp 2 prompts. REAL: real system field. NONE: both in user (s first).
        NONE_REV: both in user, **u first** (order-ablation control). FAKE: system via fake delims."""

    # ---------- Attack primitives (§3.3) ----------

    @abstractmethod
    def build_fake_delimiter_injection(self, system_instruction: str) -> str:
        """Return a TEXT payload that textually looks like a system block.

        These are string forms of delimiters — the tokenizer may or may not collapse them
        into real special tokens (surface this via fake_delimiter_tokenization_report).
        """

    @abstractmethod
    def delimiter_strings(self) -> dict[str, str]:
        """Canonical delimiter literals (e.g. {'system_open': '<|im_start|>system\\n', ...})."""

    @abstractmethod
    def special_token_ids(self) -> dict[str, int | None]:
        """Map delimiter-name -> special token ID (None if not a special token)."""

    # ---------- Position finders (§4.2) ----------

    def find_response_first_position(self, input_ids) -> int:
        """Token position whose residual predicts the first response token.

        In a causal LM fed a prompt ending with the assistant generation prompt,
        this is the last input position.
        """
        n = len(input_ids) if isinstance(input_ids, list) else input_ids.shape[-1]
        return n - 1

    def find_instruction_end_position(self, bundle: PromptBundle) -> int:
        """Last token of the instruction content."""
        return bundle.instruction_token_span[1] - 1

    def find_instruction_span(self, bundle: PromptBundle) -> tuple[int, int]:
        return bundle.instruction_token_span

    @abstractmethod
    def find_delimiter_positions(self, input_ids) -> list[int]:
        """Positions of structural delimiter tokens."""

    @abstractmethod
    def is_delimiter_token(self, token_str: str) -> bool:
        """Heuristic: is this token part of a structural delimiter?"""

    @abstractmethod
    def classify_structural_role(self, pos: int, input_ids, tok=None) -> str:
        """Label a position: 'system_tag'|'user_tag'|'assistant_tag'|'content'|'end_tag'."""

    # ---------- §9.2: fake-delimiter tokenization report ----------

    @abstractmethod
    def fake_delimiter_tokenization_report(self) -> FakeDelimReport:
        """Check whether the tokenizer eats fake delimiter text as real special tokens."""

    # ---------- Shared helpers ----------

    def _locate_substring_tokens(
        self,
        text: str,
        substring: str,
        input_ids_from_template: list[int] | None = None,
    ) -> tuple[int, int]:
        """Find the inclusive-start, exclusive-end token span of `substring` in `text`.

        Uses offset_mapping from a fast tokenizer. When the caller already tokenized via
        apply_chat_template, pass the ids to sanity-check length. The offset-mapping retokenization
        uses add_special_tokens=False (the template string typically already carries BOS).
        """
        if not getattr(self.tok, "is_fast", False):
            raise RuntimeError(
                f"Tokenizer {type(self.tok).__name__} is not a fast tokenizer — "
                "offset_mapping unavailable. Install the `tokenizers`-backed variant."
            )

        char_start = text.rindex(substring)
        char_end = char_start + len(substring)

        enc = self.tok(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors=None,
        )
        offsets = enc["offset_mapping"]
        ids = enc["input_ids"]

        tok_start = None
        tok_end = None
        for i, (a, b) in enumerate(offsets):
            if a == b:  # zero-width (e.g. BOS) — skip
                continue
            if tok_start is None and b > char_start:
                tok_start = i
            if a < char_end:
                tok_end = i + 1
        if tok_start is None:
            tok_start = 0
        if tok_end is None:
            tok_end = len(ids)

        # If the caller provided the canonical ids (from apply_chat_template), reconcile lengths:
        # re-tokenization may produce a different count by 1 BOS. We accept either as long as the
        # span we extracted is within the canonical length.
        if input_ids_from_template is not None:
            canonical_len = len(input_ids_from_template)
            retok_len = len(ids)
            if retok_len != canonical_len:
                # Common case: canonical adds BOS, offset-retok with add_special_tokens=False doesn't.
                # Shift span by the delta.
                delta = canonical_len - retok_len
                if delta > 0:
                    tok_start += delta
                    tok_end += delta
        return tok_start, tok_end
