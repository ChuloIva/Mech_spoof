"""Template adapter registry."""

from __future__ import annotations

from mech_spoof.templates.base import FakeDelimReport, PromptBundle, TemplateAdapter


def get_template(name: str, tokenizer) -> TemplateAdapter:
    """Return a template adapter for the given template key."""
    from mech_spoof.templates.chatml import ChatMLAdapter
    from mech_spoof.templates.gemma import GemmaAdapter
    from mech_spoof.templates.gemma4 import Gemma4Adapter
    from mech_spoof.templates.llama3 import Llama3Adapter
    from mech_spoof.templates.mistral import MistralAdapter
    from mech_spoof.templates.phi3 import Phi3Adapter

    registry = {
        "chatml": ChatMLAdapter,
        "llama3": Llama3Adapter,
        "mistral": MistralAdapter,
        "gemma": GemmaAdapter,
        "gemma4": Gemma4Adapter,
        "phi3": Phi3Adapter,
    }
    if name not in registry:
        raise KeyError(f"Unknown template: {name}. Known: {list(registry)}")
    return registry[name](tokenizer)


__all__ = ["FakeDelimReport", "PromptBundle", "TemplateAdapter", "get_template"]
