"""Model loading and backend abstraction.

Primary backend: HuggingFace `AutoModelForCausalLM` + raw `register_forward_hook`. Robust across
ROCm and all 5 target model families. An `nnsight` backend is available optionally.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Literal

from mech_spoof.configs import MODEL_CONFIGS, ModelConfig
from mech_spoof.templates import TemplateAdapter, get_template
from mech_spoof.utils import get_logger, pick_device

logger = get_logger(__name__)


Backend = Literal["hf_hooks", "nnsight"]


@dataclass
class LoadedModel:
    """Bundle returned by `load_model`. All downstream code consumes this."""

    hf_model: Any
    tokenizer: Any
    template: TemplateAdapter
    cfg: ModelConfig
    backend: Backend
    device: str
    n_layers: int
    d_model: int
    nnsight_model: Any = None  # only set for backend="nnsight"

    def layer_module(self, layer_idx: int):
        """Return the transformer block module at the given layer index.

        Assumes Llama-style architecture: `model.model.layers[i]`. All 5 target families share this.
        """
        return self.hf_model.model.layers[layer_idx]


def load_model(
    key: str,
    backend: Backend = "hf_hooks",
    device: str = "auto",
    dtype: str | None = None,
    trust_remote_code: bool = True,
) -> LoadedModel:
    """Load one of the configured models.

    Parameters
    ----------
    key : str
        A key into MODEL_CONFIGS (e.g. "qwen", "llama3", "gemma_small").
    backend : str
        "hf_hooks" (default, recommended) or "nnsight".
    device : str
        "auto" picks cuda/mps/cpu. Pass explicit string to override.
    dtype : str, optional
        Override the per-model default dtype.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model key {key!r}. Known: {list(MODEL_CONFIGS)}")
    cfg = MODEL_CONFIGS[key]

    dev = pick_device(device)
    torch_dtype_str = dtype or cfg.dtype
    torch_dtype = getattr(torch, torch_dtype_str)

    logger.info(f"Loading {cfg.hf_id} on {dev} dtype={torch_dtype_str} backend={backend}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    hf_model.to(dev)
    hf_model.eval()

    template = get_template(cfg.template, tokenizer)

    # Infer n_layers / d_model from config
    hf_cfg = hf_model.config
    n_layers = getattr(hf_cfg, "num_hidden_layers", None) or getattr(hf_cfg, "n_layer")
    d_model = getattr(hf_cfg, "hidden_size", None) or getattr(hf_cfg, "n_embd")

    nnsight_model = None
    if backend == "nnsight":
        try:
            from nnsight import LanguageModel
        except ImportError as e:
            raise RuntimeError(
                "nnsight backend requested but nnsight not installed. "
                "Install it or use backend='hf_hooks'."
            ) from e
        nnsight_model = LanguageModel(hf_model, tokenizer=tokenizer)

    return LoadedModel(
        hf_model=hf_model,
        tokenizer=tokenizer,
        template=template,
        cfg=cfg,
        backend=backend,
        device=dev,
        n_layers=int(n_layers),
        d_model=int(d_model),
        nnsight_model=nnsight_model,
    )


def free_model(loaded: LoadedModel) -> None:
    """Release VRAM held by a loaded model."""
    try:
        import torch
    except ImportError:
        return
    loaded.hf_model = None
    loaded.nnsight_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
