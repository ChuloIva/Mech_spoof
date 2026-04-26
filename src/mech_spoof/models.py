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
    # Dotted attribute path from hf_model to the transformer block ModuleList.
    # "model.layers" for plain causal-LM checkpoints; nested (e.g.
    # "model.language_model.layers") for VL composite configs like Qwen3.5-VL.
    layers_path: str = "model.layers"

    def layer_module(self, layer_idx: int):
        """Return the transformer block module at the given layer index."""
        mod: Any = self.hf_model
        for part in self.layers_path.split("."):
            mod = getattr(mod, part)
        return mod[layer_idx]


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
    import importlib

    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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

    # AutoModelForCausalLM trips over composite multimodal configs (e.g. Qwen3.5-VL)
    # where vocab_size lives under text_config. Detect and route to the architecture
    # class named in the config; the model still accepts text-only forward calls.
    pre_cfg = AutoConfig.from_pretrained(cfg.hf_id, trust_remote_code=trust_remote_code)
    text_cfg = getattr(pre_cfg, "text_config", None)
    is_composite = text_cfg is not None and not hasattr(pre_cfg, "vocab_size")

    load_kwargs = dict(
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    if is_composite:
        archs = getattr(pre_cfg, "architectures", None) or []
        ModelClass = None
        if archs:
            transformers_mod = importlib.import_module("transformers")
            ModelClass = getattr(transformers_mod, archs[0], None)
        if ModelClass is None:
            try:
                from transformers import AutoModelForImageTextToText
                ModelClass = AutoModelForImageTextToText
            except ImportError as e:
                raise RuntimeError(
                    f"Composite config for {cfg.hf_id} (arch={archs}) and neither "
                    "the named class nor AutoModelForImageTextToText is available."
                ) from e
        logger.info(
            f"Composite config detected — loading via {getattr(ModelClass, '__name__', ModelClass)}"
        )
        hf_model = ModelClass.from_pretrained(cfg.hf_id, **load_kwargs)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(cfg.hf_id, **load_kwargs)
    hf_model.to(dev)
    hf_model.eval()

    template = get_template(cfg.template, tokenizer)

    # Infer n_layers / d_model from config (fall back to text_config for composite)
    hf_cfg = hf_model.config
    src_cfg = hf_cfg
    if not hasattr(src_cfg, "num_hidden_layers") and getattr(src_cfg, "text_config", None):
        src_cfg = src_cfg.text_config
    n_layers = getattr(src_cfg, "num_hidden_layers", None) or getattr(src_cfg, "n_layer")
    d_model = getattr(src_cfg, "hidden_size", None) or getattr(src_cfg, "n_embd")

    # Locate the transformer-block ModuleList. For plain causal-LM checkpoints
    # this is "model.layers" (Llama/Qwen/Mistral/Gemma/Phi all share it).
    # For composite VL models the layers list is nested deeper.
    layers_path = "model.layers"
    found = False
    try:
        probe = hf_model
        for part in layers_path.split("."):
            probe = getattr(probe, part)
        if isinstance(probe, nn.ModuleList) and len(probe) == int(n_layers):
            found = True
    except AttributeError:
        pass
    if not found:
        for name, module in hf_model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) == int(n_layers):
                layers_path = name
                found = True
                break
    if not found:
        raise RuntimeError(
            f"Could not locate transformer-block ModuleList of length {n_layers} on {cfg.hf_id}"
        )
    if layers_path != "model.layers":
        logger.info(f"Using nested transformer layers at: {layers_path}")

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
        layers_path=layers_path,
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
