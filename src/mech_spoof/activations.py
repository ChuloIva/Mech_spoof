"""Residual-stream extraction via forward hooks.

Two entry points:
- `extract_residual_stream(loaded, input_ids)` — full (n_layers, seq_len, d_model) tensor. Use only
  for token-trace experiments. Cost scales with seq_len × n_layers × d_model.
- `extract_at_positions(loaded, prompts, positions_fn)` — slices to specified positions per prompt
  and returns (n_prompts, n_layers, d_model). This is what the authority/refusal probes consume.
  Memory footprint is a tiny fraction of the full cache.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from mech_spoof.models import LoadedModel
from mech_spoof.utils import get_logger

logger = get_logger(__name__)


def _register_hooks(loaded: LoadedModel, storage: list):
    """Register a forward hook on each transformer block that stashes its output hidden state."""
    handles = []
    for i in range(loaded.n_layers):
        module = loaded.layer_module(i)

        def _hook(_mod, _inp, out, idx=i):
            # Block output for Llama-style models is (hidden_state, ...) or just hidden_state.
            h = out[0] if isinstance(out, tuple) else out
            storage[idx] = h.detach()

        handles.append(module.register_forward_hook(_hook))
    return handles


def extract_residual_stream(
    loaded: LoadedModel,
    input_ids,
    to_cpu: bool = True,
):
    """Run one forward pass and return (n_layers, seq_len, d_model) residual stream.

    `input_ids` may be a list, 1D tensor, or (1, seq_len) tensor. Batch size must be 1.
    """
    import torch

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(loaded.device)

    storage: list = [None] * loaded.n_layers
    handles = _register_hooks(loaded, storage)
    try:
        with torch.no_grad():
            loaded.hf_model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    stacked = torch.stack([storage[i][0] for i in range(loaded.n_layers)], dim=0)
    # shape: (n_layers, seq_len, d_model)
    if to_cpu:
        stacked = stacked.float().cpu()
    return stacked


def extract_at_positions(
    loaded: LoadedModel,
    prompt_bundles,                                # sequence of PromptBundle
    positions_fn: Callable[[LoadedModel, "PromptBundle"], list[int] | int],  # noqa: F821
    layer_indices: list[int] | None = None,
    cache_dir: Path | None = None,
    progress: bool = True,
) -> np.ndarray:
    """Extract activations for a list of prompts, sliced to specified positions.

    Parameters
    ----------
    loaded : LoadedModel
    prompt_bundles : list of PromptBundle
    positions_fn : callable(loaded, bundle) -> int | list[int]
        Returns which token position(s) to read. If a list is returned, the values are mean-pooled.
    layer_indices : list[int], optional
        Which layers to capture. Defaults to all layers.
    cache_dir : Path, optional
        If provided, caches per-prompt (n_layers, d_model) arrays on disk; skips already-cached prompts.

    Returns
    -------
    np.ndarray of shape (n_prompts, len(layer_indices), d_model)
    """
    import torch
    from tqdm.auto import tqdm

    if layer_indices is None:
        layer_indices = list(range(loaded.n_layers))

    n = len(prompt_bundles)
    out = np.zeros((n, len(layer_indices), loaded.d_model), dtype=np.float32)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    iterator = range(n)
    if progress:
        iterator = tqdm(iterator, desc=f"extract [{loaded.cfg.key}]")

    for i in iterator:
        bundle = prompt_bundles[i]

        if cache_dir is not None:
            path = cache_dir / f"act_{i:05d}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    out[i] = pickle.load(f)
                continue

        positions = positions_fn(loaded, bundle)
        if isinstance(positions, int):
            positions = [positions]

        ids = torch.tensor([bundle.input_ids], dtype=torch.long, device=loaded.device)

        storage: list = [None] * loaded.n_layers
        handles = _register_hooks(loaded, storage)
        try:
            with torch.no_grad():
                loaded.hf_model(input_ids=ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        # Stack selected layers, select positions, mean-pool positions, move to CPU float32.
        vecs = []
        for li in layer_indices:
            act = storage[li][0]  # (seq_len, d_model)
            sliced = act[positions]  # (k, d_model)
            pooled = sliced.mean(dim=0)
            vecs.append(pooled.float().cpu().numpy())
        arr = np.stack(vecs, axis=0)  # (n_layers, d_model)
        out[i] = arr

        if cache_dir is not None:
            with open(cache_dir / f"act_{i:05d}.pkl", "wb") as f:
                pickle.dump(arr, f)

        # Explicit cleanup to keep VRAM flat.
        del storage, vecs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def streaming_activation_loader(cache_dir: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Stream per-prompt activation arrays from a disk cache."""
    cache_dir = Path(cache_dir)
    for path in sorted(cache_dir.glob("act_*.pkl")):
        idx = int(path.stem.split("_")[1])
        with open(path, "rb") as f:
            yield idx, pickle.load(f)


def response_first_position(loaded: LoadedModel, bundle) -> int:
    """Default positions_fn: the last input token (whose residual predicts token 1 of response)."""
    return bundle.response_first_pos


def instruction_end_position(loaded: LoadedModel, bundle) -> int:
    """Alternative positions_fn: last token of the instruction content."""
    start, end = bundle.instruction_token_span
    return max(start, end - 1)


def instruction_span_positions(loaded: LoadedModel, bundle) -> list[int]:
    """Mean-pool over the instruction token span."""
    start, end = bundle.instruction_token_span
    return list(range(start, end))
