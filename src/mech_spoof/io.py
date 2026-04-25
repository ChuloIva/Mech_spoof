"""Result-bundle IO: save/load JSON + numpy bundles, manifest writer."""

from __future__ import annotations

import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mech_spoof.utils import git_sha, gpu_name

_PROBE_DIR_KEY_LEGACY = re.compile(r"probe_dir_layer_(\d+)")
_PROBE_DIR_KEY_V2 = re.compile(r"probe_dir__([a-zA-Z0-9_]+?)__layer_(\d+)")


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_npz(arrays: dict[str, np.ndarray], path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def save_pickle(obj: Any, path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def write_manifest(out_dir: Path, metadata: dict | None = None) -> Path:
    """Write a manifest.json with git SHA, GPU name, timestamp, extras."""
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    import torch

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "gpu_name": gpu_name(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "torch_hip": getattr(torch.version, "hip", None),
        **(metadata or {}),
    }
    path = out_dir / "manifest.json"
    save_json(manifest, path)
    return path


def save_result_bundle(
    out_dir: Path,
    json_obj: dict | None = None,
    arrays: dict[str, np.ndarray] | None = None,
    pickles: dict[str, Any] | None = None,
    manifest_extras: dict | None = None,
) -> None:
    """Save a complete experiment result bundle to out_dir."""
    out_dir = ensure_dir(Path(out_dir))
    if json_obj is not None:
        save_json(json_obj, out_dir / "result.json")
    if arrays is not None:
        save_npz(arrays, out_dir / "arrays.npz")
    if pickles is not None:
        for name, obj in pickles.items():
            save_pickle(obj, out_dir / f"{name}.pkl")
    write_manifest(out_dir, manifest_extras)


def load_result_bundle(in_dir: Path) -> dict:
    in_dir = Path(in_dir)
    out: dict = {}
    if (in_dir / "result.json").exists():
        out["result"] = load_json(in_dir / "result.json")
    if (in_dir / "arrays.npz").exists():
        out["arrays"] = load_npz(in_dir / "arrays.npz")
    if (in_dir / "manifest.json").exists():
        out["manifest"] = load_json(in_dir / "manifest.json")
    for pkl in in_dir.glob("*.pkl"):
        out[pkl.stem] = load_pickle(pkl)
    return out


def load_authority_directions(
    bundle_dir: Path,
    position: str | None = None,
) -> tuple[int, dict[int, np.ndarray], str | None] | None:
    """Load per-layer authority probe directions from an exp1 or exp1b bundle.

    Schemas:
      legacy exp1 : `probe_dir_layer_NNN`
      exp1b       : `probe_dir__<position>__layer_NNN`

    For exp1b bundles, `position` selects which extraction-position direction set to load
    (default: the bundle's `result.json:best_position`). Returns
    `(best_layer, dirs, resolved_position_or_None)`, or `None` if no bundle is found.
    """
    if bundle_dir is None or not Path(bundle_dir).exists():
        return None

    result = load_json(Path(bundle_dir) / "result.json")
    arrays = load_npz(Path(bundle_dir) / "arrays.npz")

    v2_groups: dict[str, dict[int, np.ndarray]] = {}
    for key, val in arrays.items():
        m = _PROBE_DIR_KEY_V2.fullmatch(key)
        if m:
            v2_groups.setdefault(m.group(1), {})[int(m.group(2))] = val

    if v2_groups:
        chosen = position or result.get("best_position")
        if chosen is None:
            chosen = sorted(v2_groups.keys())[0]
        if chosen not in v2_groups:
            raise KeyError(
                f"position '{chosen}' not in bundle {bundle_dir};"
                f" available: {sorted(v2_groups.keys())}"
            )
        dirs = v2_groups[chosen]
        best = int(result.get("best_layer", max(dirs)))
        return best, dirs, chosen

    legacy: dict[int, np.ndarray] = {}
    for key, val in arrays.items():
        m = _PROBE_DIR_KEY_LEGACY.fullmatch(key)
        if m:
            legacy[int(m.group(1))] = val
    if not legacy:
        return None
    return int(result["best_layer"]), legacy, None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")
