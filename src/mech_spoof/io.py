"""Result-bundle IO: save/load JSON + numpy bundles, manifest writer."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mech_spoof.utils import git_sha, gpu_name


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
