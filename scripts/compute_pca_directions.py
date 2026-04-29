"""Compute repeng-style PCA directions (pca_diff, pca_center) from the exp6
prefilled S/U activation cache shards in `drive-download-*/act_cache_{S,U}_prefill/`.

Each shard is a slice of the same 2000-row dataset. We union the shards by
prompt id, pair S_prefill ↔ U_prefill, then for each (position × layer) fit:

  - pca_diff  : PCA(1) on the per-pair difference   h_S - h_U
  - pca_center: PCA(1) on per-pair-centered points  h - midpoint(h_S, h_U)

Sign is fixed to match the mass-mean convention: projection(mean_S) > projection(mean_U).

Outputs `exp06_pca_directions.npz` with keys:
  pca_diff_dir__{position}__layer_{l:03d}
  pca_center_dir__{position}__layer_{l:03d}
plus `manifest_pca.json` summarising sign-fix votes and class separation.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

POSITIONS = ["response_first", "response_mid", "response_last"]


def load_cache(shard_dirs: list[Path], condition: str) -> dict[str, np.ndarray]:
    """Walk all shards, collect {prompt_id: acts(3,n_layers,d)} for `condition`."""
    out: dict[str, np.ndarray] = {}
    for d in shard_dirs:
        sub = d / f"act_cache_{condition}"
        if not sub.exists():
            continue
        for p in sub.glob("prompt_*.pkl"):
            with open(p, "rb") as f:
                obj = pickle.load(f)
            out[p.stem] = obj["acts"].astype(np.float32)
    return out


def _fit_pca_diff_layer(h_S: np.ndarray, h_U: np.ndarray) -> np.ndarray:
    """h_S, h_U : (n, d).  Returns unit direction (d,)."""
    diff = h_S - h_U                # (n, d)
    pca = PCA(n_components=1, whiten=False).fit(diff)
    v = pca.components_[0].astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _fit_pca_center_layer(h_S: np.ndarray, h_U: np.ndarray) -> np.ndarray:
    """h_S, h_U : (n, d).  Returns unit direction (d,)."""
    midpoint = (h_S + h_U) / 2.0    # (n, d)
    train = np.concatenate([h_S - midpoint, h_U - midpoint], axis=0)  # (2n, d)
    pca = PCA(n_components=1, whiten=False).fit(train)
    v = pca.components_[0].astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _sign_fix(v: np.ndarray, h_S: np.ndarray, h_U: np.ndarray) -> tuple[np.ndarray, float]:
    """Flip `v` so projection(mean_S) > projection(mean_U). Returns (v_signed, frac_pos_larger)."""
    proj_S = h_S @ v
    proj_U = h_U @ v
    frac_pos_larger = float(np.mean(proj_S > proj_U))
    if proj_S.mean() < proj_U.mean():
        v = -v
        frac_pos_larger = 1.0 - frac_pos_larger
    return v, frac_pos_larger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shard-glob",
        default="drive-download-20260427T174621Z-3-00*",
        help="Glob (relative to --root) for shard parent directories.",
    )
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--out-npz", type=Path, default=Path("exp06_pca_directions.npz"))
    p.add_argument(
        "--out-manifest",
        type=Path,
        default=Path("exp06_pca_directions_manifest.json"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    root: Path = args.root
    shard_dirs = sorted(root.glob(args.shard_glob))
    if not shard_dirs:
        raise RuntimeError(f"No shards matching {args.shard_glob} under {root}")
    print(f"[info] {len(shard_dirs)} shard dirs:")
    for d in shard_dirs:
        print(f"  {d.name}")

    print("[info] loading S_prefill cache...")
    cache_S = load_cache(shard_dirs, "S_prefill")
    print(f"  {len(cache_S)} prompts")
    print("[info] loading U_prefill cache...")
    cache_U = load_cache(shard_dirs, "U_prefill")
    print(f"  {len(cache_U)} prompts")

    common = sorted(set(cache_S) & set(cache_U))
    print(f"[info] paired prompt count: {len(common)}")
    if not common:
        raise RuntimeError("no paired prompts between S_prefill and U_prefill")

    S = np.stack([cache_S[k] for k in common], axis=0)   # (n, 3, n_layers, d)
    U = np.stack([cache_U[k] for k in common], axis=0)
    n, n_pos, n_layers, d = S.shape
    print(f"[info] activations stacked: S/U shape={S.shape}  (positions × layers × d_model)")
    assert n_pos == len(POSITIONS)

    arrays: dict[str, np.ndarray] = {}
    manifest: dict = {
        "n_paired": int(n),
        "n_positions": int(n_pos),
        "n_layers": int(n_layers),
        "d_model": int(d),
        "positions": POSITIONS,
        "shard_dirs": [str(d.name) for d in shard_dirs],
        "per_layer": {pos: {} for pos in POSITIONS},
    }

    for pi, pos in enumerate(POSITIONS):
        print(f"\n[fit] position={pos}")
        for layer in range(n_layers):
            h_S = S[:, pi, layer, :]
            h_U = U[:, pi, layer, :]

            v_diff   = _fit_pca_diff_layer(h_S, h_U)
            v_center = _fit_pca_center_layer(h_S, h_U)

            v_diff,   frac_diff   = _sign_fix(v_diff,   h_S, h_U)
            v_center, frac_center = _sign_fix(v_center, h_S, h_U)

            mu_S = h_S.mean(axis=0)
            mu_U = h_U.mean(axis=0)
            mm_raw = mu_S - mu_U
            mm_unit = mm_raw / (np.linalg.norm(mm_raw) + 1e-8)

            cos_diff_mm   = float(v_diff   @ mm_unit)
            cos_center_mm = float(v_center @ mm_unit)
            cos_diff_center = float(v_diff @ v_center)

            arrays[f"pca_diff_dir__{pos}__layer_{layer:03d}"]   = v_diff.astype(np.float32)
            arrays[f"pca_center_dir__{pos}__layer_{layer:03d}"] = v_center.astype(np.float32)

            manifest["per_layer"][pos][str(layer)] = {
                "frac_S_proj_larger_diff": frac_diff,
                "frac_S_proj_larger_center": frac_center,
                "cos_pca_diff_vs_mm":     cos_diff_mm,
                "cos_pca_center_vs_mm":   cos_center_mm,
                "cos_pca_diff_vs_center": cos_diff_center,
                "mm_natural_scale":       float(np.linalg.norm(mm_raw)),
            }

            if layer in (0, 5, 10, 16, 20, 25, 31):
                print(f"  L{layer:02d}: "
                      f"frac_S>U(diff)={frac_diff:.3f}  frac_S>U(ctr)={frac_center:.3f}  "
                      f"cos(diff,MM)={cos_diff_mm:+.3f}  cos(ctr,MM)={cos_center_mm:+.3f}  "
                      f"cos(diff,ctr)={cos_diff_center:+.3f}  ||MM_raw||={np.linalg.norm(mm_raw):.3f}")

    out_npz = args.out_npz if args.out_npz.is_absolute() else root / args.out_npz
    out_man = args.out_manifest if args.out_manifest.is_absolute() else root / args.out_manifest
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    with open(out_man, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[ok] wrote {out_npz}  ({len(arrays)} arrays)")
    print(f"[ok] wrote {out_man}")


if __name__ == "__main__":
    main()
