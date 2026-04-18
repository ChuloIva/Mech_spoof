"""Experiment 1 — Authority direction (§4).

Pipeline:
    1. Load model.
    2. Build Condition S (system) and Condition U (user) prompts for every structural instruction.
    3. Extract residual at response_first for every prompt at every layer.
    4. Train per-layer logistic probes. Also compute difference-in-means directions.
    5. Save probes, directions, accuracies, and the fake-delimiter tokenization report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mech_spoof.activations import extract_at_positions, response_first_position
from mech_spoof.configs import MODEL_CONFIGS
from mech_spoof.datasets.structural import build_structural_contrastive
from mech_spoof.io import save_result_bundle, save_pickle
from mech_spoof.models import free_model, load_model
from mech_spoof.probes import (
    compute_authority_direction_dim,
    cosine_agreement,
    find_best_layer,
    train_authority_probe,
)
from mech_spoof.utils import get_logger, set_seed, timer

logger = get_logger(__name__)


@dataclass
class Exp1Result:
    model_key: str
    n_layers: int
    d_model: int
    accuracies: dict[int, float]
    aurocs: dict[int, float]
    best_layer: int
    best_accuracy: float
    probe_vs_dim_cosine: dict[int, float]
    n_system: int
    n_user: int
    fake_delim_report: dict


def run_experiment_1(
    model_key: str,
    out_dir: Path,
    seed: int = 42,
    free_after: bool = True,
    cache_activations: bool = True,
) -> Exp1Result:
    """Run Experiment 1 end-to-end for one model."""
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model(model_key)
    logger.info(f"[{model_key}] n_layers={loaded.n_layers} d_model={loaded.d_model}")

    # Fake-delimiter tokenization report (§9.2)
    report = loaded.template.fake_delimiter_tokenization_report()

    with timer(f"[{model_key}] build structural dataset"):
        ds = build_structural_contrastive(loaded.template, seed=seed)
    logger.info(f"[{model_key}] {len(ds.prompts_system)} system / {len(ds.prompts_user)} user prompts")

    cache_s = out_dir / "act_cache_system" if cache_activations else None
    cache_u = out_dir / "act_cache_user" if cache_activations else None

    with timer(f"[{model_key}] extract S activations"):
        sys_acts = extract_at_positions(
            loaded, ds.prompts_system, response_first_position, cache_dir=cache_s
        )
    with timer(f"[{model_key}] extract U activations"):
        user_acts = extract_at_positions(
            loaded, ds.prompts_user, response_first_position, cache_dir=cache_u
        )

    # Train probes + diff-in-means
    with timer(f"[{model_key}] train probes"):
        probe_res = train_authority_probe(sys_acts, user_acts, seed=seed)
    dim_dirs = compute_authority_direction_dim(sys_acts, user_acts)
    agree = cosine_agreement(probe_res.directions, dim_dirs)
    best = find_best_layer(probe_res.accuracies)

    # Save
    arrays = {
        f"probe_dir_layer_{l:03d}": v for l, v in probe_res.directions.items()
    }
    arrays.update({f"dim_dir_layer_{l:03d}": v for l, v in dim_dirs.items()})
    arrays["accuracies"] = np.array([probe_res.accuracies[i] for i in range(loaded.n_layers)])
    arrays["aurocs"] = np.array([probe_res.aurocs[i] for i in range(loaded.n_layers)])

    result_json = {
        "model_key": model_key,
        "hf_id": loaded.cfg.hf_id,
        "n_layers": loaded.n_layers,
        "d_model": loaded.d_model,
        "accuracies": {int(k): float(v) for k, v in probe_res.accuracies.items()},
        "aurocs": {int(k): float(v) for k, v in probe_res.aurocs.items()},
        "best_layer": int(best),
        "best_accuracy": float(probe_res.accuracies[best]),
        "probe_vs_dim_cosine": {int(k): float(v) for k, v in agree.items()},
        "n_system": len(ds.prompts_system),
        "n_user": len(ds.prompts_user),
        "fake_delim_report": asdict(report),
        "train_idx": ds.train_idx,
        "test_idx": ds.test_idx,
    }

    save_result_bundle(
        out_dir,
        json_obj=result_json,
        arrays=arrays,
        pickles={"probes": probe_res.probes},
        manifest_extras={"experiment": "exp1_authority", "model_key": model_key},
    )
    save_pickle(probe_res, out_dir / "probe_result.pkl")

    if free_after:
        free_model(loaded)

    return Exp1Result(
        model_key=model_key,
        n_layers=loaded.n_layers,
        d_model=loaded.d_model,
        accuracies=probe_res.accuracies,
        aurocs=probe_res.aurocs,
        best_layer=best,
        best_accuracy=probe_res.accuracies[best],
        probe_vs_dim_cosine=agree,
        n_system=len(ds.prompts_system),
        n_user=len(ds.prompts_user),
        fake_delim_report=asdict(report),
    )
