"""Matplotlib visualizations for the paper's key figures (§8.2)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_layer_accuracy(accuracies: dict[int, float], model_name: str, out_path: Path) -> Path:
    """Probe accuracy curve across layers — Exp 1 basic result."""
    layers = sorted(accuracies.keys())
    vals = [accuracies[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, vals, marker="o", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance (0.5)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe accuracy (system vs user)")
    ax.set_title(f"{model_name} — authority probe accuracy by layer")
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_probe_accuracy_overlay(per_model: dict[str, dict], out_path: Path) -> Path:
    """Figure 1: accuracy curves for all models, x-axis normalized to relative depth [0,1]."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, data in per_model.items():
        exp1 = data.get("exp1_authority")
        if exp1 is None:
            continue
        acc = exp1["accuracies"]
        layers = sorted(int(l) for l in acc.keys())
        n = exp1["n_layers"]
        xs = [l / max(n - 1, 1) for l in layers]
        ys = [acc[str(l)] if str(l) in acc else acc[l] for l in layers]
        ax.plot(xs, ys, marker="", linewidth=2, label=key)
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
    ax.set_xlabel("Relative layer depth")
    ax.set_ylabel("Probe accuracy (system vs user)")
    ax.set_title("Authority probe accuracy across models")
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_authority_refusal_cosine(per_model: dict[str, dict], out_path: Path) -> Path:
    """Figure 2: cosine(authority, refusal) by relative layer depth, one line per model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, data in per_model.items():
        exp3 = data.get("exp3_refusal")
        if exp3 is None:
            continue
        cos_map = exp3["geometry"]["cosine_by_layer"]
        layers = sorted(int(l) for l in cos_map.keys())
        n_layers = max(layers) + 1
        xs = [l / max(n_layers - 1, 1) for l in layers]
        ys = [cos_map[str(l)] if str(l) in cos_map else cos_map[l] for l in layers]
        ax.plot(xs, ys, marker="", linewidth=2, label=key)
    ax.axhline(0.0, color="gray", linestyle="--")
    ax.set_xlabel("Relative layer depth")
    ax.set_ylabel("cos(authority, refusal)")
    ax.set_title("Authority vs refusal direction cosine similarity")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_conflict_compliance_bars(per_model: dict[str, dict], out_path: Path) -> Path:
    """Figure 3: compliance rate for REAL / NONE / FAKE per model."""
    models = []
    real_vals, none_vals, fake_vals = [], [], []
    for key, data in per_model.items():
        exp2 = data.get("exp2_conflict")
        if exp2 is None:
            continue
        summary = exp2["summary"]
        models.append(key)
        real_vals.append(summary.get("REAL", {}).get("compliance_rate", np.nan))
        none_vals.append(summary.get("NONE", {}).get("compliance_rate", np.nan))
        fake_vals.append(summary.get("FAKE", {}).get("compliance_rate", np.nan))

    if not models:
        return out_path

    x = np.arange(len(models))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, real_vals, w, label="REAL (system field)")
    ax.bar(x, none_vals, w, label="NONE (both in user)")
    ax.bar(x + w, fake_vals, w, label="FAKE (delimiter injection)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("System-instruction compliance rate")
    ax.set_title("Conflict-test compliance across conditions")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_attack_prediction_scatter(per_model: dict[str, dict], out_path: Path) -> Path:
    """Figure 4: scatter of authority probe score vs behavioral refusal, colored by attack type."""
    fig, ax = plt.subplots(figsize=(9, 6))
    attack_colors: dict[str, str] = {}
    cmap = plt.get_cmap("tab10")

    n = 0
    for key, data in per_model.items():
        exp4 = data.get("exp4_attacks")
        if exp4 is None:
            continue
        for row in exp4["rows"]:
            atype = row["attack_type"]
            if atype not in attack_colors:
                attack_colors[atype] = cmap(len(attack_colors) % 10)
            y = 0 if row["refused"] else 1   # 1 = complied
            ax.scatter(
                row["authority_score"], y, s=15, alpha=0.6,
                color=attack_colors[atype],
            )
            n += 1
    for atype, color in attack_colors.items():
        ax.scatter([], [], color=color, label=atype)
    ax.set_xlabel("Authority probe score (at best layer)")
    ax.set_ylabel("Complied (1) vs refused (0)")
    ax.set_title(f"Attack probe score vs behavioral compliance (n={n})")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_token_trace(trace: list[dict], out_path: Path, title: str = "") -> Path:
    """Figure 5: token-by-token authority + refusal projection for a multi-turn injection."""
    positions = [t["position"] for t in trace]
    auth = [t["authority_score"] for t in trace]
    ref = [t["refusal_score"] for t in trace]
    delim_positions = [t["position"] for t in trace if t.get("is_delimiter")]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(positions, auth, linewidth=1.5, label="authority")
    ax.plot(positions, ref, linewidth=1.5, label="refusal")
    for p in delim_positions:
        ax.axvline(p, color="gray", alpha=0.25, linewidth=0.5)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Projection score")
    ax.set_title(title or "Token-by-token projections (vertical: delimiter positions)")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_summary_table(per_model: dict[str, dict], out_path: Path):
    """Table 1: per-model headline numbers."""
    from mech_spoof.experiments.exp5_comparative import aggregate_results
    # We already have per_model loaded — just reuse rows logic here for portability.
    import pandas as pd

    rows = []
    for key, data in per_model.items():
        exp1 = data.get("exp1_authority", {})
        exp3 = data.get("exp3_refusal", {})
        exp2 = data.get("exp2_conflict", {})
        rows.append({
            "model": key,
            "n_layers": exp1.get("n_layers"),
            "best_layer": exp1.get("best_layer"),
            "best_accuracy": exp1.get("best_accuracy"),
            "auth_vs_refusal_cos": exp3.get("cosine_at_best_authority_layer"),
            "fake_compliance": exp2.get("summary", {}).get("FAKE", {}).get("compliance_rate"),
            "real_compliance": exp2.get("summary", {}).get("REAL", {}).get("compliance_rate"),
        })
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
