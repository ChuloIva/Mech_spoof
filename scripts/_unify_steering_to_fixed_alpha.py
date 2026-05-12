"""Switch nb15+nb16 to fixed-α (repeng-canonical) steering for both models.

Why
---
Previously: `per_layer_sigma=True` meant the actual injection at layer L was
`k × unit × raw_norm[L]`, where `raw_norm = ‖harmful_mean − harmless_mean‖`.
That's a per-axis 'natural unit' — fine within one model, but on Llama 70B
the refusal natural-norm dwarfs the SU natural-norm by 13× (vs 3× on Qwen),
so identical `k` values inject vastly different magnitudes across methods,
breaking cross-model and cross-method comparison. Llama refusal at any
tested k collapsed to repeating-token gibberish for this reason.

Fix: drop the per-axis natural-unit. Use fixed-α scalar steering on unit
direction vectors — the canonical repeng / Arditi setup. Same α grid for
both methods on both models. Now 'SU at α=−1.0' and 'refusal at α=−1.0'
inject exactly the same vector magnitude, and the cross-model claim is
apples-to-apples.

Also pins POSITION_REF=-3 on both models (Llama nb15 was using -1, which
has the largest natural-norm and is anyway noisier than -3).

Affected notebooks
------------------
  - 15_jailbreak_steering_qwen35_4b.ipynb
  - 15_jailbreak_steering_llama33_70b.ipynb
  - 16_su_eval_axis_readout_qwen35_4b.ipynb
  - 16_su_eval_axis_readout_llama33_70b.ipynb

Idempotent: re-running on already-patched notebooks is a no-op (detected
via FIXED_ALPHA_MARKER in the rewritten direction-registry cell).
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / 'notebooks'

FIXED_ALPHA_MARKER = 'FIXED_ALPHA_STEERING_v1'

# ---------------------------------------------------------------------------
# Direction-registry cell: identical body for nb15 (Qwen + Llama). nb16 has a
# slightly different body (uses load_npz aliases EXP6 vs EXP6_NPZ etc.), so
# we generate the cell text per-notebook below.
# ---------------------------------------------------------------------------

def dir_registry_nb15(model: str) -> str:
    """nb15 direction-registry cell (Qwen or Llama variant)."""
    if model == 'qwen':
        steer_layers = 'list(range(16, 32))'
        exp6_load = (
            "exp6_pca = load_npz(EXP6_PCA)\n"
            "arrs6    = load_npz(EXP6_ARRAYS)\n"
            "_su_src  = exp6_pca\n"
            "_su_arrs = arrs6\n"
        )
    else:
        steer_layers = 'list(range(40, 80))  # last half of llama 70B'
        exp6_load = (
            "exp6 = load_npz(EXP6_NPZ)\n"
            "_su_src  = exp6\n"
            "_su_arrs = exp6  # llama bundles pca_center_dir + mm_raw in one npz\n"
        )
    return (
        f"# {FIXED_ALPHA_MARKER}\n"
        "# Fixed-α (repeng-canonical) steering: dirs are UNIT vectors, sigma=1.0.\n"
        "# 'k' in CONDITIONS is now a scalar α — the literal multiplier on the unit\n"
        "# direction, identical units for SU and refusal so cross-method and\n"
        "# cross-model comparisons are apples-to-apples.\n"
        f"STEER_LAYERS = {steer_layers}\n"
        "POSITION_SU  = 'response_last'\n"
        "POSITION_REF = -3   # canonical Arditi position; matches both models\n"
        "PER_LAYER_SIGMA = False  # legacy flag, kept for manifest compatibility\n"
        "\n"
        "def _unitize(v):\n"
        "    v = v.astype(np.float32)\n"
        "    return v / (np.linalg.norm(v) + 1e-8)\n"
        "\n"
        "# --- SU (unit) ---\n"
        f"{exp6_load}"
        "su_dirs  = {l: _unitize(_su_src[f'pca_center_dir__{POSITION_SU}__layer_{l:03d}']) for l in STEER_LAYERS}\n"
        "su_raw   = {l: _su_arrs[f'mm_raw__{POSITION_SU}__layer_{l:03d}'].astype(np.float32) for l in STEER_LAYERS}\n"
        "su_norms = {l: float(np.linalg.norm(su_raw[l])) for l in STEER_LAYERS}\n"
        "su_sigma = 1.0\n"
        "\n"
        "# --- refusal (unit) ---\n"
        "ref_arrs  = load_npz(REFUSAL_NPZ)\n"
        "ref_dirs  = {l: _unitize(ref_arrs[f'refusal__mm_dir__pos_{POSITION_REF:+d}__layer_{l:03d}']) for l in STEER_LAYERS}\n"
        "ref_raw   = {l: ref_arrs[f'refusal__mm_raw__pos_{POSITION_REF:+d}__layer_{l:03d}'].astype(np.float32) for l in STEER_LAYERS}\n"
        "ref_norms = {l: float(np.linalg.norm(ref_raw[l])) for l in STEER_LAYERS}\n"
        "ref_sigma = 1.0\n"
        "\n"
        "METHODS = {\n"
        "    'SU':      {'dirs': su_dirs,  'sigma': su_sigma},\n"
        "    'refusal': {'dirs': ref_dirs, 'sigma': ref_sigma},\n"
        "}\n"
        "\n"
        "def _cos(a, b):\n"
        "    a = a / (np.linalg.norm(a) + 1e-8); b = b / (np.linalg.norm(b) + 1e-8)\n"
        "    return float(np.dot(a, b))\n"
        "cos_per_layer = {l: _cos(su_dirs[l], ref_dirs[l]) for l in STEER_LAYERS}\n"
        "\n"
        "print(f'fixed-α steering. POSITION_REF={POSITION_REF}')\n"
        "print(f'  SU  unit-norm-check    median={np.median([np.linalg.norm(v) for v in su_dirs.values()]):.4f}')\n"
        "print(f'  ref unit-norm-check    median={np.median([np.linalg.norm(v) for v in ref_dirs.values()]):.4f}')\n"
        "print(f'  SU  natural-scale norm median={np.median(list(su_norms.values())):.2f} (raw_norm of mm-diff per layer; informational only)')\n"
        "print(f'  ref natural-scale norm median={np.median(list(ref_norms.values())):.2f}')\n"
        "print(f'  cos(SU, ref)           median={np.median(list(cos_per_layer.values())):+.3f}')\n"
    )


def dir_registry_nb16(model: str) -> str:
    """nb16 direction-registry cell. Llama variant has eval-aware stripped already
    (we just need to switch off PER_LAYER_SIGMA there). Qwen variant still has
    eval-aware blocks — preserve them, only change SU + refusal scaling."""
    if model == 'llama':
        return (
            f"# {FIXED_ALPHA_MARKER}\n"
            "# Fixed-α (repeng-canonical) steering — see nb15 for rationale.\n"
            "STEER_LAYERS = list(range(40, 80))  # last half of llama 70B's 80 layers\n"
            "POSITION_SU     = 'response_last'\n"
            "POSITION_REF    = -3       # canonical Arditi position\n"
            "PER_LAYER_SIGMA = False    # legacy flag, kept for manifest compatibility\n"
            "\n"
            "def _unitize(v):\n"
            "    v = v.astype(np.float32)\n"
            "    return v / (np.linalg.norm(v) + 1e-8)\n"
            "\n"
            "# --- SU (unit; readable + steerable) ---\n"
            "exp6 = load_npz(EXP6_NPZ)\n"
            "arrs6 = exp6  # llama bundles pca_center_dir + mm_raw in one npz\n"
            "su_unit  = {l: _unitize(exp6[f'pca_center_dir__{POSITION_SU}__layer_{l:03d}']) for l in STEER_LAYERS}\n"
            "su_raw   = {l: arrs6[f'mm_raw__{POSITION_SU}__layer_{l:03d}'].astype(np.float32) for l in STEER_LAYERS}\n"
            "su_norms = {l: float(np.linalg.norm(su_raw[l])) for l in STEER_LAYERS}\n"
            "su_dirs  = dict(su_unit)\n"
            "su_sigma = 1.0\n"
            "\n"
            "# --- refusal (unit; readable + steerable) ---\n"
            "dirs_arrs = load_npz(DIRS_NPZ)\n"
            "ref_unit  = {l: _unitize(dirs_arrs[f'refusal__mm_dir__pos_{POSITION_REF:+d}__layer_{l:03d}']) for l in STEER_LAYERS}\n"
            "ref_raw   = {l: dirs_arrs[f'refusal__mm_raw__pos_{POSITION_REF:+d}__layer_{l:03d}'].astype(np.float32) for l in STEER_LAYERS}\n"
            "ref_norms = {l: float(np.linalg.norm(ref_raw[l])) for l in STEER_LAYERS}\n"
            "ref_dirs  = dict(ref_unit)\n"
            "ref_sigma = 1.0\n"
            "\n"
            "READ_LAYERS = list(range(loaded.n_layers))\n"
            "ref_unit_read = ref_unit\n"
            "su_unit_read  = su_unit\n"
            "\n"
            "METHODS = {\n"
            "    'SU':      {'dirs': su_dirs,  'sigma': su_sigma},\n"
            "    'refusal': {'dirs': ref_dirs, 'sigma': ref_sigma},\n"
            "}\n"
            "\n"
            "def _cos(a, b):\n"
            "    a = a / (np.linalg.norm(a) + 1e-8); b = b / (np.linalg.norm(b) + 1e-8)\n"
            "    return float(np.dot(a, b))\n"
            "cos_su_ref = {l: _cos(su_unit[l], ref_unit[l]) for l in STEER_LAYERS}\n"
            "\n"
            "print(f'fixed-α steering. POSITION_REF={POSITION_REF}')\n"
            "print(f'  SU  natural-scale norm median={np.median(list(su_norms.values())):.2f}')\n"
            "print(f'  ref natural-scale norm median={np.median(list(ref_norms.values())):.2f}')\n"
            "print(f'  cos(SU, refusal) median={np.median(list(cos_su_ref.values())):+.3f}')\n"
        )
    else:
        # Qwen nb16: keep eval-aware blocks (still useful for that notebook's appendix)
        # but switch SU + refusal to fixed-α.
        return (
            f"# {FIXED_ALPHA_MARKER}\n"
            "# Fixed-α (repeng-canonical) steering — see nb15 for rationale.\n"
            "STEER_LAYERS    = list(range(16, 32))\n"
            "POSITION_SU     = 'response_last'\n"
            "POSITION_REF    = -3       # canonical Arditi position\n"
            "EVAL_VARIANT    = 'contrastive_dataset'\n"
            "PER_LAYER_SIGMA = False    # legacy flag, kept for manifest compatibility\n"
            "\n"
            "def _unitize(v):\n"
            "    v = v.astype(np.float32)\n"
            "    return v / (np.linalg.norm(v) + 1e-8)\n"
            "\n"
            "# --- SU (unit) ---\n"
            "exp6_pca = load_npz(EXP6_PCA)\n"
            "arrs6    = load_npz(EXP6_ARRAYS)\n"
            "su_unit  = {l: _unitize(exp6_pca[f'pca_center_dir__{POSITION_SU}__layer_{l:03d}']) for l in STEER_LAYERS}\n"
            "su_raw   = {l: arrs6[f'mm_raw__{POSITION_SU}__layer_{l:03d}'].astype(np.float32) for l in STEER_LAYERS}\n"
            "su_norms = {l: float(np.linalg.norm(su_raw[l])) for l in STEER_LAYERS}\n"
            "su_dirs  = dict(su_unit)\n"
            "su_sigma = 1.0\n"
            "\n"
            "# --- refusal (unit) + eval_aw (readout only) ---\n"
            "dirs_arrs = load_npz(DIRS_NPZ)\n"
            "ref_unit  = {l: _unitize(dirs_arrs[f'refusal__mm_dir__pos_{POSITION_REF:+d}__layer_{l:03d}']) for l in STEER_LAYERS}\n"
            "ref_raw   = {l: dirs_arrs[f'refusal__mm_raw__pos_{POSITION_REF:+d}__layer_{l:03d}'].astype(np.float32) for l in STEER_LAYERS}\n"
            "ref_norms = {l: float(np.linalg.norm(ref_raw[l])) for l in STEER_LAYERS}\n"
            "ref_dirs  = dict(ref_unit)\n"
            "ref_sigma = 1.0\n"
            "\n"
            "# Eval-aware: ALL layers (read upstream of steering range too as sanity check)\n"
            "READ_LAYERS = list(range(loaded.n_layers))\n"
            "eval_unit    = {l: _unitize(dirs_arrs[f'eval_aw__mm_dir__{EVAL_VARIANT}__layer_{l:03d}']) for l in READ_LAYERS}\n"
            "eval_test    = {l: dirs_arrs[f'eval_aw__test_mean__{EVAL_VARIANT}__layer_{l:03d}'].astype(np.float32) for l in READ_LAYERS}\n"
            "eval_deploy  = {l: dirs_arrs[f'eval_aw__deploy_mean__{EVAL_VARIANT}__layer_{l:03d}'].astype(np.float32) for l in READ_LAYERS}\n"
            "test_anchor   = {l: float(eval_test[l]   @ eval_unit[l]) for l in READ_LAYERS}\n"
            "deploy_anchor = {l: float(eval_deploy[l] @ eval_unit[l]) for l in READ_LAYERS}\n"
            "\n"
            "ref_unit_read = ref_unit\n"
            "su_unit_read  = su_unit\n"
            "\n"
            "METHODS = {\n"
            "    'SU':      {'dirs': su_dirs,  'sigma': su_sigma},\n"
            "    'refusal': {'dirs': ref_dirs, 'sigma': ref_sigma},\n"
            "}\n"
            "\n"
            "def _cos(a, b):\n"
            "    a = a / (np.linalg.norm(a) + 1e-8); b = b / (np.linalg.norm(b) + 1e-8)\n"
            "    return float(np.dot(a, b))\n"
            "cos_su_eval  = {l: _cos(su_unit[l],  eval_unit[l]) for l in STEER_LAYERS}\n"
            "cos_ref_eval = {l: _cos(ref_unit[l], eval_unit[l]) for l in STEER_LAYERS}\n"
            "cos_su_ref   = {l: _cos(su_unit[l],  ref_unit[l])  for l in STEER_LAYERS}\n"
            "\n"
            "print(f'fixed-α steering. POSITION_REF={POSITION_REF}')\n"
            "print(f'  SU  natural-scale norm median={np.median(list(su_norms.values())):.2f}')\n"
            "print(f'  ref natural-scale norm median={np.median(list(ref_norms.values())):.2f}')\n"
            "print(f'  cos(SU, ref) median={np.median(list(cos_su_ref.values())):+.3f}')\n"
        )


# ---------------------------------------------------------------------------
# Conditions: same unified α grid for both models. nb15 is the dense grid for
# the dose-response sweep; nb16 is sparser (forward only, just enough for a
# leverage slope fit).
# ---------------------------------------------------------------------------

NB15_CONDITIONS = '''# Unified fixed-α grid for both methods. Same scalar units for SU and refusal,
# enabling apples-to-apples cross-method (and cross-model) comparison.
# Negative α = jailbreak direction; positive α = defense booster.
CONDITIONS = [
    ('baseline',         None,       0.0),
    # SU sweep
    ('SU α=-3.0',        'SU',      -3.0),
    ('SU α=-2.0',        'SU',      -2.0),
    ('SU α=-1.5',        'SU',      -1.5),
    ('SU α=-1.0',        'SU',      -1.0),
    ('SU α=-0.7',        'SU',      -0.7),
    ('SU α=-0.5',        'SU',      -0.5),
    ('SU α=-0.3',        'SU',      -0.3),
    ('SU α=-0.15',       'SU',      -0.15),
    ('SU α=+0.15',       'SU',      +0.15),
    ('SU α=+0.3',        'SU',      +0.3),
    ('SU α=+0.5',        'SU',      +0.5),
    ('SU α=+1.0',        'SU',      +1.0),
    # Refusal sweep — same α grid for direct comparison
    ('refusal α=-3.0',   'refusal', -3.0),
    ('refusal α=-2.0',   'refusal', -2.0),
    ('refusal α=-1.5',   'refusal', -1.5),
    ('refusal α=-1.0',   'refusal', -1.0),
    ('refusal α=-0.7',   'refusal', -0.7),
    ('refusal α=-0.5',   'refusal', -0.5),
    ('refusal α=-0.3',   'refusal', -0.3),
    ('refusal α=-0.15',  'refusal', -0.15),
    ('refusal α=+0.15',  'refusal', +0.15),
    ('refusal α=+0.3',   'refusal', +0.3),
    ('refusal α=+0.5',   'refusal', +0.5),
    ('refusal α=+1.0',   'refusal', +1.0),
]
print(f'{len(CONDITIONS)} conditions: 1 baseline + 12 SU + 12 refusal on unified α grid')
for n, m, k in CONDITIONS:
    print(f'  {n:<18s}  method={m!s:<8s}  α={k:+.2f}')
'''

NB16_CONDITIONS = '''# Sparser unified α grid for the leverage-curve readout (forward only).
# Same α set for SU and refusal so the leverage-slope comparison is apples-to-apples.
CONDITIONS = [
    ('baseline',        None,       0.0),
    ('SU α=-1.5',       'SU',      -1.5),
    ('SU α=-1.0',       'SU',      -1.0),
    ('SU α=-0.5',       'SU',      -0.5),
    ('SU α=-0.3',       'SU',      -0.3),
    ('SU α=-0.15',      'SU',      -0.15),
    ('SU α=+0.3',       'SU',      +0.3),
    ('SU α=+1.0',       'SU',      +1.0),
    ('refusal α=-1.5',  'refusal', -1.5),
    ('refusal α=-1.0',  'refusal', -1.0),
    ('refusal α=-0.5',  'refusal', -0.5),
    ('refusal α=-0.3',  'refusal', -0.3),
    ('refusal α=-0.15', 'refusal', -0.15),
    ('refusal α=+0.3',  'refusal', +0.3),
    ('refusal α=+1.0',  'refusal', +1.0),
]
for n, m, k in CONDITIONS:
    print(f'  {n:<18s}  method={m!s:<8s}  α={k:+.2f}')
'''


# ---------------------------------------------------------------------------
# Per-notebook patch plans
# ---------------------------------------------------------------------------

PATCHES = [
    {
        'nb': '15_jailbreak_steering_qwen35_4b.ipynb',
        'rewrites': [
            ('STEER_LAYERS = list(range(16, 32))', dir_registry_nb15('qwen')),
            ("CONDITIONS = [\n    ('baseline'",      NB15_CONDITIONS),
        ],
    },
    {
        'nb': '15_jailbreak_steering_llama33_70b.ipynb',
        'rewrites': [
            ("STEER_LAYERS = list(range(40, 80))  # last half of llama 70B's 80 layers",
             dir_registry_nb15('llama')),
            ("CONDITIONS = [\n    ('baseline'",      NB15_CONDITIONS),
        ],
    },
    {
        'nb': '16_su_eval_axis_readout_qwen35_4b.ipynb',
        'rewrites': [
            ('STEER_LAYERS    = list(range(16, 32))', dir_registry_nb16('qwen')),
            ("CONDITIONS = [\n    ('baseline'",       NB16_CONDITIONS),
        ],
    },
    {
        'nb': '16_su_eval_axis_readout_llama33_70b.ipynb',
        'rewrites': [
            ("STEER_LAYERS = list(range(40, 80))  # last half of llama 70B's 80 layers",
             dir_registry_nb16('llama')),
            ("CONDITIONS = [\n    ('baseline'",       NB16_CONDITIONS),
        ],
    },
]


def _src(c):
    s = c.get('source', '')
    return ''.join(s) if isinstance(s, list) else s


def _set_src(c, text: str):
    split = text.split('\n')
    c['source'] = [s + '\n' for s in split[:-1]] + ([split[-1]] if split[-1] else [])


def find_one(nb, fp):
    matches = [(i, c) for i, c in enumerate(nb['cells'])
               if c.get('cell_type') == 'code' and fp in _src(c)]
    if len(matches) == 0:
        raise RuntimeError(f'no code cell matches fingerprint: {fp!r}')
    if len(matches) > 1:
        raise RuntimeError(f'fingerprint {fp!r} ambiguous, matches {len(matches)} cells')
    return matches[0]


def main():
    for plan in PATCHES:
        path = NB_DIR / plan['nb']
        nb = json.loads(path.read_text())
        if any(FIXED_ALPHA_MARKER in _src(c) for c in nb['cells']):
            print(f'  {plan["nb"]}: already on fixed-α (marker present); skipping')
            continue
        for fp, new_text in plan['rewrites']:
            _, c = find_one(nb, fp)
            _set_src(c, new_text)
            c['outputs'] = []
            c['execution_count'] = None
        path.write_text(json.dumps(nb, indent=1) + '\n')
        print(f'  {plan["nb"]}: patched {len(plan["rewrites"])} cells → fixed-α')


if __name__ == '__main__':
    main()
