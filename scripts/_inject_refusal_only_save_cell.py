"""Inject a short-circuit 'save refusal directions only' cell into nb14.

After §5 finishes computing the refusal directions, this cell writes a
refusal-only `directions.npz` + `manifest.json` to OUT_DIR and (on Colab)
zips them into a single download. Lets the user stop here without running
the eval-awareness section if they only need the refusal vectors.

Idempotent.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / 'notebooks'
TARGET = NB_DIR / '14_refusal_and_eval_awareness_directions_llama33_70b.ipynb'

MARKER = 'REFUSAL_ONLY_SHORTCIRCUIT_SAVE'

NEW_MARKDOWN = {
    'cell_type': 'markdown',
    'id': 'refusal-only-save-md',
    'metadata': {},
    'source': [
        "## 5b — Save refusal directions only (short-circuit)\n",
        "\n",
        "If you only need the refusal direction (i.e. you don't care about the eval-awareness mean-diff in §6), run this cell and stop. It writes a refusal-only `directions.npz` + `manifest.json` to `OUT_DIR`, and on Colab triggers a single-zip download. Notebook 15/16 only need the refusal vectors, so this is the canonical short path.\n",
    ],
}

NEW_CODE = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'refusal-only-save-code',
    'metadata': {},
    'outputs': [],
    'source': [
        "# " + MARKER + "\n",
        "# Refusal-only save: writes the same NPZ schema as the full notebook,\n",
        "# but skips every eval-aware key. Safe to run as soon as §5 finishes.\n",
        "import shutil\n",
        "\n",
        "refusal_payload = {}\n",
        "for pi, P in enumerate(POSITIONS):\n",
        "    for L in range(n_layers):\n",
        "        suffix = f'pos_{P:+d}__layer_{L:03d}'\n",
        "        refusal_payload[f'refusal__mm_dir__{suffix}']        = refusal_dir[L, pi].numpy()\n",
        "        refusal_payload[f'refusal__mm_raw__{suffix}']        = raw_diff[L, pi].numpy()\n",
        "        refusal_payload[f'refusal__harmful_mean__{suffix}']  = harmful_mean[L, pi].numpy()\n",
        "        refusal_payload[f'refusal__harmless_mean__{suffix}'] = harmless_mean[L, pi].numpy()\n",
        "\n",
        "refusal_manifest = {\n",
        "    'experiment':    'refusal_directions_only',\n",
        "    'model_key':     'llama33_70b',\n",
        "    'hf_id':         loaded.cfg.hf_id,\n",
        "    'n_layers':      n_layers,\n",
        "    'd_model':       d_model,\n",
        "    'hook':          'post-block forward hook on model.model.layers[L]',\n",
        "    'enable_thinking_supported': supports_thinking,\n",
        "    'refusal': {\n",
        "        'recipe':            'Arditi et al. 2406.11717 — mean-diff at last 5 EOI positions',\n",
        "        'positions':         POSITIONS,\n",
        "        'n_train_per_class': N_TRAIN,\n",
        "        'n_harmful_used':    int(n_harmful),\n",
        "        'n_harmless_used':   int(n_harmless),\n",
        "        'mm_natural_scale':  norms.tolist(),\n",
        "        'frac_harmful_proj_larger': frac_harmful_proj_larger.tolist(),\n",
        "    },\n",
        "}\n",
        "\n",
        "np.savez_compressed(OUT_DIR / 'directions.npz', **refusal_payload)\n",
        "(OUT_DIR / 'manifest.json').write_text(json.dumps(refusal_manifest, indent=2, default=float))\n",
        "print(f'wrote {len(refusal_payload)} refusal arrays to {OUT_DIR / \"directions.npz\"}')\n",
        "print(f'wrote manifest to {OUT_DIR / \"manifest.json\"}')\n",
        "for f in sorted(OUT_DIR.iterdir()):\n",
        "    print(f'  {f.name:<24s} {f.stat().st_size/1e6:>8.2f} MB')\n",
        "\n",
        "# Pack into a single zip for one-click download on Colab.\n",
        "zip_base = OUT_DIR.parent / 'exp_directions_llama33_70b_refusal_only'\n",
        "if zip_base.with_suffix('.zip').exists():\n",
        "    zip_base.with_suffix('.zip').unlink()\n",
        "shutil.make_archive(str(zip_base), 'zip', root_dir=OUT_DIR)\n",
        "zip_path = zip_base.with_suffix('.zip')\n",
        "print(f'\\nzipped → {zip_path}  ({zip_path.stat().st_size/1e6:.2f} MB)')\n",
        "\n",
        "try:\n",
        "    from google.colab import files\n",
        "    files.download(str(zip_path))\n",
        "    print('Colab download triggered.')\n",
        "except Exception:\n",
        "    print('Not on Colab — copy the zip off the pod manually, e.g.:')\n",
        "    print(f'  scp <pod>:{zip_path} ./')\n",
    ],
}


def already_injected(nb: dict) -> bool:
    for c in nb['cells']:
        s = ''.join(c['source']) if isinstance(c.get('source'), list) else c.get('source', '')
        if MARKER in s:
            return True
    return False


def find_anchor_index(nb: dict) -> int:
    """Insert right AFTER the cell that computes refusal_dir (the np cell that
    starts with `import numpy as np` and ends the §5 block)."""
    for i, c in enumerate(nb['cells']):
        if c.get('cell_type') != 'code':
            continue
        s = ''.join(c['source']) if isinstance(c.get('source'), list) else c.get('source', '')
        if 'refusal_dir = raw_diff / norms' in s and 'frac_harmful_proj_larger' in s:
            return i + 1
    return None


nb = json.loads(TARGET.read_text())
if already_injected(nb):
    print(f'  {TARGET.name}: short-circuit cell already present; skipping')
else:
    idx = find_anchor_index(nb)
    if idx is None:
        raise RuntimeError(f'could not find §5-end anchor cell in {TARGET.name}')
    nb['cells'].insert(idx, NEW_CODE)
    nb['cells'].insert(idx, NEW_MARKDOWN)
    TARGET.write_text(json.dumps(nb, indent=1) + '\n')
    print(f'  {TARGET.name}: inserted §5b refusal-only save at cells {idx}/{idx+1}')
