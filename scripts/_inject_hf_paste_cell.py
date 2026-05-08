"""Inject a paste-token cell into the Llama 3.3 70B notebooks.

Adds a small markdown + code cell pair right BEFORE the existing
'2 — Project root, paths, optional HF auth' cell where the user can
just paste their HF token between the quotes and run.

Idempotent: skips notebooks that already have the paste cell.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / 'notebooks'

TARGETS = [
    '14_refusal_and_eval_awareness_directions_llama33_70b.ipynb',
    '15_jailbreak_steering_llama33_70b.ipynb',
    '16_su_eval_axis_readout_llama33_70b.ipynb',
]

PASTE_MARKER = 'HF_TOKEN_PASTE'

PASTE_MARKDOWN = {
    'cell_type': 'markdown',
    'id': 'hf-paste-token',
    'metadata': {},
    'source': [
        "## 1c — Paste your Hugging Face token\n",
        "\n",
        "Llama 3.3 70B is gated. Paste your `HF_TOKEN` between the quotes below and run this cell. The token is stored only in this kernel's environment (not written to disk). If you're on Colab and have already set `HF_TOKEN` in *Secrets*, you can leave this empty.\n",
    ],
}

PASTE_CODE = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'hf-paste-token-code',
    'metadata': {},
    'outputs': [],
    'source': [
        "import os\n",
        "from huggingface_hub import login\n",
        "\n",
        "# >>> Paste your HF token between the quotes <<<\n",
        "HF_TOKEN_PASTE = \"\"  # e.g. \"hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890\"\n",
        "\n",
        "if HF_TOKEN_PASTE:\n",
        "    os.environ['HF_TOKEN'] = HF_TOKEN_PASTE\n",
        "if os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'):\n",
        "    login(token=os.environ.get('HF_TOKEN') or os.environ['HUGGING_FACE_HUB_TOKEN'],\n",
        "          add_to_git_credential=False)\n",
        "    print('logged in to Hugging Face')\n",
        "else:\n",
        "    raise RuntimeError('No HF token set. Paste it above (HF_TOKEN_PASTE = \"hf_...\") or set HF_TOKEN in env / Colab Secrets.')\n",
    ],
}


def already_injected(nb: dict) -> bool:
    for c in nb['cells']:
        s = ''.join(c['source']) if isinstance(c.get('source'), list) else c.get('source', '')
        if PASTE_MARKER in s:
            return True
    return False


def find_anchor_index(nb: dict) -> int:
    """Insert before the section-2 cell (the one that asserts HF_TOKEN). If we
    can't find that, insert before the first cell that calls `load_model`."""
    for i, c in enumerate(nb['cells']):
        s = ''.join(c['source']) if isinstance(c.get('source'), list) else c.get('source', '')
        if "assert os.environ.get('HF_TOKEN')" in s and 'login(token=' in s:
            return i
    for i, c in enumerate(nb['cells']):
        s = ''.join(c['source']) if isinstance(c.get('source'), list) else c.get('source', '')
        if 'load_model(' in s:
            return i
    return 1  # after the title


def patch_existing_auth_cell(nb: dict) -> None:
    """In the existing project-root cell that does the assert+login, drop the
    bottom assert and login (now handled by the paste cell). Keep the path
    setup and the `userdata` Colab-Secrets fallback."""
    for c in nb['cells']:
        if c.get('cell_type') != 'code':
            continue
        src = c['source']
        joined = ''.join(src) if isinstance(src, list) else src
        if "assert os.environ.get('HF_TOKEN')" not in joined:
            continue
        if 'login(token=' not in joined:
            continue
        new_lines = []
        skip = False
        for line in (src if isinstance(src, list) else joined.splitlines(keepends=True)):
            stripped = line.strip()
            if stripped.startswith("assert os.environ.get('HF_TOKEN')"):
                skip = True
                continue
            if skip and stripped.endswith("'HF_TOKEN required (Llama 3.3 is gated).'"):
                skip = False
                continue
            if stripped == 'from huggingface_hub import login':
                continue
            if stripped.startswith('login(token='):
                continue
            new_lines.append(line)
        c['source'] = new_lines
        return


for name in TARGETS:
    path = NB_DIR / name
    nb = json.loads(path.read_text())
    if already_injected(nb):
        print(f'  {name}: paste cell already present; skipping')
        continue
    patch_existing_auth_cell(nb)
    idx = find_anchor_index(nb)
    nb['cells'].insert(idx, PASTE_CODE)
    nb['cells'].insert(idx, PASTE_MARKDOWN)
    path.write_text(json.dumps(nb, indent=1) + '\n')
    print(f'  {name}: injected paste cell at position {idx}')
