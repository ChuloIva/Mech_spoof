"""Pin transformers to a version that doesn't trip torch's infer_schema bug.

`transformers >= 4.55` introduced `integrations/moe.py` which calls
`torch.library.custom_op` with string-form annotations (`'torch.Tensor'`).
Older torch (< 2.6) can't resolve those forward refs and the import dies with:

    ValueError: infer_schema(func): Parameter input has unsupported type torch.Tensor.

Fix: pin transformers to 4.46.3 — known-good with bitsandbytes 8-bit Llama 3.3
loading and current Accelerate. Updates the `!pip install ...` cell in each of
the three Llama-port notebooks.

Idempotent.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / 'notebooks'

TARGETS = [
    '14_refusal_and_eval_awareness_directions_llama33_70b.ipynb',
    '15_jailbreak_steering_llama33_70b.ipynb',
    '16_su_eval_axis_readout_llama33_70b.ipynb',
]

PIN = "'transformers==4.46.3'"

# Match the existing install line: `!pip install -q 'transformers>=4.45' ...`
PAT = re.compile(r"'transformers>=?[\d.]+'")


for name in TARGETS:
    path = NB_DIR / name
    nb = json.loads(path.read_text())
    changed = False
    for c in nb['cells']:
        if c.get('cell_type') != 'code':
            continue
        src = c['source']
        joined = ''.join(src) if isinstance(src, list) else src
        if 'pip install' not in joined or 'transformers' not in joined:
            continue
        if PIN in joined:
            continue
        new = PAT.sub(PIN, joined)
        if new == joined:
            continue
        # Re-split into list-of-lines preserving newline convention
        if isinstance(src, list):
            split = new.split('\n')
            new_lines = [s + '\n' for s in split[:-1]] + ([split[-1]] if split[-1] else [])
            c['source'] = new_lines
        else:
            c['source'] = new
        changed = True
    if changed:
        path.write_text(json.dumps(nb, indent=1) + '\n')
        print(f'  {name}: pinned transformers to {PIN}')
    else:
        print(f'  {name}: no install cell found or already pinned')
