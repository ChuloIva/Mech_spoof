"""Adapt the Llama 3.3 70B notebooks for multi-GPU runs (e.g. 3× A100).

For each of nb14/15/16:
  - Replace the bare `loaded = load_model('llama33_70b')` call with one
    that honours a `QUANT` choice ("8bit" / "bf16") and balances memory
    across all visible GPUs via `device_map="auto"` + a `max_memory` dict.
  - Print the resulting layer-to-device sharding so the user can verify.

Idempotent: re-running on already-patched notebooks is a no-op.
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

MARKER = 'MECH_SPOOF_LLAMA33_MULTI_GPU_LOAD'

# Drop-in replacement for the `loaded = load_model('llama33_70b')` line.
# We replace one line, so we can keep the surrounding cell text intact.
NEW_LOAD_BLOCK = (
    "# " + MARKER + "\n"
    "# Multi-GPU loader. Pick precision below; load_model honours device_map='auto'\n"
    "# so the 70B is automatically sharded across every visible GPU.\n"
    "import torch\n"
    "QUANT = 'bf16'  # default: full bf16 (~140 GB; fits 3x A100 80GB).\n"
    "                # alternatives: '8bit' (~70 GB, fits 1x 80GB or 3x 40GB), '4bit' (~35 GB).\n"
    "\n"
    "# Reserve ~4 GB per GPU for activations + KV cache; the rest is for weights.\n"
    "n_gpus = torch.cuda.device_count()\n"
    "if n_gpus == 0:\n"
    "    raise RuntimeError('no CUDA GPUs visible')\n"
    "per_gpu_gb = [int(torch.cuda.get_device_properties(i).total_memory / 1e9) for i in range(n_gpus)]\n"
    "HEADROOM_GB = 4\n"
    "max_memory = {i: f'{max(per_gpu_gb[i] - HEADROOM_GB, 4)}GiB' for i in range(n_gpus)}\n"
    "max_memory['cpu'] = '32GiB'   # tiny CPU offload safety valve\n"
    "print(f'visible GPUs: {n_gpus}, per-GPU memory budget: {max_memory}')\n"
    "\n"
    "# Translate QUANT for load_model: 'bf16' means no quantization (full precision).\n"
    "_quant_arg = 'none' if QUANT == 'bf16' else QUANT\n"
    "loaded = load_model(\n"
    "    'llama33_70b',\n"
    "    quantization=_quant_arg,\n"
    "    device_map='auto',\n"
    "    max_memory=max_memory,\n"
    ")\n"
)


# Tiny diagnostic block — print layer→device sharding right after load.
SHARDING_DIAG = """# Show how the 70B got sharded across GPUs (sanity check for multi-GPU loads).
from collections import Counter
_dev_counts = Counter()
for name, p in loaded.hf_model.named_parameters():
    _dev_counts[str(p.device)] += 1
print('parameter-tensor counts per device:', dict(_dev_counts))
# First param's device is what `model.generate` / `model(...)` expect inputs on.
print('first param device (input target):', next(loaded.hf_model.parameters()).device)
"""


def already_injected(nb: dict) -> bool:
    for c in nb['cells']:
        s = ''.join(c['source']) if isinstance(c.get('source'), list) else c.get('source', '')
        if MARKER in s:
            return True
    return False


def patch_load_cell(nb: dict, name: str) -> bool:
    """Find the cell that does `loaded = load_model('llama33_70b')` (with no
    extra kwargs) and rewrite it to use the multi-GPU loader. Append a
    sharding-diagnostic block if one isn't there already."""
    for c in nb['cells']:
        if c.get('cell_type') != 'code':
            continue
        src = c['source']
        if isinstance(src, list):
            joined = ''.join(src)
        else:
            joined = src
        load_pat = re.compile(
            r"^([ \t]*)loaded\s*=\s*load_model\(\s*['\"]llama33_70b['\"]\s*\)\s*$",
            re.MULTILINE,
        )
        m = load_pat.search(joined)
        if not m:
            continue
        if MARKER in joined:
            return False  # already done
        indent = m.group(1)
        block = '\n'.join(indent + ln if ln else ln for ln in NEW_LOAD_BLOCK.rstrip().splitlines())
        new = joined[:m.start()] + block + joined[m.end():]
        if SHARDING_DIAG.strip() not in new:
            new = new.rstrip() + '\n\n' + SHARDING_DIAG
        # Re-split into list-of-lines for ipynb convention
        split = new.split('\n')
        new_lines = [s + '\n' for s in split[:-1]] + ([split[-1]] if split[-1] else [])
        c['source'] = new_lines
        return True
    print(f"  WARN: no llama33_70b load_model cell in {name}")
    return False


for name in TARGETS:
    path = NB_DIR / name
    nb = json.loads(path.read_text())
    if already_injected(nb):
        print(f'  {name}: multi-GPU loader already present; skipping')
        continue
    changed = patch_load_cell(nb, name)
    if changed:
        path.write_text(json.dumps(nb, indent=1) + '\n')
        print(f'  {name}: patched load cell + added sharding diag')
