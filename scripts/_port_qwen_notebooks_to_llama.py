"""One-shot porter: Qwen 3.5 4B notebooks (14/15/16) → Llama 3.3 70B versions.

Reads the existing Qwen notebooks and applies a series of in-place
substitutions appropriate to each notebook. Output goes to:
  notebooks/14_refusal_and_eval_awareness_directions_llama33_70b.ipynb
  notebooks/15_jailbreak_steering_llama33_70b.ipynb
  notebooks/16_su_eval_axis_readout_llama33_70b.ipynb

Run from repo root:
  python scripts/_port_qwen_notebooks_to_llama.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks'

# ---------- Common substitutions ----------
COMMON = [
    # Model load + key
    (r"load_model\('qwen'\)", "load_model('llama33_70b')"),
    (r"model_key':\s*'qwen'", "model_key': 'llama33_70b'"),
    (r"'model_key':\s*'qwen'", "'model_key': 'llama33_70b'"),
    # HF id (do this BEFORE the bare-Qwen subs so we don't end up with `Qwen/Llama...`)
    (r"`?Qwen/Qwen3\.5-4B`?", "`meta-llama/Llama-3.3-70B-Instruct`"),
    (r"Qwen/Qwen3\.5-4B", "meta-llama/Llama-3.3-70B-Instruct"),
    # Header titles
    (r"Qwen3\.5-4B", "Llama 3.3 70B"),
    (r"Qwen 3\.5 4B", "Llama 3.3 70B"),
    (r"Qwen3\.5 4B", "Llama 3.3 70B"),
    (r"Qwen3\.5-?4B", "Llama 3.3 70B"),
    # Chat template references
    (r"Qwen3 chat template", "Llama 3.3 chat template"),
    (r"Qwen chat template", "Llama 3.3 chat template"),
    # Function name leftover
    (r"render_qwen_chat", "render_llama_chat"),
    # Filenames in code
    (r"15b_visualize_jailbreak", "15b_visualize_jailbreak_llama"),
    # Stale path references in markdown intro lists
    (r"`exp06_pca_directions\.npz`",          "`exp06_lamma/directions.npz`"),
    (r"`exp06_results/arrays\.npz`[^\n]*\n",  ""),  # drop the line entirely
    (r"`exp_directions_qwen35_4b/directions\.npz`",
     "`exp_directions_llama33_70b/directions.npz`"),
]


def write_nb(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}  ({len(nb['cells'])} cells)")


def transform_nb(src_path: Path, subs: list[tuple[str, str]],
                 cell_filter=None) -> dict:
    """Substitute on the *joined* cell source (so multi-line patterns work),
    then split back into a list of lines preserving trailing newlines."""
    nb = json.loads(src_path.read_text())
    new_cells = []
    for cell in nb['cells']:
        if cell.get('source') is None:
            new_cells.append(cell); continue
        src = cell['source']
        joined = ''.join(src) if isinstance(src, list) else src
        for pat, repl in subs:
            joined = re.sub(pat, repl, joined, flags=re.MULTILINE)
        # Re-split into lines with trailing newlines (ipynb convention).
        if isinstance(src, list):
            split = joined.split('\n')
            new_lines = [s + '\n' for s in split[:-1]] + ([split[-1]] if split[-1] else [])
            cell['source'] = new_lines
        else:
            cell['source'] = joined
        # Reset outputs (Qwen cells include cached outputs we don't want to ship)
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        if cell_filter is None or cell_filter(cell):
            new_cells.append(cell)
    nb['cells'] = new_cells
    return nb


# ============================================================
# Notebook 14 — refusal + eval-aw directions
# ============================================================
def port_14() -> None:
    src = NB / '14_refusal_and_eval_awareness_directions_qwen35_4b.ipynb'
    subs = COMMON + [
        # Output dir
        (r"exp_directions_qwen35_4b", "exp_directions_llama33_70b"),
        # VRAM check: 70B in 8-bit needs ~70 GB
        (r"Qwen 3\.5 4B in bf16 fits on a single 16 GB GPU\.",
         "Llama 3.3 70B in 8-bit (bnb) fits on a single 80 GB GPU."),
        (r"Qwen/Qwen3\.5-4B`?", "meta-llama/Llama-3.3-70B-Instruct`"),
        (r"~8 GB", "~70 GB (8-bit) / ~140 GB (bf16)"),
        # Compute estimate
        (r"\*\*5–10 min\*\* wall-clock on an A100 / H100",
         "**40–90 min** wall-clock on an H100 / A100 80GB"),
        # Token guidance
        (r"is not gated\. \*\*Tokens needed\.\*\* None",
         "is gated. **Tokens needed.** `HF_TOKEN` (Llama 3.3 is gated)"),
        (r"`Qwen/Qwen3\.5-4B` is not gated\.",
         "`meta-llama/Llama-3.3-70B-Instruct` is gated; set `HF_TOKEN`."),
        # Stale "is not gated" hedges in inline comments / docstrings
        (r"Llama 3\.3 70B is not gated, but if you swap models pick this up automatically\.",
         "Llama 3.3 70B is gated — paste your `HF_TOKEN` in §1c above."),
        (r"is not gated, but if you swap models pick this up automatically\.",
         "is gated — paste your `HF_TOKEN` in §1c above."),
        # Subtle: notebook references "exp06_results/" comparison
        (r"`exp06_results/`", "`exp06_lamma/`"),
        # Misc references to "qwen" in docs
        (r"the same model used for `exp06_results/`",
         "the same model used for `exp06_lamma/`"),
        # Title comment about chaining
        (r"`12_direction_comparison\.ipynb`",
         "`12_refusal_direction_llama33_70b.ipynb` / `13_eval_awareness_directions_llama33_70b.ipynb`"),
    ]
    nb = transform_nb(src, subs)
    # Add an HF_TOKEN assertion right after the auth block. Find the cell that
    # mentions HF_TOKEN and insert a guard line if missing.
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code': continue
        src_lines = cell['source']
        joined = ''.join(src_lines)
        if "if os.environ.get('HF_TOKEN'):" in joined and 'login(token=' in joined:
            # Replace the soft-optional auth with a required one for Llama.
            new = []
            for line in src_lines:
                if "if os.environ.get('HF_TOKEN'):" in line:
                    new.append("assert os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'), \\\n")
                    new.append("    'HF_TOKEN required (Llama 3.3 is gated). Set it before running this cell.'\n")
                    new.append("from huggingface_hub import login\n")
                    new.append("login(token=os.environ.get('HF_TOKEN') or os.environ['HUGGING_FACE_HUB_TOKEN'], add_to_git_credential=False)\n")
                elif 'from huggingface_hub import login' in line or 'login(token=' in line:
                    continue  # absorbed above
                else:
                    new.append(line)
            cell['source'] = new
            break
    write_nb(NB / '14_refusal_and_eval_awareness_directions_llama33_70b.ipynb', nb)


# ============================================================
# Notebook 15 — jailbreak steering
# ============================================================
def port_15() -> None:
    src = NB / '15_jailbreak_steering_qwen35_4b.ipynb'
    subs = COMMON + [
        # Output dir
        (r"exp15_jailbreak_steering(?!_)", "exp15_jailbreak_steering_llama33_70b"),
        # Layer-range descriptions in markdown
        (r"layers 16\.\.31",  "layers 40..79"),
        (r"layers `16\.\.31`","layers `40..79`"),
        # Path-existence check list (do these BEFORE we rename EXP6_PCA → EXP6_NPZ
        # so the line-strings still match cleanly).
        (r"'exp06_pca_directions\.npz',", "'exp06_lamma/directions.npz',"),
        (r"^[ \t]*'exp06_results/arrays\.npz',[ \t]*\n", ""),
        (r"'exp_directions_qwen35_4b/directions\.npz',",
         "'exp_directions_llama33_70b/directions.npz',"),
        # Path-print pairs — fold both old vars (EXP6_PCA, EXP6_ARRAYS) into a single EXP6_NPZ entry
        (r"\(\s*'exp06_pca',\s*EXP6_PCA\),?\s*\(?\s*'exp06_arrays',\s*EXP6_ARRAYS\),?",
         "('exp06_npz', EXP6_NPZ),"),
        # Variable rename: EXP6_PCA / EXP6_ARRAYS → EXP6_NPZ (for any straggler refs)
        (r"\bEXP6_PCA\b",    "EXP6_NPZ"),
        (r"\bEXP6_ARRAYS\b", "EXP6_NPZ"),
        # Direction-NPZ paths: Qwen used two npz files (exp06_pca_* and exp06_results/arrays.npz);
        # llama bundles everything in exp06_lamma/directions.npz.
        (r"EXP6_NPZ\s*=\s*PROJECT_ROOT\s*/\s*'exp06_pca_directions\.npz'",
         "EXP6_NPZ    = PROJECT_ROOT / 'exp06_lamma' / 'directions.npz'"),
        (r"EXP6_NPZ\s*=\s*PROJECT_ROOT\s*/\s*'exp06_results'\s*/\s*'arrays\.npz'\s*\n",
         ""),  # drop the now-redundant line
        (r"REFUSAL_NPZ\s*=\s*PROJECT_ROOT\s*/\s*'exp_directions_qwen35_4b'\s*/\s*'directions\.npz'",
         "REFUSAL_NPZ = PROJECT_ROOT / 'exp_directions_llama33_70b' / 'directions.npz'"),
        # Loader: collapse the two-file load into one and reuse keys.
        (r"exp6_pca = load_npz\(EXP6_NPZ\)\s*\n\s*arrs6\s*=\s*load_npz\(EXP6_NPZ\)",
         "exp6 = load_npz(EXP6_NPZ)\narrs6 = exp6  # llama bundles pca_center_dir + mm_raw in one npz"),
        (r"exp6_pca\[", "exp6["),
        # Steering layer range: Qwen uses 16..31 (last half of 32 layers).
        # Llama 80 layers → use 40..79 (last half) by default.
        (r"STEER_LAYERS\s*=\s*list\(range\(16,\s*32\)\)",
         "STEER_LAYERS = list(range(40, 80))  # last half of llama 70B's 80 layers"),
        # Mid-stack reference layer for diagnostics (used in expA tables/etc.; harmless if not in this nb)
        # SU steering window densification — already done in nb15; keep
        # Refusal position: nb15 currently uses pos -1; keep the same default
        # but bump the comment to make the next-cell clear.
        # VRAM / batch size: 70B is heavier; reduce BATCH_SIZE
        (r"BATCH_SIZE\s*=\s*50\b", "BATCH_SIZE = 8"),
        (r"BATCH_SIZE\s*=\s*16\b", "BATCH_SIZE = 8"),
        # Compute estimate text
        (r"On A100-80GB the full notebook runs in a few minutes\.",
         "On a single H100 the full sweep takes ~3-6 hours (70B greedy generation + MC readout)."),
        # Drop "is not gated" line from intro
        (r"`Qwen 3\.5 4B` ?(?:isn't|is not) gated[^\n]*",
         "`meta-llama/Llama-3.3-70B-Instruct` is gated; set `HF_TOKEN`."),
        # Hugging Face auth: enforce token
        # (handled per-cell after transform below)
        # Comment from prior diary about exp09 — replace with llama analogue
        (r"\(see exp09 diary\)", "(see Qwen exp09 diary; same caveat applies on llama)"),
    ]
    nb = transform_nb(src, subs)
    # Enforce HF_TOKEN like in nb14
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code': continue
        joined = ''.join(cell['source'])
        if "if os.environ.get('HF_TOKEN'):" in joined and 'login(token=' in joined:
            new = []
            seen_login = False
            for line in cell['source']:
                if "if os.environ.get('HF_TOKEN'):" in line and not seen_login:
                    new.append("assert os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'), \\\n")
                    new.append("    'HF_TOKEN required (Llama 3.3 is gated).'\n")
                    new.append("from huggingface_hub import login\n")
                    new.append("login(token=os.environ.get('HF_TOKEN') or os.environ['HUGGING_FACE_HUB_TOKEN'], add_to_git_credential=False)\n")
                    seen_login = True
                elif 'from huggingface_hub import login' in line or 'login(token=' in line:
                    continue
                else:
                    new.append(line)
            cell['source'] = new
            break
    write_nb(NB / '15_jailbreak_steering_llama33_70b.ipynb', nb)


# ============================================================
# Notebook 16 — SU/refusal/eval-aw axis readout
# ============================================================
def port_16() -> None:
    src = NB / '16_su_eval_axis_readout_qwen35_4b.ipynb'
    subs = COMMON + [
        # Output dir
        (r"exp16_su_eval_readout(?!_)", "exp16_su_eval_readout_llama33_70b"),
        # Layer-range descriptions in markdown
        (r"layers 16\.\.31",   "layers 40..79"),
        (r"layers `\[16\.\.31\]`", "layers `[40..79]`"),
        (r"layers 16\.\.31\]", "layers 40..79]"),
        (r"\[16\.\.31\]",      "[40..79]"),
        (r"L≥16",              "L≥40"),
        (r"L<16",              "L<40"),
        (r"layers 32",         "layers 80"),
        (r"32 layers",         "80 layers"),
        (r"all 32 layers",     "all 80 layers"),
        # Path-existence list (BEFORE EXP6_PCA→EXP6_NPZ rename)
        (r"'exp06_pca_directions\.npz',", "'exp06_lamma/directions.npz',"),
        (r"^[ \t]*'exp06_results/arrays\.npz',[ \t]*\n", ""),
        (r"'exp_directions_qwen35_4b/directions\.npz',",
         "'exp_directions_llama33_70b/directions.npz',"),
        # Path-print pairs — fold both old vars into one (do BEFORE the bare-var rename)
        (r"\(\s*'exp06_pca',\s*EXP6_PCA\),?\s*\(?\s*'exp06_arrays',\s*EXP6_ARRAYS\),?",
         "('exp06_npz', EXP6_NPZ),"),
        # Variable rename
        (r"\bEXP6_PCA\b",    "EXP6_NPZ"),
        (r"\bEXP6_ARRAYS\b", "EXP6_NPZ"),
        # Direction-NPZ paths
        (r"EXP6_NPZ\s*=\s*PROJECT_ROOT\s*/\s*'exp06_pca_directions\.npz'",
         "EXP6_NPZ    = PROJECT_ROOT / 'exp06_lamma' / 'directions.npz'"),
        (r"EXP6_NPZ\s*=\s*PROJECT_ROOT\s*/\s*'exp06_results'\s*/\s*'arrays\.npz'\s*\n",
         ""),
        (r"DIRS_NPZ\s*=\s*PROJECT_ROOT\s*/\s*'exp_directions_qwen35_4b'\s*/\s*'directions\.npz'",
         "DIRS_NPZ    = PROJECT_ROOT / 'exp_directions_llama33_70b' / 'directions.npz'"),
        (r"exp6_pca = load_npz\(EXP6_NPZ\)\s*\n\s*arrs6\s*=\s*load_npz\(EXP6_NPZ\)",
         "exp6 = load_npz(EXP6_NPZ)\narrs6 = exp6  # llama bundles pca_center_dir + mm_raw in one npz"),
        (r"exp6_pca\[", "exp6["),
        # Steering layer range — last half of 80
        (r"STEER_LAYERS\s*=\s*list\(range\(16,\s*32\)\)",
         "STEER_LAYERS = list(range(40, 80))  # last half of llama 70B's 80 layers"),
        # Generic L-range sub in markdown / printouts FIRST so my new MID/LEVERAGE comments aren't mangled.
        (r"L22\.\.28", "L60..72"),
        (r"L22-28",    "L60-72"),
        (r"L22–28",    "L60–72"),
        (r"L29-31",    "L73-79"),
        (r"L29–31",    "L73–79"),
        (r"L30-31",    "L77-79"),
        (r"L30–31",    "L77–79"),
        # The sample-layer index in summary printouts
        (r"\bL24\b", "L66"),
        (r"\(layer 24\)", "(layer 66)"),
        (r"L0,?\s*8,?\s*16,?\s*20,?\s*24,?\s*28,?\s*31",
         "L0, 20, 40, 50, 60, 70, 79"),
        (r"\[0, 8, 16, 20, 24, 28, 31\]",
         "[0, 20, 40, 50, 60, 66, 72, 79]"),
        # MID/LEVERAGE constants — set LAST so the L22-28 generic sub doesn't mangle the comment
        (r"MID_LAYERS\s*=\s*list\(range\(16,\s*29\)\)(?:\s*#[^\n]*)?",
         "MID_LAYERS = list(range(40, 73))  # mid + late stack on llama 70B; drops L73-79 amplification tail"),
        (r"LEVERAGE_LAYERS\s*=\s*list\(range\(22,\s*29\)\)(?:\s*#[^\n]*)?",
         "LEVERAGE_LAYERS = list(range(60, 73))  # mid-stack on llama 70B (analogue of Qwen 4B L22-28)"),
        # Batch size — 70B is heavier
        (r"BATCH_SIZE\s*=\s*16\b", "BATCH_SIZE = 8"),
        # exp15 compliance numbers — these are the Qwen numbers, leave them in
        # but flag clearly that they're placeholders to be replaced after running
        # the llama exp15.
        # Compute estimate
        (r"Few minutes on a single A100\.",
         "~30-60 min on a single H100 (70B forward passes only, no generation)."),
        # Comment header about the L31 spike
        (r"\bL31\b", "L79"),
        (r"\bL30\b", "L78"),
    ]
    nb = transform_nb(src, subs)
    # Enforce HF_TOKEN
    for cell in nb['cells']:
        if cell.get('cell_type') != 'code': continue
        joined = ''.join(cell['source'])
        if "if os.environ.get('HF_TOKEN'):" in joined and 'login(token=' in joined:
            new = []
            seen_login = False
            for line in cell['source']:
                if "if os.environ.get('HF_TOKEN'):" in line and not seen_login:
                    new.append("assert os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'), \\\n")
                    new.append("    'HF_TOKEN required (Llama 3.3 is gated).'\n")
                    new.append("from huggingface_hub import login\n")
                    new.append("login(token=os.environ.get('HF_TOKEN') or os.environ['HUGGING_FACE_HUB_TOKEN'], add_to_git_credential=False)\n")
                    seen_login = True
                elif 'from huggingface_hub import login' in line or 'login(token=' in line:
                    continue
                else:
                    new.append(line)
            cell['source'] = new
            break
    # Insert a banner cell at the top reminding the user that EXP15_COMPLIANCE
    # numbers are Qwen placeholders.
    banner_md = {
        'cell_type': 'markdown',
        'id': 'llama-port-banner',
        'metadata': {},
        'source': [
            "> **Llama 3.3 70B port note.** This notebook is the direct port of the Qwen 3.5 4B exp16 to Llama 3.3 70B.\n",
            ">\n",
            "> - Steering window: `L40..79` (last half of 80 layers; Qwen used `L16..31` of 32).\n",
            "> - Mid-stack diagnostic window: `L60..72` (analogue of Qwen's `L22..28`).\n",
            "> - The `EXP15_COMPLIANCE` dict in §13 still contains the **Qwen** compliance numbers as placeholders — replace them with the llama numbers from `exp15_jailbreak_steering_llama33_70b/expB_compliance_summary.csv` after running notebook 15 on llama.\n",
            "> - Inputs needed: `exp06_lamma/directions.npz` (already on disk) + `exp_directions_llama33_70b/directions.npz` (run notebook 14 on llama first).\n",
        ],
    }
    nb['cells'].insert(1, banner_md)
    write_nb(NB / '16_su_eval_axis_readout_llama33_70b.ipynb', nb)


# ============================================================
# Drive
# ============================================================
if __name__ == '__main__':
    port_14()
    port_15()
    port_16()
