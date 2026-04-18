# mech_spoof

Mechanistic interpretability of **instruction privilege** in LLMs: finding the "authority direction" that distinguishes system- from user-tagged content, and probing whether delimiter-injection attacks exploit it.

See `build_plan.md` for the full research spec.

## Install (local, ROCm)

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch
pip install -e ".[dev]"
# gfx1100 may need:  export HSA_OVERRIDE_GFX_VERSION=11.0.0
cp .env.example .env  # fill in ANTHROPIC_API_KEY, HF_TOKEN
```

## Install (Google Colab)

```python
!pip install -q git+https://github.com/<your-user>/mech_spoof.git
from google.colab import userdata, drive
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
drive.mount("/content/drive")
```

## Layout

- `src/mech_spoof/` — the library. Public API in `__init__.py`.
- `data/` — committed small JSONs (datasets).
- `notebooks/` — thin Colab notebooks, one per (model, experiment).
- `scripts/` — CLIs (dataset builders, experiment runners, aggregator).
- `tests/` — pytest.
- `results/` — gitignored; mirrored to Google Drive in Colab.

## Quick start

1. Build datasets once (needs `ANTHROPIC_API_KEY`):
   ```bash
   ms-build-datasets
   ```
2. Local smoke test (gemma-2-2b, fits on 12 GB):
   ```bash
   ms-smoke --experiment 1
   ```
3. Run an experiment on Colab: open `notebooks/01_authority_qwen.ipynb`, select A100 runtime, run all cells.
4. After per-model runs complete, aggregate:
   ```bash
   ms-aggregate
   ```

## Experiments

| # | Name | Module | Deps |
|---|---|---|---|
| 1 | Authority direction | `experiments.exp1_authority` | — |
| 2 | Conflict behavioral test | `experiments.exp2_conflict` | — |
| 3 | Refusal direction + geometry | `experiments.exp3_refusal` | exp1 |
| 4 | Attack evaluation + token trace | `experiments.exp4_attacks` | exp1, exp3 |
| 5 | Cross-model comparative | `experiments.exp5_comparative` | exp1-4 × all models |

## Models

Five instruct models covering four delimiter philosophies. Local box only runs `gemma_small` (gemma-2-2b-it); everything else is Colab-only.

| Key | HF ID | Template |
|---|---|---|
| `qwen` | `Qwen/Qwen2.5-7B-Instruct` | ChatML |
| `llama3` | `meta-llama/Llama-3.1-8B-Instruct` | Llama-3 headers |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` | `[INST]` |
| `gemma` | `google/gemma-2-9b-it` | `<start_of_turn>` |
| `phi3` | `microsoft/Phi-3.5-mini-instruct` | `<|system|>` |
| `gemma_small` | `google/gemma-2-2b-it` | `<start_of_turn>` (local smoke only) |
