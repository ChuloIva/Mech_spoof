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
!pip install -q git+https://github.com/ChuloIva/Mech_spoof.git
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

1. Build datasets once. Two options:
   - **vLLM (default, no API cost)** — open `notebooks/build_datasets.ipynb` on Colab (A100 or T4) and Run All. Generated JSONs land in `data/`.
   - **Claude API** — `ms-build-datasets --backend claude` (needs `ANTHROPIC_API_KEY`).
   - CLI: `ms-build-datasets --backend vllm --model Qwen/Qwen2.5-7B-Instruct`
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

**Refusal-direction method**: Exp 3 uses the OBLITERATUS prompt corpus (`source="builtin"` → 512 curated harmful/harmless pairs) and the Arditi-style diff-of-means at the last raw token by default (`wrap_mode="raw"`). Results are directly comparable with OBLITERATUS's abliteration pipeline. Other sources (`advbench`, `harmbench`, `anthropic_redteam`, `wildjailbreak`) and chat-template wrapping (`wrap_mode="chat"`) are available via `run_experiment_3` keyword args. The third-party OBLITERATUS checkout lives at `third_party/OBLITERATUS/` (AGPL-3.0).

**GPU batching**: Both Exp 1 (authority) and Exp 3 (refusal) extract activations via a batched GPU path with left-padding. Forward passes are batched (`batch_size=8` default), per-layer means accumulate on-device, and only the final `(n_layers, d_model)` direction tensor is copied to CPU. Tune `batch_size` per VRAM:
- A100 40 GB on a 7-9 B bf16 model → `batch_size=16–32`
- T4 16 GB → `batch_size=4–8`
- Local 12 GB (gemma 4 E4B 4-bit or small models) → `batch_size=2–4`

Pass via: `run_experiment_1(MODEL_KEY, OUT_DIR, batch_size=16)` / `run_experiment_3(..., batch_size=16)`.

## Models

Five instruct models covering four delimiter philosophies. Local box only runs `gemma_small` (gemma-2-2b-it); everything else is Colab-only.

| Key | HF ID | Template |
|---|---|---|
| `qwen` | `Qwen/Qwen2.5-7B-Instruct` | ChatML |
| `llama3` | `meta-llama/Llama-3.1-8B-Instruct` | Llama-3 headers |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` | `[INST]` |
| `gemma` | `google/gemma-2-9b-it` | `<start_of_turn>` |
| `phi3` | `microsoft/Phi-3.5-mini-instruct` | `<|system|>` |
| `gemma_small` | `google/gemma-4-E4B-it` | `<\|turn>` (smoke-test; ~8B bf16 → needs Colab T4+ or 4-bit local) |
