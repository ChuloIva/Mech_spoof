# Run the live red-team dashboard

```
.venv/bin/streamlit run streamlit_apps/live_redteam_dashboard.py --server.fileWatcherType=none -- --exp1b-dir /Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/exp1b_authority_conflict --exp6-dir /Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/exp06_results --exp8-dir /Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/exp08_directions --model-key qwen
```

Multi-line version (same command, easier to read / edit):

```
.venv/bin/streamlit run streamlit_apps/live_redteam_dashboard.py \
    --server.fileWatcherType=none -- \
    --exp1b-dir     /Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/exp1b_authority_conflict \
    --exp6-dir      /Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/exp06_results \
    --exp6-pca-path /Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/exp06_pca_directions.npz \
    --exp8-dir      /Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/exp08_directions \
    --model-key qwen
```

Notes:
- `--server.fileWatcherType=none` silences the harmless `torchvision` import warnings emitted by Streamlit's file watcher when it walks `transformers.*` submodules. Side effect: no hot-reload on save (which you don't want with a resident model anyway).
- Args after the `--` are passed to the dashboard itself.
- `--exp6-dir` is optional; pass it to enable the **exp6 probe (structural-authority)** tab alongside the legacy exp1b view. Drop the flag (or leave the sidebar field empty) to hide the tab.
- `--exp8-dir` is optional; pass it to enable the **exp08 directions (cosine)** tab. exp08 ships directions only (no midpoints, no val accuracies), so the tab uses cosine projection and a manual layer slider (defaults to ~75% depth).
- `--exp6-pca-path` is optional; used by the **Steering (causal)** tab to expose exp06's `pca_diff` / `pca_center` alongside MM. Auto-detected when `--exp6-dir` is set and the file is at the standard `<repo>/exp06_pca_directions.npz` location.
- The **Steering (causal)** tab appears whenever `--exp6-dir` and/or `--exp8-dir` is set. It mirrors `notebooks/07_steering_authority.ipynb`: pick a position + layer band + one or more direction sets (`exp06/MM`, `exp08/MM`, `exp06/pca_diff`, `exp08/pca_diff`, `exp06/pca_center`, `exp08/pca_center`), set `k` in σ-units, and run baseline / +k·σ / −k·σ on the current prompt. There's also a "Conflict-battery sweep" expander that runs the same 11 contradictory pairs from the notebook against one method × one sign.
- Device is auto-picked: MPS on Mac, CUDA where available, CPU otherwise. Sidebar shows the actual device.
- Thinking is forced off via `enable_thinking=False` for Qwen3.5 (chatml template).
