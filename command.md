# Run the live red-team dashboard

```
.venv/bin/streamlit run streamlit_apps/live_redteam_dashboard.py --server.fileWatcherType=none -- --exp1b-dir /Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/exp1b_authority_conflict --exp6-dir /Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/exp06_results --model-key qwen
```

Notes:
- `--server.fileWatcherType=none` silences the harmless `torchvision` import warnings emitted by Streamlit's file watcher when it walks `transformers.*` submodules. Side effect: no hot-reload on save (which you don't want with a resident model anyway).
- Args after the `--` are passed to the dashboard itself.
- `--exp6-dir` is optional; pass it to enable the **exp6 probe (structural-authority)** tab alongside the legacy exp1b view. Drop the flag (or leave the sidebar field empty) to hide the tab.
- Device is auto-picked: MPS on Mac, CUDA where available, CPU otherwise. Sidebar shows the actual device.
- Thinking is forced off via `enable_thinking=False` for Qwen3.5 (chatml template).
