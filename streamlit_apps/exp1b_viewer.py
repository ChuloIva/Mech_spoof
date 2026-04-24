"""Streamlit viewer for Experiment 1b results.

Point it at an exp1b output directory (containing `result.json`, `arrays.npz`,
`prompts.csv`) and explore per-layer / per-position probe metrics, the
perplexity distribution, and a per-pair drilldown.

Run:
    streamlit run streamlit_apps/exp1b_viewer.py -- --results /path/to/exp1b_dir

Or just launch without `--results` and paste the path into the sidebar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------------ IO

@st.cache_data(show_spinner=False)
def load_result(results_dir: str) -> dict:
    p = Path(results_dir) / "result.json"
    if not p.exists():
        raise FileNotFoundError(f"result.json not found in {results_dir}")
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_arrays(results_dir: str) -> dict[str, np.ndarray]:
    p = Path(results_dir) / "arrays.npz"
    if not p.exists():
        return {}
    with np.load(p) as data:
        return {k: data[k] for k in data.files}


@st.cache_data(show_spinner=False)
def load_prompts(results_dir: str) -> pd.DataFrame:
    p = Path(results_dir) / "prompts.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # Normalize types
    for col in ("ppl", "probe_score_best"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_manifest(results_dir: str) -> dict:
    p = Path(results_dir) / "manifest.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------------ UI

def _sidebar() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results", default="", help="Default results dir.")
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    known, _ = parser.parse_known_args(argv)

    st.sidebar.header("Experiment 1b viewer")
    default = known.results or st.session_state.get("results_dir", "")
    results_dir = st.sidebar.text_input("Results directory", value=default)
    st.session_state["results_dir"] = results_dir
    return results_dir


def _overview(result: dict, manifest: dict, prompts: pd.DataFrame) -> None:
    st.header("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", result.get("model_key", "?"))
    c2.metric("Best position", result.get("best_position", "?"))
    c3.metric("Best layer", result.get("best_layer", "?"))
    c4.metric("Best accuracy", f"{result.get('best_accuracy', float('nan')):.3f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pairs", result.get("n_pairs", "?"))
    c2.metric("Sys-following", result.get("n_system_following", "?"))
    c3.metric("Usr-following", result.get("n_user_following", "?"))
    c4.metric("Layers", result.get("n_layers", "?"))

    ppl = result.get("ppl_stats", {})
    if ppl:
        st.subheader("Perplexity summary")
        ppl_df = pd.DataFrame({
            "split": ["sys-following", "usr-following"],
            "mean": [ppl.get("sys_following_mean"), ppl.get("usr_following_mean")],
            "median": [ppl.get("sys_following_median"), ppl.get("usr_following_median")],
        })
        st.dataframe(ppl_df, use_container_width=True, hide_index=True)

    if manifest:
        with st.expander("Manifest"):
            st.json(manifest)


def _layer_curves(result: dict) -> None:
    st.header("Per-layer metrics by position")
    positions = result.get("positions") or list(result.get("accuracies_by_position", {}).keys())
    accs = result.get("accuracies_by_position", {})
    aucs = result.get("aurocs_by_position", {})
    if not positions or not accs:
        st.info("No per-position metrics found.")
        return
    rows = []
    for pos in positions:
        for layer_str, acc in accs.get(pos, {}).items():
            layer = int(layer_str)
            auc = aucs.get(pos, {}).get(str(layer), aucs.get(pos, {}).get(layer, float("nan")))
            rows.append({"position": pos, "layer": layer, "accuracy": acc, "auroc": float(auc)})
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No per-layer values recorded.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accuracy")
        chart = (
            alt.Chart(df).mark_line(point=True).encode(
                x=alt.X("layer:Q", title="Layer"),
                y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0.4, 1.05]), title="Probe accuracy"),
                color="position:N",
                tooltip=["position", "layer", alt.Tooltip("accuracy:Q", format=".3f")],
            ).properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)
    with c2:
        st.subheader("AUROC")
        chart = (
            alt.Chart(df).mark_line(point=True).encode(
                x=alt.X("layer:Q", title="Layer"),
                y=alt.Y("auroc:Q", scale=alt.Scale(domain=[0.4, 1.05]), title="AUROC"),
                color="position:N",
                tooltip=["position", "layer", alt.Tooltip("auroc:Q", format=".3f")],
            ).properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

    cos = result.get("probe_vs_dim_cosine_by_position", {})
    if cos:
        st.subheader("Probe vs diff-in-means cosine")
        rows = []
        for pos in positions:
            for layer_str, v in cos.get(pos, {}).items():
                rows.append({"position": pos, "layer": int(layer_str), "cosine": float(v)})
        df_cos = pd.DataFrame(rows)
        chart = (
            alt.Chart(df_cos).mark_line(point=True).encode(
                x="layer:Q",
                y=alt.Y("cosine:Q", scale=alt.Scale(domain=[-0.1, 1.1])),
                color="position:N",
                tooltip=["position", "layer", alt.Tooltip("cosine:Q", format=".3f")],
            ).properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)


def _axis_table(result: dict) -> None:
    st.header("Per-macro-axis breakdown (at global best pos+layer)")
    breakdown = result.get("macro_axis_breakdown", {})
    if not breakdown:
        st.info("No axis breakdown.")
        return
    rows = [
        {"axis": axis, "n": d["n"], "accuracy": d["accuracy"], "auroc": d["auroc"]}
        for axis, d in breakdown.items()
    ]
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    st.dataframe(df.style.format({"accuracy": "{:.3f}", "auroc": "{:.3f}"}),
                 use_container_width=True, hide_index=True)


def _ppl_section(prompts: pd.DataFrame) -> None:
    if prompts.empty or "ppl" not in prompts.columns:
        return
    st.header("Perplexity")

    base = prompts.dropna(subset=["ppl"]).copy()
    if base.empty:
        st.info("No PPL values recorded.")
        return

    max_ppl = float(base["ppl"].quantile(0.99))
    clip_max = st.slider("Clip PPL at (visualization only)", 1.0, max(50.0, max_ppl), min(50.0, max_ppl))
    base["ppl_clipped"] = base["ppl"].clip(upper=clip_max)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Histogram by side")
        hist = (
            alt.Chart(base).mark_bar(opacity=0.55).encode(
                x=alt.X("ppl_clipped:Q", bin=alt.Bin(maxbins=60), title="Perplexity"),
                y=alt.Y("count():Q", stack=None),
                color="side:N",
            ).properties(height=340)
        )
        st.altair_chart(hist, use_container_width=True)
    with c2:
        st.subheader("PPL vs probe score (best pos+layer)")
        scatter = (
            alt.Chart(base).mark_circle(opacity=0.45, size=30).encode(
                x=alt.X("ppl_clipped:Q", title="Perplexity (clipped)"),
                y=alt.Y("probe_score_best:Q", title="Probe score"),
                color="side:N",
                tooltip=["pair_idx", "trace", "side", "macro_axis",
                         alt.Tooltip("ppl:Q", format=".3f"),
                         alt.Tooltip("probe_score_best:Q", format=".3f")],
            ).properties(height=340)
        )
        st.altair_chart(scatter, use_container_width=True)

    st.subheader("PPL summary by macro axis")
    if "macro_axis" in base.columns:
        summary = (
            base.groupby(["macro_axis", "side"])["ppl"]
            .agg(["count", "mean", "median"])
            .reset_index()
            .rename(columns={"count": "n"})
        )
        st.dataframe(
            summary.style.format({"mean": "{:.3f}", "median": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )


def _pair_drilldown(prompts: pd.DataFrame) -> None:
    if prompts.empty or "pair_idx" not in prompts.columns:
        return
    st.header("Per-pair drilldown")
    pair_ids = sorted(prompts["pair_idx"].dropna().unique().tolist())
    if not pair_ids:
        return
    pid = st.selectbox("Pair idx", pair_ids)
    sub = prompts[prompts["pair_idx"] == pid].sort_values(["side", "trace"])
    display_cols = [c for c in
                    ["side", "trace", "label", "macro_axis", "conflict_axis",
                     "ppl", "probe_score_best", "response_preview"]
                    if c in sub.columns]
    st.dataframe(
        sub[display_cols].style.format({"ppl": "{:.3f}", "probe_score_best": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )


def _prompts_filter(prompts: pd.DataFrame) -> None:
    if prompts.empty:
        return
    st.header("Prompt table")
    with st.expander("Filter"):
        sides = sorted(prompts["side"].dropna().unique().tolist()) if "side" in prompts else []
        axes = sorted(prompts["macro_axis"].dropna().unique().tolist()) if "macro_axis" in prompts else []
        side_sel = st.multiselect("side", sides, default=sides)
        axis_sel = st.multiselect("macro_axis", axes, default=axes)
        ppl_max = float(prompts["ppl"].quantile(0.99)) if "ppl" in prompts else None
        ppl_cap = st.number_input("PPL ≤", value=ppl_max if ppl_max else 100.0)
    df = prompts.copy()
    if side_sel and "side" in df:
        df = df[df["side"].isin(side_sel)]
    if axis_sel and "macro_axis" in df:
        df = df[df["macro_axis"].isin(axis_sel)]
    if "ppl" in df:
        df = df[(df["ppl"].isna()) | (df["ppl"] <= ppl_cap)]
    st.write(f"{len(df):,} rows")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ main

def main() -> None:
    st.set_page_config(page_title="Exp1b viewer", layout="wide")
    results_dir = _sidebar()
    if not results_dir or not Path(results_dir).exists():
        st.info("Enter the path to an exp1b results directory in the sidebar.")
        return

    try:
        result = load_result(results_dir)
    except FileNotFoundError as e:
        st.error(str(e))
        return
    arrays = load_arrays(results_dir)
    prompts = load_prompts(results_dir)
    manifest = load_manifest(results_dir)

    _overview(result, manifest, prompts)
    st.divider()
    _layer_curves(result)
    st.divider()
    _axis_table(result)
    st.divider()
    _ppl_section(prompts)
    st.divider()
    _pair_drilldown(prompts)
    st.divider()
    _prompts_filter(prompts)

    if arrays:
        with st.expander("Available arrays (arrays.npz)"):
            st.write({k: list(v.shape) for k, v in arrays.items()})


if __name__ == "__main__":
    main()
