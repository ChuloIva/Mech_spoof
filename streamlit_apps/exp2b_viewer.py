"""Streamlit viewer for exp2b judged CSV files.

Run:
    streamlit run streamlit_apps/exp2b_viewer.py -- --csv /path/to/judged_nothink.csv
or paste a path into the sidebar.
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
from scipy.stats import pearsonr


# ------------------------------------------------------------------ IO

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Coerce types
    for col in ("probe_score", "n_layers_activated_above_mean", "n_layers_activated_strong"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "system_followed" in df.columns:
        s = df["system_followed"].astype(str).str.lower()
        df["system_followed_bool"] = s.where(s.isin(["true", "false"]))
        df["y"] = df["system_followed_bool"].map({"true": 1, "false": 0})
    return df


def parse_psbl(s: object) -> dict:
    if isinstance(s, str) and s.strip():
        try:
            d = json.loads(s)
            return {int(k): float(v) for k, v in d.items()}
        except Exception:
            return {}
    return {}


@st.cache_data(show_spinner=False)
def per_layer_table(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format per-layer probe scores: one row per (df_row, layer)."""
    if "probe_scores_by_layer_json" not in df.columns:
        return pd.DataFrame()
    rows = []
    for idx, row in df.iterrows():
        psbl = parse_psbl(row["probe_scores_by_layer_json"])
        for layer, score in psbl.items():
            rows.append({
                "row_idx": idx, "layer": layer, "score": score,
                "y": row.get("y"),
                "condition": row.get("condition"),
                "judge_verdict": row.get("judge_verdict"),
                "conflict_axis": row.get("conflict_axis"),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ helpers

def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int] | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
        return None
    r, p = pearsonr(x[mask], y[mask])
    return float(r), float(p), int(mask.sum())


# ------------------------------------------------------------------ sections

def _sidebar() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", default="", help="Default judged CSV path.")
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    known, _ = parser.parse_known_args(argv)

    st.sidebar.header("Exp2b judged viewer")
    default = known.csv or st.session_state.get("csv_path", "")
    csv_path = st.sidebar.text_input("Judged CSV path", value=default)
    st.session_state["csv_path"] = csv_path
    return csv_path


def _overview(df: pd.DataFrame) -> None:
    st.header("Overview")
    n = len(df)
    n_judged = int(df["y"].notna().sum()) if "y" in df else 0
    n_sys = int((df["judge_verdict"] == "system").sum())
    n_usr = int((df["judge_verdict"] == "user").sum())
    n_neither = int((df["judge_verdict"] == "neither").sum())
    n_both = int((df["judge_verdict"] == "both").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", n)
    c2.metric("Judged (S/U)", n_judged)
    c3.metric("System", n_sys)
    c4.metric("User", n_usr)
    c5.metric("Both/Neither", n_both + n_neither)

    if "y" in df and df["y"].notna().any():
        x = df["probe_score"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        rows = []
        for label, mask in [
            ("overall", np.ones(len(df), dtype=bool)),
            ("REAL", (df["condition"] == "REAL").to_numpy()),
            ("NONE", (df["condition"] == "NONE").to_numpy()),
            ("FAKE", (df["condition"] == "FAKE").to_numpy()),
        ]:
            corr = _safe_corr(x[mask], y[mask])
            if corr is None:
                rows.append({"slice": label, "n": int(mask.sum()), "r": np.nan, "p": np.nan})
            else:
                r, p, nv = corr
                rows.append({"slice": label, "n": nv, "r": r, "p": p})
        st.subheader("Best-layer correlation (probe_score vs system_followed)")
        st.dataframe(
            pd.DataFrame(rows).style.format({"r": "{:+.3f}", "p": "{:.2g}"}),
            use_container_width=True, hide_index=True,
        )


def _per_condition(df: pd.DataFrame) -> None:
    st.header("Per-condition compliance & probe stats")
    if "condition" not in df:
        return
    rows = []
    for cond, sub in df.groupby("condition"):
        n = len(sub)
        judged = sub.dropna(subset=["y"])
        rows.append({
            "condition": cond,
            "n": n,
            "n_judged": len(judged),
            "sys_followed_rate": float(judged["y"].mean()) if len(judged) else float("nan"),
            "mean_probe": float(sub["probe_score"].mean()),
            "std_probe": float(sub["probe_score"].std()),
        })
    st.dataframe(
        pd.DataFrame(rows).style.format({
            "sys_followed_rate": "{:.2%}",
            "mean_probe": "{:+.3f}",
            "std_probe": "{:.3f}",
        }),
        use_container_width=True, hide_index=True,
    )


def _per_layer_chart(df: pd.DataFrame) -> None:
    st.header("Per-layer correlation curve")
    long = per_layer_table(df)
    if long.empty:
        st.info("No probe_scores_by_layer_json column.")
        return
    rows = []
    for cond_label in ("overall", "REAL", "NONE", "FAKE"):
        if cond_label == "overall":
            sub = long
        else:
            sub = long[long["condition"] == cond_label]
        for layer, g in sub.groupby("layer"):
            corr = _safe_corr(g["score"].to_numpy(dtype=float), g["y"].to_numpy(dtype=float))
            if corr is None:
                continue
            r, p, nv = corr
            rows.append({"slice": cond_label, "layer": int(layer), "r": r, "n": nv})
    if not rows:
        st.info("Not enough variance to compute correlations.")
        return
    plot_df = pd.DataFrame(rows)
    chart = (
        alt.Chart(plot_df).mark_line(point=True).encode(
            x=alt.X("layer:Q", title="Layer"),
            y=alt.Y("r:Q", title="Pearson r"),
            color="slice:N",
            tooltip=["slice", "layer", alt.Tooltip("r:Q", format="+.3f"), "n"],
        ).properties(height=400)
    )
    rule = alt.Chart(pd.DataFrame({"r": [0]})).mark_rule(color="gray", strokeDash=[2, 2]).encode(y="r:Q")
    st.altair_chart(chart + rule, use_container_width=True)


def _scatter(df: pd.DataFrame) -> None:
    if "y" not in df or df["y"].isna().all():
        return
    st.header("Probe score vs verdict (jittered)")
    sub = df.dropna(subset=["y", "probe_score"]).copy()
    sub["y_jit"] = sub["y"] + np.random.RandomState(0).uniform(-0.08, 0.08, len(sub))
    chart = (
        alt.Chart(sub).mark_circle(size=30, opacity=0.5).encode(
            x=alt.X("probe_score:Q"),
            y=alt.Y("y_jit:Q", title="system_followed (jittered)",
                    scale=alt.Scale(domain=[-0.3, 1.3])),
            color="condition:N",
            tooltip=["pair_idx", "condition", "judge_verdict",
                     "conflict_axis",
                     alt.Tooltip("probe_score:Q", format="+.3f")],
        ).properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def _axis_table(df: pd.DataFrame) -> None:
    if "conflict_axis" not in df:
        return
    st.header("Per conflict_axis breakdown")
    rows = []
    for axis, sub in df.groupby("conflict_axis"):
        judged = sub.dropna(subset=["y"])
        corr = _safe_corr(
            sub["probe_score"].to_numpy(dtype=float),
            sub["y"].to_numpy(dtype=float),
        )
        if corr is None:
            r = np.nan; p = np.nan
        else:
            r, p, _ = corr
        rows.append({
            "axis": axis,
            "n": len(sub),
            "n_judged": len(judged),
            "sys_rate": float(judged["y"].mean()) if len(judged) else float("nan"),
            "r": r,
            "p": p,
        })
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    st.dataframe(
        out.style.format({"sys_rate": "{:.2%}", "r": "{:+.3f}", "p": "{:.2g}"}),
        use_container_width=True, hide_index=True,
    )


def _drilldown(df: pd.DataFrame) -> None:
    st.header("Per-pair drilldown")
    if "pair_idx" not in df:
        return
    pair_ids = sorted(df["pair_idx"].dropna().unique().tolist())
    pid = st.selectbox("Pair idx", pair_ids)
    sub = df[df["pair_idx"] == pid].sort_values("condition")
    if sub.empty:
        return
    st.markdown(f"**S:** {sub.iloc[0]['s_instruction']}")
    st.markdown(f"**U:** {sub.iloc[0]['u_instruction']}")
    st.caption(f"axis: {sub.iloc[0].get('conflict_axis', '?')}")
    for _, r in sub.iterrows():
        with st.expander(
            f"{r['condition']}  ·  verdict={r.get('judge_verdict')}"
            f"  ·  probe={r.get('probe_score'):+.3f}",
            expanded=False,
        ):
            st.markdown("**Response:**")
            st.text((r.get("response") or "")[:4000])
            st.markdown("**Judge reason:**")
            st.text(r.get("judge_reason", "") or "")


def _table(df: pd.DataFrame) -> None:
    st.header("Filterable table")
    cols = [c for c in df.columns if c not in ("probe_scores_by_layer_json", "judge_raw")]
    with st.expander("Filter"):
        conds = sorted(df["condition"].dropna().unique()) if "condition" in df else []
        verdicts = sorted(df["judge_verdict"].dropna().unique()) if "judge_verdict" in df else []
        cond_sel = st.multiselect("condition", conds, default=conds)
        verdict_sel = st.multiselect("judge_verdict", verdicts, default=verdicts)
    out = df.copy()
    if cond_sel and "condition" in out:
        out = out[out["condition"].isin(cond_sel)]
    if verdict_sel and "judge_verdict" in out:
        out = out[out["judge_verdict"].isin(verdict_sel)]
    st.write(f"{len(out):,} rows")
    st.dataframe(out[cols], use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ main

def main() -> None:
    st.set_page_config(page_title="Exp2b judged viewer", layout="wide")
    csv_path = _sidebar()
    if not csv_path or not Path(csv_path).exists():
        st.info("Enter the path to a judged CSV in the sidebar (or pass --csv on launch).")
        return
    df = load_csv(csv_path)

    _overview(df)
    st.divider()
    _per_condition(df)
    st.divider()
    _per_layer_chart(df)
    st.divider()
    _axis_table(df)
    st.divider()
    _scatter(df)
    st.divider()
    _drilldown(df)
    st.divider()
    _table(df)


if __name__ == "__main__":
    main()
