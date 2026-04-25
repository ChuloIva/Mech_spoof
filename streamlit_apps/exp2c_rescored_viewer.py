"""Streamlit viewer for rescored_all_positions.csv (exp2c output).

Shows per-position correlations side-by-side, per-layer × per-position curves, and
per-pair drilldowns where you can compare the three position probe scores for the
same generated response.

Run:
    streamlit run streamlit_apps/exp2c_rescored_viewer.py -- --csv /path/to/rescored_all_positions.csv
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


POSITIONS = ("response_first", "response_mid", "response_last")
CONDITIONS = ("REAL", "NONE", "FAKE")


# ------------------------------------------------------------------ IO

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for pos in POSITIONS:
        col = f"probe_score_{pos}_best_layer"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ppl_response" in df.columns:
        df["ppl_response"] = pd.to_numeric(df["ppl_response"], errors="coerce")
    if "system_followed" in df.columns:
        s = df["system_followed"].astype(str).str.lower()
        df["system_followed_bool"] = s.where(s.isin(["true", "false"]))
        df["y"] = df["system_followed_bool"].map({"true": 1, "false": 0})
    return df


def parse_psbl(s: object) -> dict[int, float]:
    if isinstance(s, str) and s.strip():
        try:
            return {int(k): float(v) for k, v in json.loads(s).items()}
        except Exception:
            return {}
    return {}


@st.cache_data(show_spinner=False)
def per_layer_long(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format per-(row, position, layer) probe scores."""
    rows = []
    for idx, r in df.iterrows():
        y = r.get("y") if "y" in df else None
        for pos in POSITIONS:
            col = f"probe_scores_{pos}_by_layer_json"
            if col not in df.columns:
                continue
            psbl = parse_psbl(r[col])
            for layer, score in psbl.items():
                rows.append({
                    "row_idx": idx,
                    "position": pos,
                    "layer": layer,
                    "score": score,
                    "y": y,
                    "condition": r.get("condition"),
                    "conflict_axis": r.get("conflict_axis"),
                    "judge_verdict": r.get("judge_verdict"),
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ helpers

def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
        return None
    r, p = pearsonr(x[mask], y[mask])
    return {"r": float(r), "p": float(p), "n": int(mask.sum())}


# ------------------------------------------------------------------ sections

def _sidebar() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", default="")
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    known, _ = parser.parse_known_args(argv)
    st.sidebar.header("Exp2c rescored viewer")
    default = known.csv or st.session_state.get("rescored_csv", "")
    p = st.sidebar.text_input("Rescored CSV path", value=default)
    st.session_state["rescored_csv"] = p
    return p


def _overview(df: pd.DataFrame) -> None:
    st.header("Overview")
    n = len(df)
    n_judged = int(df["y"].notna().sum()) if "y" in df else 0
    best_layer = int(df["best_layer"].iloc[0]) if "best_layer" in df else None
    best_position = df["best_position"].iloc[0] if "best_position" in df else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", n)
    c2.metric("Judged (S/U)", n_judged)
    c3.metric("Best layer (training)", best_layer if best_layer is not None else "?")
    c4.metric("Best position (training)", best_position or "?")


def _correlation_grid(df: pd.DataFrame) -> None:
    st.header("Best-layer correlations: position × condition")
    if "y" not in df or df["y"].isna().all():
        st.info("No `system_followed` labels — can't compute correlations.")
        return
    y_full = df["y"].to_numpy(dtype=float)

    rows = []
    for pos in POSITIONS:
        col = f"probe_score_{pos}_best_layer"
        if col not in df.columns:
            continue
        x_full = df[col].to_numpy(dtype=float)
        for label, mask in [
            ("overall", np.ones(len(df), dtype=bool)),
            *[(c, (df["condition"] == c).to_numpy()) for c in CONDITIONS],
        ]:
            corr = _safe_corr(x_full[mask], y_full[mask])
            if corr is None:
                rows.append({"position": pos, "slice": label, "n": int(mask.sum()),
                             "r": np.nan, "p": np.nan})
            else:
                rows.append({"position": pos, "slice": label, **corr})

    table = pd.DataFrame(rows)
    pivot_r = table.pivot(index="position", columns="slice", values="r").reindex(POSITIONS)
    pivot_n = table.pivot(index="position", columns="slice", values="n").reindex(POSITIONS)
    st.subheader("Pearson r")
    st.dataframe(
        pivot_r.style.format("{:+.3f}").background_gradient(cmap="RdBu_r", vmin=-0.5, vmax=0.5),
        use_container_width=True,
    )
    st.subheader("n")
    st.dataframe(pivot_n.style.format("{:.0f}"), use_container_width=True)
    st.caption("Use this grid to pick the best (position × condition) probe for downstream work.")


def _per_layer_curves(df: pd.DataFrame) -> None:
    st.header("Per-layer correlation curve, all positions")
    if "y" not in df or df["y"].isna().all():
        st.info("No labels.")
        return
    long = per_layer_long(df)
    if long.empty:
        st.info("No per-layer JSON columns found.")
        return

    cond_choice = st.radio(
        "Condition slice", ("overall", *CONDITIONS), horizontal=True, key="cond_layer"
    )
    sub = long if cond_choice == "overall" else long[long["condition"] == cond_choice]

    rows = []
    for (pos, layer), g in sub.groupby(["position", "layer"]):
        corr = _safe_corr(g["score"].to_numpy(dtype=float),
                          g["y"].to_numpy(dtype=float))
        if corr is None:
            continue
        rows.append({"position": pos, "layer": int(layer), "r": corr["r"], "n": corr["n"]})
    if not rows:
        st.info("Not enough variance in this slice.")
        return
    plot_df = pd.DataFrame(rows)
    chart = (
        alt.Chart(plot_df).mark_line(point=True).encode(
            x=alt.X("layer:Q", title="Layer"),
            y=alt.Y("r:Q", title="Pearson r"),
            color="position:N",
            tooltip=["position", "layer", alt.Tooltip("r:Q", format="+.3f"), "n"],
        ).properties(height=420)
    )
    rule = alt.Chart(pd.DataFrame({"r": [0]})).mark_rule(color="gray", strokeDash=[2, 2]).encode(y="r:Q")
    st.altair_chart(chart + rule, use_container_width=True)


def _position_scatter(df: pd.DataFrame) -> None:
    st.header("Probe score scatter by position")
    if "y" not in df or df["y"].isna().all():
        return
    pos = st.selectbox("Position", POSITIONS, index=0, key="scatter_pos")
    col = f"probe_score_{pos}_best_layer"
    if col not in df.columns:
        st.info(f"{col} not in CSV.")
        return
    sub = df.dropna(subset=["y", col]).copy()
    sub["y_jit"] = sub["y"] + np.random.RandomState(0).uniform(-0.08, 0.08, len(sub))
    chart = (
        alt.Chart(sub).mark_circle(size=30, opacity=0.5).encode(
            x=alt.X(f"{col}:Q", title=f"probe score @ {pos}, best layer"),
            y=alt.Y("y_jit:Q", title="system_followed (jittered)",
                    scale=alt.Scale(domain=[-0.3, 1.3])),
            color="condition:N",
            tooltip=["pair_idx", "condition", "judge_verdict", "conflict_axis",
                     alt.Tooltip(f"{col}:Q", format="+.3f")],
        ).properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


def _axis_table(df: pd.DataFrame) -> None:
    st.header("Per conflict_axis breakdown")
    if "conflict_axis" not in df:
        return
    pos = st.selectbox("Position", POSITIONS, index=0, key="axis_pos")
    col = f"probe_score_{pos}_best_layer"
    rows = []
    for axis, sub in df.groupby("conflict_axis"):
        judged = sub.dropna(subset=["y"])
        corr = _safe_corr(sub[col].to_numpy(dtype=float), sub["y"].to_numpy(dtype=float)) if col in sub else None
        rows.append({
            "axis": axis, "n": len(sub), "n_judged": len(judged),
            "sys_rate": float(judged["y"].mean()) if len(judged) else float("nan"),
            "r": corr["r"] if corr else float("nan"),
            "p": corr["p"] if corr else float("nan"),
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
    pid = st.selectbox("Pair idx", pair_ids, key="drill_pair")
    sub = df[df["pair_idx"] == pid].sort_values("condition")
    if sub.empty:
        return
    head = sub.iloc[0]
    st.markdown(f"**S:** {head['s_instruction']}")
    st.markdown(f"**U:** {head['u_instruction']}")
    st.caption(f"axis: {head.get('conflict_axis', '?')}")
    for _, r in sub.iterrows():
        scores = " · ".join(
            f"{p.split('_')[1]}={r[f'probe_score_{p}_best_layer']:+.3f}"
            for p in POSITIONS if pd.notna(r.get(f"probe_score_{p}_best_layer"))
        )
        with st.expander(
            f"{r['condition']}  ·  verdict={r.get('judge_verdict')}  ·  ppl={r.get('ppl_response', float('nan')):.2f}"
            f"  ·  {scores}",
            expanded=False,
        ):
            st.markdown("**Response preview:**")
            st.text((r.get("response_preview") or "")[:4000])
            st.markdown("**Judge reason:**")
            st.text(r.get("judge_reason", "") or "")


def _table(df: pd.DataFrame) -> None:
    st.header("Filterable table")
    drop = [c for c in df.columns if c.endswith("_by_layer_json") or c == "y"]
    cols = [c for c in df.columns if c not in drop]
    with st.expander("Filter"):
        conds = sorted(df["condition"].dropna().unique()) if "condition" in df else []
        verdicts = sorted(df["judge_verdict"].dropna().unique()) if "judge_verdict" in df else []
        cond_sel = st.multiselect("condition", conds, default=conds, key="tbl_cond")
        verdict_sel = st.multiselect("judge_verdict", verdicts, default=verdicts, key="tbl_v")
    out = df.copy()
    if cond_sel and "condition" in out:
        out = out[out["condition"].isin(cond_sel)]
    if verdict_sel and "judge_verdict" in out:
        out = out[out["judge_verdict"].isin(verdict_sel)]
    st.write(f"{len(out):,} rows")
    st.dataframe(out[cols], use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ main

def main() -> None:
    st.set_page_config(page_title="Exp2c rescored viewer", layout="wide")
    csv_path = _sidebar()
    if not csv_path or not Path(csv_path).exists():
        st.info("Enter the path to a rescored_all_positions.csv in the sidebar.")
        return
    df = load_csv(csv_path)

    _overview(df)
    st.divider()
    _correlation_grid(df)
    st.divider()
    _per_layer_curves(df)
    st.divider()
    _axis_table(df)
    st.divider()
    _position_scatter(df)
    st.divider()
    _drilldown(df)
    st.divider()
    _table(df)


if __name__ == "__main__":
    main()
