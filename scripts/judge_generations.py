"""Standalone judging CLI.

Decouples the (slow, GPU-bound) generation phase from the (fast, swappable) judging
phase. Takes a generations.jsonl produced by exp2b — or any compatible jsonl — runs
a vLLM-served judge over each row, and writes a CSV with verdicts and probe scores
joined onto every record.

Usage
-----

    python scripts/judge_generations.py \
        --input  /path/to/exp2b_conflict_evolved/generations.jsonl \
        --output /path/to/exp2b_conflict_evolved/judged.csv \
        --judge  Qwen/Qwen2.5-7B-Instruct

Each input row must contain at minimum:
    pair_idx, condition, s_instruction, u_instruction,
    s_gold, u_gold, response

Optional fields are passed through to the CSV (`probe_score`, `probe_scores_by_layer`,
`conflict_axis`, `n_layers_activated_above_mean`, etc.)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Make `mech_spoof` importable when run from the repo root without install.
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from mech_spoof.eval.llm_judge_vllm import JudgeRow, judge_with_vllm  # noqa: E402


CSV_FIELDS = [
    "pair_idx", "condition", "conflict_axis",
    "s_instruction", "u_instruction",
    "s_gold", "u_gold", "response",
    "judge_verdict", "system_followed", "judge_reason",
    "probe_score",
    "n_layers_activated_above_mean", "n_layers_activated_strong",
    "probe_scores_by_layer_json",
]


def load_generations(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_judge_rows(rows: list[dict]) -> list[JudgeRow]:
    out = []
    for i, r in enumerate(rows):
        for k in ("s_instruction", "u_instruction", "s_gold", "u_gold", "response"):
            if k not in r:
                raise KeyError(f"row {i} missing required field {k!r}")
        out.append(JudgeRow(
            s_instruction=r["s_instruction"],
            u_instruction=r["u_instruction"],
            s_gold=r["s_gold"],
            u_gold=r["u_gold"],
            response=r["response"],
            extras={"row_idx": i},
        ))
    return out


def write_csv(rows: list[dict], verdicts: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r, v in zip(rows, verdicts):
            verdict = v.get("verdict")
            if verdict == "system":
                followed = "true"
            elif verdict == "user":
                followed = "false"
            else:
                followed = ""
            psbl = r.get("probe_scores_by_layer")
            row_out = {
                "pair_idx": r.get("pair_idx", ""),
                "condition": r.get("condition", ""),
                "conflict_axis": r.get("conflict_axis", ""),
                "s_instruction": r.get("s_instruction", ""),
                "u_instruction": r.get("u_instruction", ""),
                "s_gold": r.get("s_gold", ""),
                "u_gold": r.get("u_gold", ""),
                "response": r.get("response", ""),
                "judge_verdict": verdict or "",
                "system_followed": followed,
                "judge_reason": v.get("reason", "") or "",
                "probe_score": r.get("probe_score", ""),
                "n_layers_activated_above_mean": r.get("n_layers_activated_above_mean", ""),
                "n_layers_activated_strong": r.get("n_layers_activated_strong", ""),
                "probe_scores_by_layer_json": (
                    json.dumps(psbl) if isinstance(psbl, dict) else (psbl or "")
                ),
            }
            w.writerow(row_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge a generations.jsonl with a vLLM model.")
    ap.add_argument("--input", required=True, help="Path to generations.jsonl")
    ap.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <input_dir>/judged.csv",
    )
    ap.add_argument(
        "--judge",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace ID of the judge model (vLLM-compatible).",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable <think> blocks for Qwen3+ judges (default: off — needed for JSON output).",
    )
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only judge the first N rows (useful for smoke tests).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.parent / "judged.csv"

    rows = load_generations(in_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} rows from {in_path}")

    judge_rows = to_judge_rows(rows)
    verdicts = judge_with_vllm(
        judge_rows,
        model_id=args.judge,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        free_after=True,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
    )

    write_csv(rows, verdicts, out_path)
    n_sys = sum(1 for v in verdicts if v.get("verdict") == "system")
    n_usr = sum(1 for v in verdicts if v.get("verdict") == "user")
    n_other = len(verdicts) - n_sys - n_usr
    print(
        f"Wrote {len(rows)} judged rows to {out_path}"
        f"  (system={n_sys}  user={n_usr}  both/neither/none={n_other})"
    )


if __name__ == "__main__":
    main()
