"""Download HelpSteer2 and inspect its structure for the conflict-pair repurposing."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from datasets import load_dataset


def main() -> None:
    ds = load_dataset("nvidia/HelpSteer2")
    print("=== splits ===")
    for split, dset in ds.items():
        print(f"  {split}: {len(dset)} rows")

    train = ds["train"]
    print("\n=== columns ===")
    print(train.column_names)
    print("\n=== feature types ===")
    for k, v in train.features.items():
        print(f"  {k}: {v}")

    print("\n=== first example (truncated) ===")
    ex = train[0]
    for k, v in ex.items():
        s = str(v)
        print(f"  {k}: {s[:300]}{'...' if len(s) > 300 else ''}")

    # Score stats
    score_fields = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    print("\n=== score distributions (train) ===")
    for f in score_fields:
        vals = [r[f] for r in train if isinstance(r.get(f), (int, float))]
        if not vals:
            continue
        counts = Counter(vals)
        print(f"  {f}: min={min(vals)} max={max(vals)} "
              f"dist={dict(sorted(counts.items()))}")

    # Group responses by prompt to form pairs
    print("\n=== pairing by prompt ===")
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in train:
        groups[r["prompt"]].append(r)

    sizes = Counter(len(v) for v in groups.values())
    print(f"  unique prompts: {len(groups)}")
    print(f"  responses-per-prompt distribution: {dict(sorted(sizes.items()))}")

    # For pairs, compute attribute-delta availability
    pair_prompts = [p for p, rs in groups.items() if len(rs) == 2]
    print(f"  prompts with exactly 2 responses: {len(pair_prompts)}")

    def delta(rs: list[dict], field: str) -> float:
        return abs(rs[0][field] - rs[1][field])

    for f in ["verbosity", "complexity", "helpfulness"]:
        deltas = [delta(groups[p], f) for p in pair_prompts]
        big = sum(1 for d in deltas if d >= 2)
        bigger = sum(1 for d in deltas if d >= 3)
        print(f"  |Δ {f}|: mean={sum(deltas)/len(deltas):.2f}  "
              f">=2: {big}  >=3: {bigger}")

    # Show a pair with a big verbosity delta — what would the repurposed S/U look like?
    print("\n=== sample pair with large verbosity delta ===")
    cand = sorted(
        pair_prompts,
        key=lambda p: -abs(groups[p][0]["verbosity"] - groups[p][1]["verbosity"]),
    )[:3]
    for p in cand:
        rs = groups[p]
        rs_sorted = sorted(rs, key=lambda r: r["verbosity"])
        lo, hi = rs_sorted[0], rs_sorted[1]
        print("\n--- PROMPT ---")
        print(p[:400] + ("..." if len(p) > 400 else ""))
        print(f"\nLOW-verbosity response (verb={lo['verbosity']}, comp={lo['complexity']}):")
        print(lo["response"][:400] + ("..." if len(lo["response"]) > 400 else ""))
        print(f"\nHIGH-verbosity response (verb={hi['verbosity']}, comp={hi['complexity']}):")
        print(hi["response"][:400] + ("..." if len(hi["response"]) > 400 else ""))

    # Export a small repurposed sample so we can eyeball feasibility
    print("\n=== exporting 20 repurposed pairs to scripts/hs2_sample.jsonl ===")
    repurposed = []
    for p in cand + sorted(
        pair_prompts,
        key=lambda p: -abs(groups[p][0]["complexity"] - groups[p][1]["complexity"]),
    )[:17]:
        rs = groups[p]
        vd = abs(rs[0]["verbosity"] - rs[1]["verbosity"])
        cd = abs(rs[0]["complexity"] - rs[1]["complexity"])
        axis = "verbosity" if vd >= cd else "complexity"
        rs_sorted = sorted(rs, key=lambda r: r[axis])
        lo, hi = rs_sorted[0], rs_sorted[1]
        if axis == "verbosity":
            s_instr = "Give a thorough, detailed response with complete explanations."
            u_instr = "Be concise. Answer briefly in as few words as possible."
        else:
            s_instr = "Use advanced technical language and go deep into the subject."
            u_instr = "Explain simply, as if to a beginner, avoiding jargon."
        repurposed.append({
            "user_prompt": p,
            "axis": axis,
            "delta": vd if axis == "verbosity" else cd,
            "s_instruction": s_instr,
            "u_instruction": u_instr,
            "s_aligned_response": hi["response"],
            "u_aligned_response": lo["response"],
            "s_attr_score": hi[axis],
            "u_attr_score": lo[axis],
        })

    out = "/Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/scripts/hs2_sample.jsonl"
    with open(out, "w") as f:
        for r in repurposed:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(repurposed)} rows to {out}")


if __name__ == "__main__":
    main()
