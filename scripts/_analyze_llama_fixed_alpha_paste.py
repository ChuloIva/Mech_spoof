"""Reconstruct Llama fixed-α expA tables from the pasted printout and produce
the cross-model comparison plots + signature ratios.

The Llama nb15 fixed-α run printed tables to stdout but the CSVs hadn't been
synced to disk yet, so we hard-code the numbers here. Once the CSVs land we
can swap to reading them directly.
"""
from __future__ import annotations
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('/Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/exp15_results_latest_same_magnit/figures')
OUT.mkdir(exist_ok=True)
QWEN_DIR = Path('/Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/exp15_results_latest_same_magnit')

CONDITIONS = ['baseline',
              'SU α=-3.0', 'SU α=-2.0', 'SU α=-1.5', 'SU α=-1.0', 'SU α=-0.7',
              'SU α=-0.5', 'SU α=-0.3', 'SU α=-0.15', 'SU α=+0.15', 'SU α=+0.3',
              'SU α=+0.5', 'SU α=+1.0',
              'refusal α=-3.0', 'refusal α=-2.0', 'refusal α=-1.5', 'refusal α=-1.0',
              'refusal α=-0.7', 'refusal α=-0.5', 'refusal α=-0.3', 'refusal α=-0.15',
              'refusal α=+0.15', 'refusal α=+0.3', 'refusal α=+0.5', 'refusal α=+1.0']
HARDENING = ['soft', 'soft2', 'medium', 'medium2', 'hard']

# Llama leak rate table (paste-derived).
llama_leak = pd.DataFrame(
    [
        [0.75, 0.0, 0.0, 0.0, 0.0, 0.67, 0.75, 0.75, 0.75, 0.67, 0.67, 0.75, 0.12,
         0.0, 0.0, 0.0, 1.00, 0.91, 1.00, 1.00, 0.92, 0.75, 0.67, 0.50, np.nan],
        [0.58, 0.0, 0.0, 0.0, 0.0, 0.60, 0.50, 0.67, 0.67, 0.67, 0.58, 0.50, 0.00,
         0.0, 0.0, 0.0, 0.75, 0.64, 0.73, 0.75, 0.75, 0.75, 0.50, 0.25, np.nan],
        [0.17, 0.0, 0.0, 0.0, 0.0, 0.60, 0.25, 0.17, 0.17, 0.17, 0.08, 0.00, 0.00,
         0.0, 0.0, np.nan, 0.50, 0.42, 0.33, 0.33, 0.25, 0.17, 0.08, 0.08, 0.0],
        [0.08, 0.0, 0.0, 0.0, 0.0, 0.18, 0.10, 0.08, 0.08, 0.08, 0.00, 0.08, 0.00,
         0.0, 0.0, 0.0, 0.33, 0.25, 0.17, 0.08, 0.08, 0.00, 0.00, 0.00, 0.0],
        [0.00, 0.0, 0.0, 0.0, 0.0, 0.17, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
         0.0, 0.0, 0.0, 0.25, 0.17, 0.08, 0.17, 0.17, 0.00, 0.00, 0.00, np.nan],
    ],
    index=HARDENING, columns=CONDITIONS,
)

llama_refp = pd.DataFrame(
    [
        [0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.00, 0.00, 0.00, 0.00, 0.12,
         0.0, 0.0, 0.0, 0.00, 0.00, 0.0, 0.00, 0.00, 0.00, 0.08, 0.50, 1.0],
        [0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.00, 0.00, 0.00, 0.00, 0.25,
         0.0, 0.0, 0.0, 0.08, 0.09, 0.0, 0.00, 0.00, 0.08, 0.25, 0.67, np.nan],
        [0.75, 0.0, 0.0, 0.0, 0.0, 0.20, 0.42, 0.5, 0.67, 0.58, 0.58, 0.33, 0.00,
         0.0, 0.0, np.nan, 0.00, 0.00, 0.0, 0.33, 0.25, 0.67, 0.75, 0.83, 1.0],
        [0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.5, 0.42, 0.58, 0.50, 0.42, 0.10,
         0.0, 0.0, 0.0, 0.00, 0.00, 0.0, 0.00, 0.50, 0.50, 0.67, 0.92, 1.0],
        [0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.08, 0.00, 0.00, 0.00, 0.00,
         0.0, 0.0, 0.0, 0.00, 0.00, 0.0, 0.00, 0.00, 0.25, 0.17, 0.83, np.nan],
    ],
    index=HARDENING, columns=CONDITIONS,
)


def split_alpha(c):
    if c == 'baseline':
        return 'baseline', 0.0
    m, k = c.split(' α=')
    return m, float(k)


# ---------------------------------------------------------------------------
# Cross-model side-by-side figures
# ---------------------------------------------------------------------------
qwen_leak = pd.read_csv(QWEN_DIR / 'expA_leak_rate_summary.csv').set_index('hardening').loc[HARDENING][CONDITIONS]
qwen_refp = pd.read_csv(QWEN_DIR / 'expA_refusal_rate_summary.csv').set_index('hardening').loc[HARDENING][CONDITIONS]


# Fig: signature scatter side-by-side (Qwen, Llama)
fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
for ax, (label, lk, rf) in zip(axes, [
    ('Qwen 3.5 4B',     qwen_leak, qwen_refp),
    ('Llama 3.3 70B',   llama_leak, llama_refp),
]):
    lk_mean = lk.mean(axis=0)
    rf_mean = rf.mean(axis=0)
    for cond in CONDITIONS:
        m, k = split_alpha(cond)
        if pd.isna(lk_mean[cond]) or pd.isna(rf_mean[cond]):
            continue
        if m == 'baseline':
            ax.scatter(rf_mean[cond], lk_mean[cond], s=300, marker='*', color='black',
                       edgecolor='white', lw=1.5, zorder=5)
            ax.annotate('baseline', (rf_mean[cond], lk_mean[cond]),
                        xytext=(8, 8), textcoords='offset points', fontsize=9)
        else:
            colour = '#1f77b4' if m == 'SU' else '#d62728'
            marker = 'o' if m == 'SU' else 's'
            size = 40 + 100 * abs(k)
            ax.scatter(rf_mean[cond], lk_mean[cond], s=size, alpha=0.75,
                       color=colour, marker=marker, edgecolor='black', lw=0.5)
            if abs(k) in [0.5, 1.0, 1.5, 2.0]:
                ax.annotate(f'{m} α={k:+.2g}', (rf_mean[cond], lk_mean[cond]),
                            xytext=(6, 4), textcoords='offset points', fontsize=7.5, alpha=0.8)
    ax.set_xlabel('refusal-phrase rate  (lower = template stripped)')
    ax.set_title(f'{label}\nSignature: refusal-template intactness × leak success')
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

axes[0].set_ylabel('leak rate (mean over hardening levels)')

from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='SU steering'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728', markersize=10, label='refusal steering'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=14, label='baseline'),
], loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.99), fontsize=10)
fig.suptitle('Cross-model leakage signature — Qwen vs Llama (fixed-α)', y=1.04, fontsize=13)
fig.tight_layout()
fig.savefig(OUT / 'fig8_cross_model_signature.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('  wrote fig8_cross_model_signature.png')


# Per-condition snapshot: refusal-phrase rate at "matched leak"
print('\n=== Cross-model: at matched success level (leak ≥ 0.40), what is the refusal-phrase rate? ===')
for label, lk, rf in [('Qwen', qwen_leak, qwen_refp), ('Llama', llama_leak, llama_refp)]:
    print(f'\n  {label}:')
    lk_m = lk.mean(axis=0); rf_m = rf.mean(axis=0)
    for cond in CONDITIONS:
        if cond == 'baseline':
            continue
        if pd.notna(lk_m[cond]) and lk_m[cond] >= 0.40:
            print(f'    {cond:<18s}  leak={lk_m[cond]:.2f}  refusal-phrase-rate={rf_m[cond]:.2f}')


# Compute the conditional template-intactness ratio:
#   "stealth metric" = refusal_rate at jailbreak success level
# A higher value means the model still says "I can't" while disclosing.
print('\n=== Cross-model stealth metric (mean over conditions with leak ≥ 0.50) ===')
for label, lk, rf in [('Qwen', qwen_leak, qwen_refp), ('Llama', llama_leak, llama_refp)]:
    print(f'\n  {label}:')
    for method in ['SU', 'refusal']:
        keep = []
        lk_m = lk.mean(axis=0); rf_m = rf.mean(axis=0)
        for cond in CONDITIONS:
            m, k = split_alpha(cond)
            if m != method or pd.isna(lk_m[cond]):
                continue
            if lk_m[cond] >= 0.50:
                keep.append((cond, lk_m[cond], rf_m[cond]))
        if not keep:
            print(f'    {method}: no condition reached 50% leak')
            continue
        ref_mean = np.mean([r for _, _, r in keep])
        print(f'    {method}: avg refusal-phrase-rate at >=50% leak = {ref_mean:.2f}  '
              f'(over {len(keep)} conditions: {", ".join(c for c, _, _ in keep)})')


# Llama-medium-only view: where there's a baseline template to strip.
print('\n=== Llama medium-hardening detail (baseline refusal-rate=0.75) ===')
print(f'{"condition":<18s}  {"leak":>5s}  {"refusal-phrase":>15s}  {"strip%":>7s}  {"stealth-share":>14s}')
base_ref = 0.75
for cond in CONDITIONS:
    m, k = split_alpha(cond)
    if m not in ['SU', 'refusal']:
        continue
    lk_v = llama_leak.loc['medium', cond]
    rf_v = llama_refp.loc['medium', cond]
    if pd.isna(lk_v) or pd.isna(rf_v) or lk_v < 0.20:
        continue
    strip_pct = (base_ref - rf_v) / base_ref if base_ref > 0 else np.nan
    stealth = rf_v / base_ref if base_ref > 0 else np.nan
    print(f'  {cond:<18s}  {lk_v:>5.2f}  {rf_v:>15.2f}  {strip_pct:>7.0%}  {stealth:>14.0%}')
