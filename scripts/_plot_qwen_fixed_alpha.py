"""Plot the Qwen fixed-α run from exp15_results_latest_same_magnit/.

Generates:
  fig1_expB_compliance_stack.png — stacked bars (refused/comply/degen) per condition
  fig2_expB_mc_dose_response.png — MC logit-diff Δ vs α, both methods overlaid
  fig3_expA_leak_heatmap.png     — leak rate heatmap (hardening × condition)
  fig4_expA_refusal_heatmap.png  — refusal rate heatmap
  fig5_expA_logitdiff_heatmap.png — A−B logit-diff Δ heatmap
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path('/Users/ivanculo/Desktop/Projects/Mech_Spoof/Mech_spoof/exp15_results_latest_same_magnit')
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)

mpl.rcParams.update({'font.size': 10, 'axes.titlesize': 11})


def split_alpha(cond: str):
    """('SU α=-1.0',) → ('SU', -1.0); 'baseline' → ('baseline', 0.0)"""
    if cond == 'baseline':
        return 'baseline', 0.0
    method, k = cond.split(' α=')
    return method, float(k)


def order_conditions(conds):
    """Sort: baseline first, then SU by α, then refusal by α."""
    parsed = [(c, *split_alpha(c)) for c in conds]
    out = []
    for m in ['baseline', 'SU', 'refusal']:
        rows = sorted([p for p in parsed if p[1] == m], key=lambda x: x[2])
        out.extend([r[0] for r in rows])
    return out


# ---------------------------------------------------------------------------
# fig 1: expB compliance stacked bars
# ---------------------------------------------------------------------------
df = pd.read_csv(ROOT / 'expB_compliance_summary.csv')
order = order_conditions(df.condition.tolist())
df = df.set_index('condition').loc[order].reset_index()

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(df))
ax.bar(x, df.refused, color='#2ca02c', label='refused', alpha=0.85)
ax.bar(x, df.complied, bottom=df.refused, color='#d62728', label='complied', alpha=0.85)
ax.bar(x, df.degenerate, bottom=df.refused + df.complied, color='#7f7f7f', label='degenerate', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(df.condition, rotation=60, ha='right')
ax.axvline(0.5, color='black', lw=0.5)
# Boundary between SU and refusal
methods = [split_alpha(c)[0] for c in df.condition]
for i in range(1, len(methods)):
    if methods[i] != methods[i-1]:
        ax.axvline(i - 0.5, color='black', lw=0.7, ls='--', alpha=0.5)
ax.set_ylabel('fraction of generations')
ax.set_ylim(0, 1.0)
ax.set_title('expB AdvBench-30: compliance under fixed-α steering — Qwen 3.5 4B\n'
             '(refused=safe; complied=jailbreak; degenerate=token-loop)')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'fig1_expB_compliance_stack.png', dpi=140)
plt.close(fig)
print('  wrote fig1_expB_compliance_stack.png')


# ---------------------------------------------------------------------------
# fig 2: expB MC logit-diff dose-response
# ---------------------------------------------------------------------------
delta = pd.read_csv(ROOT / 'expB_mc_logit_delta_vs_baseline.csv')
delta[['method', 'alpha']] = delta.condition.apply(lambda c: pd.Series(split_alpha(c)))

fig, ax = plt.subplots(figsize=(9, 5.5))
for method, colour, marker in [('SU', '#1f77b4', 'o'), ('refusal', '#d62728', 's')]:
    sub = delta[delta.method == method].sort_values('alpha')
    in_dist = sub[sub.n_in_dist > 0]
    ood = sub[sub.n_in_dist == 0]
    ax.plot(in_dist.alpha, in_dist.delta_mean, marker=marker, ms=7, lw=1.8,
            color=colour, label=f'{method}  (in-dist)')
    if len(ood):
        ax.scatter(ood.alpha, [0]*len(ood), marker='x', s=80, color=colour,
                   label=f'{method}  (degenerate / OOD)', alpha=0.5)

base = delta[delta.method == 'baseline']
ax.scatter(base.alpha, base.delta_mean, marker='*', s=200, color='black', label='baseline', zorder=5)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel('steering coefficient α  (× unit direction)')
ax.set_ylabel('Δ MC logit-diff [A=comply − B=refuse]  vs baseline')
ax.set_title('expB AdvBench MC dose-response — Qwen 3.5 4B (fixed-α)\n'
             'Refusal axis has 3× the leverage of SU on the comply/refuse decision')
ax.legend(loc='upper left'); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'fig2_expB_mc_dose_response.png', dpi=140)
plt.close(fig)
print('  wrote fig2_expB_mc_dose_response.png')


# ---------------------------------------------------------------------------
# Heatmap helper — hardening × condition
# ---------------------------------------------------------------------------
HARDENING_ORDER = ['soft', 'soft2', 'medium', 'medium2', 'hard']


def heatmap(csv_path, title, savename, cmap='RdYlGn_r', vmin=0, vmax=1, fmt='.2f'):
    df = pd.read_csv(csv_path)
    df = df.set_index('hardening').loc[HARDENING_ORDER]
    cols = [c for c in df.columns]
    cols_ordered = order_conditions(cols)
    df = df[cols_ordered]
    # Insert a visual gap by adding a NaN column at the SU/refusal boundary
    sep_idx = next(i for i, c in enumerate(cols_ordered) if split_alpha(c)[0] == 'refusal')
    arr = df.values
    fig, ax = plt.subplots(figsize=(14, 3.6))
    im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols_ordered)))
    ax.set_xticklabels(cols_ordered, rotation=60, ha='right')
    ax.set_yticks(range(len(HARDENING_ORDER)))
    ax.set_yticklabels(HARDENING_ORDER)
    # cell annotations
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='#888')
            else:
                ax.text(j, i, f'{v:{fmt}}', ha='center', va='center', fontsize=7,
                        color='white' if (vmax - vmin) * 0.5 < abs(v - (vmin + vmax) / 2) else 'black')
    # SU/refusal divider
    ax.axvline(sep_idx - 0.5, color='black', lw=1.2)
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / savename, dpi=140)
    plt.close(fig)
    print(f'  wrote {savename}')


heatmap(ROOT / 'expA_leak_rate_summary.csv',
        'expA prompt-leakage: leak rate per (hardening × condition) — Qwen 3.5 4B (fixed-α)\n'
        '— = generations OOD/degenerate; higher = more secret-disclosure',
        'fig3_expA_leak_heatmap.png',
        cmap='Reds', vmin=0, vmax=1)

heatmap(ROOT / 'expA_refusal_rate_summary.csv',
        'expA prompt-leakage: refusal-phrase rate (regex) — Qwen 3.5 4B (fixed-α)\n'
        '— = OOD; lower = fewer "I cannot" / "I won\'t" phrases',
        'fig4_expA_refusal_heatmap.png',
        cmap='Blues', vmin=0, vmax=1)


# ---------------------------------------------------------------------------
# fig 5: expA logit-diff Δ heatmap (raw decision shift, signed)
# ---------------------------------------------------------------------------
df = pd.read_csv(ROOT / 'expA_mc_logit_delta_vs_baseline.csv')
df = df.set_index('hardening').loc[HARDENING_ORDER]
cols_ordered = order_conditions([c for c in df.columns])
df = df[cols_ordered]
arr = df.values
absmax = np.nanmax(np.abs(arr))
fig, ax = plt.subplots(figsize=(14, 3.6))
im = ax.imshow(arr, aspect='auto', cmap='RdBu_r', vmin=-absmax, vmax=absmax)
ax.set_xticks(range(len(cols_ordered)))
ax.set_xticklabels(cols_ordered, rotation=60, ha='right')
ax.set_yticks(range(len(HARDENING_ORDER)))
ax.set_yticklabels(HARDENING_ORDER)
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        v = arr[i, j]
        if np.isnan(v):
            ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='#888')
        else:
            ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=7, color='black')
sep_idx = next(i for i, c in enumerate(cols_ordered) if split_alpha(c)[0] == 'refusal')
ax.axvline(sep_idx - 0.5, color='black', lw=1.2)
fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
ax.set_title('expA prompt-leakage: Δ MC logit-diff [A=disclose − B=refuse] vs baseline — Qwen 3.5 4B\n'
             'Red = pushed toward disclose; blue = pushed toward refuse. Refusal axis has 3-5× more leverage.')
fig.tight_layout()
fig.savefig(OUT / 'fig5_expA_logitdiff_heatmap.png', dpi=140)
plt.close(fig)
print('  wrote fig5_expA_logitdiff_heatmap.png')


# ---------------------------------------------------------------------------
# fig 6: side-by-side dose-response on expA mean across hardenings
# ---------------------------------------------------------------------------
df_lk = pd.read_csv(ROOT / 'expA_leak_rate_summary.csv').set_index('hardening').loc[HARDENING_ORDER]
df_rf = pd.read_csv(ROOT / 'expA_refusal_rate_summary.csv').set_index('hardening').loc[HARDENING_ORDER]
df_ld = pd.read_csv(ROOT / 'expA_mc_logit_delta_vs_baseline.csv').set_index('hardening').loc[HARDENING_ORDER]

cols = order_conditions([c for c in df_lk.columns])
parsed = [split_alpha(c) for c in cols]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
for ax, (label, df_) in zip(axes, [
    ('leak rate (mean over hardenings)', df_lk),
    ('refusal-phrase rate', df_rf),
    ('Δ MC logit-diff [disclose − refuse]', df_ld),
]):
    means = df_[cols].mean(axis=0)
    for method, colour, marker in [('SU', '#1f77b4', 'o'), ('refusal', '#d62728', 's')]:
        xs = [p[1] for p in parsed if p[0] == method]
        ys = [means[c] for c, p in zip(cols, parsed) if p[0] == method]
        ax.plot(xs, ys, marker=marker, ms=6, lw=1.8, color=colour, label=method)
    base_y = means.get('baseline', np.nan)
    ax.axhline(base_y, color='black', lw=0.6, ls='--', label=f'baseline = {base_y:.2f}')
    ax.axvline(0, color='#888', lw=0.5)
    ax.set_xlabel('α (× unit dir)')
    ax.set_title(label)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle('expA prompt-leakage dose-response (mean across 5 hardening levels) — Qwen 3.5 4B', y=1.02)
fig.tight_layout()
fig.savefig(OUT / 'fig6_expA_dose_response.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('  wrote fig6_expA_dose_response.png')

print(f'\nAll figures in {OUT}')
