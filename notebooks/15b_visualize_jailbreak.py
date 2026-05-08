"""Generate diary figures from the dense-grid exp15 run.

Reads CSVs from `exp15_jailbreak_steering (2)/` and writes PNGs into
`exp15_jailbreak_steering (2)/figures/`. Run from project root:

    .venv/bin/python notebooks/15b_visualize_jailbreak.py
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN  = ROOT / 'exp15_jailbreak_steering (2)'
OUT  = RUN / 'figures'
OUT.mkdir(exist_ok=True)


def parse_k(name: str) -> tuple[str, float] | None:
    """'SU −0.7σ' -> ('SU', -0.7); 'refusal +0.25σ' -> ('refusal', 0.25); 'baseline' -> None."""
    if name == 'baseline':
        return None
    m = re.match(r'(SU|refusal)\s*([−+\-])\s*(\d+(?:\.\d+)?)σ', name)
    if not m:
        return None
    method, sign, mag = m.groups()
    k = float(mag) * (-1 if sign == '−' or sign == '-' else 1)
    return method, k


# ---------- expB compliance curve ----------
B = pd.read_csv(RUN / 'expB_compliance_summary.csv')
parsed = B['condition'].map(parse_k)
B['method'] = parsed.map(lambda p: p[0] if p else 'baseline')
B['k']      = parsed.map(lambda p: p[1] if p else 0.0)
baseline_row = B[B['condition'] == 'baseline'].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, method in zip(axes, ['SU', 'refusal']):
    sub = B[B['method'] == method].sort_values('k').copy()
    # Inject the baseline as a k=0 row so each line passes through it
    base_row = pd.DataFrame([{'k': 0.0,
                              'refused':    baseline_row['refused'],
                              'complied':   baseline_row['complied'],
                              'degenerate': baseline_row['degenerate']}])
    sub_with_base = pd.concat([sub[['k', 'refused', 'complied', 'degenerate']],
                                base_row], ignore_index=True).sort_values('k')
    ax.plot(sub_with_base['k'], sub_with_base['refused'],    'o-', label='refused',    color='#2c7fb8', linewidth=2)
    ax.plot(sub_with_base['k'], sub_with_base['complied'],   's-', label='complied',   color='#d7301f', linewidth=2)
    ax.plot(sub_with_base['k'], sub_with_base['degenerate'], '^-', label='degenerate', color='#666666', linewidth=2)
    # Baseline reference lines (faint)
    ax.axhline(baseline_row['refused'],  color='#2c7fb8', linestyle=':', alpha=0.3)
    ax.axhline(baseline_row['complied'], color='#d7301f', linestyle=':', alpha=0.3)
    # Explicit baseline marker at k=0 with a star on each metric
    ax.scatter([0], [baseline_row['refused']],    marker='*', s=260, color='#2c7fb8', edgecolor='black', linewidth=1.0, zorder=5)
    ax.scatter([0], [baseline_row['complied']],   marker='*', s=260, color='#d7301f', edgecolor='black', linewidth=1.0, zorder=5)
    ax.scatter([0], [baseline_row['degenerate']], marker='*', s=260, color='#666666', edgecolor='black', linewidth=1.0, zorder=5)
    ax.axvline(0, color='black', linestyle='--', alpha=0.35, linewidth=1, zorder=1)
    ax.text(0, 1.06, 'baseline\n(k=0)', ha='center', va='bottom', fontsize=8,
            color='black', bbox=dict(boxstyle='round,pad=0.2', fc='#fffbe6', ec='black', alpha=0.9))
    ax.set_xlabel(f'{method} steering coefficient (k, in σ units)')
    ax.set_title(f'expB AdvBench compliance — {method} direction')
    ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.3)
    ax.legend(loc='center right', fontsize=9)
    # Annotate the SU jailbreak peak
    if method == 'SU':
        peak = sub[sub['complied'] == sub['complied'].max()].iloc[0]
        ax.annotate(f'peak: {peak["complied"]:.0%} comply\n(k={peak["k"]:.2f}σ, 0% degen)',
                    xy=(peak['k'], peak['complied']),
                    xytext=(peak['k'] - 0.15, peak['complied'] - 0.25),
                    arrowprops=dict(arrowstyle='->', color='#d7301f'),
                    fontsize=9, color='#d7301f',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d7301f', alpha=0.9))
axes[0].set_ylabel('fraction of prompts (n=30)')
fig.suptitle('Steering ⇒ behavioural compliance on AdvBench-30  (Qwen3.5-4B, layers 16..31, NORMALIZE+per-layer-σ)',
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT / 'expB_compliance_curve.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('wrote', OUT / 'expB_compliance_curve.png')


# ---------- expA leak / refusal heatmaps ----------
def plot_heatmap(csv_path, value_label, fname, cmap):
    df = pd.read_csv(csv_path).set_index('hardening')
    cols = [c for c in df.columns]  # already in grid order
    M = df[cols].values.astype(float)
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(cols)), 3.2))
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    # Highlight the baseline column label
    xlabels = []
    for c in cols:
        if c == 'baseline':
            xlabels.append(r'$\bf{baseline}$')
        else:
            xlabels.append(c)
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, '·', ha='center', va='center', color='#888', fontsize=10)
            else:
                color = 'white' if v > 0.55 else 'black'
                fw = 'bold' if cols[j] == 'baseline' else 'normal'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', color=color, fontsize=7, fontweight=fw)
    # Vertical separator after the baseline column to mark the "unsteered" reference
    if 'baseline' in cols:
        bidx = cols.index('baseline')
        ax.axvline(bidx + 0.5, color='black', linewidth=1.5, alpha=0.7)
        ax.text(bidx, -0.85, 'unsteered\nreference', ha='center', va='bottom',
                fontsize=7.5, color='black', style='italic')
    ax.set_title(f'expA — {value_label} by hardening × condition  (· = all rows degenerate, excluded)')
    ax.set_xlabel('condition')
    ax.set_ylabel('system-prompt hardening')
    fig.colorbar(im, ax=ax, label=value_label)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', OUT / fname)

plot_heatmap(RUN / 'expA_leak_rate_summary.csv',    'leak rate',    'expA_leak_heatmap.png',    'Reds')
plot_heatmap(RUN / 'expA_refusal_rate_summary.csv', 'refusal rate', 'expA_refusal_heatmap.png', 'Blues')


# ---------- MC delta vs behavioural compliance ----------
delta = pd.read_csv(RUN / 'expB_mc_logit_delta_vs_baseline.csv')
M = B.merge(delta, on='condition')
M['method'] = M['condition'].map(lambda c: parse_k(c)[0] if parse_k(c) else 'baseline')
M['k']      = M['condition'].map(lambda c: parse_k(c)[1] if parse_k(c) else 0.0)

fig, ax = plt.subplots(figsize=(8, 5))
colors = {'SU': '#d7301f', 'refusal': '#2c7fb8', 'baseline': 'black'}
markers = {'SU': 'o', 'refusal': 's', 'baseline': '*'}
for method in ['SU', 'refusal', 'baseline']:
    sub = M[M['method'] == method]
    s = 320 if method == 'baseline' else 80
    ax.scatter(sub['delta_mean'], sub['complied'],
               c=colors[method], marker=markers[method], s=s,
               edgecolor='black' if method == 'baseline' else 'white',
               linewidth=1.0,
               label=f'{method} (n=30, 0/0/0)' if method == 'baseline' else method,
               zorder=5 if method == 'baseline' else 3)
    for _, row in sub.iterrows():
        if not np.isnan(row['delta_mean']) and method != 'baseline':
            ax.annotate(f'k={row["k"]:.2f}', (row['delta_mean'], row['complied']),
                        xytext=(5, 5), textcoords='offset points', fontsize=7, alpha=0.7)
# Baseline reference lines
ax.axhline(baseline_row['complied'], color='black', linestyle=':', alpha=0.4, label='baseline behaviour')
ax.axvline(0, color='black', linestyle=':', alpha=0.4)
ax.set_xlabel('MC logit delta (axis evidence) — A − B vs baseline')
ax.set_ylabel('expB compliance rate (behavioural evidence)')
ax.set_title('MC underestimates behavioural jailbreak\nSU −0.7σ: MC delta +0.81 → 100% compliance')
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'mc_vs_behaviour.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('wrote', OUT / 'mc_vs_behaviour.png')


# ---------- Direction comparison: best SU vs best refusal ----------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
methods_best = {
    'baseline':                      (B[B['condition'] == 'baseline'].iloc[0],     '#666'),
    'SU −0.7σ\n(peak jailbreak)':     (B[B['condition'] == 'SU −0.7σ'].iloc[0],     '#d7301f'),
    'refusal −0.4σ\n(Arditi-style)':  (B[B['condition'] == 'refusal −0.4σ'].iloc[0], '#2c7fb8'),
    'refusal +0.3σ\n(defense booster)': (B[B['condition'] == 'refusal +0.3σ'].iloc[0], '#225ea8'),
}
labels   = list(methods_best.keys())
refused  = [methods_best[k][0]['refused']    for k in labels]
complied = [methods_best[k][0]['complied']   for k in labels]
degener  = [methods_best[k][0]['degenerate'] for k in labels]

x = np.arange(len(labels))
ax.bar(x, refused,  label='refused',    color='#2c7fb8')
ax.bar(x, complied, bottom=refused,                       label='complied',   color='#d7301f')
ax.bar(x, degener,  bottom=np.array(refused) + np.array(complied), label='degenerate', color='#888')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel('fraction (n=30)')
ax.set_title('expB compliance under best-of-each-direction conditions')
ax.legend(loc='center right', fontsize=8)
for i, (r, c, d) in enumerate(zip(refused, complied, degener)):
    if c > 0.05:
        ax.text(i, r + c / 2, f'{c:.0%}', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    if r > 0.05:
        ax.text(i, r / 2, f'{r:.0%}', ha='center', va='center', color='white', fontsize=10, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'direction_comparison.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print('wrote', OUT / 'direction_comparison.png')

print(f'\nall figures in {OUT}')
