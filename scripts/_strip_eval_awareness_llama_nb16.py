"""Strip eval-awareness readout from Llama nb16, leaving only SU + refusal axes.

For the cross-model 'more with less' paper, eval-awareness is no longer
load-bearing. nb14 §5b (the refusal-only short-circuit) doesn't produce
eval-aware keys, so leaving them in nb16 would crash on a fresh refusal-only
NPZ. This script rewrites every cell that touches eval-awareness:

  - §4 direction registry: drop eval_unit/eval_test/eval_deploy + anchor calcs
  - §8 capture loop: drop eval_proj
  - §9 aggregate: drop eval_proj columns
  - §10 plots: drop eval-axis panels; keep refusal-axis; add SU-axis panel
  - §11 manifest: drop eval keys
  - §12 mid-stack focus: drop content-gated; keep refusal-axis mid-stack only
  - §13 leverage curves: refusal-axis only (drop eval_gated subplot)
  - markdown title/abstract/commentary: rewrite without eval mentions

Idempotent: re-running on already-stripped notebook is a no-op (detected via
STRIPPED_MARKER in the title cell).
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks' / '16_su_eval_axis_readout_llama33_70b.ipynb'

STRIPPED_MARKER = 'EVAL_AWARENESS_STRIPPED_v1'


def _src(c):
    s = c.get('source', '')
    return ''.join(s) if isinstance(s, list) else s


def _set_src(c, text: str):
    split = text.split('\n')
    c['source'] = [s + '\n' for s in split[:-1]] + ([split[-1]] if split[-1] else [])


# ----- new cell contents -----------------------------------------------------

TITLE_MD = (
    "# 16 — SU vs refusal axis leverage readout (Llama 3.3 70B)\n"
    "\n"
    "<!-- " + STRIPPED_MARKER + " -->\n"
    "\n"
    "**Question.** When SU steering jailbreaks the model, *how much* does the residual stream actually move on the refusal feature axis — compared to a refusal-direction push that lands the same compliance rate? This notebook is the mechanism half of the cross-model 'more with less' story.\n"
    "\n"
    "We forward-pass AdvBench-30 (and a held-out neutral set) under each steering condition with `ResidualSteerer` installed (no generation), capture the post-block residual at each layer, and project onto:\n"
    "\n"
    "1. The **refusal axis** (Arditi mean-diff direction) — does SU push refusal off baseline? By how much per σ vs the refusal-direction sweep?\n"
    "2. The **SU axis itself** — sanity check that steering does push along its own direction; calibrates the leverage-curve slope for SU.\n"
    "\n"
    "**Why this is load-bearing for the paper.** The behavioural sweep (nb15) shows SU jailbreaks at lower σ with less degenerate output. But that's a behavioural claim. The leverage curve here turns it into a *mechanistic* claim: **for equal compliance, SU moves the refusal feature less than refusal-direction steering does** (= 'less feature movement on the refusal axis', the second of the four 'more-with-less' quantities).\n"
    "\n"
    "**Conditions.** Baseline + `SU −0.5σ` (no jailbreak) + `SU −0.65σ` (cliff edge) + `SU −0.7σ` (peak) + `SU −0.85σ` (post-cliff) + `refusal −0.4σ` (refusal-axis jailbreak peak) + `refusal +0.3σ` (defense booster). Same 7 conditions as Qwen for direct cross-model comparison.\n"
    "\n"
    "**Self-contained / Colab-portable.** Same clone-and-load pattern as nb15. ~30 prompts × 7 conditions × 2 pools = ~420 forward passes (no generation). ~30 min on 3× A100 80GB.\n"
    "\n"
    "**Inputs needed.**\n"
    "- `exp06_lamma/directions.npz` — S/U PCA directions (steerable + readable).\n"
    "- `exp_directions_llama33_70b_refusal_only/directions.npz` — refusal direction (steerable + readable).\n"
    "- `data/advbench_harmful.json` — AdvBench-30 prompts.\n"
)

PORT_BANNER_MD = (
    "> **Llama 3.3 70B port note.** Direct port of the Qwen 3.5 4B exp16, with eval-awareness stripped (refusal-only NPZ is the canonical Llama input).\n"
    ">\n"
    "> - Steering window: `L40..79` (last half of 80 layers; Qwen used `L16..31` of 32).\n"
    "> - Mid-stack leverage window: `L60..72` (analogue of Qwen's `L22..28`).\n"
    "> - The `EXP15_COMPLIANCE` dict in §13 still contains the **Qwen** compliance numbers as placeholders — replace with Llama numbers from `exp15_jailbreak_steering_llama33_70b/expB_compliance_summary.csv` after running notebook 15.\n"
    "> - Inputs: `exp06_lamma/directions.npz` + `exp_directions_llama33_70b_refusal_only/directions.npz` (both force-tracked in git).\n"
)

DIR_REGISTRY_CODE = '''STEER_LAYERS = list(range(40, 80))  # last half of llama 70B's 80 layers
POSITION_SU     = 'response_last'
POSITION_REF    = -3       # exp15 winner for refusal
PER_LAYER_SIGMA = True

def _unitize(v):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)

# --- SU (steerable + readable) ---
exp6 = load_npz(EXP6_NPZ)
arrs6 = exp6  # llama bundles pca_center_dir + mm_raw in one npz
su_unit  = {l: _unitize(exp6[f'pca_center_dir__{POSITION_SU}__layer_{l:03d}']) for l in STEER_LAYERS}
su_raw   = {l: arrs6[f'mm_raw__{POSITION_SU}__layer_{l:03d}'].astype(np.float32) for l in STEER_LAYERS}
su_norms = {l: float(np.linalg.norm(su_raw[l])) for l in STEER_LAYERS}
if PER_LAYER_SIGMA:
    su_dirs  = {l: su_unit[l] * su_norms[l] for l in STEER_LAYERS}
    su_sigma = 1.0
else:
    su_dirs  = dict(su_unit)
    su_sigma = float(np.median(list(su_norms.values())))

# --- refusal (steerable + readable) ---
dirs_arrs = load_npz(DIRS_NPZ)
ref_unit  = {l: _unitize(dirs_arrs[f'refusal__mm_dir__pos_{POSITION_REF:+d}__layer_{l:03d}']) for l in STEER_LAYERS}
ref_raw   = {l: dirs_arrs[f'refusal__mm_raw__pos_{POSITION_REF:+d}__layer_{l:03d}'].astype(np.float32) for l in STEER_LAYERS}
ref_norms = {l: float(np.linalg.norm(ref_raw[l])) for l in STEER_LAYERS}
if PER_LAYER_SIGMA:
    ref_dirs  = {l: ref_unit[l] * ref_norms[l] for l in STEER_LAYERS}
    ref_sigma = 1.0
else:
    ref_dirs  = dict(ref_unit)
    ref_sigma = float(np.median(list(ref_norms.values())))

# We read out at every layer, even outside the steering window, as an upstream sanity check.
READ_LAYERS = list(range(loaded.n_layers))
ref_unit_read = ref_unit
su_unit_read  = su_unit

METHODS = {
    'SU':      {'dirs': su_dirs,  'sigma': su_sigma},
    'refusal': {'dirs': ref_dirs, 'sigma': ref_sigma},
}

def _cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-8); b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))
cos_su_ref = {l: _cos(su_unit[l], ref_unit[l]) for l in STEER_LAYERS}

print(f'PER_LAYER_SIGMA = {PER_LAYER_SIGMA}')
print(f'SU σ      = {su_sigma:.3f}   per-layer norms range [{min(su_norms.values()):.3f}, {max(su_norms.values()):.3f}]')
print(f'refusal σ = {ref_sigma:.3f}   per-layer norms range [{min(ref_norms.values()):.3f}, {max(ref_norms.values()):.3f}]')
print()
print('cos(SU, refusal) per steered layer:')
print(f'  mean={np.mean(list(cos_su_ref.values())):+.3f}  '
      f'min={min(cos_su_ref.values()):+.3f}  max={max(cos_su_ref.values()):+.3f}')
'''

DIR_REGISTRY_MD = (
    "## 4 — Build direction registry\n"
    "\n"
    "Two roles for the directions, both steerable AND readable:\n"
    "\n"
    "- **`SU/exp06_pca_center`**. Same as nb15, layers 40..79, position `response_last`.\n"
    "- **`refusal/pos-3`**. Same as nb15.\n"
    "\n"
    "We compute `cos(SU, refusal)` per layer up-front; this is the geometric-overlap baseline for interpreting cross-axis movement under steering.\n"
)

CAPTURE_MD = (
    "## 8 — Sweep: capture residuals + project onto axes\n"
    "\n"
    "Per (condition, pool, prompt, layer), record:\n"
    "\n"
    "- `refusal_proj` = `h · refusal_unit` — the headline quantity for the leverage curve.\n"
    "- `su_proj` = `h · su_unit` — sanity check that steering pushes along its own direction.\n"
    "- `h_norm` = `‖h‖` — debugging; with `NORMALIZE=True`, baseline and steered should match closely.\n"
    "\n"
    "Both projections are only available at layers 40..79 (where the unit was extracted at the matching position).\n"
)

CAPTURE_CODE = '''from tqdm.auto import tqdm

rows = []
for cond_name, method, k in tqdm(CONDITIONS, desc='conditions'):
    for pool_name, pool in POOLS.items():
        captures = forward_capture_batch(pool, method, k)
        for L in READ_LAYERS:
            H = captures[L]                                 # (N, d)
            hn = np.linalg.norm(H, axis=-1)                 # (N,)
            if L in STEER_LAYERS:
                rf = H @ ref_unit_read[L]
                su = H @ su_unit_read[L]
            else:
                rf = np.full(H.shape[0], np.nan)
                su = np.full(H.shape[0], np.nan)
            for i in range(H.shape[0]):
                rows.append({
                    'condition': cond_name,
                    'method': method or 'none',
                    'k': k,
                    'pool': pool_name,
                    'prompt_idx': i,
                    'layer': L,
                    'refusal_proj': float(rf[i]),
                    'su_proj':      float(su[i]),
                    'h_norm':       float(hn[i]),
                })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / 'projections_per_prompt.csv', index=False)
print(f'rows = {len(df)}, conditions = {df.condition.nunique()}, layers = {df.layer.nunique()}')
df.head()
'''

AGG_CODE = '''agg = (df.groupby(['condition', 'method', 'k', 'pool', 'layer'])
         [['refusal_proj', 'su_proj', 'h_norm']]
         .agg(['mean', 'std'])
         .reset_index())
agg.columns = ['_'.join(c).rstrip('_') for c in agg.columns]
agg.to_csv(OUT_DIR / 'projections_agg.csv', index=False)

# Δ vs baseline at the (pool, layer) level — the headline quantity.
base = (df[df.condition == 'baseline']
        .groupby(['pool', 'layer'])
        [['refusal_proj', 'su_proj']]
        .mean()
        .rename(columns={'refusal_proj': 'refusal_proj_base',
                          'su_proj': 'su_proj_base'})
        .reset_index())
delta = df.merge(base, on=['pool', 'layer'])
delta['refusal_proj_delta'] = delta['refusal_proj'] - delta['refusal_proj_base']
delta['su_proj_delta']      = delta['su_proj']      - delta['su_proj_base']

delta_agg = (delta.groupby(['condition', 'method', 'k', 'pool', 'layer'])
                  [['refusal_proj_delta', 'su_proj_delta']]
                  .agg(['mean', 'std'])
                  .reset_index())
delta_agg.columns = ['_'.join(c).rstrip('_') for c in delta_agg.columns]
delta_agg.to_csv(OUT_DIR / 'projections_delta_vs_baseline.csv', index=False)
print(f'agg rows = {len(agg)}; delta rows = {len(delta_agg)}')
agg.head()
'''

PLOTS_MD = (
    "## 10 — Plots\n"
    "\n"
    "Three panels, all on the harmful pool (steered layers L40..79):\n"
    "\n"
    "1. **Refusal-axis Δ vs baseline per layer** — the headline plot. Per-condition curves of `Δ(h · refusal_unit)`. The vertical separation between SU curves and the refusal-direction curves at matched compliance is the 'less feature movement on the refusal axis' claim.\n"
    "2. **SU-axis Δ vs baseline per layer** — sanity check that SU steering does push along its own direction; calibrates the leverage normalisation.\n"
    "3. **Refusal-axis Δ, neutral pool** — content control. If the refusal axis only moves on harmful prompts under SU, the SU push is content-gated rather than a global feature drift.\n"
)

PLOTS_CODE = '''import matplotlib.pyplot as plt

FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

ORDER = [c for c, _, _ in CONDITIONS]
COLOURS = {
    'baseline':       '#444444',
    'SU −0.5σ':       '#1f77b4',
    'SU −0.65σ':      '#ff7f0e',
    'SU −0.7σ':       '#d62728',
    'SU −0.85σ':      '#9467bd',
    'refusal −0.4σ':  '#2ca02c',
    'refusal +0.3σ':  '#17becf',
}

def _plot_per_layer(values, ylabel, title, savepath, layers=READ_LAYERS, hl_layers=STEER_LAYERS):
    fig, ax = plt.subplots(figsize=(10, 5))
    for cond in ORDER:
        ys = [values.get((cond, L), float('nan')) for L in layers]
        ax.plot(layers, ys, marker='o', ms=3, label=cond, color=COLOURS.get(cond))
    ax.axvspan(min(hl_layers) - 0.5, max(hl_layers) + 0.5, color='#fff5d8', alpha=0.5, zorder=-1)
    ax.set_xlabel('layer'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8, loc='best'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(savepath, dpi=130); plt.show()

def _table(metric_col, agg_df, pool):
    sub = agg_df[agg_df.pool == pool]
    return {(r['condition'], int(r['layer'])): float(r[metric_col]) for _, r in sub.iterrows()}

# Plot 1 — refusal-axis Δ, harmful pool (HEADLINE)
_plot_per_layer(
    values=_table('refusal_proj_delta_mean', delta_agg, 'harmful'),
    ylabel='Δ (h · refusal_unit)  vs baseline',
    title='Refusal-axis shift under steering — harmful prompts (HEADLINE)',
    savepath=FIG_DIR / 'refusal_axis_delta_harmful.png',
    layers=STEER_LAYERS,
)

# Plot 2 — SU-axis Δ, harmful pool (sanity)
_plot_per_layer(
    values=_table('su_proj_delta_mean', delta_agg, 'harmful'),
    ylabel='Δ (h · su_unit)  vs baseline',
    title='SU-axis shift under steering — harmful prompts (sanity)',
    savepath=FIG_DIR / 'su_axis_delta_harmful.png',
    layers=STEER_LAYERS,
)

# Plot 3 — refusal-axis Δ, neutral pool (content control)
_plot_per_layer(
    values=_table('refusal_proj_delta_mean', delta_agg, 'neutral'),
    ylabel='Δ (h · refusal_unit)  vs baseline',
    title='Refusal-axis shift under steering — NEUTRAL prompts (content control)',
    savepath=FIG_DIR / 'refusal_axis_delta_neutral.png',
    layers=STEER_LAYERS,
)
'''

MANIFEST_CODE = '''manifest = {
    'experiment': '16_su_refusal_axis_readout',
    'model': loaded.cfg.hf_id,
    'steer_layers': STEER_LAYERS,
    'read_layers': READ_LAYERS,
    'positions': {'SU': POSITION_SU, 'refusal': POSITION_REF,
                   'read_position': 'last_prompt_token'},
    'sigma': {'SU': float(su_sigma), 'refusal': float(ref_sigma)},
    'per_layer_sigma': PER_LAYER_SIGMA,
    'normalize': NORMALIZE,
    'cos_per_layer': {
        'SU_ref': cos_su_ref,
    },
    'conditions': [{'name': n, 'method': m, 'k': k} for n, m, k in CONDITIONS],
    'pools': {name: len(pool) for name, pool in POOLS.items()},
}
(OUT_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print('wrote', OUT_DIR / 'manifest.json')
print('outputs:')
for p in sorted(OUT_DIR.rglob('*')):
    if p.is_file():
        print(' ', p.relative_to(OUT_DIR))
'''

WHAT_TO_LOOK_FOR_MD = (
    "## What to look for\n"
    "\n"
    "**Headline claim — 'less feature movement on the refusal axis'.** At matched behavioural compliance (e.g. `SU −0.7σ` ≈ 100% comply vs `refusal −0.4σ` ≈ 87% comply), the refusal-axis Δ should be substantially *smaller* under SU steering than under refusal-direction steering. Quantified by §13's slope-of-feature-shift-per-σ — SU's slope on the refusal axis should be flatter than the refusal-direction sweep's slope.\n"
    "\n"
    "**Sanity:** SU steering pushes along the SU axis (Plot 2 should show clean monotonic shift with k); refusal steering doesn't push on SU (cos is small).\n"
    "\n"
    "**Content gating:** if Plot 3 (refusal axis on neutral prompts) is much flatter than Plot 1 (harmful prompts), the SU push is content-gated — only routes through the refusal feature when the prompt is harmful. Surprising result if the pattern is symmetric.\n"
    "\n"
    "**Surprises to flag:**\n"
    "- Refusal-axis Δ shifts at layers *upstream* of the steered range (L<40). Measurement artefact — no causal pathway upstream of the hook.\n"
    "- `SU −0.5σ` (no jailbreak in nb15) already shows large refusal-axis shift. Means the refusal feature moves freely below the behavioural threshold — feature movement is necessary but not sufficient for compliance.\n"
    "- `refusal −0.4σ` jailbreaks in nb15 but with *less* refusal-axis Δ than `SU −0.7σ`. Would invert the headline claim — flag immediately.\n"
)

MIDSTACK_MD = (
    "## 12 — Mid-stack focus\n"
    "\n"
    "Zoomed view of refusal-axis Δ at L60..72 (mid + late stack), dropping the L73-79 amplification tail so the y-axis isn't dominated by the late-stack spike. This is the slice §13 averages over for the leverage-curve summary.\n"
)

MIDSTACK_CODE = '''MID_LAYERS = list(range(40, 73))  # mid + late stack on llama 70B; drops L73-79 amplification tail

# Refusal-axis Δ, harmful pool, MID-STACK ONLY
_plot_per_layer(
    values=_table('refusal_proj_delta_mean', delta_agg, 'harmful'),
    ylabel='Δ (h · refusal_unit)  vs baseline',
    title='Refusal-axis shift — harmful prompts (mid-stack L40-72)',
    savepath=FIG_DIR / 'refusal_axis_delta_harmful_midstack.png',
    layers=MID_LAYERS,
    hl_layers=MID_LAYERS,
)

# Refusal-axis Δ, neutral pool, MID-STACK ONLY (content control)
_plot_per_layer(
    values=_table('refusal_proj_delta_mean', delta_agg, 'neutral'),
    ylabel='Δ (h · refusal_unit)  vs baseline',
    title='Refusal-axis shift — neutral prompts (mid-stack L40-72)',
    savepath=FIG_DIR / 'refusal_axis_delta_neutral_midstack.png',
    layers=MID_LAYERS,
    hl_layers=MID_LAYERS,
)

# Summary table at L66 (representative mid-stack layer) for the writeup
print('\\nSummary at L66 (representative mid-stack):')
print(f'{"condition":<16s}  {"refusal_Δ_harm":>14s}  {"refusal_Δ_neut":>14s}  '
      f'{"content_gated":>13s}  {"su_Δ_harm":>10s}')
for cond, _, _ in CONDITIONS:
    rh = _table('refusal_proj_delta_mean', delta_agg, 'harmful').get((cond, 66), float('nan'))
    rn = _table('refusal_proj_delta_mean', delta_agg, 'neutral').get((cond, 66), float('nan'))
    sh = _table('su_proj_delta_mean',      delta_agg, 'harmful').get((cond, 66), float('nan'))
    print(f'{cond:<16s}  {rh:>+14.3f}  {rn:>+14.3f}  {rh-rn:>+13.3f}  {sh:>+10.3f}')
'''

LEVERAGE_MD = (
    "## 13 — Per-σ leverage curves\n"
    "\n"
    "For each method, plot mid-stack mean refusal-axis Δ vs steering coefficient `k`, with nb15 compliance overlaid as marker size. The slope of `refusal_Δ` vs `k` quantifies the **per-σ leverage** of each attack on the refusal feature. The slope ratio (refusal-method ÷ SU-method) is the headline 'less feature movement per unit jailbreak' number for the paper.\n"
    "\n"
    "Compliance numbers below are still pasted from Qwen exp15 — replace with the Llama numbers from `exp15_jailbreak_steering_llama33_70b/expB_compliance_summary.csv` after running notebook 15 on Llama.\n"
)

LEVERAGE_CODE = '''# Compliance from exp15 (PLACEHOLDER: Qwen numbers — replace with llama numbers post-nb15 run)
EXP15_COMPLIANCE = {
    'baseline':       0.00,
    'SU −0.5σ':       0.00,
    'SU −0.65σ':      0.80,
    'SU −0.7σ':       1.00,
    'SU −0.85σ':      0.90,
    'refusal −0.4σ':  0.87,
    'refusal +0.3σ':  0.00,
}

# Mid-stack mean feature shift per condition (avg over L60..72 — past the steering-injection layers,
# before the L73-79 amplification kicks in)
LEVERAGE_LAYERS = list(range(60, 73))  # mid-stack on llama 70B (analogue of Qwen 4B L22-28)

def _midstack_mean(metric_col, agg_df, pool, condition):
    sub = agg_df[(agg_df.pool == pool) & (agg_df.condition == condition)]
    sub = sub[sub.layer.isin(LEVERAGE_LAYERS)]
    return float(sub[metric_col].mean())

# Build per-condition summary table
import math
summary = []
for cond, method, k in CONDITIONS:
    summary.append({
        'condition': cond,
        'method':    method or 'baseline',
        'k':         k,
        'refusal_Δ_mid':  _midstack_mean('refusal_proj_delta_mean', delta_agg, 'harmful', cond),
        'su_Δ_mid':       _midstack_mean('su_proj_delta_mean',      delta_agg, 'harmful', cond),
        'compliance':     EXP15_COMPLIANCE.get(cond, math.nan),
    })
summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_DIR / 'leverage_summary.csv', index=False)
print('Mid-stack mean (L60–72):')
print(summary_df.to_string(index=False))

# --- Leverage plot: refusal-axis Δ vs k, separated by method ---
fig, ax = plt.subplots(figsize=(8, 5.5))

for method, marker, colour in [('SU', 'o', '#d62728'), ('refusal', 's', '#2ca02c')]:
    sub = summary_df[summary_df.method == method].sort_values('k')
    if len(sub) == 0: continue
    sizes = 20 + 380 * sub['compliance'].fillna(0).values
    ax.scatter(sub.k, sub['refusal_Δ_mid'], s=sizes, alpha=0.75, marker=marker,
               color=colour, edgecolor='black', linewidth=0.5,
               label=f'{method}  (size = compliance)')
    ax.plot(sub.k, sub['refusal_Δ_mid'], '--', color=colour, alpha=0.4, lw=1)
    if len(sub) >= 2:
        slope, intercept = np.polyfit(sub.k.values, sub['refusal_Δ_mid'].values, 1)
        xs = np.linspace(sub.k.min(), sub.k.max(), 20)
        ax.plot(xs, slope * xs + intercept, '-', color=colour, lw=1.2, alpha=0.6)
        ax.text(sub.k.iloc[-1], sub['refusal_Δ_mid'].iloc[-1],
                f'  slope={slope:+.2f}', color=colour, fontsize=9, va='center')

base_row = summary_df[summary_df.method == 'baseline']
if len(base_row):
    ax.scatter(base_row.k, base_row['refusal_Δ_mid'], s=40, marker='x', color='black', label='baseline')
ax.axhline(0, color='#888', lw=0.5); ax.axvline(0, color='#888', lw=0.5)
ax.set_xlabel('steering coefficient k  (× σ)')
ax.set_ylabel('Δ (h · refusal_unit)  mid-stack mean')
ax.set_title('Refusal-axis leverage: feature shift per σ (Llama 3.3 70B)')
ax.legend(fontsize=9, loc='best'); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / 'leverage_curves.png', dpi=130)
plt.show()

# Slope ratios — quantify the leverage advantage
print('\\nLeverage (slope of refusal-axis shift per unit k):')
slopes = {}
for method in ['SU', 'refusal']:
    sub = summary_df[summary_df.method == method].sort_values('k')
    if len(sub) >= 2:
        slope, _ = np.polyfit(sub.k.values, sub['refusal_Δ_mid'].values, 1)
        slopes[method] = slope
        print(f'  {method:<8s}  slope = {slope:+.3f}  per σ')

if 'SU' in slopes and 'refusal' in slopes and slopes['SU'] != 0:
    ratio = abs(slopes['refusal'] / slopes['SU'])
    print(f'\\n  refusal / SU slope ratio = {ratio:.2f}x  '
          f'(refusal-direction steering moves the refusal feature {ratio:.1f}× more per σ than SU)')

# Compliance-per-feature efficiency: at the jailbreak peak, how much refusal_Δ does
# each method spend per percentage point of compliance gained?
print('\\nFeature-spend efficiency at jailbreak peaks:')
for cond in ['SU −0.7σ', 'refusal −0.4σ']:
    sub = summary_df[summary_df.condition == cond]
    if len(sub) == 0: continue
    row = sub.iloc[0]
    if row.compliance > 0:
        eff = row['refusal_Δ_mid'] / row.compliance
        print(f'  {cond:<16s}  refusal_Δ = {row["refusal_Δ_mid"]:+.2f}  '
              f'compliance = {row.compliance:.2%}  '
              f'(refusal_Δ per unit compliance = {eff:+.2f})')
'''


# ----- content-fingerprint rewrites -----------------------------------------
# Match each target cell by a unique substring of its current source. Each entry:
#   (kind, fingerprint_substring, new_text)
# Order matters for deletes (which we apply last in reverse-index order).

PROMPT_SET_MD = (
    "## 7 — Prompt set\n"
    "\n"
    "Two pools:\n"
    "\n"
    "- **harmful**: AdvBench-30 (same as nb15).\n"
    "- **neutral**: a small held-out set of benign requests. Controls for whether SU shifts the refusal axis on *any* input or only on harmful prompts (content gating).\n"
    "\n"
    "Both go through the same generic helpful-assistant system prompt (matches nb15 expB).\n"
)

REWRITES = [
    ('markdown', '# 16 — SU steering\'s effect on the eval-aware axis',           TITLE_MD),
    ('markdown', '## 7 — Prompt set',                                              PROMPT_SET_MD),
    ('markdown', 'Llama 3.3 70B port note.',                                       PORT_BANNER_MD),
    ('markdown', '## 4 — Build direction registry',                                DIR_REGISTRY_MD),
    ('code',     'STEER_LAYERS = list(range(40, 80))',                              DIR_REGISTRY_CODE),
    ('markdown', '## 8 — Sweep: capture residuals',                                CAPTURE_MD),
    ('code',     'rows = []\nfor cond_name, method, k in tqdm',                    CAPTURE_CODE),
    ('code',     "agg = (df.groupby(['condition', 'method', 'k', 'pool'",           AGG_CODE),
    ('markdown', '## 10 — Plots',                                                   PLOTS_MD),
    ('code',     'import matplotlib.pyplot as plt\n\nFIG_DIR',                      PLOTS_CODE),
    ('code',     "manifest = {\n    'experiment': '16_su_eval_axis_readout'",      MANIFEST_CODE),
    ('markdown', '## What to look for',                                             WHAT_TO_LOOK_FOR_MD),
    ('markdown', '## 12 — Mid-stack focus',                                         MIDSTACK_MD),
    ('code',     'MID_LAYERS = list(range(40, 73))',                                MIDSTACK_CODE),
    ('markdown', '## 13 — Per-σ leverage curves',                                   LEVERAGE_MD),
    ('code',     '# Compliance from exp15',                                         LEVERAGE_CODE),
]


def find_one(nb, kind, fp):
    matches = [(i, c) for i, c in enumerate(nb['cells'])
               if c.get('cell_type') == kind and fp in _src(c)]
    if len(matches) == 0:
        raise RuntimeError(f'no {kind} cell matches fingerprint: {fp!r}')
    if len(matches) > 1:
        raise RuntimeError(f'fingerprint {fp!r} ambiguous, matches {len(matches)} cells')
    return matches[0]


def main():
    nb = json.loads(NB.read_text())

    # Idempotence check: look for the marker in any cell.
    for c in nb['cells']:
        if STRIPPED_MARKER in _src(c):
            print(f'  {NB.name}: already stripped (marker {STRIPPED_MARKER!r}); skipping')
            return

    n_rewritten = 0
    for kind, fp, new_text in REWRITES:
        i, c = find_one(nb, kind, fp)
        _set_src(c, new_text)
        if kind == 'code':
            c['outputs'] = []
            c['execution_count'] = None
        n_rewritten += 1

    NB.write_text(json.dumps(nb, indent=1) + '\n')
    print(f'  {NB.name}: rewrote {n_rewritten} cells, eval-awareness stripped')


if __name__ == '__main__':
    main()
