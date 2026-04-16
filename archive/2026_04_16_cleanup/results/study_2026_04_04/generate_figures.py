#!/usr/bin/env python3
"""Generate publication-quality figures for Meta-CoT Control-V5 study report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
    'figure.constrained_layout.use': True,
})

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9',
          '#E69F00', '#000000', '#882255', '#44AA99', '#332288']

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Data from control_v5_eval_readout_2026_04_04.md (n=90 pilot)
# ============================================================
models = [
    'base_sft', 'all_sft', 'verify_sft', 'redirect_sft',
    'E3', 'E5', 'E8', 'E9', 'E9b', 'E9c', 'E10'
]
short_labels = [
    'Base\nSFT', 'All\nSFT', 'Verify\nSFT', 'Redirect\nSFT',
    'E3', 'E5', 'E8', 'E9', 'E9b', 'E9c', 'E10'
]

acc       = [42.2, 33.3, 36.7, 35.6, 30.0, 40.0, 38.9, 41.1, 40.0, 36.7, 35.6]
aime_acc  = [13.3, 6.7,  3.3, 10.0,  3.3,  3.3, 10.0,  6.7,  6.7,  6.7,  6.7]
conf_cov  = [0.0, 88.9, 82.2, 48.9, 100.0, 1.1, 23.3, 91.1, 0.0,  7.8,  5.6]
ece       = [None, 0.398, 0.492, 0.160, 0.515, 0.790, 0.322, 0.301, None, 0.264, 0.340]
wrong_hi  = [None, 51.7, 80.7, 0.0, 71.4, 0.0, 3.6, 50.9, 0.0, 1.8, 0.0]

# Behavior rates (from summary_evalnode_v4.md)
verify_rate   = [0.0, 16.7, 13.3, 0.0, 25.6, 0.0, 0.0, 67.8, 0.0, 0.0, 0.0]
redirect_rate = [0.0, 25.6, 0.0, 0.0, 23.3, 1.1, 0.0, 23.3, 0.0, 0.0, 0.0]
diagnosis_rate= [4.4, 18.9, 4.4, 0.0,  8.9, 5.6, 0.0, 23.3, 0.0, 0.0, 0.0]

# AIME-specific behavior (from summary per benchmark)
aime_verify   = [0.0, 26.7, 26.7, 0.0, 23.3, 0.0, 0.0, 23.3, 0.0, 0.0, 0.0]
aime_redirect = [0.0, 50.0, 0.0,  0.0, 40.0, 3.3, 0.0, 53.3, 0.0, 0.0, 0.0]
aime_diagnosis= [10.0,36.7, 10.0, 0.0, 16.7, 13.3,0.0, 56.7, 0.0, 0.0, 0.0]

# Failure mode counts (from failure_analysis_2026_04_04.json)
failure_modes = {
    'E3':  {'overconfident_verify_failed': 45, 'reasoning_error_after_control': 14, 'diagnosis_without_recovery': 4, 'no_meta_signal': 0},
    'E5':  {'no_meta_signal': 54, 'overconfident_verify_failed': 0, 'reasoning_error_after_control': 0, 'diagnosis_without_recovery': 0},
    'E8':  {'no_meta_signal': 36, 'single_redirect_without_recovery': 13, 'diagnosis_without_recovery': 4, 'late_meta_append': 2},
    'E9':  {'overconfident_verify_failed': 15, 'single_redirect_without_recovery': 13, 'no_meta_signal': 7, 'late_meta_append': 6, 'single_verify_without_correction': 6, 'diagnosis_without_recovery': 4, 'reasoning_error_after_control': 1, 'single_intervention_only': 1},
    'E10': {'no_meta_signal': 53, 'single_redirect_without_recovery': 4, 'diagnosis_without_recovery': 1},
}

# ============================================================
# Figure 1: Accuracy vs ECE scatter
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for i, m in enumerate(models):
    if ece[i] is not None:
        marker = 'o' if 'sft' not in models[i].lower() else 's'
        size = 120 if models[i] == 'base_sft' else 80
        ax.scatter(ece[i], acc[i], c=COLORS[i], s=size, marker=marker,
                   edgecolors='black', linewidths=0.5, zorder=5)
        offset_x = 0.015
        offset_y = 0.5
        if models[i] == 'E9':
            offset_y = -1.5
        elif models[i] == 'E8':
            offset_x = -0.04
        ax.annotate(models[i].replace('qwen3_metacot_control_v5_', ''),
                    (ece[i], acc[i]),
                    xytext=(ece[i]+offset_x, acc[i]+offset_y),
                    fontsize=9, ha='left')

ax.set_xlabel('ECE (Expected Calibration Error) ↓')
ax.set_ylabel('Accuracy (%) ↑')
ax.set_title('Figure 1: Accuracy–Calibration Trade-off (n=90 pilot)')
ax.axhline(y=42.2, color='gray', linestyle='--', alpha=0.5, label='Base SFT accuracy')
ax.legend(loc='lower left')
ax.set_xlim(0.1, 0.85)
ax.set_ylim(28, 45)
fig.savefig(os.path.join(OUT_DIR, 'fig1_accuracy_vs_ece.png'), bbox_inches='tight')
plt.close(fig)
print("Figure 1 saved: fig1_accuracy_vs_ece.png")

# ============================================================
# Figure 2: Confidence Coverage vs Wrong-High-Conf
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for i, m in enumerate(models):
    if wrong_hi[i] is not None and conf_cov[i] > 0:
        ax.scatter(conf_cov[i], wrong_hi[i], c=COLORS[i], s=100,
                   edgecolors='black', linewidths=0.5, zorder=5)
        ax.annotate(models[i], (conf_cov[i], wrong_hi[i]),
                    xytext=(conf_cov[i]+2, wrong_hi[i]+2),
                    fontsize=9)

# Mark the "escape zone" (low coverage)
ax.axvspan(0, 15, alpha=0.08, color='red', label='Reward escape zone\n(coverage < 15%)')
ax.axhline(y=10, color='green', linestyle='--', alpha=0.4, label='Target: wrong_hi < 10%')

ax.set_xlabel('Confidence Coverage (%) →')
ax.set_ylabel('Wrong High-Confidence Rate (%) ↓')
ax.set_title('Figure 2: Coverage–Overconfidence Trade-off')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(-5, 110)
ax.set_ylim(-5, 90)
fig.savefig(os.path.join(OUT_DIR, 'fig2_coverage_vs_wrong_hi.png'), bbox_inches='tight')
plt.close(fig)
print("Figure 2 saved: fig2_coverage_vs_wrong_hi.png")

# ============================================================
# Figure 3: Failure Mode Distribution (stacked bar)
# ============================================================
all_modes = ['no_meta_signal', 'overconfident_verify_failed',
             'single_redirect_without_recovery', 'diagnosis_without_recovery',
             'late_meta_append', 'single_verify_without_correction',
             'reasoning_error_after_control', 'single_intervention_only']
mode_labels = ['No meta signal', 'Overconf verify failed',
               'Redirect w/o recovery', 'Diagnosis w/o recovery',
               'Late meta append', 'Verify w/o correction',
               'Reasoning error', 'Single intervention']

fm_models = ['E3', 'E5', 'E8', 'E9', 'E10']
data_matrix = []
for mode in all_modes:
    row = [failure_modes[m].get(mode, 0) for m in fm_models]
    data_matrix.append(row)
data_matrix = np.array(data_matrix)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(fm_models))
width = 0.6
bottom = np.zeros(len(fm_models))

mode_colors = ['#cccccc', '#d62728', '#ff7f0e', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

for j, (mode_label, color) in enumerate(zip(mode_labels, mode_colors)):
    vals = data_matrix[j]
    if vals.sum() > 0:
        ax.bar(x, vals, width, label=mode_label, bottom=bottom, color=color, edgecolor='white', linewidth=0.5)
        bottom += vals

ax.set_xlabel('Experiment')
ax.set_ylabel('Number of Wrong Cases')
ax.set_title('Figure 3: Failure Mode Distribution by RL Experiment (n=90)')
ax.set_xticks(x)
ax.set_xticklabels(fm_models)
ax.legend(loc='upper right', fontsize=8, ncol=2)
fig.savefig(os.path.join(OUT_DIR, 'fig3_failure_modes.png'), bbox_inches='tight')
plt.close(fig)
print("Figure 3 saved: fig3_failure_modes.png")

# ============================================================
# Figure 4: AIME Behavior Breakdown (grouped bar)
# ============================================================
aime_models_idx = [0, 1, 4, 7]  # base_sft, all_sft, E3, E9
aime_labels = [models[i] for i in aime_models_idx]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(aime_labels))
w = 0.2

bars_v = [aime_verify[i] for i in aime_models_idx]
bars_r = [aime_redirect[i] for i in aime_models_idx]
bars_d = [aime_diagnosis[i] for i in aime_models_idx]

ax.bar(x - w, bars_v, w, label='Verify rate (%)', color=COLORS[0])
ax.bar(x,     bars_r, w, label='Redirect rate (%)', color=COLORS[1])
ax.bar(x + w, bars_d, w, label='Diagnosis rate (%)', color=COLORS[2])

ax.set_xlabel('Model')
ax.set_ylabel('Rate (%)')
ax.set_title('Figure 4: AIME-2024 Metacognitive Behavior Rates')
ax.set_xticks(x)
ax.set_xticklabels(aime_labels)
ax.legend()
ax.set_ylim(0, 65)
fig.savefig(os.path.join(OUT_DIR, 'fig4_aime_behavior.png'), bbox_inches='tight')
plt.close(fig)
print("Figure 4 saved: fig4_aime_behavior.png")

# ============================================================
# Figure 5: Multi-panel summary (2x2)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Panel A: Accuracy by benchmark
ax = axes[0, 0]
gsm = [80.0, 63.3, 66.7, 63.3, 63.3, 80.0, 80.0, 90.0, 83.3, 73.3, 66.7]
math_ = [33.3, 30.0, 40.0, 33.3, 23.3, 36.7, 26.7, 26.7, 30.0, 30.0, 33.3]
x = np.arange(len(models))
ax.bar(x - 0.2, gsm, 0.35, label='GSM8K', color=COLORS[0], alpha=0.8)
ax.bar(x + 0.2, math_, 0.35, label='MATH-500', color=COLORS[1], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_sft','').replace('control_v5_','') for m in models], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Accuracy (%)')
ax.set_title('(A) Per-Benchmark Accuracy')
ax.legend(fontsize=8)

# Panel B: Overall accuracy ranking
ax = axes[0, 1]
sorted_idx = np.argsort(acc)[::-1]
colors_sorted = [COLORS[i] for i in sorted_idx]
ax.barh(range(len(models)), [acc[i] for i in sorted_idx],
        color=colors_sorted, edgecolor='black', linewidth=0.3)
ax.set_yticks(range(len(models)))
ax.set_yticklabels([models[i] for i in sorted_idx], fontsize=9)
ax.set_xlabel('Overall Accuracy (%)')
ax.set_title('(B) Accuracy Ranking')
ax.axvline(x=42.2, color='red', linestyle='--', alpha=0.5, label='Base SFT')
ax.legend(fontsize=8)

# Panel C: ECE comparison (valid models only)
ax = axes[1, 0]
valid = [(m, e) for m, e in zip(models, ece) if e is not None]
valid_sorted = sorted(valid, key=lambda x: x[1])
vm, ve = zip(*valid_sorted)
ax.barh(range(len(vm)), ve, color=COLORS[2], edgecolor='black', linewidth=0.3)
ax.set_yticks(range(len(vm)))
ax.set_yticklabels(vm, fontsize=9)
ax.set_xlabel('ECE ↓')
ax.set_title('(C) Calibration (ECE)')

# Panel D: Confidence coverage
ax = axes[1, 1]
valid_cc = [(m, c) for m, c in zip(models, conf_cov) if c > 0]
if valid_cc:
    vc_models, vc_vals = zip(*sorted(valid_cc, key=lambda x: x[1], reverse=True))
    ax.barh(range(len(vc_models)), vc_vals, color=COLORS[3], edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(len(vc_models)))
    ax.set_yticklabels(vc_models, fontsize=9)
ax.set_xlabel('Confidence Coverage (%)')
ax.set_title('(D) Meta Emission Coverage')
ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% floor')
ax.legend(fontsize=8)

fig.suptitle('Meta-CoT Control-V5: Multi-Panel Summary (n=90 pilot)', fontsize=14, y=1.02)
fig.savefig(os.path.join(OUT_DIR, 'fig5_multipanel_summary.png'), bbox_inches='tight')
plt.close(fig)
print("Figure 5 saved: fig5_multipanel_summary.png")

print("\nAll figures generated successfully.")
