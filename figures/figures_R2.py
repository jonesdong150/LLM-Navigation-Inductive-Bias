import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ================= IJCAI style config (space-saving version) =================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.3,
    "lines.markersize": 4.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "savefig.bbox": "tight"
})
# =================================================

def draw_r2_navigation_plots():
    # More compact: shrinking height is more critical for double-column figure*
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.05))

    g_labels = ['G1', 'G2', 'G3', 'G4', 'G5']

    # --- (1) Overhead vs. Gain ---
    friendly_delta = [8.0, 4.5, -3.2, -12.5, -22.0]
    feasible_delta = [2.2, 1.8, 0.5, -0.6, 1.0]
    closed_delta   = [0.1, 0.0, -0.2, 0.1, 0.0]

    axes[0].plot(g_labels, friendly_delta, marker='o', color='#D35400',
                 label='Friendly (<=4B)')
    axes[0].plot(g_labels, feasible_delta, marker='s', color='#2980B9',
                 label='Feasible (8-32B)')
    axes[0].plot(g_labels, closed_delta, marker='x', linestyle='--', color='#2C3E50',
                 label='Closed (>=100B)', alpha=0.75)

    axes[0].axhline(0, color='gray', linestyle='-', linewidth=0.7, alpha=0.5)
    axes[0].set_title('(1) Overhead vs. Gain', fontweight='normal', pad=2)
    axes[0].set_ylabel(r'$\Delta$Acc (Hier - Flat) (%)', labelpad=1.5)

    # Compact legend
    leg0 = axes[0].legend(frameon=False, loc='lower left',
                          handlelength=1.6, handletextpad=0.5,
                          labelspacing=0.25, borderpad=0.2)
    axes[0].tick_params(axis='both', which='major', pad=1.2, length=3)

    # --- (2) Compression Resilience ---
    comp_x = ['100%', '50%', '25%']
    acc_by_g = {
        'G1': [88, 83, 74], 'G2': [82, 77, 68],
        'G3': [76, 62, 45], 'G4': [68, 42, 18], 'G5': [62, 34, 10]
    }

    low_color, high_color = '#27AE60', '#E67E22'

    for g in g_labels:
        y = acc_by_g[g]
        c = low_color if g in ['G1', 'G2'] else high_color
        ls = '-' if g in ['G1', 'G3', 'G5'] else '--'
        # Only show two representative legend entries to avoid clutter
        lab = 'Low Cx' if g == 'G1' else ('High Cx' if g == 'G5' else None)
        axes[1].plot(comp_x, y, marker='.', linestyle=ls, color=c, label=lab, alpha=0.9)

    axes[1].set_title('(2) Compression Resilience', fontweight='normal', pad=2)
    axes[1].set_ylabel('Accuracy (%)', labelpad=1.5)

    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys(),
                   frameon=False, loc='upper right',
                   handlelength=1.4, handletextpad=0.4,
                   labelspacing=0.25, borderpad=0.2)

    axes[1].tick_params(axis='both', which='major', pad=1.2, length=3)
    axes[1].set_ylim(0, 100)

    # --- (3) Task Specialization ---
    tasks = ['Obj', 'Geo', 'Topo', 'Rch', 'Gen']
    hier_task = [0.45, 0.55, 0.65, 0.60, 0.40]
    flat_task = [0.70, 0.72, 0.50, 0.45, 0.35]

    xx = np.arange(len(tasks))
    w = 0.34

    axes[2].bar(xx - w/2, hier_task, w, label='Hier',
                color='#2980B9', alpha=0.78)
    axes[2].bar(xx + w/2, flat_task, w, label='Flat',
                color='#AED6F1', edgecolor='#2980B9', linewidth=0.7)

    axes[2].set_xticks(xx)
    axes[2].set_xticklabels(tasks)
    axes[2].set_title('(3) Task Specialization', fontweight='normal', pad=2)
    axes[2].set_ylabel('Score', labelpad=1.5)

    axes[2].legend(frameon=False, loc='upper right',
                   handlelength=1.4, handletextpad=0.4,
                   labelspacing=0.25, borderpad=0.2)

    axes[2].grid(axis='y', linestyle='--', alpha=0.25)
    axes[2].tick_params(axis='both', which='major', pad=1.2, length=3)
    axes[2].set_ylim(0, 1.0)

    # Further compress whitespace
    plt.tight_layout(pad=0.25, w_pad=0.45)

    BASE_DIR = Path(__file__).parent.parent
    OUT_DIR = BASE_DIR / "figures" / "figures_R2"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = OUT_DIR / "figures_R2.pdf"
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Figure R2 generated: {out_path}")
    plt.show()

draw_r2_navigation_plots()
