import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ================= IJCAI style config (space-saving version) =================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.4,
    "lines.markersize": 5.5,
    "mathtext.fontset": "stix",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.axisbelow": True
})

# ================= Data (aggregated means by parameter scale) =================
data = {
    'Scale': ['0.6B', '4B', '8B', '32B', '>100B'],
    'Flat': [73.2, 78.5, 84.1, 89.5, 95.8],
    'Hierarchical': [78.5, 82.1, 81.3, 88.2, 94.5],
    'Clustered': [68.4, 65.2, 58.6, 52.4, 48.2]
}
df = pd.DataFrame(data)

# ================= Canvas (further compressed) =================
# Compared to the original (3.3, 2.6), further shrunk for IJCAI single-column compact layout
fig, ax = plt.subplots(figsize=(3.05, 2.15))

# Keep original colors (don't change if your paper uses a global palette)
colors = ['#333333', '#2E86C1', '#C0392B']

x = np.arange(len(df['Scale']))

# ================= Draw trend lines =================
ax.plot(x, df['Flat'], marker='o', color=colors[0],
        label='Flat (Baseline)', zorder=3)

ax.plot(x, df['Hierarchical'], marker='s', color=colors[1],
        label='Hierarchical', linestyle='--', zorder=4)

ax.plot(x, df['Clustered'], marker='^', color=colors[2],
        label='Clustered', linestyle=':', zorder=2)

# ================= Key annotation: Crossover (smaller, more restrained) =================
# Original xytext was too tall; place text closer after shrinking
ax.annotate(
    'Crossover',
    xy=(2, 81.3),                 # 8B hier point
    xytext=(1.15, 86.0),          # closer, less space-consuming
    arrowprops=dict(
        arrowstyle='->',
        lw=0.8,
        color='black',
        shrinkA=2, shrinkB=2
    ),
    fontsize=7,
    fontweight='normal'
)

# ================= Detail refinements =================
ax.set_xticks(x)
ax.set_xticklabels(df['Scale'])

ax.set_xlabel('Model Scale (Parameters)', labelpad=2)
ax.set_ylabel('Average Accuracy (%)', labelpad=2)

# Title: not bold + smaller + smaller pad
ax.set_title('Scaling Law of Linguistic Inductive Bias',
             fontweight='normal', pad=4)

# Y-axis range consistent with original
ax.set_ylim(40, 105)

# More compact ticks to reduce whitespace
ax.tick_params(axis='both', which='major', pad=1.5, length=3)

# Legend: smaller, no exaggerated border, space-efficient
leg = ax.legend(
    frameon=True,
    loc='lower left',
    borderpad=0.25,
    handlelength=1.6,
    handletextpad=0.5,
    labelspacing=0.25
)
leg.get_frame().set_linewidth(0.6)
leg.get_frame().set_alpha(0.9)

# Tighter layout (tight_layout + bbox_inches=tight double insurance)
plt.tight_layout(pad=0.15)

# Save
BASE_DIR = Path(__file__).parent.parent
OUT_DIR = BASE_DIR / "figures" / "figures_R1"
os.makedirs(OUT_DIR, exist_ok=True)
out_path = OUT_DIR / "figR1_structure_scaling.pdf"
plt.savefig(out_path, format='pdf', bbox_inches='tight')

plt.show()
