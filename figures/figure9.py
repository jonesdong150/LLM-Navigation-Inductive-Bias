import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "final_stats" / "r3_dimension_conflict.csv"
OUT_DIR = "figures9"

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)

# -------- Normalize model names (optional but recommended) --------
def norm_model(m: str) -> str:
    s = str(m).strip()
    # Normalize common model name variants
    if s.lower() == "gpt-5.2":
        return "ChatGPT-5.2"
    if s.lower() == "qwen3-32b" or s.lower() == "qwen3-32b".replace("b","b"):
        return "Qwen3-32B"
    if s.lower() == "qwen3-14b":
        return "Qwen3-14B"
    # Qwen3-0.6B / 1.7B / 4B keep as-is
    return s

df["Model_norm"] = df["Model"].apply(norm_model)

# Filter to non-conflict data only
nonc = df[df["Is_Conflict"] == False].copy()

# Define model and cue type display order
model_order = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-14B", "Qwen3-32B", "ChatGPT-5.2"]
# Filter to models present in the data
model_order = [m for m in model_order if m in set(nonc["Model_norm"].unique())]

cue_order = ["full", "topo_hist", "geom_rule_hist", "sem_rule_hist"]

# Aggregate data (mean by model and cue type)
agg = (nonc.groupby(["Model_norm", "Cue_Type"], as_index=False)
           .agg(sr=("PathGen_Acc", "mean")))

# Pivot for plotting
pivot = (agg.pivot(index="Model_norm", columns="Cue_Type", values="sr")
            .reindex(index=model_order, columns=cue_order))

# -------- Plot config --------
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
})

# Create figure (keep original size, remove title)
fig, ax = plt.subplots(figsize=(7.2, 2.9), constrained_layout=True)

x = np.arange(len(model_order))
bar_w = 0.18

# Draw grouped bar chart
for i, cue in enumerate(cue_order):
    y = pivot[cue].values
    ax.bar(x + (i - (len(cue_order)-1)/2)*bar_w, y, width=bar_w, label=cue)

# Axis config (keep only essential elements; title left to caption)
ax.set_xticks(x)
ax.set_xticklabels(model_order, rotation=20, ha="right")
ax.set_ylim(0, 1.0)
ax.set_ylabel("PathGen Success Rate")
ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend(frameon=False, ncol=2, fontsize=8)

# Save (simplified filenames)
pdf_path = os.path.join(OUT_DIR, "fig9_nonconflict.pdf")
svg_path = os.path.join(OUT_DIR, "fig9_nonconflict.svg")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

print("Saved:", pdf_path)
print("Saved:", svg_path)
