import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Config
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "final_stats" / "r3_dimension_conflict.csv"
OUT_DIR = "figures10"

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)

# Normalize model names
def norm_model(m: str) -> str:
    s = str(m).strip()
    s_low = s.lower()
    if s_low == "gpt-5.2":
        return "ChatGPT-5.2"
    if s_low == "qwen3-14b":
        return "Qwen3-14B"
    if s_low == "qwen3-32b":
        return "Qwen3-32B"
    return s

df["Model_norm"] = df["Model"].apply(norm_model)

# Aggregate accuracy by (model, cue type, conflict flag)
agg = (df.groupby(["Model_norm", "Cue_Type", "Is_Conflict"], as_index=False)
         .agg(acc=("PathGen_Acc", "mean")))

# Pivot: keep both non-conflict and conflict data for CSI calculation
piv = agg.pivot_table(index=["Model_norm", "Cue_Type"], columns="Is_Conflict", values="acc")
piv = piv.rename(columns={False: "nonconf", True: "conf"}).reset_index()

# Key fix: drop cue types missing either conflict or non-conflict data (e.g., full)
piv = piv.dropna(subset=["nonconf", "conf"]).copy()

# Compute CSI (relative drop), avoid division by zero
eps = 1e-6
piv["CSI"] = (piv["nonconf"] - piv["conf"]) / np.maximum(piv["nonconf"], eps)

# Define display order
model_order = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-14B", "Qwen3-32B", "ChatGPT-5.2"]
model_order = [m for m in model_order if m in set(piv["Model_norm"].unique())]

# Keep only cue types that have conflict data (full is automatically filtered by dropna)
cue_order = ["topo_hist", "geom_rule_hist", "sem_rule_hist"]
cue_order = [c for c in cue_order if c in set(piv["Cue_Type"].unique())]

# Convert to heatmap matrix format
mat = (piv.pivot(index="Model_norm", columns="Cue_Type", values="CSI")
          .reindex(index=model_order, columns=cue_order))

# Plot style config
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
})

# Create figure
fig, ax = plt.subplots(figsize=(7.2, 2.6), constrained_layout=True)

# Draw heatmap
im = ax.imshow(mat.values, aspect="auto")

# Axis config
ax.set_xticks(np.arange(len(cue_order)))
ax.set_xticklabels(cue_order, rotation=15, ha="right")
ax.set_yticks(np.arange(len(model_order)))
ax.set_yticklabels(model_order)

# Annotate values
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        val = mat.values[i, j]
        if np.isfinite(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

ax.set_xlabel("Cue combination (conflict-enabled)")
ax.set_ylabel("Model")

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("CSI (relative drop)")

# Save
pdf_path = os.path.join(OUT_DIR, "fig10_csi.pdf")
svg_path = os.path.join(OUT_DIR, "fig10_csi.svg")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

# Log output
print("Saved:", pdf_path)
print("Saved:", svg_path)
print("Cues plotted:", cue_order)
