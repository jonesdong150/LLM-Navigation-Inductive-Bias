import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "final_stats" / "r2_complexity_resilience.csv"
OUT_DIR = "figures6"
FIG_NAME = "fig6p_r2_complexity_curves_large_models"

# Supplementary models
LARGE_MODELS = ["Qwen3-14b", "Qwen3-32b", "gpt-5.2"]

METRIC_COL = "Overall_Acc"

FIG_W, FIG_H = 7.2, 2.6   # Consistent with Figure 6
DPI = 300

# =========================
# Helpers
# =========================
def ensure_outdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def normalize_variant(v):
    v = str(v).lower().strip()
    if v in ["flat", "hier", "hierarchical", "clustered", "clus"]:
        if v == "hierarchical":
            return "hier"
        if v == "clus":
            return "clustered"
        return v
    return v

def grade_key(g):
    if pd.isna(g):
        return None
    s = str(g)
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else None

# =========================
# Load
# =========================
df = pd.read_csv(CSV_PATH)

df["Variant_norm"] = df["Variant"].apply(normalize_variant)
df["Grade_num"] = df["Grade"].apply(grade_key)

df = df[df["Grade_num"].isin([1,2,3,4,5])]
df = df[df["Variant_norm"].isin(["flat", "hier", "clustered"])]
df = df[df["Model"].isin(LARGE_MODELS)]

# =========================
# Aggregate (scene-level CI if available)
# =========================
if "Scene" in df.columns:
    agg = (
        df.groupby(["Model", "Variant_norm", "Grade_num"], as_index=False)
          .agg(mean_acc=(METRIC_COL, "mean"),
               std_acc=(METRIC_COL, "std"),
               n=(METRIC_COL, "count"))
    )
    agg["ci95"] = 1.96 * agg["std_acc"].fillna(0) / np.sqrt(agg["n"].clip(lower=1))
else:
    agg = (
        df.groupby(["Model", "Variant_norm", "Grade_num"], as_index=False)
          .agg(mean_acc=(METRIC_COL, "mean"))
    )
    agg["ci95"] = np.nan

# =========================
# Plot
# =========================
ensure_outdir(OUT_DIR)

fig, axes = plt.subplots(1, len(LARGE_MODELS), figsize=(FIG_W, FIG_H), sharey=True)

if len(LARGE_MODELS) == 1:
    axes = [axes]

x = np.array([1,2,3,4,5])
x_labels = [f"G{i}" for i in x]

for ax, model in zip(axes, LARGE_MODELS):
    sub = agg[agg["Model"] == model]
    for v in ["flat", "hier", "clustered"]:
        sv = sub[sub["Variant_norm"] == v].sort_values("Grade_num")
        if sv.empty:
            continue
        ax.errorbar(
            x,
            sv["mean_acc"].values,
            yerr=sv["ci95"].values,
            marker="o",
            linewidth=1.5,
            capsize=2,
            label=v
        )
    ax.set_title(model)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Spatial Complexity")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

axes[0].set_ylabel("Overall Accuracy")
axes[0].set_ylim(0, 1.0)

handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.08),
    fontsize=9
)

fig.tight_layout()

pdf_path = os.path.join(OUT_DIR, f"{FIG_NAME}.pdf")
svg_path = os.path.join(OUT_DIR, f"{FIG_NAME}.svg")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

print("Saved:", pdf_path)
print("Saved:", svg_path)
