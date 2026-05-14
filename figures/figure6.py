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
FIG_NAME = "fig6_r2_complexity_curves"

# Main-study models (as reported in the paper)
MAIN_MODELS = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]

# Metric to plot (default Overall_Acc)
METRIC_COL = "Overall_Acc"

FIG_W, FIG_H = 7.2, 2.6   # IJCAI double-column friendly
DPI = 300

# =========================
# Helpers
# =========================
def ensure_outdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def normalize_variant(v: str):
    v = str(v).lower().strip()
    # Abstracted variants are plotted in Figure 7; filter out here
    if v in ["flat", "hier", "hierarchical", "clustered", "clus"]:
        if v == "hierarchical":
            return "hier"
        if v == "clus":
            return "clustered"
        return v
    return v

def grade_key(g):
    """
    Convert grade labels like 'G1'/'g1'/1 into sortable integer 1..5
    """
    if pd.isna(g):
        return None
    s = str(g).strip()
    m = re.search(r'(\d+)', s)
    if m:
        return int(m.group(1))
    return None

# =========================
# Load & detect columns
# =========================
df = pd.read_csv(CSV_PATH)

# Column name compatibility
# Model column
model_col = "Model" if "Model" in df.columns else None
if model_col is None:
    raise ValueError("Cannot find 'Model' column in CSV.")

# Variant column
variant_col = "Variant" if "Variant" in df.columns else None
if variant_col is None:
    raise ValueError("Cannot find 'Variant' column in CSV.")

# Grade/Gradient column (allow several names)
grade_col_candidates = ["Gradient", "Grade", "G", "Complexity", "complexity", "gradient"]
grade_col = None
for c in grade_col_candidates:
    if c in df.columns:
        grade_col = c
        break
if grade_col is None:
    raise ValueError(f"Cannot find a grade column among {grade_col_candidates}.")

# Scene column optional
scene_col = "Scene" if "Scene" in df.columns else None

# Metric col
if METRIC_COL not in df.columns:
    raise ValueError(f"Metric column '{METRIC_COL}' not found. Available columns: {list(df.columns)}")

# Normalize
df["Variant_norm"] = df[variant_col].apply(normalize_variant)
df["Grade_num"] = df[grade_col].apply(grade_key)

# Keep only G1..G5
df = df[df["Grade_num"].isin([1,2,3,4,5])].copy()

# Keep only base variants
base_variants = ["flat", "hier", "clustered"]
df = df[df["Variant_norm"].isin(base_variants)].copy()

# Filter to available MAIN_MODELS
available_models = set(df[model_col].unique().tolist())
models = [m for m in MAIN_MODELS if m in available_models]
if len(models) == 0:
    # fallback: take first three models found
    models = sorted(list(available_models))[:3]

# =========================
# Aggregate
# =========================
# If we have per-scene entries, compute mean across scenes per (model, variant, grade)
# and 95% CI across scenes
group_cols = [model_col, "Variant_norm", "Grade_num"]

if scene_col is not None:
    agg = (
        df.groupby(group_cols, as_index=False)
          .agg(mean_acc=(METRIC_COL, "mean"),
               std_acc=(METRIC_COL, "std"),
               n=(METRIC_COL, "count"))
    )
    agg["ci95"] = 1.96 * (agg["std_acc"].fillna(0) / np.sqrt(agg["n"].clip(lower=1)))
else:
    # already aggregated: just mean; no CI
    agg = (
        df.groupby(group_cols, as_index=False)
          .agg(mean_acc=(METRIC_COL, "mean"))
    )
    agg["ci95"] = np.nan

# =========================
# Plot (small multiples)
# =========================
ensure_outdir(OUT_DIR)
fig, axes = plt.subplots(1, len(models), figsize=(FIG_W, FIG_H), sharey=True)

if len(models) == 1:
    axes = [axes]

x = np.array([1,2,3,4,5])
x_labels = [f"G{i}" for i in x]

for ax, m in zip(axes, models):
    sub = agg[agg[model_col] == m].copy()
    for v in base_variants:
        sv = sub[sub["Variant_norm"] == v].sort_values("Grade_num")
        if sv.empty:
            continue
        y = sv["mean_acc"].values
        yerr = sv["ci95"].values
        # No explicit colors: rely on default cycle
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=2, label=v)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_title(m)
    ax.set_xlabel("Spatial Complexity")

axes[0].set_ylabel(f"{METRIC_COL} (Accuracy)")
axes[0].set_ylim(0, 1.0)
for ax in axes:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

# one shared legend
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=9)

fig.tight_layout()

pdf_path = os.path.join(OUT_DIR, f"{FIG_NAME}.pdf")
svg_path = os.path.join(OUT_DIR, f"{FIG_NAME}.svg")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

print("Saved:", pdf_path)
print("Saved:", svg_path)
