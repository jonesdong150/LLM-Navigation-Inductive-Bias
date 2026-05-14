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
OUT_DIR = "figures7"
FIG_NAME = "fig7_abstraction_strength_heatmaps"

# Metric to plot: Overall_Acc or PathGen_Acc
METRIC_COL = "Overall_Acc"

# Whether to plot only main-study models (recommended: main paper uses main-study models;
# supplementary figure covers large models)
MAIN_MODELS = ["Qwen3-0.6b", "Qwen3-1.7b", "Qwen3-4b"]

FIG_W, FIG_H = 7.2, 3.1  # IJCAI double-column friendly
DPI = 300

# =========================
# Helpers
# =========================
def ensure_outdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def grade_key(g):
    if pd.isna(g):
        return None
    s = str(g)
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else None

def norm_variant(v):
    v = str(v).lower().strip()
    mapping = {
        "hierarchical": "hier",
        "clus": "clustered",
        "cluster": "clustered",
    }
    v = mapping.get(v, v)
    return v

def mean_over_scenes(df, group_cols, metric):
    # scene-level mean
    return df.groupby(group_cols, as_index=False).agg(val=(metric, "mean"))

# =========================
# Load
# =========================
df = pd.read_csv(CSV_PATH)

for col in ["Model", "Variant", "Grade"]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

if METRIC_COL not in df.columns:
    raise ValueError(f"Metric column '{METRIC_COL}' not found in CSV.")

df["Variant_norm"] = df["Variant"].apply(norm_variant)
df["Grade_num"] = df["Grade"].apply(grade_key)
df = df[df["Grade_num"].isin([1,2,3,4,5])].copy()

# Keep only models of interest (fallback: keep all if not found)
avail_models = set(df["Model"].unique())
models = [m for m in MAIN_MODELS if m in avail_models]
if len(models) == 0:
    models = sorted(list(avail_models))

# We will create one Figure 7 per model (recommended for clarity),
# but IJCAI page limit suggests: show ONLY Qwen3-4B in main paper,
# and move other models to appendix.
# Here we generate: (1) main figure for Qwen3-4B if exists; else first model.
main_model = "Qwen3-4B" if "Qwen3-4B" in avail_models else models[0]

# Define abstraction variants
hier_variants = ["hier", "hier_50", "hier_25"]
clus_variants = ["clustered", "clustered_50", "clustered_25"]

# Filter to the main model
sub = df[df["Model"] == main_model].copy()

# Aggregate over scenes if present
group_cols = ["Variant_norm", "Grade_num"]
agg = mean_over_scenes(sub, group_cols, METRIC_COL)

# Pivot to matrices
def build_matrix(variants):
    mat = (
        agg[agg["Variant_norm"].isin(variants)]
        .pivot(index="Variant_norm", columns="Grade_num", values="val")
        .reindex(index=variants, columns=[1,2,3,4,5])
    )
    return mat

mat_h = build_matrix(hier_variants)
mat_c = build_matrix(clus_variants)

# Determine a shared color scale to make the two heatmaps comparable
all_vals = pd.concat([mat_h.stack(), mat_c.stack()], axis=0).dropna()
vmin = float(all_vals.min()) if len(all_vals) else 0.0
vmax = float(all_vals.max()) if len(all_vals) else 1.0

# =========================
# Plot
# =========================
ensure_outdir(OUT_DIR)
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=False, constrained_layout=True)

def plot_heat(ax, mat, title):
    im = ax.imshow(mat.values, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([f"G{i}" for i in [1,2,3,4,5]])
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)

    # annotate values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xlabel("Spatial Complexity")
    ax.set_ylabel("Abstraction Variant")

    return im

im1 = plot_heat(axes[0], mat_h, f"(a) Hierarchical family ({main_model})")
im2 = plot_heat(axes[1], mat_c, f"(b) Clustered family ({main_model})")

# colorbar (shared)
cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.04, pad=0.02)
cbar.set_label(f"{METRIC_COL}")

# fig.tight_layout()

pdf_path = os.path.join(OUT_DIR, f"{FIG_NAME}_{main_model}.pdf")
svg_path = os.path.join(OUT_DIR, f"{FIG_NAME}_{main_model}.svg")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

print("Main model:", main_model)
print("Saved:", pdf_path)
print("Saved:", svg_path)
