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
OUT_DIR = "figures7_all"
FIG_PREFIX = "fig7_abstraction_strength_heatmaps"

# Metric to plot
METRIC_COL = "Overall_Acc"   # or "PathGen_Acc"

# If None -> export all models in CSV.
# Otherwise, specify a list like:
# MODELS_TO_EXPORT = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-14B", "Qwen3-32B", "ChatGPT-5.2"]
MODELS_TO_EXPORT = None

FIG_W, FIG_H = 7.2, 3.1
DPI = 300

# =========================
# Helpers
# =========================
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def grade_key(g):
    if pd.isna(g):
        return None
    s = str(g).strip()
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else None

def norm_variant(v):
    v = str(v).lower().strip()

    # normalize separators
    v = v.replace("-", "_")

    mapping = {
        "hierarchical": "hier",
        "hierarchy": "hier",
        "clus": "clustered",
        "cluster": "clustered",
    }
    v = mapping.get(v, v)

    # handle common patterns like "hier50" -> "hier_50"
    v = re.sub(r'^(hier)(\d+)$', r'\1_\2', v)
    v = re.sub(r'^(clustered)(\d+)$', r'\1_\2', v)

    return v

def safe_model_filename(name: str) -> str:
    # make it filesystem-friendly
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', name)

def build_matrix(agg, variants):
    mat = (
        agg[agg["Variant_norm"].isin(variants)]
        .pivot(index="Variant_norm", columns="Grade_num", values="val")
        .reindex(index=variants, columns=[1, 2, 3, 4, 5])
    )
    return mat

def plot_heat(ax, mat, title, vmin, vmax):
    im = ax.imshow(mat.values, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([f"G{i}" for i in [1, 2, 3, 4, 5]])

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

# =========================
# Load
# =========================
df = pd.read_csv(CSV_PATH)

# Accept either "Grade" or "Gradient" column name
grade_col = None
for c in ["Grade", "Gradient", "grade", "gradient"]:
    if c in df.columns:
        grade_col = c
        break
if grade_col is None:
    raise ValueError("Cannot find Grade/Gradient column. Expected one of: Grade, Gradient.")

for col in ["Model", "Variant"]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

if METRIC_COL not in df.columns:
    raise ValueError(f"Metric column '{METRIC_COL}' not found. Available: {list(df.columns)}")

df["Variant_norm"] = df["Variant"].apply(norm_variant)
df["Grade_num"] = df[grade_col].apply(grade_key)
df = df[df["Grade_num"].isin([1, 2, 3, 4, 5])].copy()

# Aggregate over scenes if present; otherwise just mean
group_cols = ["Model", "Variant_norm", "Grade_num"]
if "Scene" in df.columns:
    agg = df.groupby(group_cols, as_index=False).agg(val=(METRIC_COL, "mean"))
else:
    agg = df.groupby(group_cols, as_index=False).agg(val=(METRIC_COL, "mean"))

# Decide models to export
all_models = sorted(agg["Model"].unique().tolist())
models = MODELS_TO_EXPORT if MODELS_TO_EXPORT is not None else all_models

ensure_outdir(OUT_DIR)

# Define variant families
hier_variants = ["hier", "hier_50", "hier_25"]
clus_variants = ["clustered", "clustered_50", "clustered_25"]

# =========================
# Export per model
# =========================
exported = 0
skipped = 0

for model in models:
    sub = agg[agg["Model"] == model].copy()
    if sub.empty:
        print(f"[SKIP] No rows for model: {model}")
        skipped += 1
        continue

    mat_h = build_matrix(sub, hier_variants)
    mat_c = build_matrix(sub, clus_variants)

    # Check availability
    if mat_h.isna().all().all() and mat_c.isna().all().all():
        print(f"[SKIP] No required variants (hier/clustered families) for model: {model}")
        skipped += 1
        continue

    # Shared color scale per model (recommended)
    all_vals = pd.concat([mat_h.stack(), mat_c.stack()], axis=0).dropna()
    if len(all_vals) == 0:
        print(f"[SKIP] No numeric values for model: {model}")
        skipped += 1
        continue
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    # Plot
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H),
        constrained_layout=True
    )

    im1 = plot_heat(axes[0], mat_h, f"(a) Hierarchical family ({model})", vmin, vmax)
    im2 = plot_heat(axes[1], mat_c, f"(b) Clustered family ({model})", vmin, vmax)

    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.045, pad=0.02)
    cbar.set_label(f"{METRIC_COL}")

    # Save
    safe_name = safe_model_filename(model)
    pdf_path = os.path.join(OUT_DIR, f"{FIG_PREFIX}_{safe_name}.pdf")
    svg_path = os.path.join(OUT_DIR, f"{FIG_PREFIX}_{safe_name}.svg")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    exported += 1
    print(f"[OK] {model} -> {pdf_path} / {svg_path}")

print(f"\nDone. Exported: {exported}, Skipped: {skipped}, OutDir: {OUT_DIR}")
