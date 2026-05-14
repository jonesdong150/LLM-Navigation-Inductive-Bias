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
CSV_PATH = BASE_DIR / "final_stats" / "r1_structure_scaling.csv"
OUT_DIR = "figures5"
FIG_NAME = "fig5_structure_scaling"

# Paper-friendly sizing (double-column friendly)
FIG_W, FIG_H = 7.2, 3.2   # inches; suitable for IJCAI double-column layout
DPI = 300

# =========================
# Helpers
# =========================
def parse_params_b(model_name: str):
    """
    Extract parameter size in billions from model string if available.
    e.g., 'Qwen3-0.6B' -> 0.6, 'Qwen3-14b' -> 14
    Return None if not parseable (e.g., proprietary models).
    """
    s = model_name.lower()
    m = re.search(r'(\d+(\.\d+)?)\s*b', s)
    if m:
        return float(m.group(1))
    return None

def ensure_outdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# =========================
# Load
# =========================
df = pd.read_csv(CSV_PATH)

# Basic checks
required_cols = {"Model", "Scene", "Variant", "Overall_Acc"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Normalize variant names (just in case)
df["Variant"] = df["Variant"].str.lower().str.strip()
variant_order = ["flat", "hier", "clustered"]

# Decide model order:
# - Sort parseable models by params
# - Put non-parseable models (e.g., gpt-5.2) at the end
models = sorted(df["Model"].unique().tolist())
model_params = {m: parse_params_b(m) for m in models}

parseable = [m for m in models if model_params[m] is not None]
nonparseable = [m for m in models if model_params[m] is None]

parseable_sorted = sorted(parseable, key=lambda m: model_params[m])
model_order = parseable_sorted + sorted(nonparseable)

# Create pretty x labels
def pretty_label(m):
    p = model_params[m]
    if p is None:
        return m  # e.g., gpt-5.2
    # keep original but make B explicit
    # if original already has B/b, keep it
    if re.search(r'[Bb]', m):
        return m
    return f"{m} ({p}B)"

x_labels = [pretty_label(m) for m in model_order]
x = np.arange(len(model_order))

# Aggregate: mean across scenes (Set-A has 10 scenes)
agg = (
    df.groupby(["Model", "Variant"], as_index=False)
      .agg(mean_acc=("Overall_Acc", "mean"),
           std_acc=("Overall_Acc", "std"),
           n=("Overall_Acc", "count"))
)

# Prepare pivot for plotting
pivot_mean = agg.pivot(index="Model", columns="Variant", values="mean_acc").reindex(model_order)
pivot_std  = agg.pivot(index="Model", columns="Variant", values="std_acc").reindex(model_order)
pivot_n    = agg.pivot(index="Model", columns="Variant", values="n").reindex(model_order)

# 95% CI across scenes (t-approx; n small but OK for plotting; you can switch to bootstrap later)
# If std is NaN (n=1), fallback to 0
ci95 = 1.96 * (pivot_std.fillna(0) / np.sqrt(pivot_n.fillna(1)))

# Delta vs flat
delta_hier = pivot_mean["hier"] - pivot_mean["flat"]
delta_clus = pivot_mean["clustered"] - pivot_mean["flat"]

# =========================
# Plot
# =========================
ensure_outdir(OUT_DIR)
fig = plt.figure(figsize=(FIG_W, FIG_H))

# Two panels side-by-side
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.35)

# ---- Panel (a): Overall Acc vs Model (ordered by size)
ax1 = fig.add_subplot(gs[0, 0])

for v in variant_order:
    if v not in pivot_mean.columns:
        continue
    y = pivot_mean[v].values
    yerr = ci95[v].values if v in ci95.columns else None
    # No explicit colors: rely on matplotlib defaults
    ax1.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=2, label=v)

ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, rotation=25, ha="right")
ax1.set_ylabel("Overall Accuracy")
ax1.set_title("(a) Overall performance vs. model scale")
ax1.set_ylim(0, 1.0)
ax1.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
ax1.legend(frameon=False, fontsize=9)

# ---- Panel (b): Delta vs Flat
ax2 = fig.add_subplot(gs[0, 1])

bar_w = 0.38
ax2.bar(x - bar_w/2, delta_hier.values, width=bar_w, label="hier - flat")
ax2.bar(x + bar_w/2, delta_clus.values, width=bar_w, label="clustered - flat")

ax2.axhline(0, linewidth=1.0)
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, rotation=25, ha="right")
ax2.set_ylabel("Δ Accuracy vs. Flat")
ax2.set_title("(b) Relative gains over Flat")
ax2.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
ax2.legend(frameon=False, fontsize=9)

# Tight layout and save (vector)
fig.tight_layout()
pdf_path = os.path.join(OUT_DIR, f"{FIG_NAME}.pdf")
svg_path = os.path.join(OUT_DIR, f"{FIG_NAME}.svg")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

print("Saved:", pdf_path)
print("Saved:", svg_path)
