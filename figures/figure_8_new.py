import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Global style (paper-friendly)
# =========================
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
})

# =========================
# Config (key adjustment: reduce height)
# =========================
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "final_stats" / "r2_complexity_resilience.csv"
OUT_DIR = "figures8"
FIG_NAME = "fig8_task_complexity_interaction"

MODEL = "gpt-5.2"
VARIANTS = ["flat", "hier", "hier_50", "hier_25"]
TASKS = ["ObjectLocation", "GeometryYN", "TopologyYN", "ReachabilityYN", "PathGen"]
TASK_DISPLAY_NAMES = {
    "ObjectLocation": "ObjLoc",
    "GeometryYN": "GeomYN",
    "TopologyYN": "TopoYN",
    "ReachabilityYN": "ReachYN",
    "PathGen": "PathGen"
}

TASK_COL_CANDIDATES = {
    "ObjectLocation": ["Task_ObjectLocation_Acc", "ObjectLocation_Acc", "ObjectLocation", "ObjLoc_Acc", "objloc_acc"],
    "GeometryYN":     ["Task_GeometryYN_Acc",     "GeometryYN_Acc",     "GeometryYN",     "GeomYN_Acc", "geom_yn_acc"],
    "TopologyYN":     ["Task_TopologyYN_Acc",     "TopologyYN_Acc",     "TopologyYN",     "TopoYN_Acc", "topo_yn_acc"],
    "ReachabilityYN": ["Task_ReachabilityYN_Acc", "ReachabilityYN_Acc", "ReachabilityYN", "ReachYN_Acc","reach_yn_acc"],
    "PathGen":        ["Task_PathGen_Acc",        "PathGen_Acc",        "PathGen",        "PathGen_SR", "pathgen_acc"],
}

# Key adjustment: keep width at 12.0 (wide enough), reduce height from 5.0 to 3.5
FIG_W, FIG_H = 12.0, 3.5
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
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None

def norm_variant(v):
    v = str(v).lower().strip()
    v = v.replace("-", "_")
    mapping = {"hierarchical": "hier", "clus": "clustered", "cluster": "clustered"}
    v = mapping.get(v, v)
    v = re.sub(r"^(hier)(\d+)$", r"\1_\2", v)
    v = re.sub(r"^(clustered)(\d+)$", r"\1_\2", v)
    return v

def find_grade_col(df):
    for c in ["Grade", "Gradient", "grade", "gradient", "G"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot find Grade/Gradient column in CSV.")

def pick_task_col(df, task_name):
    for c in TASK_COL_CANDIDATES[task_name]:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find a column for task '{task_name}'. Tried: {TASK_COL_CANDIDATES[task_name]}\n"
        f"Available columns: {list(df.columns)}"
    )

def plot_heat(ax, mat, title, vmin, vmax, tasks, task_display_names, show_y=False, annotate_fontsize=7):
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=10)

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([f"G{i}" for i in [1, 2, 3, 4, 5]])
    ax.set_xlabel("Spatial Complexity", labelpad=8)

    if show_y:
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels([task_display_names[t] for t in tasks])
        ax.set_ylabel("Task", labelpad=8)
    else:
        ax.set_yticks([])
        ax.set_ylabel("")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=annotate_fontsize)

    return im

# =========================
# Load
# =========================
df = pd.read_csv(CSV_PATH)

if "Model" not in df.columns or "Variant" not in df.columns:
    raise ValueError("CSV must contain columns: Model, Variant")

grade_col = find_grade_col(df)
df["Grade_num"] = df[grade_col].apply(grade_key)
df = df[df["Grade_num"].isin([1, 2, 3, 4, 5])].copy()

df["Variant_norm"] = df["Variant"].apply(norm_variant)

avail_models = set(df["Model"].unique())
if MODEL not in avail_models:
    raise ValueError(f"Model '{MODEL}' not found. Available: {sorted(avail_models)}")

sub = df[(df["Model"] == MODEL) & (df["Variant_norm"].isin(VARIANTS))].copy()
if sub.empty:
    raise ValueError(
        f"No rows after filtering. Check MODEL/VARIANTS.\n"
        f"MODEL={MODEL}, VARIANTS={VARIANTS}\n"
        f"Available variants for this model: {sorted(df[df['Model']==MODEL]['Variant_norm'].unique())}"
    )

task_cols = {t: pick_task_col(sub, t) for t in TASKS}

group_cols = ["Variant_norm", "Grade_num"]
agg = sub.groupby(group_cols, as_index=False).agg({c: "mean" for c in task_cols.values()})

mats = {}
for v in VARIANTS:
    vsub = agg[agg["Variant_norm"] == v].copy().set_index("Grade_num").reindex([1, 2, 3, 4, 5])
    mat = np.full((len(TASKS), 5), np.nan, dtype=float)
    for i, t in enumerate(TASKS):
        col = task_cols[t]
        mat[i, :] = vsub[col].values if col in vsub.columns else np.nan
    mats[v] = mat

all_vals = np.concatenate([mats[v].ravel() for v in VARIANTS])
all_vals = all_vals[np.isfinite(all_vals)]
vmin = float(all_vals.min()) if len(all_vals) else 0.0
vmax = float(all_vals.max()) if len(all_vals) else 1.0

# =========================
# Plot (optimize spacing)
# =========================
ensure_outdir(OUT_DIR)

fig, axes = plt.subplots(
    1, len(VARIANTS),
    figsize=(FIG_W, FIG_H),
    constrained_layout=False,
    sharey=False
)
# Further reduce subplot spacing to use width efficiently (from 0.15 to 0.1)
fig.subplots_adjust(wspace=0.1)

if len(VARIANTS) == 1:
    axes = [axes]

last_im = None
for idx, (ax, v) in enumerate(zip(axes, VARIANTS)):
    title = v
    if v == "hier_50":
        title = "hier (50%)"
    elif v == "hier_25":
        title = "hier (25%)"
    last_im = plot_heat(
        ax, mats[v], title, vmin, vmax,
        TASKS, TASK_DISPLAY_NAMES,
        show_y=(idx == 0),
        annotate_fontsize=7
    )

# Colorbar adapted to compact layout
cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.012, pad=0.015)
cbar.set_label("Accuracy", labelpad=8)

pdf_path = os.path.join(OUT_DIR, f"{FIG_NAME}_{MODEL}.pdf")
svg_path = os.path.join(OUT_DIR, f"{FIG_NAME}_{MODEL}.svg")
fig.savefig(pdf_path, bbox_inches="tight", dpi=DPI)
fig.savefig(svg_path, bbox_inches="tight", dpi=DPI)
plt.close(fig)

print("Saved:", pdf_path)
print("Saved:", svg_path)
print("Model:", MODEL)
print("Variants:", VARIANTS)
print("Task columns used:", task_cols)
print("Grade column used:", grade_col)
