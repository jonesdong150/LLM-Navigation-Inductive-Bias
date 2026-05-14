import os
import json
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Compact paper-friendly style config (for single-column figures)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Global smaller fonts
    "font.size": 7.0,
    "axes.titlesize": 7.6,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.4,

    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.0,
})

# Path config
BASE_DIR = Path(__file__).parent.parent
IN_PATH = BASE_DIR / "analysis_ieq_conflict" / "context_metrics_summary.json"
OUT_DIR = "figures_conflict"

def mkdirp(p):
    os.makedirs(p, exist_ok=True)

def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_model_name(name: str) -> str:
    n = name.strip()
    low = n.lower()
    if low in {"gpt-5.2", "chatgpt-5.2"}:
        return "ChatGPT-5.2"
    if "gemini" in low:
        return "Gemini-2.5"
    n = re.sub(r"^qwen", "Qwen", n, flags=re.IGNORECASE)
    n = re.sub(r"^llama", "Llama", n, flags=re.IGNORECASE)
    n = re.sub(r"(\d+(?:\.\d+)?)b\b", r"\1B", n, flags=re.IGNORECASE)
    return n

def parse_param_b(name_norm: str):
    m = re.search(r"(\d+(?:\.\d+)?)B\b", name_norm)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def assign_tier(name_norm: str) -> str:
    if name_norm in {"ChatGPT-5.2", "Gemini-2.5"}:
        return "Tier3"
    b = parse_param_b(name_norm)
    if b is None:
        return "Other"
    if b <= 4.0:
        return "Tier1"
    return "Tier2"

def tier_label_clean(tier_key: str) -> str:
    if tier_key == "Tier1": return "<=4B"
    if tier_key == "Tier2": return "4B-32B"
    if tier_key == "Tier3": return ">=100B"
    return tier_key

def tier_aggregate(results: dict):
    tiers = {}
    seen = set()

    for raw_name, mobj in results.items():
        name = normalize_model_name(raw_name)
        uniq = name if name not in seen else f"{name}<{raw_name}>"
        seen.add(uniq)

        tier = assign_tier(name)
        tiers.setdefault(tier, {"names": [], "C1": {}, "C2": {}})
        tiers[tier]["names"].append(name)

        c1 = mobj.get("C1", {})
        c2 = mobj.get("C2", {})

        def push(dst, key, val):
            dst.setdefault(key, [])
            dst[key].append(float(val))

        # C1
        push(tiers[tier]["C1"], "topo", c1.get("mean_acc_topo", 0.0))
        push(tiers[tier]["C1"], "geom", c1.get("mean_acc_geom", 0.0))
        push(tiers[tier]["C1"], "dom",  c1.get("cue_dominance_score", 0.0))

        # C2
        push(tiers[tier]["C2"], "sahi", c2.get("SAHI_score", 0.0))
        drops = c2.get("per_task_drop_mean", {})
        if drops:
            avg_drop = np.mean(list(drops.values()))
            push(tiers[tier]["C2"], "avg_drop", avg_drop)
        else:
            push(tiers[tier]["C2"], "avg_drop", 0.0)

    stats = {}
    order = ["Tier1", "Tier2", "Tier3"]
    for tier in order:
        if tier not in tiers:
            continue
        obj = tiers[tier]
        stats[tier] = {"n_models": len(obj["names"]), "C1": {}, "C2": {}}
        for sec in ["C1", "C2"]:
            for k, arr in obj[sec].items():
                a = np.array(arr, dtype=float)
                stats[tier][sec][k] = {
                    "mean": float(np.mean(a)),
                    "std":  float(np.std(a)),
                }
    return stats

# -------------------------
# C1 plot (compact)
# -------------------------
def draw_c1_tier(stats, out_dir):
    mkdirp(out_dir)
    tiers = [t for t in ["Tier1", "Tier2", "Tier3"] if t in stats]

    topo = np.array([stats[t]["C1"]["topo"]["mean"] for t in tiers])
    geom = np.array([stats[t]["C1"]["geom"]["mean"] for t in tiers])
    dom  = np.array([stats[t]["C1"]["dom"]["mean"]  for t in tiers])
    topo_std = [stats[t]["C1"]["topo"]["std"] for t in tiers]
    geom_std = [stats[t]["C1"]["geom"]["std"] for t in tiers]
    dom_std  = [stats[t]["C1"]["dom"]["std"]  for t in tiers]

    x = np.arange(len(tiers))
    w = 0.34

    # Further shrink canvas
    fig, ax1 = plt.subplots(figsize=(2.85, 2.05))

    # Bar with thinner borders + smaller errorbar caps
    ax1.bar(x - w/2, topo, width=w, label='Topo', color='#aec7e8',
            edgecolor='black', linewidth=0.45, zorder=2)
    ax1.errorbar(x - w/2, topo, yerr=topo_std, fmt='none',
                 capsize=2.0, linewidth=0.8, color='black', zorder=3)

    ax1.bar(x + w/2, geom, width=w, label='Geom', color='#ffbb78',
            edgecolor='black', linewidth=0.45, zorder=2)
    ax1.errorbar(x + w/2, geom, yerr=geom_std, fmt='none',
                 capsize=2.0, linewidth=0.8, color='black', zorder=3)

    ax1.set_ylabel('Acc', labelpad=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([tier_label_clean(t) for t in tiers])
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='both', which='major', pad=1.0, length=2.6)

    ax2 = ax1.twinx()
    ax2.plot(x, dom, marker='o', linestyle='--', color='black',
             label=r'$\Delta$Acc', linewidth=1.0, markersize=2.7, zorder=5)
    ax2.errorbar(x, dom, yerr=dom_std, fmt='none',
                 capsize=2.0, linewidth=0.7, color='black', zorder=5)
    ax2.axhline(0, linestyle=':', alpha=0.45, color='gray', linewidth=0.7)
    ax2.set_ylabel('Dom.', labelpad=1.2)
    ax2.set_ylim(-0.1, 0.6)
    ax2.tick_params(axis='y', which='major', pad=1.0, length=2.6)

    # Compact legend (merged, upper left)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(h1 + h2, l1 + l2, loc='upper left',
                     frameon=True, borderpad=0.2,
                     handlelength=1.2, handletextpad=0.4,
                     labelspacing=0.2)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.9)

    ax1.set_title('(C1) Cue Dominance', fontweight='normal', pad=2)

    fig.tight_layout(pad=0.12)
    out_path = os.path.join(out_dir, "figC1_tier_cue_dominance.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")

# -------------------------
# C2 plot (compact)
# -------------------------
def draw_c2_tier_single(stats, out_dir):
    mkdirp(out_dir)
    tiers = [t for t in ["Tier1", "Tier2", "Tier3"] if t in stats]

    sahi      = [stats[t]["C2"]["sahi"]["mean"]     for t in tiers]
    avg_drop  = [stats[t]["C2"]["avg_drop"]["mean"] for t in tiers]
    sahi_std     = [stats[t]["C2"]["sahi"]["std"]     for t in tiers]
    avg_drop_std = [stats[t]["C2"]["avg_drop"]["std"] for t in tiers]

    x = np.arange(len(tiers))

    fig, ax1 = plt.subplots(figsize=(2.85, 2.05))

    ax1.bar(x, sahi, width=0.48, label='SAHI', color='#c5b0d5',
            edgecolor='black', linewidth=0.45, zorder=2)
    ax1.errorbar(x, sahi, yerr=sahi_std, fmt='none',
                 capsize=2.0, linewidth=0.8, color='black', zorder=3)
    ax1.set_ylabel('SAHI', labelpad=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([tier_label_clean(t) for t in tiers])
    ax1.set_ylim(0, 0.5)
    ax1.tick_params(axis='both', which='major', pad=1.0, length=2.6)

    ax2 = ax1.twinx()
    ax2.plot(x, avg_drop, marker='o', linestyle='--', color='black',
             label=r'Avg $\Delta$Acc', linewidth=1.0, markersize=2.7, zorder=5)
    ax2.errorbar(x, avg_drop, yerr=avg_drop_std, fmt='none',
                 capsize=2.0, linewidth=0.7, color='black', zorder=5)
    ax2.set_ylabel(r'$\Delta$Acc', labelpad=1.2)
    ax2.set_ylim(0, 0.5)
    ax2.tick_params(axis='y', which='major', pad=1.0, length=2.6)

    # Compact legend, upper right
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(h1 + h2, l1 + l2, loc='upper right',
                     frameon=True, borderpad=0.2,
                     handlelength=1.2, handletextpad=0.4,
                     labelspacing=0.2)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.9)

    ax1.set_title("(C2) Semantic Interference", fontweight='normal', pad=2)

    fig.tight_layout(pad=0.12)
    out_path = os.path.join(out_dir, "figC2_tier_semantic_achilles.pdf")
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")

def main():
    mkdirp(OUT_DIR)
    summary = load_summary(IN_PATH)
    stats = tier_aggregate(summary["results"])
    draw_c1_tier(stats, OUT_DIR)
    draw_c2_tier_single(stats, OUT_DIR)

if __name__ == "__main__":
    main()
