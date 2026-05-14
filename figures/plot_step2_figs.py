#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

BASE = Path(__file__).parent.parent
ANALYSIS_ROOT = BASE / "analysis_ieq"
OUT_DIR = BASE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]
VARIANTS = ["flat", "hier", "clustered"]

def model_size_b(model_name: str) -> float:
    # "Qwen3-0.6B" -> 0.6
    s = model_name.split("-")[-1].upper().replace("B", "")
    return float(s)

def load_all_scene_summaries():
    """
    Returns:
      data[model]["overall"][variant] = (correct_sum, n_sum)
      data[model]["by_task"][task][variant] = (correct_sum, n_sum)
    """
    data = {}
    for m in MODELS:
        mdir = ANALYSIS_ROOT / m
        if not mdir.exists():
            print(f"[WARN] Missing dir: {mdir}")
            continue

        overall = {v: {"correct": 0, "n": 0} for v in VARIANTS}
        by_task = defaultdict(lambda: {v: {"correct": 0, "n": 0} for v in VARIANTS})

        for p in sorted(mdir.glob("*_summary.json")):
            with open(p, "r", encoding="utf-8") as f:
                js = json.load(f)

            # overall
            ov = js.get("overall", {})
            for v in VARIANTS:
                overall[v]["correct"] += int(ov.get(v, {}).get("correct", 0))
                overall[v]["n"] += int(ov.get(v, {}).get("n", 0))

            # by task type
            bt = js.get("by_task_type", {})
            for task, task_stats in bt.items():
                for v in VARIANTS:
                    by_task[task][v]["correct"] += int(task_stats.get(v, {}).get("correct", 0))
                    by_task[task][v]["n"] += int(task_stats.get(v, {}).get("n", 0))

        data[m] = {"overall": overall, "by_task": by_task}
    return data

def acc(correct: int, n: int) -> float:
    return (correct / n) if n else 0.0

def fig1_overall_acc(data):
    # Bar chart: model x variant overall accuracy
    models = [m for m in MODELS if m in data]
    x = range(len(models))

    # compute
    vals = {v: [] for v in VARIANTS}
    for m in models:
        for v in VARIANTS:
            c = data[m]["overall"][v]["correct"]
            n = data[m]["overall"][v]["n"]
            vals[v].append(acc(c, n))

    # plot
    plt.figure()
    width = 0.25
    offsets = {"flat": -width, "hier": 0.0, "clustered": width}

    for v in VARIANTS:
        plt.bar([i + offsets[v] for i in x], vals[v], width=width, label=v)

    plt.xticks(list(x), models, rotation=0)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Fig1: Overall accuracy by model and representation")
    plt.legend()

    out = OUT_DIR / "fig1_overall_accuracy.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"[OK] Wrote {out}")

def fig2_delta_vs_scale(data):
    # Line chart: model size vs delta accuracy (hier-flat, clustered-flat)
    models = [m for m in MODELS if m in data]
    models_sorted = sorted(models, key=model_size_b)
    xs = [model_size_b(m) for m in models_sorted]

    deltas_h = []
    deltas_c = []
    for m in models_sorted:
        ov = data[m]["overall"]
        a_flat = acc(ov["flat"]["correct"], ov["flat"]["n"])
        a_hier = acc(ov["hier"]["correct"], ov["hier"]["n"])
        a_clus = acc(ov["clustered"]["correct"], ov["clustered"]["n"])
        deltas_h.append(a_hier - a_flat)
        deltas_c.append(a_clus - a_flat)

    plt.figure()
    plt.plot(xs, deltas_h, marker="o", label="deltaAcc (hier - flat)")
    plt.plot(xs, deltas_c, marker="o", label="deltaAcc (clustered - flat)")

    plt.xticks(xs, [f"{x}B" for x in xs])
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("delta Accuracy")
    plt.title("Fig2: Structural gain vs model scale (non-monotonic scaling)")
    plt.legend()

    out = OUT_DIR / "fig2_delta_vs_scale.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"[OK] Wrote {out}")

def fig3_task_delta_bars(data):
    # For each task_type, compute avg delta across models (or show per model)
    # Here: plot per-task deltas for each model as grouped bars (2 deltas each model is too busy),
    # so we do: per task, show (hier-flat) and (clustered-flat) averaged across models.
    tasks = set()
    for m in data:
        tasks.update(data[m]["by_task"].keys())
    tasks = sorted(tasks)

    dh = []
    dc = []
    for task in tasks:
        # average delta across models weighted by n
        sum_h, sum_c, sum_n = 0.0, 0.0, 0
        for m in data:
            bt = data[m]["by_task"].get(task, None)
            if not bt:
                continue
            a_flat = acc(bt["flat"]["correct"], bt["flat"]["n"])
            a_hier = acc(bt["hier"]["correct"], bt["hier"]["n"])
            a_clus = acc(bt["clustered"]["correct"], bt["clustered"]["n"])
            # weight by flat n (same as others typically)
            n = bt["flat"]["n"]
            sum_h += (a_hier - a_flat) * n
            sum_c += (a_clus - a_flat) * n
            sum_n += n
        dh.append(sum_h / sum_n if sum_n else 0.0)
        dc.append(sum_c / sum_n if sum_n else 0.0)

    x = range(len(tasks))
    plt.figure(figsize=(10, 4.5))
    width = 0.35
    plt.bar([i - width/2 for i in x], dh, width=width, label="deltaAcc (hier - flat)")
    plt.bar([i + width/2 for i in x], dc, width=width, label="deltaAcc (clustered - flat)")

    plt.xticks(list(x), tasks, rotation=30, ha="right")
    plt.axhline(0.0, linewidth=1)
    plt.ylim(-1.0, 1.0)
    plt.ylabel("delta Accuracy")
    plt.title("Fig3: Structural gain by task type (avg over models, weighted)")
    plt.legend()

    out = OUT_DIR / "fig3_task_delta_bars.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"[OK] Wrote {out}")

def main():
    data = load_all_scene_summaries()
    if not data:
        raise SystemExit("[FATAL] No summaries found. Expected under analysis_ieq/<model>/")

    fig1_overall_acc(data)
    fig2_delta_vs_scale(data)
    fig3_task_delta_bars(data)

    print("\nAll figures saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
