import os
import json
from typing import Dict, Any, List, Tuple

# Ensure these paths match your environment
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "analysis_ieq_conflict")
OUT_PATH = os.path.join(ROOT_DIR, "context_metrics_summary.json")

CONDITIONS = ["combo_topo", "combo_geom", "conflict_sem"]
CORE_TASKS = ["ObjectLocation", "TopologyYN", "ReachabilityYN", "PathGen"]
ALL_TASKS = CORE_TASKS + ["GeometryYN"]

EPS = 1e-9

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def get_task_acc(task_obj: Dict[str, Any]) -> Tuple[float, int]:
    """
    Get accuracy directly, without FVR penalty.
    Returns: (accuracy, count)
    """
    acc = float(task_obj.get("accuracy", 0.0))
    cnt = int(task_obj.get("count", 0))
    return acc, cnt

def aggregate_condition(scene_data: Dict[str, Any], condition: str) -> Dict[str, Any]:
    """
    From a single scene's summary.json, compute:
    - Per-task accuracy
    - Weighted average accuracy for CORE tasks
    - Weighted average accuracy for ALL tasks
    """
    cond = scene_data.get(condition, {})
    tasks = cond.get("tasks", {})

    per_task = {}
    for t in ALL_TASKS:
        if t in tasks:
            acc, cnt = get_task_acc(tasks[t])
        else:
            acc, cnt = 0.0, 0
        per_task[t] = {"acc": acc, "count": cnt}

    def weighted_mean(task_list: List[str]) -> float:
        num, den = 0.0, 0
        for t in task_list:
            acc = per_task[t]["acc"]
            cnt = per_task[t]["count"]
            num += acc * cnt
            den += cnt
        return num / den if den > 0 else 0.0

    return {
        "per_task": per_task,
        "core_acc": weighted_mean(CORE_TASKS),
        "all_acc": weighted_mean(ALL_TASKS),
    }

def mean_over_scenes(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def compute_model_metrics(model_dir: str) -> Dict[str, Any]:
    """
    Read summary.json from all scenes in the model directory and aggregate metrics:
    - C1: Topological dominance vs geometric dominance
    - C2: Semantic conflict detection (SAHI)
    """
    scene_files = sorted(
        f for f in os.listdir(model_dir)
        if f.startswith("scene_conflict_") and f.endswith("_summary.json")
    )

    if not scene_files:
        return {"error": f"no scene summary files found in {model_dir}"}

    scene_level = []
    for sf in scene_files:
        path = os.path.join(model_dir, sf)
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        per_cond = {}
        for c in CONDITIONS:
            per_cond[c] = aggregate_condition(data, c)

        # Compute C2 semantic interference performance drop (Topology - Conflict)
        topo_tasks = per_cond["combo_topo"]["per_task"]
        conf_tasks = per_cond["conflict_sem"]["per_task"]
        per_task_drop = {}
        for t in CORE_TASKS:
            per_task_drop[t] = topo_tasks[t]["acc"] - conf_tasks[t]["acc"]

        scene_level.append({
            "scene_file": sf,
            "combo_topo_core_acc": per_cond["combo_topo"]["core_acc"],
            "combo_geom_core_acc": per_cond["combo_geom"]["core_acc"],
            "conflict_sem_core_acc": per_cond["conflict_sem"]["core_acc"],
            "combo_geom_geomYN_acc": per_cond["combo_geom"]["per_task"]["GeometryYN"]["acc"],
            "per_task_drop_core": per_task_drop,
        })

    # Average across scenes
    topo_core_avg = mean_over_scenes([x["combo_topo_core_acc"] for x in scene_level])
    geom_core_avg = mean_over_scenes([x["combo_geom_core_acc"] for x in scene_level])
    conf_core_avg = mean_over_scenes([x["conflict_sem_core_acc"] for x in scene_level])
    geom_geomYN_avg = mean_over_scenes([x["combo_geom_geomYN_acc"] for x in scene_level])

    # C1: Compute cue dominance score
    cue_dominance = topo_core_avg - geom_core_avg

    # C2: Compute Semantic Achilles' Heel Index (SAHI)
    # SAHI = (Shield - Conflict) / Shield
    shield = topo_core_avg
    sahI = max(0.0, (shield - conf_core_avg) / (shield + EPS))

    # Mean drop per task
    per_task_drop_mean = {}
    for t in CORE_TASKS:
        per_task_drop_mean[t] = mean_over_scenes([x["per_task_drop_core"][t] for x in scene_level])

    return {
        "n_scenes": len(scene_level),
        "C1": {
            "mean_acc_topo": topo_core_avg,
            "mean_acc_geom": geom_core_avg,
            "cue_dominance_score": cue_dominance,
            "geom_specialization_geomYN": geom_geomYN_avg,
            "note": "Computed on CORE_TASKS (excluding GeometryYN) to ensure topology fairness."
        },
        "C2": {
            "shield_acc_topo": shield,
            "conflict_acc_sem": conf_core_avg,
            "SAHI_score": sahI,
            "per_task_drop_mean": per_task_drop_mean,
            "pathgen_fragility": per_task_drop_mean.get("PathGen", 0.0),
            "note": "Quantifies performance degradation under semantic ambiguity."
        },
        "scene_level_raw": scene_level  # Preserve raw data for subsequent figure analysis
    }

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Error: ROOT_DIR {ROOT_DIR} does not exist.")
        return

    models = sorted(
        d for d in os.listdir(ROOT_DIR)
        if os.path.isdir(os.path.join(ROOT_DIR, d))
        and not d.startswith(".")
    )

    out = {
        "config": {
            "core_tasks": CORE_TASKS,
            "all_tasks": ALL_TASKS,
            "conditions": CONDITIONS
        },
        "results": {}
    }

    for m in models:
        model_dir = os.path.join(ROOT_DIR, m)
        print(f"Processing model: {m}...")
        out["results"][m] = compute_model_metrics(model_dir)

    with open(OUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Summary data saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
