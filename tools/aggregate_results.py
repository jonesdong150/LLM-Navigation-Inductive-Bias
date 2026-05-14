#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate raw evaluation results into final_stats CSV files.

Reads per-scene summary JSONs from eval/ and produces the CSV files
needed by the figure-generation scripts.

Usage:
    python tools/aggregate_results.py

Output:
    final_stats/r1_structure_scaling.csv
    final_stats/r2_complexity_resilience.csv
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
OUT_DIR = PROJECT_ROOT / "final_stats"


def collect_r1_stats() -> List[Dict]:
    """Collect R1 (structure scaling) stats from eval/R1/{model}/scene_*_summary.json."""
    rows = []
    r1_dir = EVAL_DIR / "R1"
    if not r1_dir.exists():
        print(f"[WARN] eval/R1 not found at {r1_dir}")
        return rows

    for model_dir in sorted(r1_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for summary_file in sorted(model_dir.glob("scene_*_summary.json")):
            scene_name = summary_file.stem.replace("_summary", "")

            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for variant_name, variant_data in data.items():
                if not isinstance(variant_data, dict):
                    continue
                if "overall_accuracy" not in variant_data:
                    continue

                row = {
                    "Model": model_name,
                    "Scene": scene_name,
                    "Variant": variant_name,
                    "Overall_Acc": variant_data.get("overall_accuracy", 0.0),
                    "Format_Violation_Rate": variant_data.get("overall_fvr", 0.0),
                }

                # Per-task accuracies
                tasks = variant_data.get("tasks", {})
                for task_type, task_info in tasks.items():
                    if isinstance(task_info, dict):
                        row[f"{task_type}_Acc"] = task_info.get("accuracy", 0.0)
                        row[f"{task_type}_FVR"] = task_info.get("format_violation_rate", 0.0)

                rows.append(row)

    return rows


def collect_r2_stats() -> List[Dict]:
    """Collect R2 (complexity resilience) stats from eval/R2/{model}/scene_*_summary.json."""
    rows = []
    r2_dir = EVAL_DIR / "R2"
    if not r2_dir.exists():
        print(f"[WARN] eval/R2 not found at {r2_dir}")
        return rows

    for model_dir in sorted(r2_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for summary_file in sorted(model_dir.glob("scene_complex_*_summary.json")):
            scene_name = summary_file.stem.replace("_summary", "")

            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Determine grade from scene name
            # scene_complex_01-02 -> G1, 03-04 -> G2, etc.
            import re
            m = re.search(r'(\d+)', scene_name)
            scene_num = int(m.group(1)) if m else 1
            grade_num = (scene_num + 1) // 2
            grade = f"G{grade_num}"

            for variant_name, variant_data in data.items():
                if not isinstance(variant_data, dict):
                    continue
                if "overall_accuracy" not in variant_data:
                    continue

                row = {
                    "Model": model_name,
                    "Scene": scene_name,
                    "Variant": variant_name,
                    "Grade": grade,
                    "Overall_Acc": variant_data.get("overall_accuracy", 0.0),
                    "Format_Violation_Rate": variant_data.get("overall_fvr", 0.0),
                }

                tasks = variant_data.get("tasks", {})
                for task_type, task_info in tasks.items():
                    if isinstance(task_info, dict):
                        row[f"{task_type}_Acc"] = task_info.get("accuracy", 0.0)
                        row[f"{task_type}_FVR"] = task_info.get("format_violation_rate", 0.0)

                rows.append(row)

    return rows


def collect_conflict_stats() -> List[Dict]:
    """Collect conflict stats from eval/conflict/{model}/scene_*_summary.json."""
    rows = []
    conflict_dir = EVAL_DIR / "conflict"
    if not conflict_dir.exists():
        print(f"[WARN] eval/conflict not found at {conflict_dir}")
        return rows

    for model_dir in sorted(conflict_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for summary_file in sorted(model_dir.glob("scene_conflict_*_summary.json")):
            scene_name = summary_file.stem.replace("_summary", "")

            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for variant_name, variant_data in data.items():
                if not isinstance(variant_data, dict):
                    continue
                if "overall_accuracy" not in variant_data:
                    continue

                row = {
                    "Model": model_name,
                    "Scene": scene_name,
                    "Variant": variant_name,
                    "Overall_Acc": variant_data.get("overall_accuracy", 0.0),
                    "Format_Violation_Rate": variant_data.get("overall_fvr", 0.0),
                }

                tasks = variant_data.get("tasks", {})
                for task_type, task_info in tasks.items():
                    if isinstance(task_info, dict):
                        row[f"{task_type}_Acc"] = task_info.get("accuracy", 0.0)
                        row[f"{task_type}_FVR"] = task_info.get("format_violation_rate", 0.0)

                rows.append(row)

    return rows


def write_csv(rows: List[Dict], output_path: str):
    """Write rows to CSV, auto-detecting all fieldnames."""
    if not rows:
        print(f"[WARN] No data to write to {output_path}")
        return

    # Collect all fieldnames across all rows
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[OK] Wrote {len(rows)} rows to {output_path}")


def main():
    print("=" * 60)
    print("Evaluation Results Aggregator")
    print("=" * 60)

    # R1 stats
    print("\n[R1] Aggregating structure scaling results...")
    r1_rows = collect_r1_stats()
    if r1_rows:
        write_csv(r1_rows, str(OUT_DIR / "r1_structure_scaling.csv"))
    else:
        print("[SKIP] No R1 evaluation data found (eval/R1/ is empty or missing).")
        print("       Run inference first, then re-run this script.")

    # R2 stats
    print("\n[R2] Aggregating complexity resilience results...")
    r2_rows = collect_r2_stats()
    if r2_rows:
        write_csv(r2_rows, str(OUT_DIR / "r2_complexity_resilience.csv"))
    else:
        print("[SKIP] No R2 evaluation data found (eval/R2/ is empty or missing).")
        print("       Run inference first, then re-run this script.")

    # Conflict stats
    print("\n[Conflict] Aggregating conflict sensitivity results...")
    conflict_rows = collect_conflict_stats()
    if conflict_rows:
        write_csv(conflict_rows, str(OUT_DIR / "conflict_sensitivity.csv"))
    else:
        print("[SKIP] No conflict evaluation data found (eval/conflict/ is empty or missing).")
        print("       Run inference first, then re-run this script.")

    print("\n" + "=" * 60)
    print("Aggregation complete.")
    print(f"Output directory: {OUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
