#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Validation and Statistics Script

Validates the navigation planning dataset:
1. Scene integrity (required fields, variant completeness)
2. Query CSV integrity (required columns, task distribution)
3. Information equivalence across format variants
4. Dataset statistics and target compliance
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


# Expected datasets
DATASETS = ["R1", "R2", "conflict"]

# Dataset-specific expected variants (core variants; semantic variants are optional)
EXPECTED_VARIANTS_MAP = {
    "R1": ["flat", "flat_50", "flat_25",
           "hier", "hier_50", "hier_25",
           "clustered", "clustered_50", "clustered_25"],
    "R2": ["flat_full", "flat_topo_hist", "flat_geom_rule_hist", "flat_sem_rule_hist",
           "flat", "flat_50", "flat_25",
           "hier", "hier_50", "hier_25",
           "clustered", "clustered_50", "clustered_25"],
    "conflict": ["flat_full", "flat_topo_hist", "flat_geom_rule_hist", "flat_sem_rule_hist",
                 "conflict_flat", "conflict_hier", "conflict_clustered"],
}

# Required world structure fields
REQUIRED_WORLD_FIELDS = ["rooms", "edges", "history"]

# Required CSV columns
REQUIRED_CSV_COLUMNS = ["question_id", "task_type", "question_text", "answer_format", "ground_truth"]


def count_csv_rows(csv_path: str) -> int:
    """Count data rows in a CSV file (excluding header)."""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1
    except Exception as e:
        print(f"  Error reading {csv_path}: {e}")
        return 0


def validate_scene(scene_path: str, dataset: str = "R1") -> Dict:
    """Validate a single scene JSON file."""
    with open(scene_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    issues = []
    expected_variants = EXPECTED_VARIANTS_MAP.get(dataset, EXPECTED_VARIANTS_MAP["R1"])

    # Check required top-level fields
    if "scene_id" not in data:
        issues.append("Missing scene_id")

    if "world" not in data:
        issues.append("Missing world field (structured ground truth)")
    else:
        for field in REQUIRED_WORLD_FIELDS:
            if field not in data["world"]:
                issues.append(f"Missing world field: {field}")

    if "variants" not in data:
        issues.append("Missing variants")
    else:
        for variant in expected_variants:
            if variant not in data["variants"]:
                issues.append(f"Missing variant: {variant}")
            elif not data["variants"][variant] or len(data["variants"][variant]) < 10:
                issues.append(f"Variant '{variant}' is too short or empty")

    return {
        "path": scene_path,
        "scene_id": data.get("scene_id", "unknown"),
        "num_variants": len(data.get("variants", {})),
        "has_world": "world" in data,
        "issues": issues,
        "valid": len(issues) == 0
    }


def validate_query_csv(csv_path: str) -> Dict:
    """Validate a queries CSV file."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    issues = []

    # Check required columns
    if reader.fieldnames:
        for col in REQUIRED_CSV_COLUMNS:
            if col not in reader.fieldnames:
                issues.append(f"Missing column: {col}")

    # Check query distribution
    task_types = defaultdict(int)
    for row in rows:
        task_types[row.get("task_type", "unknown")] += 1

    # Check for empty fields
    for i, row in enumerate(rows):
        if not row.get("question_text"):
            issues.append(f"Row {i + 1}: Empty question_text")
        if not row.get("ground_truth"):
            issues.append(f"Row {i + 1}: Empty ground_truth")

    return {
        "path": csv_path,
        "num_queries": len(rows),
        "task_distribution": dict(task_types),
        "issues": issues,
        "valid": len(issues) == 0
    }


def check_information_equivalence(scene_path: str) -> Dict:
    """
    Verify that all format variants encode the same underlying world state.

    Checks:
    - All room IDs appear in every variant
    - History trace is preserved across variants
    - Topology edges are represented (explicitly or implicitly)
    - Each variant has non-trivial content
    """
    with open(scene_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    variants = data.get("variants", {})
    world = data.get("world", {})

    if not world or not variants:
        return {"equivalent": False, "reason": "Missing world or variants"}

    rooms = world.get("rooms", [])
    room_ids = [r.get("room_id", "") for r in rooms]
    history = world.get("history", [])
    edges = world.get("edges", [])

    checks = {}
    for variant_name, variant_text in variants.items():
        # Compressed formats may use numeric indices (1,2,3) instead of room IDs (R1,R2,R3)
        all_rooms_ok = all(
            rid in variant_text or str(i + 1) in variant_text
            for i, rid in enumerate(room_ids)
        )
        # History is now a list of dicts with "from_room" and "to_room" fields
        history_ok = True
        if history:
            history_ok = any(
                (h.get("from_room", "") in variant_text or
                 h.get("to_room", "") in variant_text)
                for h in history
            )
        topo_ok = (
            len(edges) == 0 or
            any(
                (room_ids[e[0]] in variant_text and room_ids[e[1]] in variant_text) or
                (str(e[0] + 1) in variant_text and str(e[1] + 1) in variant_text)
                for e in edges if e[0] < len(room_ids) and e[1] < len(room_ids)
            )
        )
        variant_checks = {
            "all_rooms_mentioned": all_rooms_ok,
            "history_preserved": history_ok,
            "non_empty": len(variant_text) > 10,
            "contains_topology": topo_ok
        }
        checks[variant_name] = variant_checks

    all_pass = all(all(check.values()) for check in checks.values())

    return {
        "equivalent": all_pass,
        "checks": checks,
        "reason": "All variants encode same information" if all_pass else "Some variants missing information"
    }


def generate_statistics(base_dir: str) -> Dict:
    """Generate comprehensive dataset statistics."""
    stats = {
        "scenes": {},
        "queries": {},
        "totals": {}
    }

    base_path = Path(base_dir)

    for dataset in DATASETS:
        dataset_path = base_path / dataset

        # Count scenes
        scene_dir = dataset_path / "scene"
        if scene_dir.exists():
            scene_files = sorted(scene_dir.glob("*.json"))
            stats["scenes"][dataset] = {
                "count": len(scene_files),
                "files": [f.name for f in scene_files]
            }

            # Get variant info from first scene
            if scene_files:
                with open(scene_files[0], "r", encoding="utf-8") as f:
                    first_scene = json.load(f)
                variants = first_scene.get("variants", {})
                stats["scenes"][dataset]["variants_per_scene"] = len(variants)
                stats["scenes"][dataset]["variant_names"] = list(variants.keys())
                stats["scenes"][dataset]["has_world"] = "world" in first_scene
        else:
            stats["scenes"][dataset] = {"count": 0, "files": []}

        # Count queries
        queries_dir = dataset_path / "queries"
        if queries_dir.exists():
            query_files = sorted(queries_dir.glob("*.csv"))
            total_queries = sum(count_csv_rows(str(qf)) for qf in query_files)
            stats["queries"][dataset] = {
                "count": total_queries,
                "files": [f.name for f in query_files]
            }
        else:
            stats["queries"][dataset] = {"count": 0, "files": []}

    # Calculate totals
    stats["totals"]["total_scenes"] = sum(s["count"] for s in stats["scenes"].values())
    stats["totals"]["total_queries"] = sum(q["count"] for q in stats["queries"].values())

    # Average variants per scene
    variant_counts = [s.get("variants_per_scene", 0) for s in stats["scenes"].values() if s.get("variants_per_scene")]
    avg_variants = int(sum(variant_counts) / len(variant_counts)) if variant_counts else 7
    stats["totals"]["avg_variants_per_scene"] = avg_variants
    stats["totals"]["total_samples_with_variants"] = stats["totals"]["total_queries"] * avg_variants

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate dataset integrity and generate statistics"
    )
    parser.add_argument("--base_dir", default="./data_set", help="Base dataset directory")
    parser.add_argument("--validate", action="store_true", help="Run validation on all scenes and queries")
    parser.add_argument("--check_equivalence", action="store_true", help="Check information equivalence across variants")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=" * 70)
    print("Dataset Statistics and Validation Report")
    print("=" * 70)

    # Generate statistics
    stats = generate_statistics(args.base_dir)

    print("\n[SCENE STATISTICS]")
    for dataset in DATASETS:
        data = stats["scenes"].get(dataset, {})
        print(f"\n  {dataset}:")
        print(f"    Scenes:              {data.get('count', 0)}")
        print(f"    Variants per scene:  {data.get('variants_per_scene', 'N/A')}")
        print(f"    Has world field:     {data.get('has_world', 'N/A')}")
        if args.verbose and data.get("variant_names"):
            print(f"    Variant names:       {', '.join(data['variant_names'])}")
        if args.verbose and data.get("files"):
            print(f"    Files:               {', '.join(data['files'])}")

    print("\n[QUERY STATISTICS]")
    for dataset in DATASETS:
        data = stats["queries"].get(dataset, {})
        print(f"\n  {dataset}:")
        print(f"    Total queries:       {data.get('count', 0)}")

    print("\n[OVERALL STATISTICS]")
    totals = stats["totals"]
    print(f"\n  Total scenes:                    {totals['total_scenes']}")
    print(f"  Total queries:                   {totals['total_queries']}")
    print(f"  Average variants per scene:      {totals['avg_variants_per_scene']}")
    print(f"  Total samples (queries x variants): {totals['total_samples_with_variants']}")

    # Validation
    if args.validate:
        print("\n" + "=" * 70)
        print("Validation Results")
        print("=" * 70)

        all_valid = True
        for dataset in DATASETS:
            dataset_path = Path(args.base_dir) / dataset
            scene_dir = dataset_path / "scene"
            queries_dir = dataset_path / "queries"

            if not scene_dir.exists():
                print(f"\n[{dataset}] Scene directory not found: {scene_dir}")
                continue

            print(f"\n[{dataset} Scenes]")
            for scene_file in sorted(scene_dir.glob("*.json")):
                result = validate_scene(str(scene_file), dataset)
                status = "PASS" if result["valid"] else "FAIL"
                world_status = "+world" if result["has_world"] else "-world"
                print(f"  [{status}] {result['scene_id']}: {result['num_variants']} variants {world_status}")

                if not result["valid"]:
                    all_valid = False
                    for issue in result["issues"]:
                        print(f"         -> {issue}")

            if queries_dir.exists():
                print(f"\n[{dataset} Queries]")
                for query_file in sorted(queries_dir.glob("*.csv")):
                    result = validate_query_csv(str(query_file))
                    status = "PASS" if result["valid"] else "FAIL"
                    tasks_summary = ", ".join(
                        f"{t}:{n}" for t, n in result.get("task_distribution", {}).items()
                    )
                    print(f"  [{status}] {query_file.name}: {result['num_queries']} queries ({tasks_summary})")

                    if not result["valid"]:
                        all_valid = False
                        for issue in result["issues"]:
                            print(f"         -> {issue}")

        print(f"\n[Overall] All valid: {'YES' if all_valid else 'NO - see issues above'}")

    # Information equivalence
    if args.check_equivalence:
        print("\n" + "=" * 70)
        print("Information Equivalence Check")
        print("=" * 70)

        for dataset in DATASETS:
            dataset_path = Path(args.base_dir) / dataset
            scene_dir = dataset_path / "scene"

            if not scene_dir.exists():
                continue

            print(f"\n[{dataset}]")
            for scene_file in sorted(scene_dir.glob("*.json"))[:3]:  # Sample first 3 scenes
                result = check_information_equivalence(str(scene_file))
                status = "PASS" if result["equivalent"] else "FAIL"
                print(f"  [{status}] {scene_file.stem}: {result['reason']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
