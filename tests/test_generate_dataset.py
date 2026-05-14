#!/usr/bin/env python3
"""Tests for the dataset generation pipeline."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.generate_dataset import (
    generate_queries_for_scene,
    write_queries_csv,
)
from tools.scene_serializer import (
    generate_all_variants,
    generate_dimension_variants,
    generate_conflict_variants,
    verify_equivalence,
    parse_world,
)

SAMPLE_WORLD = {
    "scene_name": "test_scene",
    "gradient": "G1",
    "gradient_label": "Basic",
    "rooms": [
        {"idx": 0, "room_id": "R1", "type": "Hall", "x": 0, "y": 0},
        {"idx": 1, "room_id": "R2", "type": "Kitchen", "x": 4, "y": 0},
        {"idx": 2, "room_id": "R3", "type": "Bedroom", "x": 4, "y": 4},
    ],
    "edges": [[0, 1], [1, 2]],
    "objects": {
        "0": [["Furniture", "Sofa"]],
        "1": [["Appliance", "Fridge"]],
        "2": [["Furniture", "Bed"]],
    },
    "history": [0, 1, 2],
    "rules": ["Use shortest path."],
}


def test_generate_queries():
    scene = {"world": SAMPLE_WORLD, "variants": {}}
    queries = generate_queries_for_scene(scene, num_queries=50)

    assert len(queries) == 50, f"Expected 50 queries, got {len(queries)}"

    for q in queries:
        assert "question_id" in q
        assert "task_type" in q
        assert "question_text" in q
        assert "answer_format" in q
        assert "ground_truth" in q

    task_types = set(q["task_type"] for q in queries)
    expected_types = {"ObjectLocation", "GeometryYN", "TopologyYN", "ReachabilityYN", "PathGen"}
    assert task_types == expected_types, f"Missing task types: {expected_types - task_types}"


def test_structural_variants():
    variants = generate_all_variants(SAMPLE_WORLD)
    expected = ["flat", "hier", "hier_50", "hier_25", "clustered", "clustered_50", "clustered_25"]
    for v in expected:
        assert v in variants, f"Missing structural variant: {v}"
        assert len(variants[v]) > 10, f"Variant {v} too short"


def test_dimension_variants():
    variants = generate_dimension_variants(SAMPLE_WORLD)
    expected = ["flat_full", "flat_topo_hist", "flat_geom_rule_hist", "flat_sem_rule_hist"]
    for v in expected:
        assert v in variants, f"Missing dimension variant: {v}"
        assert len(variants[v]) > 10, f"Variant {v} too short"


def test_conflict_variants():
    variants = generate_conflict_variants(SAMPLE_WORLD, seed=42)
    assert "conflict_topo" in variants
    assert "conflict_geom" in variants
    assert "conflict_sem" in variants
    assert "flat_full" in variants  # base variants included


def test_all_variants_have_rooms():
    world = parse_world(SAMPLE_WORLD)
    for gen_func in [generate_all_variants, generate_dimension_variants]:
        variants = gen_func(SAMPLE_WORLD)
        results = verify_equivalence(variants, world)
        for name, result in results.items():
            assert result, f"Equivalence check failed for variant: {name}"


if __name__ == "__main__":
    test_generate_queries()
    test_structural_variants()
    test_dimension_variants()
    test_conflict_variants()
    test_all_variants_have_rooms()
    print("All generation tests passed!")
