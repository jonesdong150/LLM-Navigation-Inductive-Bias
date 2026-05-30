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
    generate_semantic_variation,
    generate_semantic_conflict,
    verify_equivalence,
    parse_world,
)

SAMPLE_WORLD = {
    "scene_name": "test_scene",
    "gradient": "G1",
    "gradient_label": "Basic",
    "rooms": [
        {"idx": 0, "room_id": "R1", "canonical": "Hall", "synonyms": ["Foyer", "Entrance"],
         "abbr": "Hal", "attributes": ["Bright"], "x": 0, "y": 0, "w": 4, "h": 4},
        {"idx": 1, "room_id": "R2", "canonical": "Kitchen", "synonyms": ["Pantry", "Cookhouse"],
         "abbr": "Kit", "attributes": ["Clean"], "x": 4, "y": 0, "w": 4, "h": 4},
        {"idx": 2, "room_id": "R3", "canonical": "Bedroom", "synonyms": ["Chamber", "Sleeping Quarters"],
         "abbr": "Bed", "attributes": ["Quiet"], "x": 4, "y": 4, "w": 4, "h": 4},
    ],
    "edges": [[0, 1], [1, 2]],
    "objects": {
        "0": [
            {"obj_id": "O1", "canonical": "Sofa", "synonyms": ["Couch", "Settee"],
             "abbr": "Sof", "attributes": ["Red"], "name": "Sofa_1"},
        ],
        "1": [
            {"obj_id": "O2", "canonical": "Table", "synonyms": ["Counter", "Bench"],
             "abbr": "Tbl", "attributes": ["Wooden"], "name": "Table_1"},
        ],
        "2": [
            {"obj_id": "O3", "canonical": "Desk", "synonyms": ["Table", "Workbench"],
             "abbr": "Dsk", "attributes": ["Small"], "name": "Desk_1"},
        ],
    },
    "history": [
        {"object": "Key", "from_room": "R1", "to_room": "R2"},
        {"object": "Key", "from_room": "R2", "to_room": "R3"},
    ],
    "containment": {"Sofa_1": "R1", "Table_1": "R2", "Desk_1": "R3"},
    "parallel_rooms": [("R1", "R2"), ("R2", "R3")],
    "parallel_objects": [],
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
    """Test structural variants at 100%/50%/25% retention."""
    variants = generate_all_variants(SAMPLE_WORLD)
    expected = [
        "flat", "flat_50", "flat_25",
        "hier", "hier_50", "hier_25",
        "clustered", "clustered_50", "clustered_25",
    ]
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
    """Test that conflict variants include base + semantic conflict (paper's C2 regime)."""
    variants = generate_conflict_variants(SAMPLE_WORLD, seed=42)
    # Base dimension variants
    assert "flat_full" in variants
    assert "flat_topo_hist" in variants
    # Semantic conflict variants (paper's C2: duplicate labels)
    assert "conflict_flat" in variants
    assert "conflict_hier" in variants
    assert "conflict_clustered" in variants


def test_semantic_variation():
    """Test semantic variation generation."""
    sem_variants = generate_semantic_variation(SAMPLE_WORLD, seed=42)
    # Should have same structure as regular variants
    expected = [
        "flat", "flat_50", "flat_25",
        "hier", "hier_50", "hier_25",
        "clustered", "clustered_50", "clustered_25",
    ]
    for v in expected:
        assert v in sem_variants, f"Missing semantic variation variant: {v}"
        assert len(sem_variants[v]) > 10, f"Semantic variant {v} too short"


def test_semantic_conflict():
    """Test semantic conflict generation."""
    conflict_vars = generate_semantic_conflict(SAMPLE_WORLD, seed=42)
    assert "conflict_flat" in conflict_vars
    assert "conflict_hier" in conflict_vars
    assert "conflict_clustered" in conflict_vars


def test_all_variants_have_rooms():
    """Test information equivalence across all variants."""
    world = parse_world(SAMPLE_WORLD)
    for gen_func in [generate_all_variants, generate_dimension_variants]:
        variants = gen_func(SAMPLE_WORLD)
        results = verify_equivalence(variants, world)
        for name, result in results.items():
            assert result, f"Equivalence check failed for variant: {name}"


def test_query_ground_truth_consistency():
    """Test that query ground truths are consistent with world data."""
    scene = {"world": SAMPLE_WORLD, "variants": {}}
    queries = generate_queries_for_scene(scene, num_queries=50)

    # Check ObjectLocation queries
    obj_queries = [q for q in queries if q["task_type"] == "ObjectLocation"]
    for q in obj_queries:
        # Ground truth should be a valid room ID
        assert q["ground_truth"] in ["R1", "R2", "R3"], \
            f"Invalid room ID in ObjectLocation: {q['ground_truth']}"

    # Check TopologyYN queries
    topo_queries = [q for q in queries if q["task_type"] == "TopologyYN"]
    for q in topo_queries:
        assert q["ground_truth"] in ["YES", "NO"], \
            f"Invalid answer in TopologyYN: {q['ground_truth']}"

    # Check PathGen queries
    path_queries = [q for q in queries if q["task_type"] == "PathGen"]
    for q in path_queries:
        path = q["ground_truth"]
        if path != "UNREACHABLE":
            parts = path.split("->")
            assert parts[0] == "R1" or parts[0] == "R2" or parts[0] == "R3", \
                f"Invalid path start: {parts[0]}"


def test_compression_ordering():
    """Test that 25% retention produces shorter text than 100%."""
    variants = generate_all_variants(SAMPLE_WORLD)
    assert len(variants["flat_25"]) < len(variants["flat"]), \
        "Flat 25% should be shorter than 100%"
    assert len(variants["hier_25"]) < len(variants["hier"]), \
        "Hier 25% should be shorter than 100%"
    assert len(variants["clustered_25"]) < len(variants["clustered_50"]), \
        "Clustered 25% should be shorter than 50%"


if __name__ == "__main__":
    test_generate_queries()
    test_structural_variants()
    test_dimension_variants()
    test_conflict_variants()
    test_semantic_variation()
    test_semantic_conflict()
    test_all_variants_have_rooms()
    test_query_ground_truth_consistency()
    test_compression_ordering()
    print("All generation tests passed!")
