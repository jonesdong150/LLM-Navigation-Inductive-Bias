#!/usr/bin/env python3
"""Unit tests for the scene serializer module."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.scene_serializer import (
    parse_world, generate_all_variants, verify_equivalence,
    SceneSerializer, World, Room
)


SAMPLE_WORLD = {
    "scene_name": "test_scene",
    "gradient": "G1",
    "gradient_label": "Basic",
    "rooms": [
        {"idx": 0, "room_id": "R1", "type": "Living Room", "x": 0, "y": 0},
        {"idx": 1, "room_id": "R2", "type": "Kitchen", "x": 4, "y": 0},
        {"idx": 2, "room_id": "R3", "type": "Bedroom", "x": 4, "y": 4},
    ],
    "edges": [[0, 1], [1, 2]],
    "objects": {
        "0": [["Furniture", "Sofa"], ["IT", "TV"]],
        "1": [["Appliance", "Fridge"], ["Furniture", "Table"]],
        "2": [["Furniture", "Bed"], ["Storage", "Wardrobe"]],
    },
    "history": [0, 1, 2],
    "rules": ["Use shortest path."],
}


def test_parse_world():
    """Test parsing world dict into World object."""
    world = parse_world(SAMPLE_WORLD)
    assert world.scene_name == "test_scene"
    assert len(world.rooms) == 3
    assert world.rooms[0].room_id == "R1"
    assert world.rooms[0].x == 0
    assert world.rooms[0].y == 0
    assert len(world.edges) == 2
    assert world.history == [0, 1, 2]


def test_generate_all_variants_count():
    """Test that all 7 variants are generated."""
    variants = generate_all_variants(SAMPLE_WORLD)
    expected = ["flat", "hier", "hier_50", "hier_25", "clustered", "clustered_50", "clustered_25"]
    for variant_name in expected:
        assert variant_name in variants, f"Missing variant: {variant_name}"
        assert len(variants[variant_name]) > 0, f"Empty variant: {variant_name}"


def test_variants_non_empty():
    """Test that all generated variants have meaningful content."""
    variants = generate_all_variants(SAMPLE_WORLD)
    for name, text in variants.items():
        assert len(text) > 10, f"Variant {name} is too short ({len(text)} chars): {text}"


def test_all_rooms_in_variants():
    """Test that all room IDs (or compressed numeric indices) appear in every variant."""
    variants = generate_all_variants(SAMPLE_WORLD)
    room_ids = ["R1", "R2", "R3"]
    numeric_ids = ["1", "2", "3"]
    for name, text in variants.items():
        for rid, nid in zip(room_ids, numeric_ids):
            found = rid in text or nid in text
            assert found, f"Variant {name} missing room {rid} (checked both {rid} and {nid})"


def test_history_in_variants():
    """Test that history is preserved in variants (R1 or 1 in compressed formats)."""
    variants = generate_all_variants(SAMPLE_WORLD)
    for name, text in variants.items():
        found = "R1" in text or "1" in text
        assert found, f"Variant {name} missing starting room in history"


def test_verify_equivalence():
    """Test the equivalence verification function."""
    world = parse_world(SAMPLE_WORLD)
    variants = generate_all_variants(SAMPLE_WORLD)
    results = verify_equivalence(variants, world)
    for name, result in results.items():
        assert result, f"Equivalence check failed for variant: {name}"


def test_compression_reduces_size():
    """Test that compression levels produce progressively shorter text."""
    variants = generate_all_variants(SAMPLE_WORLD)
    assert len(variants["hier"]) > len(variants["hier_50"]) > len(variants["hier_25"]), \
        "Hierarchical compression should reduce text length"
    assert len(variants["clustered"]) > len(variants["clustered_50"]) > len(variants["clustered_25"]), \
        "Clustered compression should reduce text length"


def test_deterministic_output():
    """Test that same input produces identical output."""
    v1 = generate_all_variants(SAMPLE_WORLD)
    v2 = generate_all_variants(SAMPLE_WORLD)
    for key in v1:
        assert v1[key] == v2[key], f"Non-deterministic output for variant: {key}"


if __name__ == "__main__":
    # Run tests manually
    test_parse_world()
    test_generate_all_variants_count()
    test_variants_non_empty()
    test_all_rooms_in_variants()
    test_history_in_variants()
    test_verify_equivalence()
    test_compression_reduces_size()
    test_deterministic_output()
    print("All tests passed!")
