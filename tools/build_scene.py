#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based scene world generator for navigation planning experiments.

Generates structured world data (rooms, edges, objects, history) for:
- R1: Simple navigation scenes (4-6 rooms)
- R2: Complex navigation scenes (6-14 rooms, G1-G5 gradients)
- Conflict: Scenes with intentionally conflicting information cues

This module ONLY generates world structure. All text serialization
(format variants, dimension ablations, conflict rendering) is handled
by scene_serializer.py.

No human annotation or LLM-based annotation is used — generation is
purely rule-based and deterministic given a seed.
"""

import math
import random
from typing import Dict, List, Tuple

# -------------------------
# Complexity schedule (10 scenes -> 5 gradients)
# -------------------------
GRADIENTS = {
    "G1": {"rooms": 6,  "room_types": 3, "obj_per_room": 4,  "obj_types": 2, "history_steps": 2,  "label": "Basic cognition"},
    "G2": {"rooms": 8,  "room_types": 4, "obj_per_room": 6,  "obj_types": 3, "history_steps": 4,  "label": "Intermediate planning"},
    "G3": {"rooms": 10, "room_types": 5, "obj_per_room": 8,  "obj_types": 4, "history_steps": 6,  "label": "Complex modeling"},
    "G4": {"rooms": 12, "room_types": 6, "obj_per_room": 10, "obj_types": 5, "history_steps": 8,  "label": "High-load retrieval"},
    "G5": {"rooms": 14, "room_types": 7, "obj_per_room": 12, "obj_types": 6, "history_steps": 10, "label": "Extreme reasoning"},
}

# scene 1-2 => G1, 3-4 => G2, ..., 9-10 => G5
SCENE_TO_G = {1: "G1", 2: "G1", 3: "G2", 4: "G2", 5: "G3",
              6: "G3", 7: "G4", 8: "G4", 9: "G5", 10: "G5"}

ROOM_TYPE_BANK = [
    "Reception", "Corridor", "Office", "Library", "Kitchen", "Lab", "MeetingRoom",
    "Storage", "Classroom", "Studio", "Bedroom", "Bathroom", "Lounge", "Workshop"
]

OBJ_TYPE_BANK = [
    ("Furniture", ["Table", "Chair", "Bench", "Sofa", "Desk", "Shelf", "Cabinet"]),
    ("IT", ["Server", "Monitor", "Router", "Laptop", "Tablet", "Scanner", "Printer"]),
    ("Storage", ["Locker", "Rack", "Safe", "Box", "Drawer", "Crate", "FileCabinet"]),
    ("Security", ["Camera", "Sensor", "Alarm", "BadgeReader", "Lock", "Gate", "Light"]),
    ("Tools", ["Wrench", "Hammer", "Drill", "Screwdriver", "Cutter", "Tape", "Gloves"]),
    ("Digital", ["Stylus", "Kindle", "Phone", "Projector", "Microphone", "Speaker", "Mixer"]),
    ("Instruments", ["Piano", "Guitar", "Violin", "Drum", "Flute", "Amp", "Stool"]),
]

GRID_STEP = 4


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def build_grid_positions(n: int, seed: int) -> List[Tuple[int, int]]:
    """Create n positions on a grid, spaced by GRID_STEP in a snake layout."""
    random.seed(seed)
    side = max(3, int(math.ceil(math.sqrt(n))))
    positions = []
    for i in range(n):
        r = i // side
        c = i % side if r % 2 == 0 else (side - 1 - (i % side))
        positions.append((c * GRID_STEP, (side - 1 - r) * GRID_STEP))
    return positions[:n]


def build_edges_from_geometry(positions: List[Tuple[int, int]],
                              seed: int,
                              extra_loops: int = 1) -> List[Tuple[int, int]]:
    """Create edges using geometry rule (Manhattan distance == GRID_STEP)
    then add extra loops. Ensures connectivity via spanning tree."""
    random.seed(seed)
    n = len(positions)
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            if manhattan(positions[i], positions[j]) == GRID_STEP:
                cand.append((i, j))

    # Build spanning tree
    edges = []
    visited = {0}
    frontier = [0]
    while len(visited) < n:
        found = None
        random.shuffle(frontier)
        for u in frontier:
            options = [v for (a, b) in cand
                       for v in ([b] if a == u else ([a] if b == u else []))
                       if v not in visited]
            if options:
                v = random.choice(options)
                found = (u, v)
                break
        if found is None:
            u = max(visited)
            v = u + 1
            if v < n:
                found = (u, v)
            else:
                break
        u, v = found
        edges.append((min(u, v), max(u, v)))
        visited.add(v)
        frontier.append(v)

    # Add extra loops
    extra = [e for e in cand if e not in edges]
    random.shuffle(extra)
    edges.extend(extra[:extra_loops])

    return sorted(list(set(edges)))


def sample_room_types(k: int, seed: int) -> List[str]:
    random.seed(seed)
    pool = ROOM_TYPE_BANK[:]
    random.shuffle(pool)
    return pool[:k]


def sample_object_schema(obj_types: int, seed: int) -> List[Tuple[str, List[str]]]:
    random.seed(seed)
    pool = OBJ_TYPE_BANK[:]
    random.shuffle(pool)
    return pool[:obj_types]


def assign_objects_to_rooms(n_rooms: int, obj_per_room: int,
                            obj_schema: List, seed: int) -> Dict[int, List[Tuple[str, str]]]:
    random.seed(seed)
    out = {}
    for r in range(n_rooms):
        objs = []
        for _ in range(obj_per_room):
            cat, names = random.choice(obj_schema)
            name = random.choice(names)
            objs.append((cat, name))
        out[r] = objs
    return out


def build_history_path(edges: List[Tuple[int, int]], steps: int, seed: int) -> List[int]:
    """Generate a plausible movement path of length steps+1 over the graph."""
    random.seed(seed)
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    cur = 0
    path = [cur]
    for _ in range(steps):
        nxt = random.choice(adj.get(cur, [cur]))
        path.append(nxt)
        cur = nxt
    return path


def generate_world(scene_idx: int, seed: int = 1234) -> dict:
    """Generate a structured world dict for a given scene index.

    Returns a dict with scene_name, gradient, rooms, edges, objects, history, rules.
    This is the single source of truth — no text variants are generated here.
    """
    g = SCENE_TO_G.get(scene_idx, "G1")
    cfg = GRADIENTS[g]
    n_rooms = cfg["rooms"]
    n_types = cfg["room_types"]
    obj_per = cfg["obj_per_room"]
    obj_types = cfg["obj_types"]
    hist_steps = cfg["history_steps"]

    scene_name = f"scene_complex_{scene_idx:02d}"
    room_ids = [f"R{i+1}" for i in range(n_rooms)]

    types = sample_room_types(n_types, seed + 17)
    room_type = [types[i % n_types] for i in range(n_rooms)]

    pos = build_grid_positions(n_rooms, seed + 31)

    extra_loops = max(1, (scene_idx // 2))
    edges = build_edges_from_geometry(pos, seed + 43, extra_loops=extra_loops)

    schema = sample_object_schema(obj_types, seed + 59)
    objects = assign_objects_to_rooms(n_rooms, obj_per, schema, seed + 61)

    hist_path = build_history_path(edges, steps=hist_steps, seed=seed + 71)

    rooms = []
    for i in range(n_rooms):
        rooms.append({
            "idx": i,
            "room_id": room_ids[i],
            "type": room_type[i],
            "x": pos[i][0],
            "y": pos[i][1],
        })

    rules = [
        "When generating a path, output the shortest valid path under the implied connectivity unless the question specifies otherwise.",
        "Do not repeat a room in the path unless necessary.",
        "If multiple shortest paths exist, choose the lexicographically smallest by room ID sequence.",
    ]

    return {
        "scene_name": scene_name,
        "gradient": g,
        "gradient_label": cfg["label"],
        "rooms": rooms,
        "edges": edges,
        "objects": objects,
        "history": hist_path,
        "rules": rules,
    }


def generate_simple_world(scene_idx: int, seed: int = 1234) -> dict:
    """Generate a small simple world (4-6 rooms) for R1 experiments."""
    n_rooms = 4 + (scene_idx % 3)  # 4-6 rooms
    scene_name = f"scene_simple_{scene_idx:02d}"
    room_ids = [f"R{i+1}" for i in range(n_rooms)]

    random.seed(seed)
    # Simple room types for small scenes
    simple_types = ["Entrance", "Kitchen", "Living Room", "Bedroom", "Bathroom", "Study"]
    room_type = [simple_types[i % len(simple_types)] for i in range(n_rooms)]

    # Simple grid layout
    side = max(2, int(math.ceil(math.sqrt(n_rooms))))
    pos = []
    for i in range(n_rooms):
        r, c = i // side, i % side
        pos.append((c * GRID_STEP, r * GRID_STEP))

    # Simple chain + cycles
    edges = []
    for i in range(n_rooms - 1):
        if i + 1 < n_rooms:
            edges.append((i, i + 1))
    if n_rooms >= 4:
        edges.append((n_rooms - 1, 0))  # close the loop

    # Simple objects
    simple_obj_bank = [
        ("Furniture", ["Sofa", "Table", "Chair", "Bed", "Desk"]),
        ("Appliance", ["Fridge", "Microwave", "Oven"]),
        ("Storage", ["Shelf", "Wardrobe", "Cabinet"]),
    ]
    objects = {}
    for r in range(n_rooms):
        objects[r] = [(random.choice(simple_obj_bank)[0],
                       random.choice(random.choice(simple_obj_bank)[1]))
                      for _ in range(2)]

    # Short history
    hist = list(range(min(3, n_rooms)))

    rooms = []
    for i in range(n_rooms):
        rooms.append({
            "idx": i,
            "room_id": room_ids[i],
            "type": room_type[i],
            "x": pos[i][0],
            "y": pos[i][1],
        })

    rules = [
        "When generating a path, output the shortest valid path under the implied connectivity.",
        "Do not repeat a room in the path unless necessary.",
    ]

    return {
        "scene_name": scene_name,
        "gradient": "G1",
        "gradient_label": "Simple navigation",
        "rooms": rooms,
        "edges": edges,
        "objects": objects,
        "history": hist,
        "rules": rules,
    }
