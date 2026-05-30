#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based scene world generator for navigation planning experiments.

Generates structured world data using the predefined knowledge base:
- Rooms with canonical names, synonyms, abbreviations, and attributes
- Objects with canonical names, synonyms, abbreviations, and attributes
- Canonical scene graph with containment and parallel relations
- Topology, geometry (x,y,w,h), and history

All scene elements are sampled from the knowledge base per the paper's
specification. No human annotation or LLM-based annotation is used.
"""

import math
import random
from typing import Dict, List, Tuple

from tools.knowledge_base import (
    GRADIENTS, SCENE_TO_G,
    sample_rooms, sample_objects, sample_room_attrs, sample_object_attrs,
    get_room_abbr, get_object_abbr,
)

GRID_STEP = 4
ROOM_WIDTH = 4
ROOM_HEIGHT = 4


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


def assign_objects_to_rooms(n_rooms: int, obj_per_room: int,
                            obj_schema: List[Dict], seed: int,
                            room_type_indices: List[int]) -> Dict[int, List[Dict]]:
    """Assign objects from knowledge base to rooms with attributes.

    Each object uses a different synonym from its category to ensure
    meaningful grouping in the hier format (e.g., Printer(Scanner, Copier)).

    Returns dict mapping room_idx -> list of object dicts with:
      obj_id (e.g. "O1"), canonical, synonyms, abbr, attributes, name, display_name
    """
    random.seed(seed)
    out = {}
    obj_counter = {}
    for r in range(n_rooms):
        objs = []
        for j in range(obj_per_room):
            # Randomly pick an object type for each object
            obj_entry = random.choice(obj_schema)
            canonical = obj_entry["canonical"]
            obj_counter[canonical] = obj_counter.get(canonical, 0) + 1
            # Assign 1-2 random object attributes
            n_attrs = random.randint(1, 2)
            attrs = sample_object_attrs(n_attrs, seed=random.randint(0, 99999))
            # Pick a display name from canonical + synonyms to avoid repetition
            all_names = [canonical] + obj_entry["synonyms"]
            display_name = all_names[j % len(all_names)]
            objs.append({
                "obj_id": obj_entry["oid"],
                "canonical": canonical,
                "synonyms": obj_entry["synonyms"],
                "abbr": obj_entry["abbr"],
                "attributes": attrs,
                "name": f"{canonical}_{obj_counter[canonical]}",
                "display_name": display_name,
            })
        out[r] = objs
    return out


def build_history_path(edges: List[Tuple[int, int]], steps: int, seed: int,
                       n_rooms: int) -> List[Dict]:
    """Generate a plausible movement history as a list of event dicts.

    Each event: {"object": "Key", "from_room_idx": int, "to_room_idx": int}
    """
    random.seed(seed)
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    cur = 0
    events = []
    for _ in range(steps):
        nxt = random.choice(adj.get(cur, [cur]))
        events.append({
            "object": "Key",
            "from_room_idx": cur,
            "to_room_idx": nxt,
        })
        cur = nxt
    return events


def generate_world(scene_idx: int, seed: int = 1234) -> dict:
    """Generate a structured world dict for a given scene index.

    Uses knowledge base for all room/object types, attributes, synonyms.
    Returns a dict with scene_name, gradient, rooms, objects, edges, history,
    containment, parallel_rooms, parallel_objects, rules.
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

    # Sample room types from knowledge base
    room_kb_entries = sample_rooms(n_types, seed + 17)
    room_type_assign = [room_kb_entries[i % n_types] for i in range(n_rooms)]

    # Assign room attributes (1 attribute per room)
    room_attr_lists = []
    for i in range(n_rooms):
        n_attrs = random.Random(seed + 100 + i).randint(1, 2)
        attrs = sample_room_attrs(n_attrs, seed + 200 + i)
        room_attr_lists.append(attrs)

    # Build positions
    pos = build_grid_positions(n_rooms, seed + 31)

    # Build edges
    extra_loops = max(1, (scene_idx // 2))
    edges = build_edges_from_geometry(pos, seed + 43, extra_loops=extra_loops)

    # Sample object types from knowledge base
    obj_kb_entries = sample_objects(obj_types, seed + 59)

    # Assign objects to rooms
    objects = assign_objects_to_rooms(n_rooms, obj_per, obj_kb_entries, seed + 61,
                                     list(range(n_types)))

    # Build history
    hist_events = build_history_path(edges, steps=hist_steps, seed=seed + 71,
                                     n_rooms=n_rooms)

    # Build room list with KB metadata
    rooms = []
    for i in range(n_rooms):
        entry = room_type_assign[i]
        rooms.append({
            "idx": i,
            "room_id": room_ids[i],
            "canonical": entry["canonical"],
            "synonyms": entry["synonyms"],
            "abbr": entry["abbr"],
            "attributes": room_attr_lists[i],
            "x": pos[i][0],
            "y": pos[i][1],
            "w": ROOM_WIDTH,
            "h": ROOM_HEIGHT,
        })

    # Build containment relations: obj_name -> room_id
    containment = {}
    for r_idx, obj_list in objects.items():
        for obj in obj_list:
            containment[obj["name"]] = room_ids[r_idx]

    # Build parallel room relations (rooms connected by edges)
    parallel_rooms = [(room_ids[e[0]], room_ids[e[1]]) for e in edges]

    # Build parallel object relations (objects in the same room)
    parallel_objects = []
    for r_idx, obj_list in objects.items():
        for i in range(len(obj_list)):
            for j in range(i + 1, len(obj_list)):
                parallel_objects.append((obj_list[i]["name"], obj_list[j]["name"]))

    # Build history in the new format
    history = []
    for evt in hist_events:
        history.append({
            "object": evt["object"],
            "from_room": room_ids[evt["from_room_idx"]],
            "to_room": room_ids[evt["to_room_idx"]],
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
        "history": history,
        "containment": containment,
        "parallel_rooms": parallel_rooms,
        "parallel_objects": parallel_objects,
        "rules": rules,
    }


def generate_simple_world(scene_idx: int, seed: int = 1234) -> dict:
    """Generate a small simple world (4-6 rooms) for R1 experiments.

    Uses knowledge base for all room/object types, attributes, synonyms.
    """
    n_rooms = 4 + (scene_idx % 3)  # 4-6 rooms
    scene_name = f"scene_simple_{scene_idx:02d}"
    room_ids = [f"R{i+1}" for i in range(n_rooms)]

    random.seed(seed)

    # Sample room types from knowledge base
    room_kb_entries = sample_rooms(n_rooms, seed + 17)
    room_type_assign = [room_kb_entries[i % len(room_kb_entries)] for i in range(n_rooms)]

    # Assign room attributes
    room_attr_lists = []
    for i in range(n_rooms):
        attrs = sample_room_attrs(1, seed + 200 + i)
        room_attr_lists.append(attrs)

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

    # Sample objects from knowledge base
    obj_kb_entries = sample_objects(3, seed + 59)
    objects = {}
    obj_counter = {}
    for r in range(n_rooms):
        objs = []
        for j in range(2):
            obj_entry = random.choice(obj_kb_entries)
            canonical = obj_entry["canonical"]
            obj_counter[canonical] = obj_counter.get(canonical, 0) + 1
            attrs = sample_object_attrs(1, seed=random.randint(0, 99999))
            all_names = [canonical] + obj_entry["synonyms"]
            display_name = all_names[j % len(all_names)]
            objs.append({
                "obj_id": obj_entry["oid"],
                "canonical": canonical,
                "synonyms": obj_entry["synonyms"],
                "abbr": obj_entry["abbr"],
                "attributes": attrs,
                "name": f"{canonical}_{obj_counter[canonical]}",
                "display_name": display_name,
            })
        objects[r] = objs

    # Short history
    hist_events = []
    for i in range(min(2, n_rooms - 1)):
        hist_events.append({
            "object": "Key",
            "from_room_idx": i,
            "to_room_idx": i + 1,
        })

    # Build room list
    rooms = []
    for i in range(n_rooms):
        entry = room_type_assign[i]
        rooms.append({
            "idx": i,
            "room_id": room_ids[i],
            "canonical": entry["canonical"],
            "synonyms": entry["synonyms"],
            "abbr": entry["abbr"],
            "attributes": room_attr_lists[i],
            "x": pos[i][0],
            "y": pos[i][1],
            "w": ROOM_WIDTH,
            "h": ROOM_HEIGHT,
        })

    # Build containment
    containment = {}
    for r_idx, obj_list in objects.items():
        for obj in obj_list:
            containment[obj["name"]] = room_ids[r_idx]

    # Build parallel relations
    parallel_rooms = [(room_ids[e[0]], room_ids[e[1]]) for e in edges]
    parallel_objects = []
    for r_idx, obj_list in objects.items():
        for i in range(len(obj_list)):
            for j in range(i + 1, len(obj_list)):
                parallel_objects.append((obj_list[i]["name"], obj_list[j]["name"]))

    # Build history
    history = []
    for evt in hist_events:
        history.append({
            "object": evt["object"],
            "from_room": room_ids[evt["from_room_idx"]],
            "to_room": room_ids[evt["to_room_idx"]],
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
        "history": history,
        "containment": containment,
        "parallel_rooms": parallel_rooms,
        "parallel_objects": parallel_objects,
        "rules": rules,
    }
