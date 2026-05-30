#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Base Module for Spatial Scene Generation.

Defines structured knowledge bases for rooms, objects, and their attributes,
following the paper's specification for controlled scene generation.

All scene elements are sampled from these predefined knowledge bases.
"""

import random
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Room Category Library
# ============================================================================

ROOM_CATEGORIES = [
    {"rid": "R1",  "canonical": "Office",        "synonyms": ["Workspace", "Bureau"],           "abbr": "Off"},
    {"rid": "R2",  "canonical": "Corridor",       "synonyms": ["Aisle", "Hallway"],             "abbr": "Corr"},
    {"rid": "R3",  "canonical": "Meeting Room",    "synonyms": ["Conference Room", "Boardroom"], "abbr": "MR"},
    {"rid": "R4",  "canonical": "Lobby",           "synonyms": ["Entrance Hall", "Foyer"],       "abbr": "Lob"},
    {"rid": "R5",  "canonical": "Bedroom",         "synonyms": ["Sleeping Quarters", "Chamber"], "abbr": "Bed"},
    {"rid": "R6",  "canonical": "Kitchen",         "synonyms": ["Pantry", "Cookhouse"],          "abbr": "Kit"},
    {"rid": "R7",  "canonical": "Laboratory",      "synonyms": ["Lab", "Research Room"],         "abbr": "Lab"},
    {"rid": "R8",  "canonical": "Storage Room",    "synonyms": ["Storeroom", "Stockroom"],       "abbr": "Stor"},
    {"rid": "R9",  "canonical": "Bathroom",        "synonyms": ["Restroom", "Washroom"],         "abbr": "Bath"},
    {"rid": "R10", "canonical": "Classroom",       "synonyms": ["Lecture Room", "Tutorial Room"],"abbr": "CR"},
    {"rid": "R11", "canonical": "Entrance",        "synonyms": ["Entryway", "Doorway"],          "abbr": "Ent"},
    {"rid": "R12", "canonical": "Library",         "synonyms": ["Reading Room", "Archive"],      "abbr": "Lib"},
]

# ============================================================================
# Object Category Library
# ============================================================================

OBJECT_CATEGORIES = [
    {"oid": "O1", "canonical": "Chair",     "synonyms": ["Seat", "Stool"],           "abbr": "Chr"},
    {"oid": "O2", "canonical": "Desk",      "synonyms": ["Table", "Workbench"],      "abbr": "Dsk"},
    {"oid": "O3", "canonical": "Laptop",    "synonyms": ["Notebook Computer", "PC"], "abbr": "Lap"},
    {"oid": "O4", "canonical": "Table",     "synonyms": ["Counter", "Bench"],        "abbr": "Tbl"},
    {"oid": "O5", "canonical": "Sofa",      "synonyms": ["Couch", "Settee"],         "abbr": "Sof"},
    {"oid": "O6", "canonical": "Key",       "synonyms": ["Passkey", "Access Card"],  "abbr": "Key"},
    {"oid": "O7", "canonical": "Cabinet",   "synonyms": ["Cupboard", "Locker"],      "abbr": "Cab"},
    {"oid": "O8", "canonical": "Printer",   "synonyms": ["Scanner", "Copier"],       "abbr": "Prn"},
    {"oid": "O9", "canonical": "Monitor",   "synonyms": ["Display", "Screen"],       "abbr": "Mon"},
    {"oid": "O10","canonical": "Bookshelf", "synonyms": ["Shelf", "Rack"],           "abbr": "Bks"},
]

# ============================================================================
# Room Attribute Library (only for rooms)
# ============================================================================

ROOM_ATTRIBUTES = [
    "Bright", "Dark", "Messy", "Clean", "Quiet", "Crowded", "Spacious", "Narrow"
]

# ============================================================================
# Object Attribute Library (only for objects)
# ============================================================================

OBJECT_ATTRIBUTES = [
    "Red", "Blue", "Large", "Small", "Heavy", "Light", "Wooden", "Metallic"
]


# ============================================================================
# Lookup Functions
# ============================================================================

def get_room_by_canonical(name: str) -> Optional[Dict]:
    """Look up a room category by its canonical name (case-insensitive)."""
    name_lower = name.lower()
    for room in ROOM_CATEGORIES:
        if room["canonical"].lower() == name_lower:
            return room
    return None


def get_room_by_synonym(synonym: str) -> Optional[Dict]:
    """Look up a room category by one of its synonyms (case-insensitive)."""
    syn_lower = synonym.lower()
    for room in ROOM_CATEGORIES:
        if syn_lower == room["canonical"].lower():
            return room
        for s in room["synonyms"]:
            if s.lower() == syn_lower:
                return room
    return None


def get_room_synonyms(canonical: str) -> List[str]:
    """Return the synonym list for a room canonical name."""
    room = get_room_by_canonical(canonical)
    if room:
        return room["synonyms"]
    return []


def get_room_abbr(canonical: str) -> str:
    """Return the abbreviation for a room canonical name."""
    room = get_room_by_canonical(canonical)
    if room:
        return room["abbr"]
    return canonical[:3]


def get_object_by_canonical(name: str) -> Optional[Dict]:
    """Look up an object category by its canonical name (case-insensitive)."""
    name_lower = name.lower()
    for obj in OBJECT_CATEGORIES:
        if obj["canonical"].lower() == name_lower:
            return obj
    return None


def get_object_by_synonym(synonym: str) -> Optional[Dict]:
    """Look up an object category by one of its synonyms (case-insensitive)."""
    syn_lower = synonym.lower()
    for obj in OBJECT_CATEGORIES:
        if syn_lower == obj["canonical"].lower():
            return obj
        for s in obj["synonyms"]:
            if s.lower() == syn_lower:
                return obj
    return None


def get_object_synonyms(canonical: str) -> List[str]:
    """Return the synonym list for an object canonical name."""
    obj = get_object_by_canonical(canonical)
    if obj:
        return obj["synonyms"]
    return []


def get_object_abbr(canonical: str) -> str:
    """Return the abbreviation for an object canonical name."""
    obj = get_object_by_canonical(canonical)
    if obj:
        return obj["abbr"]
    return canonical[:3]


# ============================================================================
# Sampling Functions
# ============================================================================

def sample_rooms(k: int, seed: int) -> List[Dict]:
    """Sample k distinct room categories from the knowledge base."""
    random.seed(seed)
    pool = ROOM_CATEGORIES[:]
    random.shuffle(pool)
    return pool[:k]


def sample_objects(k: int, seed: int) -> List[Dict]:
    """Sample k distinct object categories from the knowledge base."""
    random.seed(seed)
    pool = OBJECT_CATEGORIES[:]
    random.shuffle(pool)
    return pool[:k]


def sample_room_attrs(k: int, seed: int) -> List[str]:
    """Sample k distinct room attributes."""
    random.seed(seed)
    pool = ROOM_ATTRIBUTES[:]
    random.shuffle(pool)
    return pool[:k]


def sample_object_attrs(k: int, seed: int) -> List[str]:
    """Sample k distinct object attributes."""
    random.seed(seed)
    pool = OBJECT_ATTRIBUTES[:]
    random.shuffle(pool)
    return pool[:k]


def pick_synonym(canonical: str, seed: int) -> str:
    """Randomly pick a synonym (or the canonical name itself) for a room/object."""
    random.seed(seed)
    room = get_room_by_canonical(canonical)
    if room:
        choices = [room["canonical"]] + room["synonyms"]
        return random.choice(choices)
    obj = get_object_by_canonical(canonical)
    if obj:
        choices = [obj["canonical"]] + obj["synonyms"]
        return random.choice(choices)
    return canonical


# ============================================================================
# Complexity Schedule
# ============================================================================

GRADIENTS = {
    "G1": {"rooms": 6,  "room_types": 3, "obj_per_room": 4,  "obj_types": 2, "history_steps": 2,  "label": "Basic cognition"},
    "G2": {"rooms": 8,  "room_types": 4, "obj_per_room": 6,  "obj_types": 3, "history_steps": 4,  "label": "Intermediate planning"},
    "G3": {"rooms": 10, "room_types": 5, "obj_per_room": 8,  "obj_types": 4, "history_steps": 6,  "label": "Complex modeling"},
    "G4": {"rooms": 12, "room_types": 6, "obj_per_room": 10, "obj_types": 5, "history_steps": 8,  "label": "High-load retrieval"},
    "G5": {"rooms": 14, "room_types": 7, "obj_per_room": 12, "obj_types": 6, "history_steps": 10, "label": "Extreme reasoning"},
}

SCENE_TO_G = {1: "G1", 2: "G1", 3: "G2", 4: "G2", 5: "G3",
              6: "G3", 7: "G4", 8: "G4", 9: "G5", 10: "G5"}


if __name__ == "__main__":
    print("=== Room Categories ===")
    for r in ROOM_CATEGORIES:
        print(f"  {r['rid']}: {r['canonical']} | Synonyms: {r['synonyms']} | Abbr: {r['abbr']}")
    print(f"\n=== Object Categories ===")
    for o in OBJECT_CATEGORIES:
        print(f"  {o['oid']}: {o['canonical']} | Synonyms: {o['synonyms']} | Abbr: {o['abbr']}")
    print(f"\n=== Room Attributes: {ROOM_ATTRIBUTES}")
    print(f"=== Object Attributes: {OBJECT_ATTRIBUTES}")
    print(f"\n=== Sample 3 rooms (seed=42): {[r['canonical'] for r in sample_rooms(3, 42)]}")
    print(f"=== Sample 3 objects (seed=42): {[o['canonical'] for o in sample_objects(3, 42)]}")
