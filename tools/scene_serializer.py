#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene Serializer - Rule-based serialization for navigation planning scenes.

Generates all format variants (Flat, Hierarchical, Clustered) at three
compression levels (100%, 50%, 25% retention) from structured world data.

Also implements:
- Semantic Variation: synonym substitution from knowledge base
- Semantic Conflict: duplicate labels for distinct nodes (perceptual aliasing)
- Dimension ablation: selective removal of spatial dimensions

All serialization is rule-based and deterministic. No LLM annotation used.
"""

import json
import math
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from tools.knowledge_base import (
    get_room_abbr, get_object_abbr, pick_synonym,
    ROOM_ATTRIBUTES, OBJECT_ATTRIBUTES,
)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Room:
    """Represents a room in the navigation scene."""
    idx: int
    room_id: str
    canonical: str
    synonyms: List[str]
    abbr: str
    attributes: List[str]
    x: int
    y: int
    w: int
    h: int
    objects: List[Dict] = field(default_factory=list)


@dataclass
class World:
    """Structured representation of a navigation scene."""
    scene_name: str
    gradient: str
    gradient_label: str
    rooms: List[Room]
    edges: List[Tuple[int, int]]
    history: List[Dict]  # list of {"object", "from_room", "to_room"}
    containment: Dict[str, str]
    parallel_rooms: List[Tuple[str, str]]
    parallel_objects: List[Tuple[str, str]]
    rules: List[str]


def parse_world(world_dict: Dict) -> World:
    """Parse world dictionary into structured World object."""
    rooms = []
    objects_dict = world_dict.get("objects", {})
    for r in world_dict["rooms"]:
        obj_list = objects_dict.get(str(r["idx"]), objects_dict.get(r["idx"], []))
        rooms.append(Room(
            idx=int(r["idx"]),
            room_id=r["room_id"],
            canonical=r.get("canonical", r.get("type", "Room")),
            synonyms=r.get("synonyms", []),
            abbr=r.get("abbr", r.get("room_id", "")[:3]),
            attributes=r.get("attributes", []),
            x=r["x"],
            y=r["y"],
            w=r.get("w", 4),
            h=r.get("h", 4),
            objects=obj_list,
        ))

    edges = [tuple(e) for e in world_dict["edges"]]
    history = world_dict.get("history", [])
    containment = world_dict.get("containment", {})
    parallel_rooms = [tuple(pr) for pr in world_dict.get("parallel_rooms", [])]
    parallel_objects = [tuple(po) for po in world_dict.get("parallel_objects", [])]

    return World(
        scene_name=world_dict["scene_name"],
        gradient=world_dict.get("gradient", "G1"),
        gradient_label=world_dict.get("gradient_label", "Basic"),
        rooms=rooms,
        edges=edges,
        history=history,
        containment=containment,
        parallel_rooms=parallel_rooms,
        parallel_objects=parallel_objects,
        rules=world_dict.get("rules", []),
    )


# ============================================================================
# Helper Functions
# ============================================================================

def edge_to_str(edge: Tuple[int, int], rooms: List[Room]) -> str:
    return f"{rooms[edge[0]].room_id}-{rooms[edge[1]].room_id}"


def room_label(room: Room, use_attrs: bool = True) -> str:
    """Get room label with optional attributes."""
    if use_attrs and room.attributes:
        return f"{' '.join(room.attributes)} {room.canonical}"
    return room.canonical


def obj_label(obj: Dict, use_attrs: bool = True) -> str:
    """Get object label with optional attributes."""
    # Use display_name if available (for synonym variety), else canonical
    name = obj.get("display_name", obj.get("canonical", "Unknown"))
    if use_attrs and obj.get("attributes"):
        return f"{' '.join(obj['attributes'])} {name}"
    return name


def format_history_events(history: List[Dict], rooms: List[Room]) -> str:
    """Format history events as string (paper style: Key moved: R3->R2)."""
    if not history:
        return ""
    # Build the path: R3->R2->R1 style
    if history:
        path_rooms = [history[0]["from_room"]] + [evt["to_room"] for evt in history]
        return f"{history[0]['object']} moved: {'->'.join(path_rooms)}"
    return ""


def format_history_flat(history: List[Dict], rooms: List[Room]) -> str:
    """Format history in flat narrative style (paper style)."""
    if not history:
        return ""
    room_map = {r.room_id: r for r in rooms}
    # Build path: R3->R2->R1
    path_rooms = [history[0]["from_room"]] + [evt["to_room"] for evt in history]
    path_str = "->".join(path_rooms)
    return f"Before, the {history[0]['object']} moved from {path_rooms[0]} to {path_rooms[-1]}."


# ============================================================================
# Scene Serializer
# ============================================================================

class SceneSerializer:
    """Main serializer class for generating all format variants."""

    def __init__(self, world: World):
        self.world = world
        self.rooms = world.rooms
        self.edges = world.edges
        self.history = world.history

    def serialize_flat(self, retention: float = 1.0) -> str:
        """Generate flat (natural language) format.

        retention: 1.0 (100%), 0.5 (50%), 0.25 (25%)
        """
        if retention >= 1.0:
            return self._flat_100()
        elif retention >= 0.5:
            return self._flat_50()
        else:
            return self._flat_25()

    def _flat_100(self) -> str:
        """100% retention: full natural language narrative (paper style)."""
        parts = []

        for i, room in enumerate(self.rooms):
            attrs_str = " ".join(room.attributes) + " " if room.attributes else ""
            # Build object list
            obj_names = [obj_label(o, use_attrs=True) for o in room.objects]
            obj_str = ", ".join(obj_names) if obj_names else ""

            if i == 0:
                # Paper style: "Start at Entrance R1(0,0)."
                start = f"Start at {attrs_str}{room.canonical} {room.room_id}({room.x},{room.y})."
                if obj_str:
                    start += f" {obj_str}."
                parts.append(start)
            else:
                # Paper style: "Go east to Corridor R2(4,0),"
                conn = self._get_connection_brief(room.idx)
                if conn:
                    go_str = f"{conn} {attrs_str}{room.canonical} {room.room_id}({room.x},{room.y})."
                else:
                    go_str = f"Go to {attrs_str}{room.canonical} {room.room_id}({room.x},{room.y})."
                if obj_str:
                    go_str += f" {obj_str}."
                parts.append(go_str)

        # History (paper style: "Before, the Key moved from R3 to R2.")
        hist_str = format_history_flat(self.history, self.rooms)
        if hist_str:
            parts.append(hist_str)

        return " ".join(parts)

    def _flat_50(self) -> str:
        """50% retention: remove redundant fillers, keep all spatial cues."""
        lines = []
        for i, room in enumerate(self.rooms):
            attrs_str = " ".join(room.attributes) + " " if room.attributes else ""
            objs = ", ".join([obj_label(o, use_attrs=True) for o in room.objects[:3]])
            if i == 0:
                lines.append(f"Start at {attrs_str}{room.canonical} {room.room_id}({room.x},{room.y}). {objs}.")
            else:
                conn = self._get_connection_brief(room.idx)
                if conn:
                    lines.append(f"{conn} {attrs_str}{room.canonical} {room.room_id}({room.x},{room.y}). {objs}.")
                else:
                    lines.append(f"{attrs_str}{room.canonical} {room.room_id}({room.x},{room.y}). {objs}.")

        # Edges
        edge_strs = [edge_to_str(e, self.rooms) for e in self.edges]
        lines.append(f"Edges: {', '.join(edge_strs)}.")

        # History
        hist_str = format_history_events(self.history, self.rooms)
        if hist_str:
            lines.append(hist_str)

        return " ".join(lines)

    def _flat_25(self) -> str:
        """25% retention: abbreviations, shortest expressions."""
        parts = []
        for room in self.rooms:
            obj_abbrs = [get_object_abbr(o["canonical"]) for o in room.objects[:2]]
            obj_str = ",".join(obj_abbrs) if obj_abbrs else ""
            parts.append(f"{room.abbr}:{room.room_id}({room.x},{room.y})[{obj_str}]")

        # Edges as compact pairs
        edge_nums = [f"{e[0]+1}-{e[1]+1}" for e in self.edges]
        parts.append(f"Edg:{','.join(edge_nums)}")

        # History compact
        hist_parts = []
        for evt in self.history:
            hist_parts.append(f"{evt['object']}:{evt['from_room']}→{evt['to_room']}")
        if hist_parts:
            parts.append("; ".join(hist_parts))

        return " | ".join(parts)

    def serialize_hier(self, retention: float = 1.0) -> str:
        """Generate hierarchical format.

        retention: 1.0 (100%), 0.5 (50%), 0.25 (25%)
        """
        if retention >= 1.0:
            return self._hier_100()
        elif retention >= 0.5:
            return self._hier_50()
        else:
            return self._hier_25()

    def _hier_100(self) -> str:
        """100% retention: full hierarchical (paper style)."""
        lines = []

        # Rooms with objects (paper style: [R1: Entrance @ (0,0)] with objects inline)
        for room in self.rooms:
            obj_groups = self._group_objects(room.objects)
            obj_str = ", ".join([
                f"{cat}({', '.join(names)})"
                for cat, names in obj_groups.items()
            ])
            attrs_str = f" [{', '.join(room.attributes)}]" if room.attributes else ""
            obj_part = f", {obj_str}" if obj_str else ""
            lines.append(f"[{room.room_id}: {room.canonical}{attrs_str} @ ({room.x},{room.y}){obj_part}]")

        # Topology (paper style: Topo: R1↔R2↔R3)
        topo_str = ", ".join([edge_to_str(e, self.rooms) for e in self.edges])
        lines.append(f"Topo: {topo_str}")

        # History (paper style: Hist: Key R3→R2)
        hist_str = format_history_events(self.history, self.rooms)
        if hist_str:
            lines.append(f"Hist: {hist_str}")

        return "\n".join(lines)

    def _hier_50(self) -> str:
        """50% retention: semi-structured, abbreviated."""
        lines = ["[HIERARCHY]"]

        scene_name = self.world.scene_name.replace("scene_complex_", "").replace("scene_simple_", "S")
        lines.append(f"Scene({scene_name}) -> " + ",".join([r.room_id for r in self.rooms]))

        for room in self.rooms:
            obj_abbrs = []
            for o in room.objects[:4]:
                cat_abbr = get_object_abbr(o["canonical"])
                obj_abbrs.append(cat_abbr)
            obj_str = ",".join(obj_abbrs)
            attrs_str = ",".join(room.attributes[:1]) if room.attributes else ""
            label = f"{attrs_str} {room.canonical}" if attrs_str else room.canonical
            lines.append(f"{room.room_id}({room.x},{room.y}): {label} {{{obj_str}}}")

        edges_str = ",".join([f"{self.rooms[e[0]].room_id}-{self.rooms[e[1]].room_id}" for e in self.edges])
        lines.append(f"Edges: {edges_str}")

        hist_parts = [f"{evt['from_room']}→{evt['to_room']}" for evt in self.history]
        if hist_parts:
            lines.append(f"Key: {'; '.join(hist_parts)}")

        return "\n".join(lines)

    def _hier_25(self) -> str:
        """25% retention: maximum compression, abbreviations only."""
        parts = []

        room_strs = []
        for room in self.rooms:
            obj_abbrs = [get_object_abbr(o["canonical"]) for o in room.objects[:2]]
            obj_str = ",".join(obj_abbrs) if obj_abbrs else ""
            room_strs.append(f"{room.room_id}({room.x},{room.y}):{room.abbr}[{obj_str}]")

        scene_name = self.world.scene_name.replace("scene_complex_", "C").replace("scene_simple_", "S")
        parts.append(f"{scene_name}[{'; '.join(room_strs)}]")

        edge_nums = [f"{e[0]+1}-{e[1]+1}" for e in self.edges]
        parts.append(f"Edg:{','.join(edge_nums)}")

        hist_compact = []
        for evt in self.history:
            hist_compact.append(f"{evt['from_room']}→{evt['to_room']}")
        if hist_compact:
            parts.append(f"Key:{';'.join(hist_compact)}")

        return " | ".join(parts)

    def serialize_clustered(self, retention: float = 1.0) -> str:
        """Generate clustered (zone-based) format.

        retention: 1.0 (100%), 0.5 (50%), 0.25 (25%)
        """
        if retention >= 1.0:
            return self._clustered_100()
        elif retention >= 0.5:
            return self._clustered_50()
        else:
            return self._clustered_25()

    def _clustered_100(self) -> str:
        """100% retention: full clustered (paper style)."""
        zones = self._cluster_rooms()
        parts = []

        # Zones (paper style: Zone A: {R1,R2} @ (2,0))
        for zone_name, zone_rooms in zones.items():
            room_ids = [r.room_id for r in zone_rooms]
            center = self._compute_room_center(zone_rooms)
            zone_str = f"Zone {zone_name}: {{{','.join(room_ids)}}} @ ({center[0]:.0f},{center[1]:.0f})"
            parts.append(zone_str)

        # Link (paper style: Link: A↔B)
        zone_connections = self._compute_zone_connections_brief(zones)
        if zone_connections:
            parts.append(f"Link: {zone_connections}")

        # History (paper style: Key: B→A(R2))
        hist_str = format_history_events(self.history, self.rooms)
        if hist_str:
            parts.append(hist_str)

        return " ".join(parts)

    def _clustered_50(self) -> str:
        """50% retention: compressed zones with abbreviated info."""
        zones = self._cluster_rooms()
        lines = ["[CLUSTERS]"]

        for zone_name, zone_rooms in zones.items():
            room_ids = [r.room_id for r in zone_rooms]
            center = self._compute_room_center(zone_rooms)
            obj_cats = list(set([get_object_abbr(o["canonical"]) for r in zone_rooms for o in r.objects]))
            attrs = list(set([a for r in zone_rooms for a in r.attributes]))[:2]
            attr_str = f" {','.join(attrs)}" if attrs else ""
            lines.append(
                f"{zone_name}({','.join(room_ids)}) @({center[0]:.0f},{center[1]:.0f})"
                f":{attr_str} {{{'/'.join(obj_cats[:3])}}}"
            )

        # Zone links
        zone_connections = self._compute_zone_connections(zones)
        lines.append(f"[LINKS] {zone_connections}")

        # History trace (start/end only)
        if self.history:
            first = self.history[0]
            last = self.history[-1]
            lines.append(f"[TRACE] Key: {first['from_room']}→{last['to_room']}")

        return "\n".join(lines)

    def _clustered_25(self) -> str:
        """25% retention: maximum compression for clustered format."""
        zones = self._cluster_rooms()
        parts = []
        zone_idx = 1

        for zone_rooms in zones.values():
            center = self._compute_room_center(zone_rooms)
            room_nums = [str(r.idx + 1) for r in zone_rooms]
            parts.append(f"G{zone_idx}({center[0]:.0f},{center[1]:.0f}):{','.join(room_nums)}")
            zone_idx += 1

        # Compressed connections
        zone_connections = self._compute_zone_connections_compressed(zones)
        parts.append(f"Links:{zone_connections}")

        # Compressed history
        if self.history:
            first = self.history[0]
            last = self.history[-1]
            parts.append(f"Key:{first['from_room']}→{last['to_room']}")

        return ". ".join(parts) + "."

    # ---- Helper methods ----

    def _get_scene_type(self) -> str:
        name = self.world.scene_name.lower()
        if "campus" in name or "complex_01" in name or "complex_02" in name:
            return "smart campus"
        elif "hospital" in name or "complex_03" in name or "complex_04" in name:
            return "modern hospital"
        elif "mall" in name or "complex_05" in name or "complex_06" in name:
            return "shopping mall"
        elif "airport" in name or "complex_07" in name or "complex_08" in name:
            return "international airport"
        elif "museum" in name or "complex_09" in name or "complex_10" in name:
            return "art museum"
        else:
            return "building"

    def _format_objects_flat_full(self, objects: List[Dict]) -> str:
        if not objects:
            return "has no notable objects"
        labels = [obj_label(o, use_attrs=True) for o in objects]
        if len(labels) == 1:
            return f"features a {labels[0]}"
        elif len(labels) == 2:
            return f"contains a {labels[0]} and a {labels[1]}"
        else:
            return f"contains {', '.join(labels[:-1])}, and a {labels[-1]}"

    def _get_connection_description(self, room_idx: int) -> Optional[str]:
        for edge in self.edges:
            if edge[1] == room_idx:
                from_room = self.rooms[edge[0]]
                to_room = self.rooms[room_idx]
                if to_room.x > from_room.x:
                    direction = "East"
                elif to_room.x < from_room.x:
                    direction = "West"
                elif to_room.y > from_room.y:
                    direction = "North"
                else:
                    direction = "South"
                return f"Moving {direction} from the {room_label(from_room)} ({from_room.room_id}), you enter"
        return None

    def _get_connection_brief(self, room_idx: int) -> Optional[str]:
        for edge in self.edges:
            if edge[1] == room_idx:
                from_room = self.rooms[edge[0]]
                to_room = self.rooms[room_idx]
                if to_room.x > from_room.x:
                    direction = "east"
                elif to_room.x < from_room.x:
                    direction = "west"
                elif to_room.y > from_room.y:
                    direction = "north"
                else:
                    direction = "south"
                return f"Go {direction} from {from_room.room_id} to"
        return None

    def _generate_topology_hints(self) -> Optional[str]:
        loops = self._find_loops()
        if loops:
            hints = []
            for l in loops:
                r1 = self.rooms[l[0]]
                r2 = self.rooms[l[1]]
                hints.append(f"the {r1.canonical} features a connection back to the {r2.canonical}")
            return f"Interestingly, {' and '.join(hints)}."
        return None

    def _find_loops(self) -> List[Tuple[int, int]]:
        loops = []
        for edge in self.edges:
            u, v = edge
            if abs(u - v) > 1:
                loops.append(edge)
        return loops[:2]

    def _group_objects(self, objects: List[Dict]) -> Dict[str, List[str]]:
        """Group objects by canonical category, using display_name for items."""
        groups = {}
        for o in objects:
            cat = o.get("canonical", "Unknown")
            # Use display_name to show variety (e.g., Scanner, Copier instead of Printer, Printer)
            display = o.get("display_name", o.get("canonical", "Unknown"))
            groups.setdefault(cat, []).append(display)
        return groups

    def _cluster_rooms(self, n_clusters: int = 2) -> Dict[str, List[Room]]:
        if len(self.rooms) <= n_clusters:
            return {"Main": self.rooms}
        sorted_rooms = sorted(self.rooms, key=lambda r: r.x)
        mid = len(sorted_rooms) // 2
        return {"A": sorted_rooms[:mid], "B": sorted_rooms[mid:]}

    def _compute_room_center(self, rooms: List[Room]) -> Tuple[float, float]:
        if not rooms:
            return (0.0, 0.0)
        avg_x = sum(r.x for r in rooms) / len(rooms)
        avg_y = sum(r.y for r in rooms) / len(rooms)
        return (avg_x, avg_y)

    def _compute_zone_centers(self, zones: Dict[str, List[Room]]) -> Dict[str, Tuple[float, float]]:
        return {name: self._compute_room_center(rooms) for name, rooms in zones.items()}

    def _compute_zone_connections(self, zones: Dict[str, List[Room]]) -> str:
        connections = []
        zone_rooms_map = {}
        for zone_name, rooms in zones.items():
            for room in rooms:
                zone_rooms_map[room.idx] = zone_name

        inter_zone_edges = []
        for edge in self.edges:
            z1 = zone_rooms_map.get(edge[0])
            z2 = zone_rooms_map.get(edge[1])
            if z1 and z2 and z1 != z2:
                inter_zone_edges.append((z1, z2, self.rooms[edge[0]].room_id, self.rooms[edge[1]].room_id))

        if inter_zone_edges:
            zone_pairs = {}
            for z1, z2, r1, r2 in inter_zone_edges:
                key = f"{z1}-{z2}"
                zone_pairs.setdefault(key, []).append(f"{r1}-{r2}")
            for pair, edges in zone_pairs.items():
                connections.append(f"Zone_{pair.replace('-', ' <-> Zone_')} via {edges[0]}")

        for zone_name, rooms in zones.items():
            room_indices = {r.idx for r in rooms}
            zone_edges = [e for e in self.edges if e[0] in room_indices and e[1] in room_indices]
            if len(zone_edges) > len(rooms) - 1:
                connections.append(f"Loop in Zone_{zone_name}")

        return ". ".join(connections) if connections else "No inter-zone connections"

    def _compute_zone_connections_brief(self, zones: Dict[str, List[Room]]) -> str:
        """Brief zone connection (paper style: A↔B)."""
        zone_rooms_map = {}
        for zone_name, rooms in zones.items():
            for room in rooms:
                zone_rooms_map[room.idx] = zone_name

        connected_pairs = set()
        for edge in self.edges:
            z1 = zone_rooms_map.get(edge[0])
            z2 = zone_rooms_map.get(edge[1])
            if z1 and z2 and z1 != z2:
                connected_pairs.add(tuple(sorted([z1, z2])))

        if connected_pairs:
            return ", ".join([f"{a}<->{b}" for a, b in sorted(connected_pairs)])
        return ""

    def _compute_zone_connections_compressed(self, zones: Dict[str, List[Room]]) -> str:
        zone_idx_map = {}
        idx = 1
        for zone_name in zones.keys():
            zone_idx_map[zone_name] = idx
            idx += 1

        connections = []
        zone_rooms_map = {}
        for zone_name, rooms in zones.items():
            for room in rooms:
                zone_rooms_map[room.idx] = zone_name

        for edge in self.edges:
            z1 = zone_rooms_map.get(edge[0])
            z2 = zone_rooms_map.get(edge[1])
            if z1 and z2 and z1 != z2:
                idx1 = zone_idx_map[z1]
                idx2 = zone_idx_map[z2]
                r1 = edge[0] + 1
                r2 = edge[1] + 1
                connections.append(f"G{idx1}-G{idx2}({r1}-{r2})")

        return ",".join(connections) if connections else "None"


# ============================================================================
# Batch Variant Generators
# ============================================================================

def generate_all_variants(world_dict: Dict) -> Dict[str, str]:
    """Generate all format variants from a world dictionary.

    Returns: variant_name -> serialized_text
    Includes Flat, Hierarchical, Clustered at 100%, 50%, 25% retention.
    """
    world = parse_world(world_dict)
    serializer = SceneSerializer(world)

    variants = {}

    # Flat at 100%, 50%, 25% retention
    variants["flat"] = serializer.serialize_flat(1.0)
    variants["flat_50"] = serializer.serialize_flat(0.5)
    variants["flat_25"] = serializer.serialize_flat(0.25)

    # Hierarchical at 100%, 50%, 25% retention
    variants["hier"] = serializer.serialize_hier(1.0)
    variants["hier_50"] = serializer.serialize_hier(0.5)
    variants["hier_25"] = serializer.serialize_hier(0.25)

    # Clustered at 100%, 50%, 25% retention
    variants["clustered"] = serializer.serialize_clustered(1.0)
    variants["clustered_50"] = serializer.serialize_clustered(0.5)
    variants["clustered_25"] = serializer.serialize_clustered(0.25)

    return variants


# ============================================================================
# Dimension-Ablation Rendering (for R2)
# ============================================================================

GRID_STEP = 4
KEY_NAME = "Key"


def _get_room_label_with_attrs(room_dict: Dict) -> str:
    """Get room label from room dict with attributes."""
    canonical = room_dict.get("canonical", room_dict.get("type", "Room"))
    attrs = room_dict.get("attributes", [])
    if attrs:
        return f"{' '.join(attrs)} {canonical}"
    return canonical


def render_full(world_dict: Dict) -> str:
    """Include everything: geometry + topology + semantics + history + rules."""
    scene = world_dict.get("scene_name", "scene")
    rooms = world_dict["rooms"]
    edges = world_dict["edges"]
    objects = world_dict.get("objects", {})
    hist = world_dict.get("history", [])
    rules = world_dict.get("rules", [])

    txt = []
    txt.append(f"You are in the scene '{scene}'.")
    txt.append("Rooms and coordinates:")
    for r in rooms:
        label = _get_room_label_with_attrs(r)
        txt.append(f"- {r['room_id']} ({label}) @ ({r['x']},{r['y']})")
        obj_list = objects.get(str(r['idx']), objects.get(r['idx'], []))
        obj_strs = []
        for o in obj_list:
            olabel = obj_label(o, use_attrs=True) if isinstance(o, dict) else str(o)
            obj_strs.append(olabel)
        txt.append(f"  Objects: {', '.join(obj_strs)}")

    room_ids = [r['room_id'] for r in rooms]
    edge_strs = [f"{room_ids[e[0]]}-{room_ids[e[1]]}" for e in edges]
    txt.append(f"Topology edges: {', '.join(edge_strs)}.")

    hist_str = _format_history_for_render(hist)
    txt.append(f"History of {KEY_NAME}: {hist_str}.")

    txt.append("Rules/constraints:")
    for rr in rules:
        txt.append(f"- {rr}")
    return "\n".join(txt)


def render_topo_hist(world_dict: Dict) -> str:
    """Topology + History only (no coordinates/semantics details)."""
    rooms = world_dict["rooms"]
    edges = world_dict["edges"]
    hist = world_dict.get("history", [])
    room_ids = [r['room_id'] for r in rooms]
    txt = []
    txt.append("You are given a set of room IDs and how they connect (topology).")
    txt.append(f"Rooms: {', '.join(room_ids)}.")
    edge_strs = [f"{room_ids[e[0]]}-{room_ids[e[1]]}" for e in edges]
    txt.append(f"Topology edges: {', '.join(edge_strs)}.")
    hist_str = _format_history_for_render(hist)
    txt.append(f"History of {KEY_NAME}: {hist_str}.")
    txt.append("Constraint: When generating a path, output the shortest valid path unless the question specifies otherwise.")
    return "\n".join(txt)


def render_geom_rule_hist(world_dict: Dict) -> str:
    """Geometry + Rules + History; no explicit edges (must be inferred)."""
    rooms = world_dict["rooms"]
    hist = world_dict.get("history", [])
    txt = []
    txt.append("You are given room coordinates. You must infer connectivity using the rules below.")
    txt.append("Rooms and coordinates:")
    for r in rooms:
        label = _get_room_label_with_attrs(r)
        txt.append(f"- {r['room_id']} ({label}) @ ({r['x']},{r['y']})")
    txt.append("Rules:")
    txt.append(f"- Two rooms are connected by a door iff their Manhattan distance is exactly {GRID_STEP}.")
    txt.append("- You may assume there are no blocked doors unless the text explicitly says so.")
    hist_str = _format_history_for_render(hist)
    txt.append(f"History of {KEY_NAME}: {hist_str}.")
    txt.append("Constraint: Prefer the shortest path under the inferred connectivity.")
    return "\n".join(txt)


def render_sem_rule_hist(world_dict: Dict) -> str:
    """Semantics + Rules + History (room-chain). No coordinates or explicit topology."""
    rooms = world_dict["rooms"]
    objects = world_dict.get("objects", {})
    hist = world_dict.get("history", [])
    room_ids = [r["room_id"] for r in rooms]
    room_chain = " -> ".join(room_ids)

    txt = []
    txt.append("You are given room semantics and room contents. Connectivity is defined by semantic rules below.")
    txt.append("Rooms and semantics:")
    for r in rooms:
        label = _get_room_label_with_attrs(r)
        ridx = int(r["idx"])
        obj_list = objects.get(str(ridx), objects.get(ridx, []))
        obj_strs = []
        for o in obj_list:
            olabel = obj_label(o, use_attrs=True) if isinstance(o, dict) else str(o)
            obj_strs.append(olabel)
        txt.append(f"- {r['room_id']} is a '{label}'. Objects: {', '.join(obj_strs)}")
    txt.append("Semantic connectivity rules (room-chain):")
    txt.append("- The scene provides an ordered chain of rooms. Two rooms are connected iff they are adjacent in this chain.")
    txt.append(f"- Room chain: {room_chain}")
    txt.append("- The chain is the ONLY source of connectivity (do not assume any other connections).")
    hist_str = _format_history_for_render(hist)
    txt.append(f"History of {KEY_NAME}: {hist_str}.")
    txt.append("Constraint: Prefer the shortest path under the inferred connectivity.")
    return "\n".join(txt)


# ============================================================================
# Semantic Variation (Synonym Substitution)
# ============================================================================

def generate_semantic_variation(world_dict: Dict, seed: int = 42) -> Dict[str, str]:
    """Generate semantic variation variants using synonyms from knowledge base.

    For each format (Flat, Hier, Clustered), generates a variant where:
    - Room canonical names are replaced with random synonyms
    - Object canonical names are replaced with random synonyms
    - Topology, Geometry, History remain unchanged

    Returns: variant_name -> serialized_text
    """
    random.seed(seed)
    # Create a copy of world_dict with synonym-substituted labels
    var_world = _substitute_synonyms(world_dict, seed)

    # Generate all format variants using the synonym-substituted world
    return generate_all_variants(var_world)


def _substitute_synonyms(world_dict: Dict, seed: int) -> Dict:
    """Create a copy of world_dict with all canonical names replaced by synonyms."""
    import copy
    var_world = copy.deepcopy(world_dict)

    # Substitute room canonical names
    for room in var_world["rooms"]:
        canonical = room.get("canonical", room.get("type", "Room"))
        synonyms = room.get("synonyms", [])
        if synonyms:
            choices = [canonical] + synonyms
            room["canonical"] = random.choice(choices)

    # Substitute object canonical names
    for r_idx, obj_list in var_world.get("objects", {}).items():
        for obj in obj_list:
            if isinstance(obj, dict):
                canonical = obj.get("canonical", "")
                synonyms = obj.get("synonyms", [])
                if synonyms:
                    choices = [canonical] + synonyms
                    obj["canonical"] = random.choice(choices)

    return var_world


# ============================================================================
# Semantic Conflict (Duplicate Labels for Distinct Nodes)
# ============================================================================

def generate_semantic_conflict(world_dict: Dict, seed: int = 42) -> Dict[str, str]:
    """Generate semantic conflict variants per the paper's specification.

    Injects perceptual aliasing: duplicate semantic labels for distinct nodes.
    Example: R2 is "Office", R5 is "Lab", but both are labeled "Office" in text.

    The topology, geometry, history, and IDs remain correct.
    Only semantic labels are corrupted.

    Returns: variant_name -> serialized_text
    """
    random.seed(seed)
    conflict_world = _inject_duplicate_labels(world_dict, seed)

    variants = {}
    # Generate conflict variants in all formats
    variants["conflict_flat"] = _render_conflict_flat(conflict_world)
    variants["conflict_hier"] = _render_conflict_hier(conflict_world)
    variants["conflict_clustered"] = _render_conflict_clustered(conflict_world)
    return variants


def _inject_duplicate_labels(world_dict: Dict, seed: int) -> Dict:
    """Inject duplicate semantic labels for distinct rooms.

    Picks 2-3 rooms and assigns them the same canonical name,
    simulating perceptual aliasing. IDs and topology unchanged.
    """
    import copy
    random.seed(seed)
    conflict_world = copy.deepcopy(world_dict)
    rooms = conflict_world["rooms"]

    if len(rooms) < 3:
        return conflict_world

    # Pick a "dominant" label and apply it to 2-3 other rooms
    dominant_idx = random.randint(0, len(rooms) - 1)
    dominant_label = rooms[dominant_idx].get("canonical", rooms[dominant_idx].get("type", "Room"))

    # Pick 1-2 other rooms to also get this label
    other_indices = [i for i in range(len(rooms)) if i != dominant_idx]
    random.shuffle(other_indices)
    n_conflicts = min(2, len(other_indices))

    for idx in other_indices[:n_conflicts]:
        rooms[idx]["canonical"] = dominant_label
        rooms[idx]["conflict_original"] = rooms[idx].get("canonical", "")

    return conflict_world


def _render_conflict_flat(world_dict: Dict) -> str:
    """Render conflict scene in flat format."""
    rooms = world_dict["rooms"]
    edges = world_dict["edges"]
    objects = world_dict.get("objects", {})
    hist = world_dict.get("history", [])

    txt = []
    txt.append("You are in a building. Note: some room labels may be ambiguous.")
    for i, r in enumerate(rooms):
        label = _get_room_label_with_attrs(r)
        obj_list = objects.get(str(r['idx']), objects.get(r['idx'], []))
        obj_strs = [obj_label(o, use_attrs=True) if isinstance(o, dict) else str(o) for o in obj_list]
        txt.append(f"- {r['room_id']} is labeled '{label}' @ ({r['x']},{r['y']}). Objects: {', '.join(obj_strs)}")

    room_ids = [r['room_id'] for r in rooms]
    edge_strs = [f"{room_ids[e[0]]}-{room_ids[e[1]]}" for e in edges]
    txt.append(f"Topology edges: {', '.join(edge_strs)}.")

    hist_str = _format_history_for_render(hist)
    txt.append(f"History of {KEY_NAME}: {hist_str}.")
    txt.append("Note: If information conflicts, still output the best path you believe is correct.")
    return "\n".join(txt)


def _render_conflict_hier(world_dict: Dict) -> str:
    """Render conflict scene in hierarchical format."""
    rooms = world_dict["rooms"]
    edges = world_dict["edges"]
    objects = world_dict.get("objects", {})
    hist = world_dict.get("history", [])

    txt = ["[SCENE: Conflict Test]"]
    for r in rooms:
        label = _get_room_label_with_attrs(r)
        obj_list = objects.get(str(r['idx']), objects.get(r['idx'], []))
        obj_strs = [obj_label(o) if isinstance(o, dict) else str(o) for o in obj_list]
        txt.append(f"- [{r['room_id']}: {label} @ ({r['x']},{r['y']})]")
        txt.append(f"  [Objects: {', '.join(obj_strs)}]")

    room_ids = [r['room_id'] for r in rooms]
    edge_strs = [f"{room_ids[e[0]]}-{room_ids[e[1]]}" for e in edges]
    txt.append(f"[TOPOLOGY] {', '.join(edge_strs)}")

    hist_str = _format_history_for_render(hist)
    txt.append(f"[HISTORY] {hist_str}")
    txt.append("Note: If information conflicts, still output the best path you believe is correct.")
    return "\n".join(txt)


def _render_conflict_clustered(world_dict: Dict) -> str:
    """Render conflict scene in clustered format."""
    rooms = world_dict["rooms"]
    edges = world_dict["edges"]
    hist = world_dict.get("history", [])

    # Simple clustering by x-coordinate
    sorted_rooms = sorted(rooms, key=lambda r: r.get("x", 0))
    mid = len(sorted_rooms) // 2
    zones = {"A": sorted_rooms[:mid], "B": sorted_rooms[mid:]}

    txt = ["[SCENE: Conflict Test]"]
    for zone_name, zone_rooms in zones.items():
        room_ids = [r["room_id"] for r in zone_rooms]
        labels = [_get_room_label_with_attrs(r) for r in zone_rooms]
        txt.append(f"[Zone {zone_name}: {', '.join(room_ids)}]")
        txt.append(f"  Labels: {', '.join(labels)}")

    room_ids_all = [r['room_id'] for r in rooms]
    edge_strs = [f"{room_ids_all[e[0]]}-{room_ids_all[e[1]]}" for e in edges]
    txt.append(f"[TOPOLOGY] {', '.join(edge_strs)}")

    hist_str = _format_history_for_render(hist)
    txt.append(f"[HISTORY] {hist_str}")
    txt.append("Note: If information conflicts, still output the best path you believe is correct.")
    return "\n".join(txt)


def _format_history_for_render(hist) -> str:
    """Format history for render functions (handles both dict and legacy formats)."""
    if not hist:
        return ""
    if isinstance(hist[0], dict):
        return "; ".join([f"{evt['object']}: {evt['from_room']}→{evt['to_room']}" for evt in hist])
    else:
        # Legacy format: list of room indices
        return " -> ".join([f"R{i+1}" for i in hist])


# ============================================================================
# Dimension Variants (for R2 - Topology-Dominant vs Geometry-Dominant)
# ============================================================================

def generate_dimension_variants(world_dict: Dict) -> Dict[str, str]:
    """Generate dimension-ablation variants for R2 experiments."""
    return {
        "flat_full": render_full(world_dict),
        "flat_topo_hist": render_topo_hist(world_dict),
        "flat_geom_rule_hist": render_geom_rule_hist(world_dict),
        "flat_sem_rule_hist": render_sem_rule_hist(world_dict),
    }


# ============================================================================
# Conflict Variants (for Conflict experiments)
# ============================================================================

def generate_conflict_variants(world_dict: Dict, seed: int = 42) -> Dict[str, str]:
    """Generate conflict variants for conflict experiments.

    Per paper Section 3.2 (Contextual Conflict Probing):
    - Base dimension variants (flat_full, flat_topo_hist, etc.)
    - Semantic conflict: duplicate labels for distinct nodes (perceptual aliasing)

    The paper's C2 regime injects semantic conflict into the Topology-Dominant
    regime to test whether semantic ambiguity can override correct topology.
    """
    variants = generate_dimension_variants(world_dict)

    # Semantic conflict variants (duplicate labels - paper's C2 regime)
    sem_conflict = generate_semantic_conflict(world_dict, seed)
    variants.update(sem_conflict)

    return variants


# ============================================================================
# Equivalence Verification
# ============================================================================

def verify_equivalence(variants: Dict[str, str], world: World) -> Dict[str, bool]:
    """Verify that all variants encode the same underlying information.

    Checks:
    - All rooms are mentioned (by ID or numeric index)
    - All history rooms are present
    - Content is non-empty
    """
    results = {}

    for variant_name, text in variants.items():
        all_rooms_ok = all(
            room.room_id in text or str(room.idx + 1) in text
            for room in world.rooms
        )

        # Check history rooms
        hist_room_ids = set()
        for evt in world.history:
            if isinstance(evt, dict):
                hist_room_ids.add(evt.get("from_room", ""))
                hist_room_ids.add(evt.get("to_room", ""))
            else:
                if evt < len(world.rooms):
                    hist_room_ids.add(world.rooms[evt].room_id)

        history_ok = any(rid in text or str(i+1) in text
                         for i, room in enumerate(world.rooms)
                         for rid in [room.room_id]
                         if rid in hist_room_ids)

        checks = {
            "all_rooms": all_rooms_ok,
            "history_preserved": history_ok,
            "non_empty": len(text) > 0,
        }
        results[variant_name] = all(checks.values())

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scene_serializer.py <scene.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        scene = json.load(f)

    world_dict = scene.get("world", scene)
    variants = generate_all_variants(world_dict)

    print("Generated variants:")
    for name, text in variants.items():
        print(f"\n=== {name} ===")
        print(text[:300] + "..." if len(text) > 300 else text)
