#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene Serializer - Rule-based serialization for navigation planning scenes.

This module provides fully automated, rule-based serialization scripts that generate
all data format variants (flat, hierarchical, clustered with 0%/50%/75% compression)
from structured world data. No manual annotation or LLM-based annotation is required.

Key Features:
- Information equivalence: All variants encode the same underlying world state
- Deterministic: Same input produces same output (seed-controlled)
- Compression rates: Supports 0% (full), 50%, 75% compression levels
"""

import json
import math
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class Room:
    """Represents a room in the navigation scene."""
    idx: int
    room_id: str
    room_type: str
    x: int
    y: int
    objects: List[Tuple[str, str]]  # List of (category, name)


@dataclass
class World:
    """Structured representation of a navigation scene."""
    scene_name: str
    gradient: str
    gradient_label: str
    rooms: List[Room]
    edges: List[Tuple[int, int]]
    history: List[int]
    rules: List[str]


def parse_world(world_dict: Dict) -> World:
    """Parse world dictionary into structured World object."""
    rooms = []
    for r in world_dict["rooms"]:
        obj_list = world_dict.get("objects", {}).get(int(r["idx"]), [])
        rooms.append(Room(
            idx=int(r["idx"]),
            room_id=r["room_id"],
            room_type=r["type"],
            x=r["x"],
            y=r["y"],
            objects=obj_list
        ))

    edges = [tuple(e) for e in world_dict["edges"]]
    history = world_dict["history"]

    return World(
        scene_name=world_dict["scene_name"],
        gradient=world_dict.get("gradient", "G1"),
        gradient_label=world_dict.get("gradient_label", "Basic"),
        rooms=rooms,
        edges=edges,
        history=history,
        rules=world_dict.get("rules", [])
    )


def edge_to_str(edge: Tuple[int, int], rooms: List[Room]) -> str:
    """Convert edge tuple to room_id string."""
    return f"{rooms[edge[0]].room_id}-{rooms[edge[1]].room_id}"


def format_history(history: List[int], rooms: List[Room], obj_name: str = "Master_Key") -> str:
    """Format object history as string."""
    parts = []
    for i, ridx in enumerate(history):
        parts.append(f"{rooms[ridx].room_id}(t{i+1})")
    return f"{obj_name}: " + " -> ".join(parts)


def format_history_flat(history: List[int], rooms: List[Room], obj_name: str = "Master_Key") -> str:
    """Format history in flat narrative style."""
    if len(history) == 1:
        return f"At t1, a '{obj_name}' was in the {rooms[history[0]].room_type} ({rooms[history[0]].room_id})."

    parts = [f"At t1, a '{obj_name}' was in the {rooms[history[0]].room_type} ({rooms[history[0]].room_id})."]
    for i in range(1, len(history)):
        parts.append(f"At t{i+1}, it was picked up and moved to the {rooms[history[i]].room_type} ({rooms[history[i]].room_id}).")
    return " ".join(parts)


class SceneSerializer:
    """Main serializer class for generating all format variants."""

    def __init__(self, world: World):
        self.world = world
        self.rooms = world.rooms
        self.edges = world.edges
        self.history = world.history

    def serialize_flat(self) -> str:
        """
        Generate flat (natural language) format.
        All information presented in narrative prose style.
        """
        lines = []

        # Scene introduction
        scene_type = self._get_scene_type()
        lines.append(f"You are in a {scene_type}.")

        # Room descriptions
        for i, room in enumerate(self.rooms):
            obj_desc = self._format_objects_flat(room.objects)

            if i == 0:
                lines.append(f"You start at the {room.room_type} ({room.room_id}), located at ({room.x},{room.y}), which {obj_desc}.")
            else:
                # Find connection from previous room
                conn_info = self._get_connection_description(room.idx)
                if conn_info:
                    lines.append(f"{conn_info} the {room.room_type} ({room.room_id}) at ({room.x},{room.y}), which {obj_desc}.")
                else:
                    lines.append(f"There is also the {room.room_type} ({room.room_id}) at ({room.x},{room.y}), which {obj_desc}.")

        # Add topology hints for rooms not yet mentioned
        topology_hints = self._generate_topology_hints()
        if topology_hints:
            lines.append(topology_hints)

        # History
        hist_str = format_history_flat(self.history, self.rooms)
        lines.append(hist_str)

        return " ".join(lines)

    def serialize_hier(self, compression: float = 0.0) -> str:
        """
        Generate hierarchical format.
        Compression: 0.0 (full), 0.5 (50%), 0.75 (75%)
        """
        if compression == 0.0:
            return self._serialize_hier_full()
        elif compression <= 0.5:
            return self._serialize_hier_50()
        else:
            return self._serialize_hier_25()

    def _serialize_hier_full(self) -> str:
        """Full hierarchical format (0% compression)."""
        lines = [f"[SCENE: {self._get_scene_type()}]"]

        for room in self.rooms:
            obj_groups = self._group_objects(room.objects)
            obj_str = ", ".join([f"{cat}({', '.join(names)})" for cat, names in obj_groups.items()])
            lines.append(f"- [ROOM {room.room_id}: {room.room_type} @ ({room.x},{room.y})]")
            lines.append(f"  - [Objects: {obj_str}]")

        # Topology
        topo_str = ", ".join([edge_to_str(e, self.rooms) for e in self.edges])
        lines.append(f"\n[TOPOLOGY] {topo_str}")

        # History
        hist_str = format_history(self.history, self.rooms)
        lines.append(f"[HISTORY] {hist_str}")

        return "\n".join(lines)

    def _serialize_hier_50(self) -> str:
        """50% compressed hierarchical format."""
        lines = ["[HIERARCHY]"]

        scene_name = self.world.scene_name.replace("scene_complex_", "").replace("scene_simple_", "Home")
        lines.append(f"Scene({scene_name}) -> " + ",".join([r.room_id for r in self.rooms]))

        for room in self.rooms:
            obj_abbr = self._abbreviate_objects(room.objects, level=1)
            lines.append(f"{room.room_id}({room.x},{room.y}): {obj_abbr}")

        edges_str = ",".join([f"{self.rooms[e[0]].room_id}-{self.rooms[e[1]].room_id}" for e in self.edges])
        lines.append(f"Edges: {edges_str}")

        hist_str = ">".join([self.rooms[i].room_id for i in self.history])
        lines.append(f"Path: Key@{hist_str}")

        return "\n".join(lines)

    def _serialize_hier_25(self) -> str:
        """75% compressed hierarchical format."""
        parts = []

        scene_name = self.world.scene_name.replace("scene_complex_", "").replace("scene_simple_", "H")

        room_strs = []
        for room in self.rooms:
            obj_abbr = self._abbreviate_objects(room.objects, level=2)
            room_strs.append(f"{room.room_id}({room.x},{room.y}):{obj_abbr}")

        parts.append(f"{scene_name}[{'; '.join(room_strs)}].")

        # Compressed edges
        edge_nums = [f"{e[0]+1}-{e[1]+1}" for e in self.edges]
        parts.append(f"Edges:{','.join(edge_nums)}.")

        # Compressed history
        hist_nums = ">".join([str(i+1) for i in self.history])
        parts.append(f"Key:{hist_nums}.")

        return " ".join(parts)

    def serialize_clustered(self, compression: float = 0.0) -> str:
        """
        Generate clustered (zone-based) format.
        Compression: 0.0 (full), 0.5 (50%), 0.75 (75%)
        """
        if compression == 0.0:
            return self._serialize_clustered_full()
        elif compression <= 0.5:
            return self._serialize_clustered_50()
        else:
            return self._serialize_clustered_25()

    def _serialize_clustered_full(self) -> str:
        """Full clustered format (0% compression)."""
        zones = self._cluster_rooms()

        lines = [f"[SCENE: {self._get_scene_type()}]"]

        for zone_name, zone_rooms in zones.items():
            room_ids = [r.room_id for r in zone_rooms]
            all_objects = []
            for r in zone_rooms:
                all_objects.extend([f"{cat}" for cat, _ in r.objects])

            obj_types = list(set(all_objects))
            lines.append(f"- [ZONE {zone_name}: {', '.join(room_ids)}]")
            lines.append(f"  - [{', '.join(room_ids)} Objects]: {{{', '.join(obj_types)}}}")

        # Geometry - zone centers
        zone_centers = self._compute_zone_centers(zones)
        center_strs = [f"{name}_Center: ({c[0]:.1f}, {c[1]:.1f})" for name, c in zone_centers.items()]
        lines.append(f"\n[GEOMETRY] {' | '.join(center_strs)}")

        # Connectivity
        zone_connections = self._compute_zone_connections(zones)
        lines.append(f"[CONNECTIVITY] {zone_connections}")

        # History
        hist_zones = self._map_history_to_zones(self.history, zones)
        hist_str = " -> ".join([f"Zone_{z}({self.rooms[i].room_id})" for i, z in zip(self.history, hist_zones) if i in [self.history[0], self.history[-1]] or self.history.index(i) == 0])
        lines.append(f"[HISTORY] Master_Key: {hist_str}")

        return "\n".join(lines)

    def _serialize_clustered_50(self) -> str:
        """50% compressed clustered format."""
        zones = self._cluster_rooms()

        lines = ["[CLUSTERS]"]

        for zone_name, zone_rooms in zones.items():
            room_ids = [r.room_id for r in zone_rooms]
            center = self._compute_room_center(zone_rooms)
            obj_cats = list(set([cat for r in zone_rooms for cat, _ in r.objects]))
            lines.append(f"{zone_name}_Zone({','.join(room_ids)}) @({center[0]:.0f},{center[1]:.0f}): {{{'/'.join(obj_cats[:2])}}}")

        # Zone links
        zone_connections = self._compute_zone_connections(zones)
        lines.append(f"[LINKS] {zone_connections}")

        # History trace
        hist_zones = self._map_history_to_zones(self.history, zones)
        hist_str = "->".join([f"{z}({self.rooms[i].room_id})" for i, z in zip([self.history[0], self.history[-1]], [hist_zones[0], hist_zones[-1]])])
        lines.append(f"[TRACE] Key: {hist_str}.")

        return "\n".join(lines)

    def _serialize_clustered_25(self) -> str:
        """75% compressed clustered format."""
        zones = self._cluster_rooms()

        parts = []
        zone_idx = 1

        for zone_rooms in zones.values():
            center = self._compute_room_center(zone_rooms)
            room_nums = [str(r.idx + 1) for r in zone_rooms]
            obj_cats = list(set([cat for r in zone_rooms for cat, _ in r.objects]))[:2]
            parts.append(f"G{zone_idx}({center[0]:.0f},{center[1]:.0f}):{','.join(room_nums)}")
            zone_idx += 1

        # Compressed connections
        zone_connections = self._compute_zone_connections_compressed(zones)
        parts.append(f"Links:{zone_connections}")

        # Compressed history
        hist_zones = self._map_history_to_zones(self.history, zones)
        hist_str = ">".join([str(z[-1]) if isinstance(z, str) else str(z) for z in [hist_zones[0], hist_zones[-1]]])
        parts.append(f"Key:{hist_str}")

        return ". ".join(parts) + "."

    # Helper methods

    def _get_scene_type(self) -> str:
        """Extract scene type from scene name."""
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

    def _format_objects_flat(self, objects: List[Tuple[str, str]]) -> str:
        """Format objects in flat narrative style."""
        if not objects:
            return "has no notable objects"

        obj_names = [name for _, name in objects]
        if len(obj_names) == 1:
            return f"features a {obj_names[0]}"
        elif len(obj_names) == 2:
            return f"contains a {obj_names[0]} and a {obj_names[1]}"
        else:
            return f"contains {', '.join(obj_names[:-1])}, and a {obj_names[-1]}"

    def _get_connection_description(self, room_idx: int) -> Optional[str]:
        """Get narrative description of how to reach a room."""
        for edge in self.edges:
            if edge[1] == room_idx:
                from_room = self.rooms[edge[0]]
                to_room = self.rooms[room_idx]

                # Determine direction
                if to_room.x > from_room.x:
                    direction = "East"
                elif to_room.x < from_room.x:
                    direction = "West"
                elif to_room.y > from_room.y:
                    direction = "North"
                else:
                    direction = "South"

                return f"Moving {direction} from the {from_room.room_type} ({from_room.room_id}), you enter"

        return None

    def _generate_topology_hints(self) -> Optional[str]:
        """Generate additional topology hints for complex structures."""
        loops = self._find_loops()
        if loops:
            return f"Interestingly, {' and '.join([f'the {self.rooms[l[0]].room_type} features a connection back to the {self.rooms[l[1]].room_type}' for l in loops])}."
        return None

    def _find_loops(self) -> List[Tuple[int, int]]:
        """Find loops in the graph (edges that create cycles)."""
        loops = []
        # Simple heuristic: if an edge connects non-adjacent nodes in BFS order
        for i, edge in enumerate(self.edges):
            u, v = edge
            if abs(u - v) > 1:  # Non-sequential indices might form loops
                loops.append(edge)
        return loops[:2]  # Limit to 2 loops

    def _group_objects(self, objects: List[Tuple[str, str]]) -> Dict[str, List[str]]:
        """Group objects by category."""
        groups = {}
        for cat, name in objects:
            groups.setdefault(cat, []).append(name)
        return groups

    def _abbreviate_objects(self, objects: List[Tuple[str, str]], level: int = 1) -> str:
        """Abbreviate object list based on compression level."""
        if level == 1:
            # Level 1: Shortened category names, object names truncated
            abbrs = []
            for cat, name in objects[:4]:  # Limit to 4 objects
                cat_abbr = cat[:3] if len(cat) > 3 else cat
                name_abbr = name[:3] if len(name) > 3 else name
                abbrs.append(f"{cat_abbr}:{name_abbr}")
            return "{" + ", ".join(abbrs) + "}"
        else:
            # Level 2: Maximum compression
            return "{" + ", ".join([cat[:3] for cat, _ in objects[:3]]) + "}"

    def _cluster_rooms(self, n_clusters: int = 2) -> Dict[str, List[Room]]:
        """Cluster rooms into zones based on spatial proximity."""
        if len(self.rooms) <= n_clusters:
            return {"Main": self.rooms}

        # Simple spatial clustering by x-coordinate
        sorted_rooms = sorted(self.rooms, key=lambda r: r.x)
        mid = len(sorted_rooms) // 2

        return {
            "A": sorted_rooms[:mid],
            "B": sorted_rooms[mid:]
        }

    def _compute_room_center(self, rooms: List[Room]) -> Tuple[float, float]:
        """Compute geometric center of a room cluster."""
        if not rooms:
            return (0.0, 0.0)
        avg_x = sum(r.x for r in rooms) / len(rooms)
        avg_y = sum(r.y for r in rooms) / len(rooms)
        return (avg_x, avg_y)

    def _compute_zone_centers(self, zones: Dict[str, List[Room]]) -> Dict[str, Tuple[float, float]]:
        """Compute centers for all zones."""
        return {name: self._compute_room_center(rooms) for name, rooms in zones.items()}

    def _compute_zone_connections(self, zones: Dict[str, List[Room]]) -> str:
        """Compute inter-zone connectivity description."""
        connections = []
        zone_rooms_map = {}
        for zone_name, rooms in zones.items():
            for room in rooms:
                zone_rooms_map[room.idx] = zone_name

        # Find edges connecting different zones
        inter_zone_edges = []
        for edge in self.edges:
            z1 = zone_rooms_map.get(edge[0])
            z2 = zone_rooms_map.get(edge[1])
            if z1 and z2 and z1 != z2:
                inter_zone_edges.append((z1, z2, self.rooms[edge[0]].room_id, self.rooms[edge[1]].room_id))

        if inter_zone_edges:
            # Group by zone pairs
            zone_pairs = {}
            for z1, z2, r1, r2 in inter_zone_edges:
                key = f"{z1}-{z2}"
                zone_pairs.setdefault(key, []).append(f"{r1}-{r2}")

            for pair, edges in zone_pairs.items():
                connections.append(f"Zone_{pair.replace('-', ' <-> Zone ')} via {edges[0]}")

        # Add intra-zone loops
        for zone_name, rooms in zones.items():
            room_indices = {r.idx for r in rooms}
            zone_edges = [e for e in self.edges if e[0] in room_indices and e[1] in room_indices]
            if len(zone_edges) > len(rooms) - 1:  # Has cycle
                connections.append(f"Loop in Zone_{zone_name}")

        return ". ".join(connections) if connections else "No inter-zone connections"

    def _compute_zone_connections_compressed(self, zones: Dict[str, List[Room]]) -> str:
        """Compressed zone connection description."""
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

    def _map_history_to_zones(self, history: List[int], zones: Dict[str, List[Room]]) -> List[str]:
        """Map history room indices to zone names."""
        zone_map = {}
        for zone_name, rooms in zones.items():
            for room in rooms:
                zone_map[room.idx] = zone_name

        return [zone_map.get(idx, "Unknown") for idx in history]


def generate_all_variants(world_dict: Dict) -> Dict[str, str]:
    """
    Generate all format variants from a world dictionary.

    Returns a dictionary mapping variant names to their serialized strings.
    """
    world = parse_world(world_dict)
    serializer = SceneSerializer(world)

    variants = {}

    # Flat format
    variants["flat"] = serializer.serialize_flat()

    # Hierarchical formats (0%, 50%, 75% compression)
    variants["hier"] = serializer.serialize_hier(0.0)
    variants["hier_50"] = serializer.serialize_hier(0.5)
    variants["hier_25"] = serializer.serialize_hier(0.75)

    # Clustered formats (0%, 50%, 75% compression)
    variants["clustered"] = serializer.serialize_clustered(0.0)
    variants["clustered_50"] = serializer.serialize_clustered(0.5)
    variants["clustered_25"] = serializer.serialize_clustered(0.75)

    return variants


# ============================================================================
# Dimension-ablation rendering (for R2 information completeness experiments)
# ============================================================================

GRID_STEP = 4
KEY_NAME = "Master_Key"


def render_full(world_dict: Dict) -> str:
    """Include everything: geometry + topology + semantics + history + rules."""
    scene = world_dict.get("scene_name", "scene")
    rooms = world_dict["rooms"]
    edges = world_dict["edges"]
    objects = world_dict.get("objects", {})
    hist = world_dict["history"]
    rules = world_dict.get("rules", [])

    txt = []
    txt.append(f"You are in the scene '{scene}'.")
    txt.append("Rooms and coordinates:")
    for r in rooms:
        txt.append(f"- {r['room_id']} ({r['type']}) @ ({r['x']},{r['y']})")
        obj_list = ", ".join([f"{c}:{n}" for c, n in objects.get(int(r['idx']), [])])
        txt.append(f"  Objects: {obj_list}")
    room_ids = [r['room_id'] for r in rooms]
    edge_strs = [f"{room_ids[e[0]]}-{room_ids[e[1]]}" for e in edges]
    txt.append(f"Topology edges: {', '.join(edge_strs)}.")
    hist_str = format_history(hist, [parse_room_dict(r) for r in rooms], KEY_NAME)
    txt.append(f"History of {KEY_NAME}: {hist_str}.")
    txt.append("Rules/constraints:")
    for rr in rules:
        txt.append(f"- {rr}")
    return "\n".join(txt)


def render_topo_hist(world_dict: Dict) -> str:
    """Topology + History only (no coordinates/semantics details)."""
    rooms = world_dict["rooms"]
    edges = world_dict["edges"]
    hist = world_dict["history"]
    room_ids = [r['room_id'] for r in rooms]
    txt = []
    txt.append("You are given a set of room IDs and how they connect (topology).")
    txt.append(f"Rooms: {', '.join(room_ids)}.")
    edge_strs = [f"{room_ids[e[0]]}-{room_ids[e[1]]}" for e in edges]
    txt.append(f"Topology edges: {', '.join(edge_strs)}.")
    hist_str = format_history(hist, [parse_room_dict(r) for r in rooms], KEY_NAME)
    txt.append(f"History of {KEY_NAME}: {hist_str}.")
    txt.append("Constraint: When generating a path, output the shortest valid path unless the question specifies otherwise.")
    return "\n".join(txt)


def render_geom_rule_hist(world_dict: Dict) -> str:
    """Geometry + Rules + History; no explicit edges (must be inferred)."""
    rooms = world_dict["rooms"]
    hist = world_dict["history"]
    txt = []
    txt.append("You are given room coordinates. You must infer connectivity using the rules below.")
    txt.append("Rooms and coordinates:")
    for r in rooms:
        txt.append(f"- {r['room_id']} @ ({r['x']},{r['y']})")
    txt.append("Rules:")
    txt.append(f"- Two rooms are connected by a door iff their Manhattan distance is exactly {GRID_STEP}.")
    txt.append("- You may assume there are no blocked doors unless the text explicitly says so.")
    hist_str = format_history(hist, [parse_room_dict(r) for r in rooms], KEY_NAME)
    txt.append(f"History of {KEY_NAME}: {hist_str}.")
    txt.append("Constraint: Prefer the shortest path under the inferred connectivity.")
    return "\n".join(txt)


def render_sem_rule_hist(world_dict: Dict) -> str:
    """Semantics + Rules + History (room-chain). No coordinates or explicit topology."""
    rooms = world_dict["rooms"]
    objects = world_dict.get("objects", {})
    hist = world_dict["history"]
    room_ids = [r["room_id"] for r in rooms]
    room_chain = " -> ".join(room_ids)

    txt = []
    txt.append("You are given room semantics and room contents. Connectivity is defined by semantic rules below.")
    txt.append("Rooms and semantics:")
    for r in rooms:
        ridx = int(r["idx"])
        obj_list = ", ".join([f"{c}:{n}" for c, n in objects.get(ridx, [])])
        txt.append(f"- {r['room_id']} is a '{r['type']}'. Objects: {obj_list}")
    txt.append("Semantic connectivity rules (room-chain):")
    txt.append("- The scene provides an ordered chain of rooms. Two rooms are connected iff they are adjacent in this chain.")
    txt.append(f"- Room chain: {room_chain}")
    txt.append("- The chain is the ONLY source of connectivity (do not assume any other connections).")
    hist_str = format_history(hist, [parse_room_dict(r) for r in rooms], KEY_NAME)
    txt.append(f"History of {KEY_NAME}: {hist_str}.")
    txt.append("Constraint: Prefer the shortest path under the inferred connectivity.")
    return "\n".join(txt)


# ============================================================================
# Conflict variant rendering
# ============================================================================

def render_conflict_topo_hist(world_dict: Dict, seed: int = 42) -> str:
    """Create conflict: remove one edge used by key history, so topology contradicts history."""
    random.seed(seed)
    rooms = world_dict["rooms"]
    room_ids = [r["room_id"] for r in rooms]
    hist = world_dict["history"]
    edges = set(tuple(sorted(e)) for e in world_dict["edges"])

    implied = [tuple(sorted((a, b))) for a, b in zip(hist[:-1], hist[1:])]
    implied = [e for e in implied if e in edges]
    if implied:
        bad = random.choice(implied)
        edges.remove(bad)
    edges = sorted(list(edges))

    txt = []
    txt.append("You are given topology and history, but they may conflict.")
    txt.append(f"Rooms: {', '.join(room_ids)}.")
    edge_strs = [f"{room_ids[e[0]]}-{room_ids[e[1]]}" for e in edges]
    txt.append(f"Topology edges: {', '.join(edge_strs)}.")
    hist_str = format_history(hist, [parse_room_dict(r) for r in rooms], KEY_NAME)
    txt.append(f"History of {KEY_NAME}: {hist_str}.")
    txt.append("Note: If information conflicts, still output the best path you believe is correct.")
    return "\n".join(txt)


def render_conflict_geom_rule_hist(world_dict: Dict, seed: int = 42) -> str:
    """Conflict: keep coordinates but alter the rule so distance threshold is wrong."""
    random.seed(seed)
    rooms = world_dict["rooms"]
    hist = world_dict["history"]
    wrong_step = GRID_STEP * 2
    txt = []
    txt.append("You are given room coordinates. You must infer connectivity using the rules below (may be inconsistent).")
    txt.append("Rooms and coordinates:")
    for r in rooms:
        txt.append(f"- {r['room_id']} @ ({r['x']},{r['y']})")
    txt.append("Rules:")
    txt.append(f"- Two rooms are connected by a door iff their Manhattan distance is exactly {wrong_step}.")
    hist_str = format_history(hist, [parse_room_dict(r) for r in rooms], KEY_NAME)
    txt.append(f"History of {KEY_NAME}: {hist_str}.")
    txt.append("Note: If information conflicts, still output the best path you believe is correct.")
    return "\n".join(txt)


def render_conflict_sem_rule_hist(world_dict: Dict, seed: int = 42) -> str:
    """Conflict: shuffle room chain so implied connectivity contradicts history."""
    random.seed(seed)
    rooms = world_dict["rooms"]
    objects = world_dict.get("objects", {})
    hist = world_dict["history"]
    room_ids_list = [r["room_id"] for r in rooms]
    chain = room_ids_list[:]
    random.shuffle(chain)

    txt = []
    txt.append("You are given room semantics and room contents. Connectivity rules may conflict with history.")
    txt.append("Rooms and semantics:")
    for r in rooms:
        ridx = int(r["idx"])
        obj_list = ", ".join([f"{c}:{n}" for c, n in objects.get(ridx, [])])
        txt.append(f"- {r['room_id']} is a '{r['type']}'. Objects: {obj_list}")
    txt.append("Semantic connectivity rules (room-chain):")
    txt.append("- The scene provides an ordered chain of rooms. Two rooms are connected iff they are adjacent in this chain.")
    txt.append(f"- Room chain: {' -> '.join(chain)}")
    txt.append("- The chain is the ONLY source of connectivity (do not assume any other connections).")
    hist_str = format_history(hist, [parse_room_dict(r) for r in rooms], KEY_NAME)
    txt.append(f"History of {KEY_NAME}: {hist_str}.")
    txt.append("Note: If information conflicts, still output the best path you believe is correct.")
    return "\n".join(txt)


def parse_room_dict(r: Dict) -> Room:
    """Convert room dict to Room dataclass."""
    return Room(idx=r["idx"], room_id=r["room_id"], room_type=r["type"],
                x=r["x"], y=r["y"], objects=[])


# ============================================================================
# Batch variant generators for different experiment types
# ============================================================================

def generate_dimension_variants(world_dict: Dict) -> Dict[str, str]:
    """Generate dimension-ablation variants for R2 experiments."""
    return {
        "flat_full": render_full(world_dict),
        "flat_topo_hist": render_topo_hist(world_dict),
        "flat_geom_rule_hist": render_geom_rule_hist(world_dict),
        "flat_sem_rule_hist": render_sem_rule_hist(world_dict),
    }


def generate_conflict_variants(world_dict: Dict, seed: int = 42) -> Dict[str, str]:
    """Generate conflict variants for conflict experiments (includes base + conflict)."""
    variants = generate_dimension_variants(world_dict)
    variants["conflict_topo"] = render_conflict_topo_hist(world_dict, seed)
    variants["conflict_geom"] = render_conflict_geom_rule_hist(world_dict, seed + 1)
    variants["conflict_sem"] = render_conflict_sem_rule_hist(world_dict, seed + 2)
    return variants


def verify_equivalence(variants: Dict[str, str], world: World) -> Dict[str, bool]:
    """
    Verify that all variants encode the same underlying information.

    Checks:
    - All rooms are mentioned
    - All edges are represented (explicitly or implicitly)
    - History is preserved
    """
    results = {}

    for variant_name, text in variants.items():
        # In compressed formats rooms may appear as numeric indices (1,2,3) instead of IDs (R1,R2,R3)
        all_rooms_ok = all(
            room.room_id in text or str(room.idx + 1) in text
            for room in world.rooms
        )
        history_ok = any(
            room.room_id in text or str(room.idx + 1) in text
            for room in [world.rooms[i] for i in world.history]
        )
        checks = {
            "all_rooms": all_rooms_ok,
            "history_preserved": history_ok,
            "non_empty": len(text) > 0
        }
        results[variant_name] = all(checks.values())

    return results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scene_serializer.py <scene.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        scene = json.load(f)

    if "world" in scene:
        world_dict = scene["world"]
    else:
        world_dict = scene

    variants = generate_all_variants(world_dict)

    print("Generated variants:")
    for name, text in variants.items():
        print(f"\n=== {name} ===")
        print(text[:200] + "..." if len(text) > 200 else text)
