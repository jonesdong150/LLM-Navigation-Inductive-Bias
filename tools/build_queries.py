#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch build PathGen queries (questions.csv) from scene JSON files with a `world` field.

- --scene can be a single scene JSON or a directory containing multiple scene_*.json
- --out_dir specifies where to write <scene_basename>_questions.csv

CSV schema (aligned with your existing runner/old queries):
  question_id, task_type, question_text, answer_format, ground_truth

Only generates PathGen tasks.
"""

import argparse
import csv
import json
import os
import random
from collections import deque
from typing import Dict, List, Tuple


# ---------------- Graph utils ----------------

def build_graph(n: int, edges: List[List[int]]) -> Dict[int, List[int]]:
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for k in adj:
        adj[k] = sorted(adj[k])
    return adj


def lex_smallest_shortest_path(
    adj: Dict[int, List[int]],
    start: int,
    goal: int,
    room_ids: List[str],
) -> List[int]:
    """Shortest path with lexicographic tie-break on room_id sequence."""
    if start == goal:
        return [start]

    # BFS distance from start
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)

    if goal not in dist:
        return []  # unreachable

    # BFS distance to goal
    dist_to_goal = {goal: 0}
    q = deque([goal])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist_to_goal:
                dist_to_goal[v] = dist_to_goal[u] + 1
                q.append(v)

    total_len = dist[goal]
    path = [start]
    cur = start

    for _ in range(total_len):
        candidates = []
        for nxt in adj[cur]:
            if dist.get(nxt, 10**9) == dist[cur] + 1:
                if dist_to_goal.get(nxt, 10**9) == total_len - dist[nxt]:
                    candidates.append(nxt)
        if not candidates:
            return []
        candidates.sort(key=lambda i: room_ids[i])
        cur = candidates[0]
        path.append(cur)

    return path


def path_to_str(path: List[int], room_ids: List[str]) -> str:
    return "->".join(room_ids[i] for i in path)


# ---------------- Scene loading ----------------

def load_scene(scene_path: str) -> Tuple[dict, dict]:
    with open(scene_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "world" not in data:
        raise ValueError(f"Scene JSON missing 'world': {scene_path}")
    return data, data["world"]


def list_scene_files(scene_arg: str) -> List[str]:
    """scene_arg can be a file or a directory."""
    if os.path.isfile(scene_arg):
        return [scene_arg]
    if os.path.isdir(scene_arg):
        files = []
        for fn in os.listdir(scene_arg):
            if fn.lower().endswith(".json") and fn.startswith("scene_"):
                files.append(os.path.join(scene_arg, fn))
        files.sort()
        return files
    raise FileNotFoundError(f"--scene not found: {scene_arg}")


# ---------------- Query generation ----------------

def make_question_text(
    start_idx: int,
    goal_idx: int,
    rooms: List[dict],
    room_ids: List[str],
) -> str:
    """
    Keep it stable and easy to parse: include both semantic type and room id.
    """
    s = rooms[start_idx]
    g = rooms[goal_idx]
    return (
        f'Provide the shortest path from {s["type"]} ({room_ids[start_idx]}) '
        f'to {g["type"]} ({room_ids[goal_idx]}).'
    )


def generate_pairs(n_rooms: int, num_q: int, seed: int) -> List[Tuple[int, int]]:
    random.seed(seed)
    all_pairs = [(i, j) for i in range(n_rooms) for j in range(n_rooms) if i != j]
    random.shuffle(all_pairs)
    if num_q <= len(all_pairs):
        return all_pairs[:num_q]
    out = []
    while len(out) < num_q:
        random.shuffle(all_pairs)
        out.extend(all_pairs)
    return out[:num_q]


def is_likely_unique(adj: Dict[int, List[int]], start: int, goal: int) -> bool:
    """
    Lightweight uniqueness heuristic: if goal has multiple shortest parents, likely multiple shortest paths.
    """
    dist = {start: 0}
    parents_count = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                parents_count[v] = 1
                q.append(v)
            else:
                if dist[v] == dist[u] + 1:
                    parents_count[v] += 1
    if goal not in dist:
        return False
    return parents_count.get(goal, 1) <= 1


def build_queries_for_scene(
    scene_path: str,
    out_csv: str,
    num_questions: int,
    seed: int,
    require_unique_paths: bool,
) -> int:
    scene_json, world = load_scene(scene_path)

    rooms = world["rooms"]
    n = len(rooms)

    # room_ids in order of idx
    room_ids = [None] * n
    for r in rooms:
        room_ids[int(r["idx"])] = r["room_id"]
    if any(x is None for x in room_ids):
        raise ValueError(f"Room idx mapping incomplete: {scene_path}")

    edges = []
    for e in world["edges"]:
        edges.append([int(e[0]), int(e[1])])

    adj = build_graph(n, edges)

    # Per-scene deterministic seed (stable in batch)
    scene_seed = (seed * 1000003) ^ (abs(hash(os.path.basename(scene_path))) & 0xFFFFFFFF)
    pairs = generate_pairs(n, num_questions, scene_seed)

    rows = []
    qid = 0
    for s_idx, g_idx in pairs:
        if require_unique_paths and (not is_likely_unique(adj, s_idx, g_idx)):
            continue

        path = lex_smallest_shortest_path(adj, s_idx, g_idx, room_ids)
        if not path:
            continue

        qid += 1
        q_text = make_question_text(s_idx, g_idx, rooms, room_ids)
        ans = path_to_str(path, room_ids)

        # OLD/EXPECTED CSV SCHEMA:
        rows.append({
            "question_id": f"q{qid:02d}" if qid < 100 else f"q{qid}",
            "task_type": "PathGen",
            "question_text": q_text,
            "answer_format": "PATH",
            "ground_truth": ans,
        })

        if len(rows) >= num_questions:
            break

    if len(rows) == 0:
        raise RuntimeError(f"No questions generated for {scene_path}. Check connectivity/uniqueness.")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = ["question_id", "task_type", "question_text", "answer_format", "ground_truth"]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Scene JSON file OR directory containing scene_*.json")
    ap.add_argument("--out_dir", required=True, help="Output directory for *_questions.csv")
    ap.add_argument("--num_questions", type=int, default=50, help="Questions per scene")
    ap.add_argument("--seed", type=int, default=42, help="Base seed")
    ap.add_argument("--require_unique_paths", action="store_true", help="Try to keep only likely-unique shortest paths")
    args = ap.parse_args()

    scene_files = list_scene_files(args.scene)
    os.makedirs(args.out_dir, exist_ok=True)

    ok = 0
    for sp in scene_files:
        base = os.path.splitext(os.path.basename(sp))[0]  # scene_xxx
        out_csv = os.path.join(args.out_dir, f"{base}_questions.csv")
        n = build_queries_for_scene(
            scene_path=sp,
            out_csv=out_csv,
            num_questions=args.num_questions,
            seed=args.seed,
            require_unique_paths=args.require_unique_paths,
        )
        print(f"[OK] {base}: wrote {n} questions -> {out_csv}")
        ok += 1

    print(f"[DONE] Processed {ok} scene file(s). Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
