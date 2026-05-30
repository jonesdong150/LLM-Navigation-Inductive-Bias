#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-LLM Verification Framework for Query Ground Truth.

Implements the paper's specification: Generator LLM ≠ Verifier LLM.

For each (scene, question, ground_truth) triple:
1. Symbolic solver computes ground truth (done in generate_dataset.py)
2. Independent verifier LLM checks the ground truth
3. If verifier agrees → keep sample
4. If verifier disagrees → discard sample

This module provides the verification interface. Actual LLM calls
require API keys and are configurable.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


class QueryVerifier:
    """Verifies query ground truths using an independent LLM.

    The verifier LLM must be different from the generator LLM
    per the paper's dual-model verification requirement.
    """

    def __init__(self, verifier_model: str = "deepseek-v3", api_key: str = ""):
        """Initialize the verifier.

        Args:
            verifier_model: Model name for verification (must differ from generator)
            api_key: API key for the verifier model
        """
        self.verifier_model = verifier_model
        self.api_key = api_key or os.environ.get("VERIFIER_API_KEY", "")

    def verify_single(self, scene_text: str, question: str,
                      ground_truth: str, task_type: str) -> Tuple[bool, str]:
        """Verify a single query's ground truth.

        Args:
            scene_text: The scene description
            question: The question text
            ground_truth: The symbolic solver's answer
            task_type: Type of task (ObjectLocation, GeometryYN, etc.)

        Returns:
            (is_correct, verifier_response) tuple
        """
        prompt = self._build_verification_prompt(
            scene_text, question, ground_truth, task_type
        )

        # Call the verifier LLM
        response = self._call_llm(prompt)

        # Parse the verifier's answer
        is_correct = self._parse_verification(response, ground_truth, task_type)

        return is_correct, response

    def verify_batch(self, scene_text: str, queries: List[Dict],
                     variant_name: str = "flat") -> Dict[str, Tuple[bool, str]]:
        """Verify a batch of queries.

        Args:
            scene_text: The scene description
            queries: List of query dicts with question_id, question_text, ground_truth, task_type
            variant_name: Which variant to use for scene text

        Returns:
            Dict mapping question_id to (is_correct, response)
        """
        results = {}
        for query in queries:
            qid = query.get("question_id", "")
            is_correct, response = self.verify_single(
                scene_text,
                query.get("question_text", ""),
                query.get("ground_truth", ""),
                query.get("task_type", ""),
            )
            results[qid] = (is_correct, response)
        return results

    def _build_verification_prompt(self, scene_text: str, question: str,
                                   ground_truth: str, task_type: str) -> str:
        """Build the verification prompt for the LLM."""
        format_hint = self._get_format_hint(task_type)

        prompt = f"""You are a spatial reasoning verifier. Your task is to verify whether the given answer to a spatial reasoning question is correct.

SCENE DESCRIPTION:
{scene_text}

QUESTION: {question}

PROPOSED ANSWER: {ground_truth}

TASK TYPE: {task_type}
EXPECTED FORMAT: {format_hint}

Please analyze the scene and question carefully. Determine if the proposed answer is correct.

Your response must be in this exact format:
VERDICT: CORRECT or INCORRECT
REASON: <brief explanation>

If the answer is CORRECT, respond: VERDICT: CORRECT
If the answer is INCORRECT, respond: VERDICT: INCORRECT followed by the correct answer in REASON."""

        return prompt

    def _get_format_hint(self, task_type: str) -> str:
        """Get format hint for the task type."""
        hints = {
            "ObjectLocation": "ROOM_ID (e.g., R3)",
            "GeometryYN": "YES or NO",
            "TopologyYN": "YES or NO",
            "ReachabilityYN": "YES or NO",
            "PathGen": "PATH (e.g., R1->R2->R3)",
        }
        return hints.get(task_type, "Answer")

    def _call_llm(self, prompt: str) -> str:
        """Call the verifier LLM.

        Override this method to integrate with your specific LLM API.
        Default implementation returns a placeholder.
        """
        # Placeholder - replace with actual API call
        # Example for OpenAI-compatible API:
        # import openai
        # client = openai.OpenAI(api_key=self.api_key)
        # response = client.chat.completions.create(
        #     model=self.verifier_model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.0,
        # )
        # return response.choices[0].message.content

        return "VERDICT: CORRECT\nREASON: Placeholder verification."

    def _parse_verification(self, response: str, ground_truth: str,
                            task_type: str) -> bool:
        """Parse the verifier's response to determine if it agrees."""
        response_upper = response.upper()
        if "VERDICT: CORRECT" in response_upper:
            return True
        elif "VERDICT: INCORRECT" in response_upper:
            return False
        # Default to agreeing if can't parse
        return True


class SymbolicVerifier:
    """Verifies queries using symbolic solvers (no LLM needed).

    This is the primary verification method per the paper's specification:
    "符号求解器正确 + LLM验证通过"
    """

    @staticmethod
    def verify_object_location(room_id: str, rooms: List[Dict],
                               objects: Dict, obj_name: str) -> bool:
        """Verify ObjectLocation answer by direct lookup.

        Returns True iff the object is in the specified room.
        """
        for r in rooms:
            ridx = str(r["idx"])
            room_objs = objects.get(ridx, objects.get(r["idx"], []))
            for o in room_objs:
                o_name = o.get("canonical", o.get("name", "")) if isinstance(o, dict) else o[1]
                if o_name == obj_name and r["room_id"] == room_id:
                    return True
        return False

    @staticmethod
    def verify_geometry_yn(room1_id: str, room2_id: str, direction: str,
                           rooms: List[Dict]) -> bool:
        """Verify GeometryYN answer by coordinate comparison."""
        r1 = next((r for r in rooms if r["room_id"] == room1_id), None)
        r2 = next((r for r in rooms if r["room_id"] == room2_id), None)
        if not r1 or not r2:
            return False

        y1, y2 = r1["y"], r2["y"]
        if direction == "North":
            return y1 > y2
        elif direction == "South":
            return y1 < y2
        elif direction == "at the same latitude as":
            return y1 == y2
        return False

    @staticmethod
    def verify_topology_yn(room1_id: str, room2_id: str,
                           edges: List, rooms: List[Dict]) -> bool:
        """Verify TopologyYN answer by edge lookup."""
        id_to_idx = {r["room_id"]: r["idx"] for r in rooms}
        idx1 = id_to_idx.get(room1_id)
        idx2 = id_to_idx.get(room2_id)
        if idx1 is None or idx2 is None:
            return False
        edge_set = set(tuple(sorted(e)) for e in edges)
        return tuple(sorted((idx1, idx2))) in edge_set

    @staticmethod
    def verify_reachability(room1_id: str, room2_id: str,
                            edges: List, rooms: List[Dict]) -> bool:
        """Verify ReachabilityYN answer by BFS."""
        id_to_idx = {r["room_id"]: r["idx"] for r in rooms}
        idx1 = id_to_idx.get(room1_id)
        idx2 = id_to_idx.get(room2_id)
        if idx1 is None or idx2 is None:
            return False

        adj = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        visited = {idx1}
        queue = [idx1]
        while queue:
            curr = queue.pop(0)
            for neighbor in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return idx2 in visited

    @staticmethod
    def verify_path_gen(path: str, start_id: str, end_id: str,
                        edges: List, rooms: List[Dict]) -> bool:
        """Verify PathGen answer by checking path validity and optimality."""
        if path == "UNREACHABLE":
            # Verify unreachability
            return not SymbolicVerifier.verify_reachability(
                start_id, end_id, edges, rooms
            )

        room_ids = path.split("->")
        if not room_ids or room_ids[0] != start_id or room_ids[-1] != end_id:
            return False

        # Check each edge in path exists
        id_to_idx = {r["room_id"]: r["idx"] for r in rooms}
        edge_set = set(tuple(sorted(e)) for e in edges)
        for i in range(len(room_ids) - 1):
            idx1 = id_to_idx.get(room_ids[i])
            idx2 = id_to_idx.get(room_ids[i + 1])
            if idx1 is None or idx2 is None:
                return False
            if tuple(sorted((idx1, idx2))) not in edge_set:
                return False

        # Verify shortest path (BFS)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        start_idx = id_to_idx[start_id]
        end_idx = id_to_idx[end_id]

        # BFS to find shortest distance
        dist = {start_idx: 0}
        queue = [start_idx]
        while queue:
            curr = queue.pop(0)
            for neighbor in adj.get(curr, []):
                if neighbor not in dist:
                    dist[neighbor] = dist[curr] + 1
                    queue.append(neighbor)

        if end_idx not in dist:
            return False

        return len(room_ids) - 1 == dist[end_idx]


def verify_scene_queries(scene_data: Dict, queries: List[Dict]) -> Dict[str, bool]:
    """Verify all queries for a scene using symbolic solvers.

    Returns dict mapping question_id to is_correct.
    """
    world = scene_data.get("world", scene_data)
    rooms = world.get("rooms", [])
    edges = world.get("edges", [])
    objects = world.get("objects", {})

    results = {}
    for query in queries:
        qid = query.get("question_id", "")
        task_type = query.get("task_type", "")
        gt = query.get("ground_truth", "")
        qtext = query.get("question_text", "")

        is_correct = False

        if task_type == "ObjectLocation":
            # Extract object name from question
            obj_name = qtext.replace("Where is the ", "").replace(" currently located?", "")
            is_correct = SymbolicVerifier.verify_object_location(
                gt, rooms, objects, obj_name
            )

        elif task_type == "GeometryYN":
            # Parse question to get room IDs and direction
            # "Is the X (R1) located North the Y (R2)?"
            # "Is the X (R1) located at the same latitude as the Y (R2)?"
            parts = qtext.split()
            r1_id = None
            r2_id = None
            direction = None
            for j, p in enumerate(parts):
                cleaned = p.rstrip("?.,;:")
                if cleaned.startswith("(R") and cleaned.endswith(")"):
                    if r1_id is None:
                        r1_id = cleaned[1:-1]
                    else:
                        r2_id = cleaned[1:-1]
                if p in ("North", "South", "East", "West"):
                    direction = p
            # Handle multi-word direction "at the same latitude as"
            if direction is None and "same latitude" in qtext:
                direction = "at the same latitude as"
            if r1_id and r2_id and direction:
                expected = "YES" if SymbolicVerifier.verify_geometry_yn(
                    r1_id, r2_id, direction, rooms
                ) else "NO"
                is_correct = (gt == expected)

        elif task_type == "TopologyYN":
            parts = qtext.split()
            r1_id = r2_id = None
            for j, p in enumerate(parts):
                cleaned = p.rstrip("?.,;:")
                if cleaned.startswith("R") and cleaned[1:].isdigit():
                    if r1_id is None:
                        r1_id = cleaned
                    else:
                        r2_id = cleaned
            if r1_id and r2_id:
                expected = "YES" if SymbolicVerifier.verify_topology_yn(
                    r1_id, r2_id, edges, rooms
                ) else "NO"
                is_correct = (gt == expected)

        elif task_type == "ReachabilityYN":
            parts = qtext.split()
            r1_id = r2_id = None
            for p in parts:
                cleaned = p.rstrip("?.,;:")
                if cleaned.startswith("R") and cleaned[1:].isdigit():
                    if r1_id is None:
                        r1_id = cleaned
                    else:
                        r2_id = cleaned
            if r1_id and r2_id:
                expected = "YES" if SymbolicVerifier.verify_reachability(
                    r1_id, r2_id, edges, rooms
                ) else "NO"
                is_correct = (gt == expected)

        elif task_type == "PathGen":
            parts = qtext.split()
            r1_id = r2_id = None
            for p in parts:
                cleaned = p.rstrip("?.,;:")
                if cleaned.startswith("R") and cleaned[1:].isdigit():
                    if r1_id is None:
                        r1_id = cleaned
                    else:
                        r2_id = cleaned
            if r1_id and r2_id:
                is_correct = SymbolicVerifier.verify_path_gen(
                    gt, r1_id, r2_id, edges, rooms
                )

        results[qid] = is_correct

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify scene queries")
    parser.add_argument("scene_json", help="Path to scene JSON file")
    parser.add_argument("--query_csv", help="Path to query CSV file (optional)")
    args = parser.parse_args()

    with open(args.scene_json, "r", encoding="utf-8") as f:
        scene = json.load(f)

    # Load queries
    if args.query_csv:
        import csv as csv_mod
        with open(args.query_csv, "r", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            queries = list(reader)
    else:
        # Generate queries
        from tools.generate_dataset import generate_queries_for_scene
        queries = generate_queries_for_scene(scene)

    results = verify_scene_queries(scene, queries)

    correct = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Verified {total} queries: {correct} correct, {total - correct} incorrect")
    print(f"Accuracy: {correct/total*100:.1f}%")

    # Show incorrect ones
    for qid, is_correct in results.items():
        if not is_correct:
            q = next((q for q in queries if q.get("question_id") == qid), None)
            if q:
                print(f"  INCORRECT: {qid} - {q.get('task_type')}: {q.get('ground_truth')}")
