#!/usr/bin/env python3
"""
Local Llama model inference runner for navigation planning benchmark.
"""

import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Base path configuration
PROJECT_ROOT = Path(__file__).parent.parent
DR_DATA_BASE = PROJECT_ROOT / "data_set" / "R1"
SCENE_DIR = DR_DATA_BASE / "scene"
QUERY_DIR = DR_DATA_BASE / "queries"
PROMPT_DIR = PROJECT_ROOT / "prompts"

def load_scene(scene_name: str) -> dict:
    p = SCENE_DIR / f"{scene_name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Scene file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_queries(scene_name: str) -> List[dict]:
    p = QUERY_DIR / f"{scene_name}_questions.csv"
    if not p.exists():
        raise FileNotFoundError(f"Queries not found: {p}")

    rows = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            qid = (r.get("question_id") or r.get("query_id") or "").strip()
            if not qid or qid.startswith("#"):
                continue
            r["question_id"] = qid
            r["task_type"] = (r.get("task_type") or "").strip()
            r["answer_format"] = (r.get("answer_format") or "").strip()
            if r["task_type"]:
                rows.append(r)
    return rows

def load_prompts() -> Dict[str, str]:
    prompts = {}
    if not PROMPT_DIR.exists():
        return prompts
    for p_file in PROMPT_DIR.glob("*_ieq.txt"):
        variant_name = p_file.name.replace("_ieq.txt", "")
        with open(p_file, "r", encoding="utf-8") as f:
            prompts[variant_name] = f.read()
    return prompts

def get_strict_rules(fmt: str) -> str:
    """Enhanced output format restriction rules"""
    fmt = fmt.upper()
    if "YES_NO" in fmt:
        return (
            'For example, if you want to answer the question, the required output format should be: '
            '"Answer: YES" or "Answer: NO". Do not provide any extra explanations.'
        )
    if "ROOM_ID" in fmt:
        return (
            'Specifically, the required output format is: "Answer: Rn" '
            '(where n is the room number). '
            'CRITICAL: Provide ONLY the room ID (e.g., R1). '
            'DO NOT include room names or parentheses, and DO NOT add any extra text.'
        )
    if "PATH" in fmt:
        return (
            'Specifically, the Example of the required output format is: '
            '"Answer: Rx->Ry->Rz (x, y, z=1, 2, 3...)". '
            'Note that you must use ONLY "->" as the separator with no spaces.'
        )
    return 'Please ensure your answer follows the format: "Answer: <value>".'

def build_final_prompt(template: str, scene_text: str, query_row: dict) -> str:
    """Build prompt that meets format requirements"""
    q_text = query_row.get("question_text") or query_row.get("query_text")
    task = query_row.get("task_type", "General")
    fmt = query_row.get("answer_format", "TEXT")

    formatted_scene = f"[SCENE]\n{scene_text}"

    fmt_desc = fmt
    if fmt == "ROOM_ID":
        fmt_desc = "ROOM_ID (Strictly ONLY the ID, e.g., R1)"

    out = template.replace("{scene_text}", formatted_scene)
    out = out.replace("{query_text}", q_text)
    out = out.replace("{task_type}", task)
    out = out.replace("{answer_format}", fmt_desc)
    out = out.replace("{strict_format_rules}", get_strict_rules(fmt))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--do_sample", action="store_true")
    args = ap.parse_args()

    os.environ["SHIYAN_MODEL_PATH"] = args.model_path
    import sys
    sys.path.append(str(PROJECT_ROOT))
    from run.infer_hugface_llama import run_llm

    scene_json = load_scene(args.scene)
    queries = load_queries(args.scene)
    prompts_tpl = load_prompts()

    out_root = Path(args.out_dir) / args.model / args.scene
    out_root.mkdir(parents=True, exist_ok=True)

    sample_dir = out_root / "prompt_samples"
    sample_dir.mkdir(exist_ok=True)
    sampled_tasks = set()

    print(f"[RUNNING DR-EXP] {args.model} on {args.scene}")

    for q in tqdm(queries):
        if not q or not q.get("task_type"):
            continue

        qid = q["question_id"]
        current_task_type = q["task_type"]

        record = {
            "scene": args.scene,
            "model": args.model,
            "query_id": qid,
            "task_type": current_task_type,
            "answer_format": q["answer_format"],
            "query_text": q.get("question_text") or q.get("query_text") or "",
            "ground_truth": q.get("ground_truth", ""),
            "results": {}
        }

        for v in scene_json["variants"].keys():
            scene_text = scene_json["variants"][v]

            tpl = prompts_tpl.get(v)
            if not tpl:
                tpl = prompts_tpl.get("hier")
            if not tpl:
                tpl = prompts_tpl.get("flat")

            if not tpl:
                print(f"Warning: No prompt template found for variant {v}, skipping.")
                continue

            full_prompt = build_final_prompt(tpl, scene_text, q)

            sample_key = f"{v}_{current_task_type}"
            if sample_key not in sampled_tasks:
                with open(sample_dir / f"{sample_key}.txt", "w", encoding="utf-8") as sf:
                    sf.write(full_prompt)
                sampled_tasks.add(sample_key)

            try:
                out = run_llm(
                    prompt=full_prompt,
                    model_path=args.model_path,
                    enable_thinking=args.enable_thinking,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.do_sample
                )
            except Exception as e:
                print(f"\n[ERROR] Inference failed for {qid} in {v}: {e}")
                out = {"answer": "ERROR", "thinking": str(e)}

            ans_raw = out.get("answer", "").strip()
            clean_answer = ans_raw
            if "Answer:" in ans_raw:
                clean_answer = ans_raw.split("Answer:")[-1].strip()

            record["results"][v] = {
                "answer": clean_answer,
                "answer_raw": ans_raw,
                "format_ok": True if "Answer:" in ans_raw else False,
                "thinking": out.get("thinking", "") if args.enable_thinking else ""
            }

        with open(out_root / f"{qid}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
