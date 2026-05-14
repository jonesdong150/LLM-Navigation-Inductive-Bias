#!/usr/bin/env python3
"""
API-based inference runner for navigation planning benchmark.

Supports DashScope (Qwen series) and third-party API (GPT/Gemini series).
"""

import os
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from openai import OpenAI

# -------------------------
# API Configuration
# -------------------------

# DashScope (Qwen series) configuration
DASHSCOPE_API_KEY = ""   # Obtain from: https://www.aliyun.com/benefit
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Third-party API (GPT/Gemini series) configuration
LAOZHANG_API_KEY = ""    # Obtain from: https://api.laozhang.ai/
LAOZHANG_BASE_URL = "https://api.laozhang.ai/v1"

def get_client(model_name: str) -> OpenAI:
    """Auto-select API client based on model name."""
    model_lower = model_name.lower()

    # Qwen-prefixed models use DashScope; others use laozhang API
    if "qwen" in model_lower:
        return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
    else:
        return OpenAI(api_key=LAOZHANG_API_KEY, base_url=LAOZHANG_BASE_URL)

def run_api_inference(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = 5
) -> Dict[str, str]:
    """
    Execute API inference with retry mechanism.
    """
    messages = [{"role": "user", "content": prompt}]

    # Qwen-specific parameters (disable thinking mode)
    extra_body = {}
    if "qwen" in model_name.lower():
        extra_body["enable_thinking"] = False

    delay = 2
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body if extra_body else None
            )

            content = response.choices[0].message.content
            return {
                "answer": content,
                "thinking": ""
            }

        except Exception as e:
            print(f"\n[API ERROR] Model: {model_name} | Attempt {attempt+1}/{max_retries} | Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                return {"answer": f"API_ERROR: {str(e)}", "thinking": ""}

# -------------------------
# Data and Prompt Processing
# -------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_SCENE_DIR = PROJECT_ROOT / "data_set" / "R1" / "scene"
DEFAULT_QUERY_DIR = PROJECT_ROOT / "data_set" / "R1" / "queries"
PROMPT_DIR = PROJECT_ROOT / "prompts"

def load_scene(scene_path: Path) -> dict:
    with open(scene_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_queries(query_path: Path) -> List[dict]:
    if not query_path.exists():
        raise FileNotFoundError(f"Queries not found: {query_path}")
    rows = []
    with open(query_path, "r", encoding="utf-8") as f:
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
    fmt = fmt.upper()
    if "YES_NO" in fmt:
        return (
            'For example, if you want to answer the question, the required output format should be: '
            '"Answer: YES" or "Answer: NO". Do not provide any explanations.'
        )

    if "ROOM_ID" in fmt:
        return (
            'Specifically, the required output format is: "Answer: Rn" (where n is the room number). '
            'CRITICAL: Provide ONLY the room ID (e.g., R1). '
            'DO NOT include room names, DO NOT include parentheses, and DO NOT add any extra text.'
        )

    if "PATH" in fmt:
        return (
            'Specifically, the Example of the required output format is: "Answer: Rx->Ry->Rz '
            '(x, y, z=1, 2, 3...)". '
            'Note that you must use ONLY "->" as the separator and ensure there are no spaces or room names in the path.'
        )

    return 'Please ensure your answer follows the format: "Answer: <value>".'

def build_final_prompt(template: str, scene_text: str, query_row: dict) -> str:
    q_text = query_row.get("question_text") or query_row.get("query_text") or ""
    task = query_row.get("task_type", "General")
    fmt = query_row.get("answer_format", "TEXT")

    # Add [SCENE] tag before scene text
    formatted_scene = f"[SCENE]\n{scene_text}"

    # Inject strict format constraints for ROOM_ID tasks
    fmt_desc = fmt
    if fmt == "ROOM_ID":
        fmt_desc = "ROOM_ID (Strictly ONLY the ID, e.g., R1)"

    out = template.replace("{scene_text}", formatted_scene)
    out = out.replace("{query_text}", q_text)
    out = out.replace("{task_type}", task)
    out = out.replace("{answer_format}", fmt_desc)
    out = out.replace("{strict_format_rules}", get_strict_rules(fmt))

    return out

# -------------------------
# Main Program
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Scene name (without .json)")
    ap.add_argument("--model", required=True, help="API Model name (e.g. qwen-plus, gpt-4o)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scene_dir", default=str(DEFAULT_SCENE_DIR))
    ap.add_argument("--query_dir", default=str(DEFAULT_QUERY_DIR))
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    # Initialize client
    client = get_client(args.model)

    # Path processing
    scene_path = Path(args.scene_dir) / f"{args.scene}.json"
    query_path = Path(args.query_dir) / f"{args.scene}_questions.csv"

    # Load data
    scene_json = load_scene(scene_path)
    queries = load_queries(query_path)
    prompts_tpl = load_prompts()

    # Output directory structure: out_dir/model_name/scene_name/
    out_root = Path(args.out_dir) / args.model / args.scene
    out_root.mkdir(parents=True, exist_ok=True)

    sample_dir = out_root / "prompt_samples"
    sample_dir.mkdir(exist_ok=True)
    sampled_tasks = set()

    print(f"[RUN API] Model: {args.model} | Scene: {args.scene} | Queries: {len(queries)}")

    for q in tqdm(queries):
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

        # Dynamically iterate variants (flat, hier_50, clustered_25, etc.)
        variants_to_run = list(scene_json["variants"].keys())

        for v in variants_to_run:
            scene_text = scene_json["variants"][v]

            # Template selection logic:
            # 1. Prefer variant-specific template (e.g., combo_topo_ieq.txt)
            tpl = prompts_tpl.get(v)

            # 2. Fall back to hier template
            if not tpl:
                tpl = prompts_tpl.get("hier")

            # 3. Final fallback to flat template
            if not tpl:
                tpl = prompts_tpl.get("flat")

            if not tpl:
                print(f"Warning: No prompt template found for variant {v}, skipping.")
                continue

            full_prompt = build_final_prompt(tpl, scene_text, q)

            # Export prompt sample
            sample_key = f"{v}_{current_task_type}"
            if sample_key not in sampled_tasks:
                with open(sample_dir / f"{sample_key}.txt", "w", encoding="utf-8") as sf:
                    sf.write(full_prompt)
                sampled_tasks.add(sample_key)

            # === Core API call ===
            out = run_api_inference(
                client=client,
                model_name=args.model,
                prompt=full_prompt,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens
            )

            # Post-process results
            ans_raw = out.get("answer", "").strip()
            clean_answer = ans_raw
            # If model outputs "Answer: XXX", extract XXX
            if "Answer:" in ans_raw:
                clean_answer = ans_raw.split("Answer:")[-1].strip()

            record["results"][v] = {
                "answer": clean_answer,
                "answer_raw": ans_raw,
                "format_ok": True if "Answer:" in ans_raw else False,
                "thinking": out.get("thinking", "")
            }

        # Save results incrementally
        with open(out_root / f"{qid}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
