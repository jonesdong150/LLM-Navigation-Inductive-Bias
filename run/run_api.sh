#!/usr/bin/env bash
# API inference runner for navigation planning benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -------------------------
# Configuration
# -------------------------
# Select output path for R1, R2, or Conflict experiments
OUT_DIR="${PROJECT_DIR}/results/tmp"  # R1
# OUT_DIR="${PROJECT_DIR}/results/R2" # R2
# OUT_DIR="${PROJECT_DIR}/results/conflict"  # Conflict

# Data paths - select scene path for R1, R2, or Conflict experiments
SCENE_DATA_DIR="${PROJECT_DIR}/data_set/R1/scene" # R1
# SCENE_DATA_DIR="${PROJECT_DIR}/data_set/R2/scene" # R2
# SCENE_DATA_DIR="${PROJECT_DIR}/data_set/conflict/scene" # Conflict

QUERY_DATA_DIR="${PROJECT_DIR}/data_set/R1/queries" # R1
# QUERY_DATA_DIR="${PROJECT_DIR}/data_set/R2/queries" # R2
# QUERY_DATA_DIR="${PROJECT_DIR}/data_set/conflict/queries" # Conflict

# API model list to evaluate
# Note:
# 1. Qwen series: qwen-turbo, qwen-plus, qwen-max, qwen-long
# 2. GPT series (laozhang API): gpt-4o, gpt-4-turbo, gpt-3.5-turbo
# 3. Gemini series (laozhang API): gemini-1.5-pro, gemini-1.5-flash
MODELS="qwen3-14b,qwen3-32b,gpt-5.2,gemini-3-flash-preview"
# MODELS="gemini-2.5-flash-lite"

# Scenes to run; leave empty to auto-scan directory for all JSON files
SCENES=""

MAX_NEW_TOKENS=512
TEMPERATURE=0.0

RUN_PY="${SCRIPT_DIR}/run_api.py"

# -------------------------
# Auto-scan scenes logic
# -------------------------
if [[ -z "$SCENES" ]]; then
  if [[ -d "$SCENE_DATA_DIR" ]]; then
    # Scan directory for all JSON filenames as scene names
    mapfile -t SCENE_LIST < <(ls -1 "${SCENE_DATA_DIR}"/*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//')
  else
    echo "[ERROR] Scene directory not found: $SCENE_DATA_DIR"
    exit 1
  fi
else
  IFS=',' read -r -a SCENE_LIST <<< "$SCENES"
fi

# Convert comma-separated model string to array
IFS=',' read -r -a MODEL_LIST <<< "$MODELS"

echo "========================================================"
echo "[CONFIG] Output Dir: $OUT_DIR"
echo "[CONFIG] Scene Dir:  $SCENE_DATA_DIR"
echo "[CONFIG] Models:     ${MODEL_LIST[*]}"
echo "[CONFIG] Scenes:     ${SCENE_LIST[*]}"
echo "========================================================"

# -------------------------
# Execution loop
# -------------------------
for m in "${MODEL_LIST[@]}"; do
  for s in "${SCENE_LIST[@]}"; do
    echo
    echo ">>> [STARTING] Model: $m | Scene: $s"

    python3 "${RUN_PY}" \
      --model "$m" \
      --scene "$s" \
      --scene_dir "$SCENE_DATA_DIR" \
      --query_dir "$QUERY_DATA_DIR" \
      --out_dir "$OUT_DIR" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE"

    echo ">>> [FINISHED] Model: $m | Scene: $s"

    # Brief delay to avoid API rate limit triggers
    sleep 1
  done
done

echo
echo "[SUCCESS] All API tasks completed."
