#!/usr/bin/env bash
# Local Qwen model inference runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -------------------------
# Configuration
# -------------------------
# Select output path for R1, R2, or Conflict experiments
OUT_DIR="${PROJECT_DIR}/results/R1"  # R1
# OUT_DIR="${PROJECT_DIR}/results/R2" # R2
# OUT_DIR="${PROJECT_DIR}/results/conflict"  # Conflict
MODELS="Qwen3-0.6B,Qwen3-1.7B,Qwen3-4B"
SCENES=""  # Leave empty to auto-scan
MAX_NEW_TOKENS=256
ENABLE_THINKING=0
DO_SAMPLE=0
TEMPERATURE=0.0

RUN_PY="${SCRIPT_DIR}/run_local_qwen.py"
# Select scene path for R1, R2, or Conflict experiments
SCENE_DATA_DIR="${PROJECT_DIR}/data_set/R1/scene" # R1
# SCENE_DATA_DIR="${PROJECT_DIR}/data_set/R2/scene" # R2
# SCENE_DATA_DIR="${PROJECT_DIR}/data_set/conflict/scene" # Conflict

# Model path mapping for small models (deployment-friendly)
# Override with SHIYAN_MODEL_ROOT env var if needed
MODEL_ROOT="${SHIYAN_MODEL_ROOT:-${PROJECT_DIR}/model}"
model_path_of () {
  case "$1" in
    Qwen3-0.6B) echo "${MODEL_ROOT}/Qwen3-0.6B/Qwen/Qwen3-0.6B" ;;
    Qwen3-1.7B) echo "${MODEL_ROOT}/Qwen3-1.7B/Qwen/Qwen3-1.7B" ;;
    Qwen3-4B)   echo "${MODEL_ROOT}/Qwen3-4B/Qwen/Qwen3-4B" ;;
  esac
}

# Parse optional input arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    *) shift 1 ;;
  esac
done

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
if [[ -z "$SCENES" ]]; then
  mapfile -t SCENE_LIST < <(ls -1 "${SCENE_DATA_DIR}"/*.json | xargs -n1 basename | sed 's/\.json$//')
else
  IFS=',' read -r -a SCENE_LIST <<< "$SCENES"
fi

# -------------------------
# Execution loop
# -------------------------
for m in "${MODEL_LIST[@]}"; do
  mp=$(model_path_of "$m")
  for s in "${SCENE_LIST[@]}"; do
    echo ">>> Executing: Model=$m, Scene=$s"

    python3 "${RUN_PY}" \
      --model "$m" \
      --model_path "$mp" \
      --scene "$s" \
      --out_dir "$OUT_DIR" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      $( [[ "$ENABLE_THINKING" -eq 1 ]] && echo "--enable_thinking" ) \
      $( [[ "$DO_SAMPLE" -eq 1 ]] && echo "--do_sample --temperature $TEMPERATURE" )
  done
done

echo "[SUCCESS] All tasks finished."
