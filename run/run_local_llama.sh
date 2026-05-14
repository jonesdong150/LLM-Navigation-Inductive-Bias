#!/usr/bin/env bash
# Local Llama model inference runner
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
MODELS="Llama-3.2-1B,Llama-3.2-3B,Llama-3.1-8B"
SCENES=""  # Leave empty to auto-scan
MAX_NEW_TOKENS=256
ENABLE_THINKING=0  # Llama does not natively support thinking
DO_SAMPLE=0
TEMPERATURE=0.0

# Point to our Python logic file
RUN_PY="${SCRIPT_DIR}/run_local_llama.py"
# Select scene path for R1, R2, or Conflict experiments
SCENE_DATA_DIR="${PROJECT_DIR}/data_set/R1/scene" # R1
# SCENE_DATA_DIR="${PROJECT_DIR}/data_set/R2/scene" # R2
# SCENE_DATA_DIR="${PROJECT_DIR}/data_set/conflict/scene" # Conflict

# Model path mapping for Llama models
model_path_of () {
  case "$1" in
    Llama-3.2-1B) echo "/root/autodl-tmp/model/Llama-3.2-1B-Instruct/LLM-Research/Llama-3.2-1B-Instruct" ;;
    Llama-3.2-3B) echo "/root/autodl-tmp/model/Llama-3.2-3B-Instruct/LLM-Research/Llama-3.2-3B-Instruct" ;;
    Llama-3.1-8B) echo "/root/autodl-tmp/model/Meta-Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3.1-8B-Instruct" ;;
    # Add other Llama versions as needed
  esac
}

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
if [[ -z "$SCENES" ]]; then
  mapfile -t SCENE_LIST < <(ls -1 "${SCENE_DATA_DIR}"/*.json | xargs -n1 basename | sed 's/\.json$//')
else
  IFS=',' read -r -a SCENE_LIST <<< "$SCENES"
fi

# Execution loop
for m in "${MODEL_LIST[@]}"; do
  mp=$(model_path_of "$m")
  for s in "${SCENE_LIST[@]}"; do
    echo ">>> Running Llama: Model=$m, Scene=$s"

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

echo "[SUCCESS] Llama inference tasks finished."
