#!/usr/bin/env bash
set -euo pipefail

# run_inference.sh - Run inference with a fine-tuned model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${1:-${PROJECT_DIR}/outputs/qlora/final_merged}"
CONFIG_FILE="${2:-${PROJECT_DIR}/configs/inference_config.yaml}"
PROMPT="${3:-Explain the importance of reproducible ML experiments.}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -m src.inference \
    --model_path "${MODEL_PATH}" \
    --config "${CONFIG_FILE}" \
    --prompt "${PROMPT}"
