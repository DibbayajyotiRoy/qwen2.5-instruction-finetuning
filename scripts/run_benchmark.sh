#!/usr/bin/env bash
set -euo pipefail

# run_benchmark.sh - Benchmark HF vs llama.cpp inference

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

HF_MODEL="${1:-${PROJECT_DIR}/outputs/qlora/final_merged}"
GGUF_MODEL="${2:-${PROJECT_DIR}/outputs/gguf/final_merged-q4_k_m.gguf}"
CONFIG_FILE="${3:-${PROJECT_DIR}/configs/inference_config.yaml}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -m src.benchmark \
    "${HF_MODEL}" \
    "${GGUF_MODEL}" \
    --config "${CONFIG_FILE}"
