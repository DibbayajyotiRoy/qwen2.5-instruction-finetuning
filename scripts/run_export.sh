#!/usr/bin/env bash
set -euo pipefail

# run_export.sh - Merge adapters and export to GGUF

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_DIR="${1:-${PROJECT_DIR}/outputs/qlora/final_checkpoint}"
CONFIG_FILE="${2:-${PROJECT_DIR}/configs/qlora_config.yaml}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -m src.export_gguf "${CHECKPOINT_DIR}" --config "${CONFIG_FILE}"
