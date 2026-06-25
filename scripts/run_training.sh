#!/usr/bin/env bash
set -euo pipefail

# run_training.sh - Launch Qwen2.5 instruction fine-tuning

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${1:-${PROJECT_DIR}/configs/qlora_config.yaml}"

cd "${PROJECT_DIR}"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python -m src.train --config "${CONFIG_FILE}"
