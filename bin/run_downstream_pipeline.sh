#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="/home/sfriedman/.conda/envs/vllm-qwen35/bin/python3"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting pipeline: critic_flag pass 1 (Qwen as critic, concurrency 25)..."
"${PYTHON}" "${REPO_ROOT}/bin/critic_flag.py" --model qwen --concurrency 25

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting pipeline: critic_flag pass 2 (gpt-oss as critic, concurrency 25)..."
"${PYTHON}" "${REPO_ROOT}/bin/critic_flag.py" --model gptoss --concurrency 25

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting pipeline: build_manifest pass..."
"${PYTHON}" "${REPO_ROOT}/bin/build_manifest.py"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline complete!"
