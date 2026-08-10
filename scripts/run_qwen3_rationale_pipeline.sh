#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jiawen/miniconda3/envs/appbench/bin/python}"
GPU="${GPU:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

cd "$ROOT"

GEN_ARGS=(
  --model-path /data/models/Qwen3-8B
  --prompt-file "$ROOT/prompt/rationale_prompt.txt"
  --output-suffix qwen3
  --output-dir "$ROOT/data/rationale"
  --batch-size "$BATCH_SIZE"
  --gpu "$GPU"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  GEN_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

echo "[1/2] Generating Qwen3 rationales via vLLM..."
"$PYTHON" "$ROOT/scripts/generate_vllm_qwen3_rationales.py" "${GEN_ARGS[@]}"

echo "[2/2] Evaluating concat and triple concat..."
"$PYTHON" "$ROOT/baselines/eval_qwen3_rationale_concat.py" \
  --rationale-suffix qwen3 \
  --cache-tag qwen3
