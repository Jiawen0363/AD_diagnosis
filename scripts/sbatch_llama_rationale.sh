#!/bin/bash
#SBATCH --job-name=ad_llama_rationale
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/data/jiawen/AD_diagnosis/scripts/logs/sbatch_llama_rationale_%j.out
#SBATCH --error=/data/jiawen/AD_diagnosis/scripts/logs/sbatch_llama_rationale_%j.err
#SBATCH --time=24:00:00

# Generate language rationales with Llama-3.1-8B-Instruct via vLLM,
# then evaluate concat / triple concat on train CV.
#
# Optional env overrides before submit:
#   BATCH_SIZE=8
#   GPU_MEMORY_UTILIZATION=0.85
#   MAX_SAMPLES=          # e.g. 10 for debug; empty = all 237
#   FORCE=1               # regenerate even if JSON exists
#   SKIP_EVAL=1           # generation only

set -euo pipefail

ROOT="/data/jiawen/AD_diagnosis"
cd "$ROOT"

mkdir -p "$ROOT/scripts/logs"

source /home/jiawen/miniconda3/etc/profile.d/conda.sh
conda activate appbench

export PYTHON="${PYTHON:-/home/jiawen/miniconda3/envs/appbench/bin/python}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_SAMPLES="${MAX_SAMPLES:-}"
export FORCE="${FORCE:-0}"
export SKIP_EVAL="${SKIP_EVAL:-0}"

echo "[$(date -Is)] job=$SLURM_JOB_ID node=$SLURM_NODELIST gpu=$CUDA_VISIBLE_DEVICES"
echo "[$(date -Is)] python=$PYTHON batch_size=$BATCH_SIZE"

GEN_ARGS=(
  --model-path /data/models/Llama-3.1-8B-Instruct
  --prompt-file "$ROOT/prompt/rationale_prompt.txt"
  --output-suffix llama
  --generator llama-3.1-8b-instruct-vllm
  --output-dir "$ROOT/data/rationale"
  --batch-size "$BATCH_SIZE"
  --gpu 0
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --disable-thinking
  --temperature 0.2
  --max-tokens 2048
)

if [[ -n "$MAX_SAMPLES" ]]; then
  GEN_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$FORCE" == "1" ]]; then
  GEN_ARGS+=(--force)
fi

echo "[$(date -Is)] [1/2] Generating Llama rationales..."
"$PYTHON" -u "$ROOT/scripts/generate_vllm_qwen3_rationales.py" "${GEN_ARGS[@]}"

if [[ "$SKIP_EVAL" != "1" ]]; then
  echo "[$(date -Is)] [2/2] Evaluating concat and triple concat (CV)..."
  "$PYTHON" -u "$ROOT/baselines/eval_qwen3_rationale_concat.py" \
    --rationale-suffix llama \
    --cache-tag llama \
    --rationale-generator llama-3.1-8b-instruct-vllm \
    --results-prefix llama
else
  echo "[$(date -Is)] SKIP_EVAL=1, skipping evaluation."
fi

echo "[$(date -Is)] done"
