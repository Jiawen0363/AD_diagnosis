#!/bin/bash
#SBATCH --job-name=ad_smoke_q32b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=/data/jiawen/AD_diagnosis/scripts/logs/sbatch_smoke_qwen32b_%j.out
#SBATCH --error=/data/jiawen/AD_diagnosis/scripts/logs/sbatch_smoke_qwen32b_%j.err
#SBATCH --time=02:00:00

# Smoke test: can DeepSeek-R1-Distill-Qwen-32B load and generate?
#
# Submit:
#   sbatch /data/jiawen/AD_diagnosis/scripts/sbatch_smoke_qwen32b.sh
#
# Optional env overrides:
#   LOAD_MODE=bf16_auto|bf16_single|4bit_single
#   MAX_NEW_TOKENS=32

set -euo pipefail

ROOT="/data/jiawen/AD_diagnosis"
MODEL_PATH="${MODEL_PATH:-/data/models/DeepSeek-R1-Distill-Qwen-32B}"
LOAD_MODE="${LOAD_MODE:-bf16_auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

mkdir -p "$ROOT/scripts/logs"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source /home/jiawen/miniconda3/etc/profile.d/conda.sh
conda activate appbench

export PYTHON="${PYTHON:-/home/jiawen/miniconda3/envs/appbench/bin/python}"

echo "[$(date -Is)] job=$SLURM_JOB_ID node=$SLURM_NODELIST gpu=$CUDA_VISIBLE_DEVICES"
echo "[$(date -Is)] model=$MODEL_PATH load_mode=$LOAD_MODE"

nvidia-smi || true

"$PYTHON" -u "$ROOT/scripts/smoke_test_qwen32b.py" \
  --model-path "$MODEL_PATH" \
  --load-mode "$LOAD_MODE" \
  --max-new-tokens "$MAX_NEW_TOKENS"

echo "[$(date -Is)] done"
