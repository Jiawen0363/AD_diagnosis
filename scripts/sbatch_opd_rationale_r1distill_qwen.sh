#!/bin/bash
#SBATCH --job-name=ad_opd_r1_qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/data/jiawen/AD_diagnosis/scripts/logs/sbatch_opd_rationale_r1_qwen_f%F_%j.out
#SBATCH --error=/data/jiawen/AD_diagnosis/scripts/logs/sbatch_opd_rationale_r1_qwen_f%F_%j.err
#SBATCH --time=24:00:00

# OPD/SDFT on Cookie Theft rationale data:
#   student: Qwen3-8B + LoRA          -> GPU 0
#   teacher: frozen R1-Distill-Qwen-14B -> GPU 1-3 (sharded)
#   PI in teacher prompt only (label + mmse_severity on fold-train)
#
# Submit one fold:
#   FOLD=0 sbatch /data/jiawen/AD_diagnosis/scripts/sbatch_opd_rationale_r1distill_qwen.sh

set -euo pipefail

ROOT="/data/jiawen/AD_diagnosis"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"
FOLD="${FOLD:-0}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen3-8B}"
TEACHER_MODEL="${TEACHER_MODEL:-DeepSeek-R1-Distill-Qwen-14B}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$ROOT/checkpoints/sft-qwen3-8b-rationale-lora/checkpoint-45/chat_template.jinja}"
DATA_DIR="$ROOT/data/opd/fold${FOLD}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints/opd-qwen3-rationale-r1distill-fold${FOLD}}"
CACHE_DIR="${CACHE_DIR:-$ROOT/caches/opd_qwen3_r1distill_fold${FOLD}}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRL_EXPERIMENTAL_SILENCE="${TRL_EXPERIMENTAL_SILENCE:-1}"

mkdir -p "$ROOT/scripts/logs" "$OUTPUT_DIR" "$CACHE_DIR"

source /home/jiawen/miniconda3/etc/profile.d/conda.sh
conda activate appbench
export PYTHON="${PYTHON:-/home/jiawen/miniconda3/envs/appbench/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if ! "$PYTHON" -c "import trl.experimental.sdft" 2>/dev/null; then
  echo "Installing trl (needed for OPD/SDFT)..."
  pip install -q "trl>=0.18.0"
fi

echo "[$(date -Is)] job=$SLURM_JOB_ID fold=$FOLD gpu=$CUDA_VISIBLE_DEVICES"
echo "[$(date -Is)] student=$MODEL_ROOT/$STUDENT_MODEL"
echo "[$(date -Is)] teacher=$MODEL_ROOT/$TEACHER_MODEL"
echo "[$(date -Is)] chat_template=$CHAT_TEMPLATE"
echo "[$(date -Is)] train_data=$DATA_DIR/train.jsonl"
echo "[$(date -Is)] val_data=$DATA_DIR/val.jsonl"
echo "[$(date -Is)] output=$OUTPUT_DIR"

nvidia-smi || true

"$PYTHON" -u "$ROOT/scripts/build_rationale_opd_folds.py"

args=(
  --trainer sdft
  --model_name_or_path "$MODEL_ROOT/$STUDENT_MODEL"
  --teacher_model_name_or_path "$MODEL_ROOT/$TEACHER_MODEL"
  --data_path "$DATA_DIR/train.jsonl"
  --eval_data_path "$DATA_DIR/val.jsonl"
  --output_dir "$OUTPUT_DIR"
  --cache_dir "$CACHE_DIR"
  --chat_template "$CHAT_TEMPLATE"
  --teacher_prompt_template "{prompt}\n\n{privileged_context}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --learning_rate "${LEARNING_RATE:-5e-6}"
  --max_length "${MAX_LENGTH:-2048}"
  --max_prompt_length "${MAX_PROMPT_LENGTH:-1536}"
  --max_new_tokens "${MAX_NEW_TOKENS:-512}"
  --beta "${BETA:-0.5}"
  --temperature "${TEMPERATURE:-0.9}"
  --distillation_mode "${DISTILLATION_MODE:-topk_logits}"
  --distillation_topk "${DISTILLATION_TOPK:-100}"
  --save_steps "${SAVE_STEPS:-100}"
  --logging_steps "${LOGGING_STEPS:-10}"
)

"$PYTHON" -u "$ROOT/opd/train_opd.py" "${args[@]}"

echo "[$(date -Is)] done fold=$FOLD"
