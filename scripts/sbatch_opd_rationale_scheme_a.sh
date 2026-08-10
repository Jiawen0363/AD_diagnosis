#!/bin/bash
#SBATCH --job-name=ad_opd_a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/jiawen/AD_diagnosis/scripts/logs/sbatch_opd_rationale_a_f%F_%j.out
#SBATCH --error=/data/jiawen/AD_diagnosis/scripts/logs/sbatch_opd_rationale_a_f%F_%j.err
#SBATCH --time=24:00:00

# Scheme A OPD for one CV fold:
#   student: Llama-3.1-8B-Instruct + LoRA
#   teacher: frozen same-backbone base model via SDFT (supports privileged_context)
#   train PI: label + mmse_severity only on fold-train split
#
# Note: TRL DistillationTrainer does not support teacher_prompt_template / PI.
# SDFT uses the student backbone as frozen teacher while student trains with LoRA.
#
# Submit all 5 folds:
#   for f in 0 1 2 3 4; do FOLD=$f sbatch --job-name=ad_opd_a_f${f} \
#     --export=ALL,FOLD=$f /data/jiawen/AD_diagnosis/scripts/sbatch_opd_rationale_scheme_a.sh
#   done

set -euo pipefail

ROOT="/data/jiawen/AD_diagnosis"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"
FOLD="${FOLD:-0}"

STUDENT_MODEL="${STUDENT_MODEL:-Llama-3.1-8B-Instruct}"
TRAINER="${TRAINER:-sdft}"
TEACHER_MODEL_KIND="${TEACHER_MODEL_KIND:-base}"
DATA_DIR="$ROOT/data/opd/fold${FOLD}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints/opd-llama31-rationale-schemeA-fold${FOLD}}"
CACHE_DIR="${CACHE_DIR:-$ROOT/caches/opd_llama31_fold${FOLD}}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$ROOT/scripts/logs" "$OUTPUT_DIR" "$CACHE_DIR"

source /home/jiawen/miniconda3/etc/profile.d/conda.sh
conda activate appbench
export PYTHON="${PYTHON:-/home/jiawen/miniconda3/envs/appbench/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if ! "$PYTHON" -c "import trl.experimental.sdft" 2>/dev/null; then
  echo "Installing trl (needed for OPD/SDFT/distillation)..."
  pip install -q "trl>=0.18.0"
fi

echo "[$(date -Is)] job=$SLURM_JOB_ID fold=$FOLD gpu=$CUDA_VISIBLE_DEVICES"
echo "[$(date -Is)] student=$MODEL_ROOT/$STUDENT_MODEL"
echo "[$(date -Is)] trainer=$TRAINER teacher_kind=$TEACHER_MODEL_KIND"
echo "[$(date -Is)] train_data=$DATA_DIR/train.jsonl"
echo "[$(date -Is)] val_data=$DATA_DIR/val.jsonl"
echo "[$(date -Is)] output=$OUTPUT_DIR"

nvidia-smi || true

"$PYTHON" -u "$ROOT/scripts/build_rationale_opd_folds.py"

"$PYTHON" -u "$ROOT/opd/train_opd.py" \
  --trainer "$TRAINER" \
  --model_name_or_path "$MODEL_ROOT/$STUDENT_MODEL" \
  --data_path "$DATA_DIR/train.jsonl" \
  --eval_data_path "$DATA_DIR/val.jsonl" \
  --output_dir "$OUTPUT_DIR" \
  --cache_dir "$CACHE_DIR" \
  --teacher_model_kind "$TEACHER_MODEL_KIND" \
  --teacher_prompt_template "{prompt}\n\n{privileged_context}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-5e-6}" \
  --max_length "${MAX_LENGTH:-2048}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH:-1536}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
  --lmbda "${LMBDA:-1.0}" \
  --beta "${BETA:-0.5}" \
  --temperature "${TEMPERATURE:-0.9}" \
  --distillation_mode "${DISTILLATION_MODE:-topk_logits}" \
  --distillation_topk "${DISTILLATION_TOPK:-100}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --logging_steps "${LOGGING_STEPS:-10}"

echo "[$(date -Is)] done fold=$FOLD"
