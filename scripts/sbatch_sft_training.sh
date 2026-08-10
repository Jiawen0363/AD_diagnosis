#!/bin/bash
#SBATCH --job-name=ad_sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/data/jiawen/AD_diagnosis/scripts/logs/sbatch_sft_%x_%j.out
#SBATCH --error=/data/jiawen/AD_diagnosis/scripts/logs/sbatch_sft_%x_%j.err
#SBATCH --time=24:00:00

# LoRA SFT: transcript + instruction -> DeepSeek rationale
#
# Submit:
#   MODEL_KEY=qwen3   sbatch --job-name=ad_sft_qwen3   --export=ALL,MODEL_KEY scripts/sbatch_sft_training.sh
#   MODEL_KEY=llama31 sbatch --job-name=ad_sft_llama31 --export=ALL,MODEL_KEY scripts/sbatch_sft_training.sh
#
# Local debug:
#   MODEL_KEY=qwen3 bash scripts/sbatch_sft_training.sh

set -euo pipefail

MODEL_KEY="${MODEL_KEY:-}"
if [[ -z "$MODEL_KEY" ]]; then
  echo "MODEL_KEY is required: qwen3 or llama31" >&2
  exit 1
fi

ROOT="/data/jiawen/AD_diagnosis"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"
DATA_PATH="${DATA_PATH:-$ROOT/data/sft/rationale_sft_train.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints}"
CACHE_PATH="${CACHE_PATH:-$ROOT/caches/sft_${MODEL_KEY}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source /home/jiawen/miniconda3/etc/profile.d/conda.sh
conda activate appbench
PYTHON="${PYTHON:-/home/jiawen/miniconda3/envs/appbench/bin/python}"

case "$MODEL_KEY" in
  qwen3)
    STUDENT_MODEL="Qwen3-8B"
    OUTPUT_DIR="$CHECKPOINT_ROOT/sft-qwen3-8b-rationale-lora"
    ;;
  llama31)
    STUDENT_MODEL="Llama-3.1-8B-Instruct"
    OUTPUT_DIR="$CHECKPOINT_ROOT/sft-llama31-8b-rationale-lora"
    ;;
  *)
    echo "Unknown MODEL_KEY: $MODEL_KEY (expected qwen3 or llama31)" >&2
    exit 1
    ;;
esac

mkdir -p "$CACHE_PATH" "$OUTPUT_DIR" "$ROOT/scripts/logs"
cd "$ROOT"

echo "[$(date -Is)] SFT start job=${SLURM_JOB_ID:-local} model=$STUDENT_MODEL key=$MODEL_KEY gpu=$CUDA_VISIBLE_DEVICES"
echo "  model_path: $MODEL_ROOT/$STUDENT_MODEL"
echo "  data_path:  $DATA_PATH"
echo "  output_dir: $OUTPUT_DIR"
echo "  template:   vicuna_v1.1"

"$PYTHON" fastchat/finetune.py \
  --model_name_or_path "$MODEL_ROOT/$STUDENT_MODEL" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --cache_path "$CACHE_PATH" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --lr_scheduler_type linear \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-10}" \
  --fp16 True \
  --gradient_checkpointing True \
  --cutoff_len "${CUTOFF_LEN:-2048}" \
  --template_name vicuna_v1.1 \
  --lora_r "${LORA_R:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --lora_target_modules q_proj v_proj

echo "[$(date -Is)] SFT done checkpoint=$OUTPUT_DIR"
