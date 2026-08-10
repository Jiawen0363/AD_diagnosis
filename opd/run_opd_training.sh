#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangjian/Coding-Tutor-Extension}"
MODEL_ROOT="${MODEL_ROOT:-/data_old/models}"

student_model="${STUDENT_MODEL:-Llama-3.1-8B-Instruct}"

DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/validate_reward/output/dialogue/vanilla/first_iter/deepseek-v4-flash/all_levels_combined_train_imitation_deepseek-v4-flash.train80_namespaces.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/opd/checkpoint/${student_model}-opd-pi}"
CACHE_DIR="${CACHE_DIR:-$PROJECT_ROOT/caches}"

# Default path: same-model on-policy self-distillation (SDFT). The student sees
# the tutor prompt without reference knowledge; the frozen base teacher sees the
# same dialogue plus privileged reference knowledge.
TRAINER="${TRAINER:-sdft}"
TEACHER_MODEL_KIND="${TEACHER_MODEL_KIND:-base}"
QWEN_CHAT_TEMPLATE="${QWEN_CHAT_TEMPLATE:-$PROJECT_ROOT/scripts/run/qwen3_nonthinking.jinja}"

if [[ -z "${CHAT_TEMPLATE:-}" ]] && [[ "$student_model" == Qwen* ]]; then
    CHAT_TEMPLATE="$QWEN_CHAT_TEMPLATE"
fi

echo "Starting OPD/SDFT training..."
echo "Student: $MODEL_ROOT/$student_model"
echo "Teacher: same backbone ($TEACHER_MODEL_KIND)"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
    echo "Chat template: $CHAT_TEMPLATE"
fi

cd "$PROJECT_ROOT"

args=(
    --trainer "$TRAINER" \
    --model_name_or_path "$MODEL_ROOT/$student_model" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --teacher_model_kind "$TEACHER_MODEL_KIND" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --learning_rate "${LEARNING_RATE:-5e-6}" \
    --weight_decay "${WEIGHT_DECAY:-0.0}" \
    --warmup_ratio "${WARMUP_RATIO:-0.03}" \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --save_steps "${SAVE_STEPS:-100}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-4}" \
    --max_length "${MAX_LENGTH:-2048}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH:-1536}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
    --lmbda "${LMBDA:-1.0}" \
    --beta "${BETA:-0.5}" \
    --temperature "${TEMPERATURE:-0.9}" \
    --num_generations "${NUM_GENERATIONS:-1}" \
    --distillation_mode "${DISTILLATION_MODE:-topk_logits}" \
    --distillation_topk "${DISTILLATION_TOPK:-100}" \
    --lora_r "${LORA_R:-8}" \
    --lora_alpha "${LORA_ALPHA:-16}" \
    --lora_dropout "${LORA_DROPOUT:-0.05}" \
    --lora_target_modules ${LORA_TARGET_MODULES:-q_proj v_proj}
)

if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
    args+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi

if [[ -n "${MAX_STEPS:-}" ]]; then
    args+=(--max_steps "$MAX_STEPS")
fi

if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
    args+=(--chat_template "$CHAT_TEMPLATE")
fi

python "$PROJECT_ROOT/opd/train_opd.py" "${args[@]}"
