#!/bin/bash
#SBATCH --job-name=ad_opd_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/jiawen/AD_diagnosis/scripts/logs/sbatch_opd_pi_eval_%j.out
#SBATCH --error=/data/jiawen/AD_diagnosis/scripts/logs/sbatch_opd_pi_eval_%j.err
#SBATCH --time=24:00:00

# DEPRECATED: previously pointed at Coding Tutor opd-pi checkpoints under opd/checkpoint/.
# Use Cookie Theft OPD checkpoints under checkpoints/opd-*-rationale-r1distill-fold*
# (see checkpoints/AD_OPD_MANIFEST.json) and write a new eval script for those paths.
#
# Generate rationales from OPD LoRA checkpoints, then run concat/triple-concat
# classification + regression CV and test evaluation.
#
# Optional env:
#   MODEL=llama|qwen|both   (default: both)
#   FORCE=1                 regenerate rationales
#   SKIP_EVAL=1             generation only
#   MAX_SAMPLES=10          debug subset

set -euo pipefail

echo "ERROR: sbatch_opd_pi_eval.sh is deprecated (removed Coding Tutor opd-pi checkpoints)." >&2
echo "See checkpoints/AD_OPD_MANIFEST.json for valid Cookie Theft OPD checkpoints." >&2
exit 1

ROOT="/data/jiawen/AD_diagnosis"
MODEL_ROOT="${MODEL_ROOT:-/data/models}"
MODEL="${MODEL:-both}"
FORCE="${FORCE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

mkdir -p "$ROOT/scripts/logs"

source /home/jiawen/miniconda3/etc/profile.d/conda.sh
conda activate appbench
export PYTHON="${PYTHON:-/home/jiawen/miniconda3/envs/appbench/bin/python}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[$(date -Is)] job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES model=$MODEL"

run_llama() {
  local gen_args=(
    --adapter-path "$ROOT/opd/checkpoint/Llama-3.1-8B-Instruct-opd-pi/checkpoint-135"
    --base-model-path "$MODEL_ROOT/Llama-3.1-8B-Instruct"
    --prompt-file "$ROOT/prompt/rationale_prompt.txt"
    --output-suffix opd_llama31
    --output-dir "$ROOT/data/rationale"
    --template-name vicuna_v1.1
    --generator llama-3.1-8b-opd-pi-lora
    --batch-size 4
    --temperature 0.2
    --max-new-tokens 512
  )
  if [[ -n "$MAX_SAMPLES" ]]; then gen_args+=(--max-samples "$MAX_SAMPLES"); fi
  if [[ "$FORCE" == "1" ]]; then gen_args+=(--force); fi

  echo "[$(date -Is)] [llama] generating rationales..."
  "$PYTHON" -u "$ROOT/scripts/generate_lora_rationales.py" "${gen_args[@]}"

  if [[ "$SKIP_EVAL" != "1" ]]; then
    echo "[$(date -Is)] [llama] evaluating concat / triple concat..."
    "$PYTHON" -u "$ROOT/baselines/eval_qwen3_rationale_concat.py" \
      --rationale-suffix opd_llama31 \
      --cache-tag opd_llama31 \
      --rationale-generator llama-3.1-8b-opd-pi-lora \
      --results-prefix opd_llama31
  fi
}

run_qwen() {
  local gen_args=(
    --adapter-path "$ROOT/opd/checkpoint/Qwen3-8B-opd-pi/checkpoint-135"
    --base-model-path "$MODEL_ROOT/Qwen3-8B"
    --prompt-file "$ROOT/prompt/rationale_prompt.txt"
    --output-suffix opd_qwen3
    --output-dir "$ROOT/data/rationale"
    --use-chat-template
    --generator qwen3-8b-opd-pi-lora
    --batch-size 2
    --temperature 0.2
    --max-new-tokens 512
  )
  if [[ -n "$MAX_SAMPLES" ]]; then gen_args+=(--max-samples "$MAX_SAMPLES"); fi
  if [[ "$FORCE" == "1" ]]; then gen_args+=(--force); fi

  echo "[$(date -Is)] [qwen] generating rationales..."
  "$PYTHON" -u "$ROOT/scripts/generate_lora_rationales.py" "${gen_args[@]}"

  if [[ "$SKIP_EVAL" != "1" ]]; then
    echo "[$(date -Is)] [qwen] evaluating concat / triple concat..."
    "$PYTHON" -u "$ROOT/baselines/eval_qwen3_rationale_concat.py" \
      --rationale-suffix opd_qwen3 \
      --cache-tag opd_qwen3 \
      --rationale-generator qwen3-8b-opd-pi-lora \
      --results-prefix opd_qwen3
  fi
}

case "$MODEL" in
  llama) run_llama ;;
  qwen) run_qwen ;;
  both) run_llama; run_qwen ;;
  *) echo "Unknown MODEL=$MODEL (use llama|qwen|both)"; exit 1 ;;
esac

echo "[$(date -Is)] done"
