#!/usr/bin/env bash
# Submit only missing Cookie Theft OPD jobs (skip folds with checkpoint-48).
#
# Correct setup:
#   Llama-3.1-8B-Instruct  + DeepSeek-R1-Distill-Llama-8B  teacher
#   Qwen3-8B               + DeepSeek-R1-Distill-Qwen-14B teacher
#
# Usage:
#   bash /data/jiawen/AD_diagnosis/scripts/submit_opd_r1distill_pending.sh
#   BACKBONE=llama|qwen|both   (default: both)
#   FOLDS="3 4"                (default: 0 1 2 3 4)

set -euo pipefail

ROOT="/data/jiawen/AD_diagnosis"
BACKBONE="${BACKBONE:-both}"
FOLDS="${FOLDS:-0 1 2 3 4}"

has_checkpoint() {
  local kind="$1"
  local fold="$2"
  local dir
  if [[ "$kind" == "llama" ]]; then
    dir="$ROOT/checkpoints/opd-llama31-rationale-r1distill-fold${fold}/checkpoint-48"
  else
    dir="$ROOT/checkpoints/opd-qwen3-rationale-r1distill-fold${fold}/checkpoint-48"
  fi
  [[ -d "$dir" && -f "$dir/adapter_model.safetensors" ]]
}

submit_llama() {
  local fold="$1"
  if has_checkpoint llama "$fold"; then
    echo "Skip Llama fold $fold (checkpoint-48 exists)"
    return
  fi
  echo "Submitting Llama fold $fold"
  FOLD="$fold" sbatch \
    --job-name="ad_opd_r1_llama_f${fold}" \
    --export=ALL,FOLD="$fold" \
    "$ROOT/scripts/sbatch_opd_rationale_r1distill.sh"
}

submit_qwen() {
  local fold="$1"
  if has_checkpoint qwen "$fold"; then
    echo "Skip Qwen fold $fold (checkpoint-48 exists)"
    return
  fi
  echo "Submitting Qwen fold $fold"
  FOLD="$fold" sbatch \
    --job-name="ad_opd_r1_qwen_f${fold}" \
    --export=ALL,FOLD="$fold" \
    "$ROOT/scripts/sbatch_opd_rationale_r1distill_qwen.sh"
}

for f in $FOLDS; do
  if [[ "$BACKBONE" == "both" || "$BACKBONE" == "llama" ]]; then
    submit_llama "$f"
  fi
  if [[ "$BACKBONE" == "both" || "$BACKBONE" == "qwen" ]]; then
    submit_qwen "$f"
  fi
done

echo "Done. Monitor with: squeue -u $USER"
