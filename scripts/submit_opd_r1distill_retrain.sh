#!/usr/bin/env bash
# Submit 5-fold OPD retraining for Llama + R1-Distill-Llama-8B and Qwen + R1-Distill-Qwen-14B.
#
# Usage:
#   bash /data/jiawen/AD_diagnosis/scripts/submit_opd_r1distill_retrain.sh
#   FOLDS="0 1" bash ...   # subset of folds

set -euo pipefail

ROOT="/data/jiawen/AD_diagnosis"
FOLDS="${FOLDS:-0 1 2 3 4}"

for f in $FOLDS; do
  echo "Submitting fold $f: Llama student + R1-Distill-Llama-8B"
  FOLD="$f" sbatch \
    --job-name="ad_opd_r1_llama_f${f}" \
    --export=ALL,FOLD="$f" \
    "$ROOT/scripts/sbatch_opd_rationale_r1distill.sh"

  echo "Submitting fold $f: Qwen student + R1-Distill-Qwen-14B"
  FOLD="$f" sbatch \
    --job-name="ad_opd_r1_qwen_f${f}" \
    --export=ALL,FOLD="$f" \
    "$ROOT/scripts/sbatch_opd_rationale_r1distill_qwen.sh"
done

echo "Done. Monitor with: squeue -u $USER"
