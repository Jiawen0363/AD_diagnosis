#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/jiawen/AD_diagnosis"
SCRIPT="$ROOT/scripts/sbatch_opd_rationale_scheme_a.sh"

echo "Building OPD fold datasets..."
python3 "$ROOT/scripts/build_rationale_opd_folds.py"

echo "Submitting 5-fold Scheme A OPD jobs..."
for f in 0 1 2 3 4; do
  job_id=$(FOLD="$f" sbatch --parsable --job-name="ad_opd_a_f${f}" --export=ALL,FOLD="$f" "$SCRIPT")
  echo "  fold $f -> job $job_id"
done

echo "Monitor: squeue -u $USER"
echo "Logs: $ROOT/scripts/logs/sbatch_opd_rationale_a_f*"
