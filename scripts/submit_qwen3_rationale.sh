#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/sbatch_qwen3_rationale.sh"

mkdir -p "$ROOT/scripts/logs"
chmod +x "$SCRIPT"

echo "Submitting Qwen3 rationale job (appbench + vLLM)..."
echo "  script: $SCRIPT"
echo "  logs:   $ROOT/scripts/logs/"

JOB_ID="$(sbatch "$SCRIPT" | awk '{print $4}')"
echo "Submitted job $JOB_ID"
echo "Monitor: squeue -j $JOB_ID"
echo "Stdout:  $ROOT/scripts/logs/sbatch_qwen3_rationale_${JOB_ID}.out"
echo "Stderr:  $ROOT/scripts/logs/sbatch_qwen3_rationale_${JOB_ID}.err"
