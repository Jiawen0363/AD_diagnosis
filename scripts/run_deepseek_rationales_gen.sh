#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-resume}"       # resume | fresh
CONCURRENCY="${2:-16}"    # optional second arg, default 16
PROMPT_FILE="${3:-prompt/cognitive-linguistic_deficits_structured.txt}"
OUTPUT_SUFFIX="${4:-}"    # e.g. compressed
OUTPUT_DIR="${5:-data}"

API_BASE="${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}"
MODEL_NAME="${DEEPSEEK_MODEL:-deepseek-v4-pro}"

if [[ "$MODE" == "fresh" ]]; then
  if [[ -n "$OUTPUT_SUFFIX" ]]; then
    SUFFIX="${OUTPUT_SUFFIX}"
    if [[ "${SUFFIX:0:1}" != "_" ]]; then
      SUFFIX="_${SUFFIX}"
    fi
  else
    SUFFIX=""
  fi
  echo "Removing previous rationale outputs (fresh run, suffix='${SUFFIX}')..."
  rm -f \
    "${OUTPUT_DIR}/ad_s2t_wav2vec_rationale${SUFFIX}.json" \
    "${OUTPUT_DIR}/control_s2t_wav2vec_rationale${SUFFIX}.json" \
    "${OUTPUT_DIR}/test_s2t_wav2vec_rationale${SUFFIX}.json"
elif [[ "$MODE" != "resume" ]]; then
  echo "Usage: $0 [resume|fresh] [concurrency] [prompt_file] [output_suffix] [output_dir]"
  exit 1
fi

echo "Starting ${MODE} generation with model=${MODEL_NAME}, concurrency=${CONCURRENCY}"
echo "Prompt file: ${PROMPT_FILE}"
echo "Output suffix: ${OUTPUT_SUFFIX:-<default>}"
echo "Output dir: ${OUTPUT_DIR}"
python -u "scripts/generate_deepseek_rationales.py" \
  --api-base "$API_BASE" \
  --model "$MODEL_NAME" \
  --concurrency "$CONCURRENCY" \
  --save-every 1 \
  --prompt-file "$PROMPT_FILE" \
  --output-suffix "$OUTPUT_SUFFIX" \
  --output-dir "$OUTPUT_DIR"
