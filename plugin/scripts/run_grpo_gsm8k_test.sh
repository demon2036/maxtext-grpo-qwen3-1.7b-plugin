#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAXTEXT_DIR="${REPO_ROOT}/third_party/maxtext"
CFG_REL="../../plugin/configs/rl_qwen3_1.7b_grpo_gsm8k_test.yml"

RUN_NAME="${RUN_NAME:-qwen3-1.7b-grpo-gsm8k-10steps-$(date +%Y%m%d-%H%M%S)}"
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-${HOME}/runs}"
HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "${BASE_OUTPUT_DIRECTORY}"

cd "${MAXTEXT_DIR}"

python3 -m src.MaxText.rl.train_rl "${CFG_REL}" \
  run_name="${RUN_NAME}" \
  base_output_directory="${BASE_OUTPUT_DIRECTORY}" \
  hf_access_token="${HF_TOKEN}" \
  log_config=True

echo "Done. Outputs at: ${BASE_OUTPUT_DIRECTORY}/${RUN_NAME}/"

