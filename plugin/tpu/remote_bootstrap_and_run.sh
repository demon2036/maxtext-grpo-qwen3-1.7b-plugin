#!/usr/bin/env bash
set -euo pipefail

# Runs on the TPU VM.

REPO_DIR="${REPO_DIR:-maxtext-grpo-plugin}"
BRANCH="${BRANCH:-main}"

VENV_DIR="${VENV_DIR:-${HOME}/venv-maxtext}"

RUN_NAME="${RUN_NAME:-qwen3-1.7b-grpo-gsm8k-10steps-$(date +%Y%m%d-%H%M%S)}"
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-${HOME}/runs}"

if [[ -d third_party/maxtext && -f plugin/scripts/run_grpo_gsm8k_test.sh ]]; then
  REPO_ROOT="$(pwd)"
  echo "[bootstrap] using existing checkout: ${REPO_ROOT}"
else
  REPO_URL="${REPO_URL:?REPO_URL is required (https://github.com/<owner>/<repo>.git)}"
  echo "[bootstrap] cloning ${REPO_URL} -> ${REPO_DIR}"
  rm -rf "${REPO_DIR}"
  git clone --depth=1 --branch "${BRANCH}" --recurse-submodules "${REPO_URL}" "${REPO_DIR}"
  REPO_ROOT="$(cd "${REPO_DIR}" && pwd)"
fi

cd "${REPO_ROOT}/third_party/maxtext"

echo "[bootstrap] python: $(python3 --version)"

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade uv
export PATH="${HOME}/.local/bin:${PATH}"

echo "[bootstrap] creating venv: ${VENV_DIR}"
uv venv --python 3.12 --seed "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[bootstrap] installing MaxText (tpu) editable + github deps"
uv pip install -e .[tpu] --resolution=lowest
install_maxtext_github_deps

echo "[bootstrap] installing post-training deps (tunix + vllm-tpu, etc.)"
bash tools/setup/setup_post_training_requirements.sh

echo "[run] starting GRPO smoke test"
cd "${REPO_ROOT}"
RUN_NAME="${RUN_NAME}" BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY}" bash plugin/scripts/run_grpo_gsm8k_test.sh
