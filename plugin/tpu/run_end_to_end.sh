#!/usr/bin/env bash
set -euo pipefail

# Control-machine script:
# - Creates a NEW TPU VM (v6e-8, europe-west4-a)
# - SSHes in and runs remote bootstrap (clone repo + install + run)

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-europe-west4-a}"
TPU_TYPE="${TPU_TYPE:-v6e-8}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is empty. Set PROJECT_ID env var or run: gcloud config set project <id>" >&2
  exit 2
fi

REPO_SLUG="${REPO_SLUG:-demon2036/maxtext-grpo-qwen3-1.7b-plugin}"
REPO_URL="https://github.com/${REPO_SLUG}.git"
REPO_DIR="${REPO_DIR:-maxtext-grpo-qwen3-1.7b-plugin}"

TPU_NAME="${TPU_NAME:-qwen3-1-7b-grpo-$(date +%Y%m%d-%H%M%S)}"

RUN_NAME="${RUN_NAME:-qwen3-1.7b-grpo-gsm8k-10steps-$(date +%Y%m%d-%H%M%S)}"

echo "[gcloud] project=${PROJECT_ID} zone=${ZONE} tpu=${TPU_NAME} type=${TPU_TYPE}"

gcloud services enable compute.googleapis.com tpu.googleapis.com --project "${PROJECT_ID}" --quiet

echo "[gcloud] creating TPU VM..."
gcloud compute tpus tpu-vm create "${TPU_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --accelerator-type "${TPU_TYPE}" \
  --version "tpu-ubuntu2204-base" \
  --quiet

echo "[gcloud] waiting for TPU to be ready..."
gcloud compute tpus tpu-vm describe "${TPU_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}" --format='value(state)'

echo "[gcloud] running remote bootstrap + training..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "set -euo pipefail; sudo apt-get update -y; sudo apt-get install -y git; \
    rm -rf '${REPO_DIR}'; \
    git clone --depth=1 --branch 'main' --recurse-submodules '${REPO_URL}' '${REPO_DIR}'; \
    cd '${REPO_DIR}'; \
    RUN_NAME='${RUN_NAME}' bash plugin/tpu/remote_bootstrap_and_run.sh"

echo "[done] TPU ${TPU_NAME} finished run_name=${RUN_NAME}"
