#!/usr/bin/env bash
set -euo pipefail

# Control-machine script:
# - Creates a NEW TPU VM (v6e-8, europe-west4-a)
# - SSHes in and starts remote bootstrap+training via nohup (so SSH returns quickly)

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-europe-west4-a}"
TPU_TYPE="${TPU_TYPE:-v6e-8}"
TPU_RUNTIME_VERSION="${TPU_RUNTIME_VERSION:-}"
TPU_VM_CREATE_FLAGS="${TPU_VM_CREATE_FLAGS:-}"
ALLOW_SPOT_FALLBACK="${ALLOW_SPOT_FALLBACK:-1}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is empty. Set PROJECT_ID env var or run: gcloud config set project <id>" >&2
  exit 2
fi

REPO_SLUG="${REPO_SLUG:-demon2036/maxtext-grpo-qwen3-1.7b-plugin}"
REPO_URL="https://github.com/${REPO_SLUG}.git"
REPO_DIR="${REPO_DIR:-maxtext-grpo-qwen3-1.7b-plugin}"

TPU_NAME="${TPU_NAME:-qwen3-1-7b-grpo-$(date +%Y%m%d-%H%M%S)}"

RUN_NAME="${RUN_NAME:-qwen3-1.7b-grpo-gsm8k-10steps-$(date +%Y%m%d-%H%M%S)}"
BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-${HOME}/runs}"

REMOTE_LOG_FILE="${REMOTE_LOG_FILE:-${HOME}/qwen3_grpo_${RUN_NAME}.log}"
REMOTE_PID_FILE="${REMOTE_PID_FILE:-${HOME}/qwen3_grpo_${RUN_NAME}.pid}"
REMOTE_DONE_FILE="${REMOTE_DONE_FILE:-${HOME}/qwen3_grpo_${RUN_NAME}.done}"

DEFAULT_RUNTIME_CANDIDATES=(tpu-ubuntu2204-base tpu-vm-base tpu-ubuntu2004-base)
if [[ -n "${TPU_RUNTIME_VERSION}" ]]; then
  if ! gcloud compute tpus tpu-vm versions describe "${TPU_RUNTIME_VERSION}" --zone "${ZONE}" >/dev/null 2>&1; then
    echo "[warn] TPU_RUNTIME_VERSION='${TPU_RUNTIME_VERSION}' not found in zone ${ZONE}; auto-selecting instead"
    TPU_RUNTIME_VERSION=""
  fi
fi
if [[ -z "${TPU_RUNTIME_VERSION}" ]]; then
  for candidate in "${DEFAULT_RUNTIME_CANDIDATES[@]}"; do
    if gcloud compute tpus tpu-vm versions describe "${candidate}" --zone "${ZONE}" >/dev/null 2>&1; then
      TPU_RUNTIME_VERSION="${candidate}"
      break
    fi
  done
fi
if [[ -z "${TPU_RUNTIME_VERSION}" ]]; then
  echo "[error] No TPU runtime version found in zone ${ZONE}. Try: gcloud compute tpus tpu-vm versions list --zone ${ZONE}" >&2
  exit 3
fi

echo "[gcloud] project=${PROJECT_ID} zone=${ZONE} tpu=${TPU_NAME} type=${TPU_TYPE} runtime_version=${TPU_RUNTIME_VERSION}"

gcloud services enable compute.googleapis.com tpu.googleapis.com --project "${PROJECT_ID}" --quiet

echo "[gcloud] creating TPU VM..."
if ! gcloud compute tpus tpu-vm create "${TPU_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --accelerator-type "${TPU_TYPE}" \
  --version "${TPU_RUNTIME_VERSION}" \
  ${TPU_VM_CREATE_FLAGS} \
  --quiet; then
  if [[ -z "${TPU_VM_CREATE_FLAGS}" && "${ALLOW_SPOT_FALLBACK}" == "1" ]]; then
    echo "[warn] create failed; retrying with --spot (set ALLOW_SPOT_FALLBACK=0 to disable)"
    gcloud compute tpus tpu-vm create "${TPU_NAME}" \
      --project "${PROJECT_ID}" \
      --zone "${ZONE}" \
      --accelerator-type "${TPU_TYPE}" \
      --version "${TPU_RUNTIME_VERSION}" \
      --spot \
      --quiet
  else
    echo "[error] TPU create failed (TPU_VM_CREATE_FLAGS='${TPU_VM_CREATE_FLAGS}')" >&2
    exit 1
  fi
fi

echo "[gcloud] waiting for TPU to be ready..."
gcloud compute tpus tpu-vm describe "${TPU_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}" --format='value(state)'

echo "[gcloud] starting remote bootstrap + training (nohup)..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "set -euo pipefail; \
    export REPO_URL='${REPO_URL}'; \
    export REPO_DIR='${REPO_DIR}'; \
    export RUN_NAME='${RUN_NAME}'; \
    export BASE_OUTPUT_DIRECTORY='${BASE_OUTPUT_DIRECTORY}'; \
    export LOG_FILE='${REMOTE_LOG_FILE}'; \
    export PID_FILE='${REMOTE_PID_FILE}'; \
    export DONE_FILE='${REMOTE_DONE_FILE}'; \
    nohup bash -lc 'set -euo pipefail; \
      sudo apt-get update -y; \
      sudo apt-get install -y git; \
      rm -rf \"\$REPO_DIR\"; \
      git clone --depth=1 --branch \"main\" --recurse-submodules \"\$REPO_URL\" \"\$REPO_DIR\"; \
      cd \"\$REPO_DIR\"; \
      RUN_NAME=\"\$RUN_NAME\" BASE_OUTPUT_DIRECTORY=\"\$BASE_OUTPUT_DIRECTORY\" bash plugin/tpu/remote_bootstrap_and_run.sh; \
      touch \"\$DONE_FILE\"' \
      > \"\$LOG_FILE\" 2>&1 & \
    echo \$! > \"\$PID_FILE\"; \
    echo \"[remote] started pid_file=\$PID_FILE log=\$LOG_FILE done=\$DONE_FILE\""

echo "[done] TPU ${TPU_NAME} started run_name=${RUN_NAME}"
echo "[next] tail logs: gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT_ID} --zone ${ZONE} --command \"tail -n 100 ${REMOTE_LOG_FILE}\""
echo "[next] check done: gcloud compute tpus tpu-vm ssh ${TPU_NAME} --project ${PROJECT_ID} --zone ${ZONE} --command \"test -f ${REMOTE_DONE_FILE} && echo DONE || echo RUNNING\""
