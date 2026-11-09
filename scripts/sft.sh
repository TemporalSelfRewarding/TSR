#!/bin/bash
set -euo pipefail
# set your conda_path and hf_token here
CONDA_PATH="${CONDA_PATH:-}"
HF_TOKEN="${HF_TOKEN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
source "${SCRIPT_DIR}/common.sh"


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
WORLD_SIZE=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')

activate_tsr_env handbook
cd "${PROJECT_ROOT}/alignment-handbook"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache}"
# Hugging Face reads HF_TOKEN directly; an empty token must not trigger login.

launch_training sft "${PROJECT_ROOT}/settings/sft/config/config.yaml"
