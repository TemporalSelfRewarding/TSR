#!/usr/bin/env bash
# Shared launch settings for the verified full-parameter reproduction.

activate_tsr_env() {
    local name=$1
    # An active virtualenv also works; Conda is optional.
    if [[ -n "${CONDA_PATH:-}" ]]; then
        source "${CONDA_PATH}/bin/activate" "${CONDA_PATH}/envs/${name}"
    fi
    export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/cache}"
    if [[ -n "${HF_TOKEN:-}" ]]; then
        export HF_TOKEN
    fi
}

launch_training() {
    local mode=$1 config=$2
    local micro accumulation
    case "$mode" in
        sft) micro=2 ;;
        dpo) micro=1 ;;
        *) echo "Unknown training mode: $mode" >&2; return 1 ;;
    esac
    if (( WORLD_SIZE < 1 || 32 % (WORLD_SIZE * micro) != 0 )); then
        echo "GPU count $WORLD_SIZE cannot preserve global batch 32 with per-device batch $micro" >&2
        return 1
    fi
    accumulation=$((32 / (WORLD_SIZE * micro)))
    local extra=()
    if [[ "$mode" == sft ]]; then
        extra+=(--per_device_eval_batch_size=2 --save_strategy=no)
        if [[ -n "${SFT_BASE_MODEL:-}" ]]; then
            extra+=("--model_name_or_path=${SFT_BASE_MODEL}")
        fi
    fi
    # ZeRO-3 retains fp32 optimizer/master weights while computing in bf16.
    # Pure-bf16 DDP at the SFT learning rate lost most updates in the prior audit.
    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file "${PROJECT_ROOT}/settings/tsr/config/deepspeed_zero3.yaml" \
        --num_processes "${WORLD_SIZE}" \
        --main_process_port "${MAIN_PROCESS_PORT:-29500}" \
        "${PROJECT_ROOT}/alignment-handbook/scripts/run_${mode}.py" "$config" \
        "--per_device_train_batch_size=${micro}" \
        "--gradient_accumulation_steps=${accumulation}" \
        "${extra[@]}"
}
