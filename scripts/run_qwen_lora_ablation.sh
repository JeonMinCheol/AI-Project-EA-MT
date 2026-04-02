#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
torchrun_bin="${TORCHRUN_BIN:-torchrun}"
model_name="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
target_locales_raw="${TARGET_LOCALE:-fr}"
source_locale="${SOURCE_LOCALE:-en}"
train_split="${TRAIN_SPLIT:-train}"
eval_split="${EVAL_SPLIT:-validation}"
output_root_override="${OUTPUT_ROOT:-}"
log_root_override="${LOG_ROOT:-}"
ablation_modes_raw="${ABLATION_MODES:-plain anchored surface retrieve rerank}"
reranker_model_path="${RERANKER_MODEL_PATH:-}"
allow_rerank_fallback="${ALLOW_RERANK_FALLBACK:-0}"
skip_existing="${SKIP_EXISTING:-1}"
dry_run="${DRY_RUN:-0}"
continue_on_error="${CONTINUE_ON_ERROR:-0}"
num_gpus="${NUM_GPUS:-1}"
gpu_ids="${GPU_IDS:-}"
default_master_port="29500"
if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]; then
    default_master_port="$((10000 + (SLURM_JOB_ID % 50000)))"
fi
master_port="${MASTER_PORT:-$default_master_port}"
torchrun_standalone="${TORCHRUN_STANDALONE:-1}"

timestamp() {
    date '+%F %T'
}

resolve_output_root() {
    local target_locale="$1"
    if [[ -n "$output_root_override" ]]; then
        if [[ ${#target_locales[@]} -gt 1 ]]; then
            printf '%s/%s\n' "$output_root_override" "$target_locale"
        else
            printf '%s\n' "$output_root_override"
        fi
        return 0
    fi

    printf '%s/results/qwen25_lora_ablation_%s\n' "$project_root" "$target_locale"
}

resolve_log_root() {
    local target_locale="$1"
    if [[ -n "$log_root_override" ]]; then
        if [[ ${#target_locales[@]} -gt 1 ]]; then
            printf '%s/%s\n' "$log_root_override" "$target_locale"
        else
            printf '%s\n' "$log_root_override"
        fi
        return 0
    fi

    if [[ ${#target_locales[@]} -gt 1 ]]; then
        printf '%s/logs/qwen_lora_ablation/%s\n' "$project_root" "$target_locale"
    else
        printf '%s/logs/qwen_lora_ablation\n' "$project_root"
    fi
}

has_long_option() {
    local needle="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

print_usage() {
    cat <<EOF
Usage:
  bash scripts/run_qwen_lora_ablation.sh [TRAIN_ARGS...]

This script runs multiple Qwen LoRA ablation jobs by calling:
  python scripts/train_qwen_lora_db.py
or, when NUM_GPUS>1:
  torchrun --nproc_per_node=NUM_GPUS scripts/train_qwen_lora_db.py

Default ablation modes:
  plain anchored surface retrieve rerank

Environment variables:
  PYTHON_BIN               Python executable. Default: python
  TORCHRUN_BIN             torchrun executable. Default: torchrun
  MODEL_NAME               HF model id. Default: Qwen/Qwen2.5-7B-Instruct
  TARGET_LOCALE            Space-separated target locale(s). Default: fr
  SOURCE_LOCALE            Source locale. Default: en
  TRAIN_SPLIT              Train split. Default: train
  EVAL_SPLIT               Eval split. Default: validation
  OUTPUT_ROOT              Output root. With multiple locales, each locale uses OUTPUT_ROOT/<locale>
  LOG_ROOT                 Log root. With multiple locales, each locale uses LOG_ROOT/<locale>
  ABLATION_MODES           Space-separated modes to run
  RERANKER_MODEL_PATH      Reranker checkpoint for rerank mode
  ALLOW_RERANK_FALLBACK    1 to allow rerank mode without checkpoint
  SKIP_EXISTING            1 to skip runs with training_summary.json
  DRY_RUN                  1 to print commands without running them
  CONTINUE_ON_ERROR        1 to continue even if one mode fails
  NUM_GPUS                 Number of GPUs per run. Default: 1
  GPU_IDS                  Optional CUDA_VISIBLE_DEVICES list, e.g. 0,1,2,3
  MASTER_PORT              torchrun master port. Default: 29500
  TORCHRUN_STANDALONE      1 to pass --standalone when NUM_GPUS>1

Examples:
  bash scripts/run_qwen_lora_ablation.sh \\
    --num-train-epochs 3 \\
    --per-device-train-batch-size 1 \\
    --gradient-accumulation-steps 16 \\
    --learning-rate 2e-4 \\
    --bf16

  TARGET_LOCALE=fr ABLATION_MODES="plain retrieve rerank" \\
  RERANKER_MODEL_PATH=artifacts/reranker.pt \\
  bash scripts/run_qwen_lora_ablation.sh --bf16

  TARGET_LOCALE="ja ar zh" ABLATION_MODES="plain anchored" \\
  bash scripts/run_qwen_lora_ablation.sh --bf16

  NUM_GPUS=4 GPU_IDS=0,1,2,3 \\
  bash scripts/run_qwen_lora_ablation.sh \\
    --per-device-train-batch-size 1 \\
    --gradient-accumulation-steps 8 \\
    --bf16
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
fi

read -r -a target_locales <<< "$target_locales_raw"
if [[ ${#target_locales[@]} -eq 0 ]]; then
    echo "[$(timestamp)] [Ablation] No target locales configured. Set TARGET_LOCALE." >&2
    exit 1
fi

read -r -a ablation_modes <<< "$ablation_modes_raw"
if [[ ${#ablation_modes[@]} -eq 0 ]]; then
    echo "[$(timestamp)] [Ablation] No modes configured. Set ABLATION_MODES." >&2
    exit 1
fi

if ! [[ "$num_gpus" =~ ^[0-9]+$ ]] || [[ "$num_gpus" -lt 1 ]]; then
    echo "[$(timestamp)] [Ablation] NUM_GPUS must be a positive integer. Current: $num_gpus" >&2
    exit 1
fi

extra_args=("$@")
train_script="$project_root/scripts/train_qwen_lora_db.py"

if [[ -n "$gpu_ids" ]]; then
    export CUDA_VISIBLE_DEVICES="$gpu_ids"
fi

if [[ "$num_gpus" -gt 1 ]] && has_long_option --device-map "${extra_args[@]}"; then
    echo "[$(timestamp)] [Ablation] --device-map should be unset when using NUM_GPUS>1 with torchrun/DDP." >&2
    exit 1
fi

run_mode() {
    local target_locale="$1"
    local mode="$2"
    local run_name
    local output_root
    local output_dir
    local log_root
    local log_file
    local -a cmd

    if [[ "$mode" == "plain" ]]; then
        run_name="plain"
    else
        run_name="entity_${mode}"
    fi

    output_root="$(resolve_output_root "$target_locale")"
    log_root="$(resolve_log_root "$target_locale")"
    output_dir="$output_root/$run_name"
    if [[ ${#target_locales[@]} -gt 1 ]]; then
        log_file="$log_root/${run_name}.log"
    else
        log_file="$log_root/${target_locale}_${run_name}.log"
    fi

    if [[ "$skip_existing" == "1" && -f "$output_dir/training_summary.json" ]]; then
        echo "[$(timestamp)] [Ablation] skip existing | target_locale=$target_locale | mode=$mode | output_dir=$output_dir"
        return 0
    fi

    if [[ "$mode" == "rerank" && -z "$reranker_model_path" && "$allow_rerank_fallback" != "1" ]]; then
        echo "[$(timestamp)] [Ablation] skip rerank | target_locale=$target_locale | reason=missing_reranker_model_path"
        return 0
    fi

    cmd=()
    if [[ "$num_gpus" -gt 1 ]]; then
        if command -v "$torchrun_bin" >/dev/null 2>&1; then
            cmd+=("$torchrun_bin")
        else
            cmd+=("$python_bin" -m torch.distributed.run)
        fi
        if [[ "$torchrun_standalone" == "1" ]]; then
            cmd+=(--standalone)
        fi
        cmd+=(--nproc_per_node "$num_gpus" --master_port "$master_port" "$train_script")
    else
        cmd+=("$python_bin" "$train_script")
    fi

    cmd+=(--output-dir "$output_dir")

    if ! has_long_option --model-name "${extra_args[@]}"; then
        cmd+=(--model-name "$model_name")
    fi
    if ! has_long_option --target-locale "${extra_args[@]}"; then
        cmd+=(--target-locale "$target_locale")
    fi
    if ! has_long_option --source-locale "${extra_args[@]}"; then
        cmd+=(--source-locale "$source_locale")
    fi
    if ! has_long_option --train-split "${extra_args[@]}"; then
        cmd+=(--train-split "$train_split")
    fi
    if ! has_long_option --eval-split "${extra_args[@]}"; then
        cmd+=(--eval-split "$eval_split")
    fi

    if [[ "$mode" != "plain" ]]; then
        cmd+=(--mode entity-aware --entity-pipeline-mode "$mode")
    fi

    if [[ "$mode" == "rerank" && -n "$reranker_model_path" ]]; then
        cmd+=(--reranker-model-path "$reranker_model_path")
    fi

    cmd+=("${extra_args[@]}")

    echo "[$(timestamp)] [Ablation] start | target_locale=$target_locale | mode=$mode | output_dir=$output_dir | num_gpus=$num_gpus | cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-auto}"
    printf '[%s] [Ablation] command |' "$(timestamp)"
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if [[ "$dry_run" == "1" ]]; then
        return 0
    fi

    mkdir -p "$output_dir" "$log_root"
    set +e
    "${cmd[@]}" 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
    set -e

    if [[ $status -ne 0 ]]; then
        echo "[$(timestamp)] [Ablation] failed | target_locale=$target_locale | mode=$mode | status=$status | log=$log_file" >&2
        if [[ "$continue_on_error" == "1" ]]; then
            return 0
        fi
        return "$status"
    fi

    echo "[$(timestamp)] [Ablation] done | target_locale=$target_locale | mode=$mode | log=$log_file"
    return 0
}

supported_ablation_modes=()
for mode in "${ablation_modes[@]}"; do
    case "$mode" in
        plain|anchored|surface|retrieve|rerank)
            supported_ablation_modes+=("$mode")
            ;;
        *)
            echo "[$(timestamp)] [Ablation] invalid mode | mode=$mode" >&2
            if [[ "$continue_on_error" != "1" ]]; then
                exit 1
            fi
            ;;
    esac
done

if [[ ${#supported_ablation_modes[@]} -eq 0 ]]; then
    echo "[$(timestamp)] [Ablation] No valid modes remain after validation." >&2
    exit 1
fi

for target_locale in "${target_locales[@]}"; do
    for mode in "${supported_ablation_modes[@]}"; do
        run_mode "$target_locale" "$mode"
    done
done

echo "[$(timestamp)] [Ablation] all requested runs processed"
