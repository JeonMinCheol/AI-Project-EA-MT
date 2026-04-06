#!/usr/bin/env bash
#SBATCH -J EA_MT_LORA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v7
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

set -euo pipefail

project_root="/data/a2019102224/AI-Project-EA-MT"
cd "$project_root"

mkdir -p "$project_root/logs"

# anchored: gold wikidata_id direct lookup만 사용
# surface: surface retrieval만 사용
# retrieve: anchored lookup + surface retrieval 후 retrieval top-1 사용
# rerank: anchored lookup + surface retrieval 후 reranker top-1 사용

target_locale="${TARGET_LOCALE:- es de zh it th tr}" # fr ja ar
ablation_modes="${ABLATION_MODES:-plain anchored surface retrieve rerank}"
reranker_model_path="${RERANKER_MODEL_PATH:-}"
allow_rerank_fallback="${ALLOW_RERANK_FALLBACK:-0}"

num_gpus="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-2}}"
if [[ ! "$num_gpus" =~ ^[0-9]+$ ]]; then
    num_gpus="$(echo "$num_gpus" | grep -oE '[0-9]+' | head -n 1)"
fi
if [[ -z "$num_gpus" ]]; then
    num_gpus="1"
fi

default_master_port="29500"
if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]; then
    default_master_port="$((10000 + (SLURM_JOB_ID % 50000)))"
fi
master_port="${MASTER_PORT:-$default_master_port}"

common_args=(
    --eval-split "${EVAL_SPLIT:-validation}"
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
    --num-train-epochs "${NUM_TRAIN_EPOCHS:-3}"
    --learning-rate "${LEARNING_RATE:-2e-4}"
)

if [[ "${USE_BF16:-1}" == "1" ]]; then
    common_args+=(--bf16)
fi

if [[ "${USE_FP16:-0}" == "1" ]]; then
    common_args+=(--fp16)
fi

echo "[SLURM] project_root=$project_root"
echo "[SLURM] target_locale=$target_locale"
echo "[SLURM] ablation_modes=$ablation_modes"
echo "[SLURM] num_gpus=$num_gpus"
echo "[SLURM] master_port=$master_port"
echo "[SLURM] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-auto}"
echo "[SLURM] reranker_model_path=${reranker_model_path:-none}"

TARGET_LOCALE="$target_locale" \
ABLATION_MODES="$ablation_modes" \
RERANKER_MODEL_PATH="$reranker_model_path" \
ALLOW_RERANK_FALLBACK="$allow_rerank_fallback" \
NUM_GPUS="$num_gpus" \
MASTER_PORT="$master_port" \
bash "$project_root/scripts/run_qwen_lora_ablation.sh" "${common_args[@]}"
