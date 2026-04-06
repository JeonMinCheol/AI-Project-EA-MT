#!/usr/bin/env bash
#SBATCH -J EA_MT_LORA
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v7
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

target_locales_raw="${TARGET_LOCALE:-es de zh it th tr}"
gpu=2
read -r -a target_locales <<< "$target_locales_raw"

for target_locale in "${target_locales[@]}"; do
  echo "[eval] start target_locale=$target_locale"

  python scripts/evaluate_qwen_lora_db.py \
    --peft-adapter-path results/qwen25_lora_ablation_"$target_locale"/entity_anchored \
    --use-training-summary \
    --merge-peft-adapter \
    --comet-num-gpus "$gpu" \
    --prediction-output-path results/qwen25_lora_eval_entity_anchored_"$target_locale".jsonl

  python scripts/evaluate_qwen_lora_db.py \
    --peft-adapter-path results/qwen25_lora_ablation_"$target_locale"/plain \
    --use-training-summary \
    --comet-num-gpus "$gpu" \
    --merge-peft-adapter \
    --prediction-output-path results/qwen25_lora_eval_plain_"$target_locale".jsonl

  python scripts/evaluate_qwen_lora_db.py \
    --peft-adapter-path results/qwen25_lora_ablation_"$target_locale"/entity_surface \
    --use-training-summary \
    --merge-peft-adapter \
    --comet-num-gpus "$gpu" \
    --prediction-output-path results/qwen25_lora_eval_entity_surface_"$target_locale".jsonl

  python scripts/evaluate_qwen_lora_db.py \
    --peft-adapter-path results/qwen25_lora_ablation_"$target_locale"/entity_retrieve \
    --use-training-summary \
    --merge-peft-adapter \
    --comet-num-gpus "$gpu" \
    --prediction-output-path results/qwen25_lora_eval_entity_retrieve_"$target_locale".jsonl
done
