# Qwen LoRA Fine-Tuning

이 저장소에는 허깅페이스 `Qwen` 계열 모델을 `PEFT LoRA`로 학습하는 DB 기반 SFT 환경이 추가되어 있습니다.

## 1. 설치

```bash
pip install -r requirements-qwen-lora.txt
```

4-bit 또는 8-bit 로딩을 쓰려면 추가로 아래를 설치하세요.

```bash
pip install bitsandbytes
```

## 2. 필수 환경 변수

기존 평가 스크립트와 동일하게 `.env` 또는 shell 환경에 아래 DB 정보가 있어야 합니다.

```bash
DB_HOST=...
DB_PORT=3306
DB_USER=...
DB_PASSWORD=...
DB_NAME=...
```

## 3. 기본 실행 예시

### Plain LoRA

```bash
python scripts/train_qwen_lora_db.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --target-locale ko \
  --output-dir results/qwen25-lora-ko \
  --mode plain \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --bf16
```

### Entity-Aware LoRA

```bash
python scripts/train_qwen_lora_db.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --target-locale ko \
  --output-dir results/qwen25-lora-ko-entity-aware \
  --mode entity-aware \
  --alias-limit 1 \
  --description-max-chars 80 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --bf16
```

### Retrieval + Reranker 기반 Entity-Aware LoRA

```bash
python scripts/train_qwen_lora_db.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --target-locale ko \
  --output-dir results/qwen25-lora-ko-rerank \
  --mode entity-aware \
  --entity-pipeline-mode rerank \
  --retrieval-top-k 10 \
  --retrieval-per-surface-k 5 \
  --reranker-model-path artifacts/reranker.pt \
  --alias-limit 1 \
  --description-max-chars 80 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --bf16
```

## 4. 주요 동작

- `train` split을 학습용으로, `validation` split을 기본 검증용으로 사용합니다.
- DB에서 번역 reference를 읽어 `prompt -> assistant target` 형태의 Qwen chat 학습 예제를 만듭니다.
- `entity-aware` 모드에서는 candidate pipeline을 거쳐 top-1 entity memory를 만든 뒤 prompt에 넣을 수 있습니다.
- loss는 assistant 응답 토큰에만 적용됩니다.
- 결과물은 `output-dir`에 LoRA adapter와 tokenizer, `training_summary.json`으로 저장됩니다.

## 4-1. LoRA 결합 평가

학습된 LoRA adapter를 base Qwen에 붙여 DB 평가를 돌릴 수 있습니다.

```bash
python scripts/evaluate_qwen_lora_db.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --peft-adapter-path results/qwen25-lora-ko \
  --merge-peft-adapter \
  --split validation \
  --target-locale ko \
  --mode entity-aware \
  --entity-pipeline-mode retrieve \
  --prediction-output-path results/qwen25_lora_eval_ko.jsonl
```

`--merge-peft-adapter`를 빼면 adapter를 붙인 상태 그대로 평가하고, 넣으면 base model에 merge한 뒤 평가합니다.

학습 때 사용한 pipeline 조건을 그대로 복원하고 싶다면 `training_summary.json`을 자동으로 읽게 할 수 있습니다.

```bash
python scripts/evaluate_qwen_lora_db.py \
  --peft-adapter-path results/qwen25_lora_ablation_ko/entity_retrieve \
  --use-training-summary \
  --merge-peft-adapter \
  --prediction-output-path results/qwen25_lora_eval_entity_retrieve_ko.jsonl
```

이 경우 `model_name`, `source_locale`, `target_locale`, `mode`, `entity_pipeline_mode`, retrieval/reranker 설정, `system_prompt`를 adapter 폴더의 `training_summary.json`에서 자동 복원합니다.

예전 실험처럼 `training_summary.json`에 `alias_limit` 또는 `description_max_chars`가 없는 경우에는 현재 평가 기본값이 사용되므로, 그 값을 바꿔 학습했었다면 평가 커맨드에 직접 다시 넣어주세요.

## 5. 자주 쓰는 옵션

```bash
--eval-split none
```

검증 없이 train split만 사용합니다.

```bash
--train-limit 128 --eval-limit 32
```

스모크 테스트용으로 일부 샘플만 사용합니다.

```bash
--lora-target-modules auto
```

Qwen 기본 projection 모듈(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)을 자동 선택합니다.

```bash
--entity-pipeline-mode anchored
--entity-pipeline-mode surface
--entity-pipeline-mode retrieve
--entity-pipeline-mode rerank
```

ablation용 candidate pipeline 설정입니다.

- `anchored`: gold `wikidata_id` direct lookup만 사용
- `surface`: surface retrieval만 사용
- `retrieve`: anchored lookup + surface retrieval 후 retrieval top-1 사용
- `rerank`: anchored lookup + surface retrieval 후 reranker top-1 사용

`rerank` 모드에서 `--reranker-model-path`를 주지 않으면 heuristic score로 fallback합니다.

```bash
--load-in-4bit
```

base model을 4-bit로 불러옵니다. `bitsandbytes`가 필요합니다.

## 6. 멀티 GPU 예시

Trainer/DDP로 돌릴 때는 보통 `device_map`을 비워두고 `torchrun`으로 실행하는 편이 안전합니다.

```bash
torchrun --nproc_per_node=4 scripts/train_qwen_lora_db.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --target-locale ko \
  --output-dir results/qwen25-lora-ko-ddp \
  --mode plain \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 3 \
  --learning-rate 2e-4 \
  --bf16
```

## 7. Ablation 스크립트

아래 스크립트는 `plain / anchored / surface / retrieve / rerank` 실험을 순차 실행합니다.

```bash
bash scripts/run_qwen_lora_ablation.sh \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --bf16
```

`rerank` 실험까지 포함하려면 reranker checkpoint를 같이 넘기세요.

```bash
RERANKER_MODEL_PATH=artifacts/reranker.pt \
bash scripts/run_qwen_lora_ablation.sh --bf16
```

기본적으로 reranker checkpoint가 없으면 `rerank` 모드는 건너뜁니다. 실행 전 커맨드만 보고 싶으면 `DRY_RUN=1`을 사용하면 됩니다.

멀티 GPU로 돌릴 때는 `NUM_GPUS`를 주면 스크립트가 자동으로 `torchrun`을 사용합니다.

```bash
NUM_GPUS=4 GPU_IDS=0,1,2,3 \
bash scripts/run_qwen_lora_ablation.sh \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --bf16
```

멀티 GPU 실행에서는 `--device-map`을 같이 주지 않는 것을 권장합니다.

## 8. SLURM run.sh

루트의 [run.sh](/data/a2019102224/AI-Project-EA-MT/run.sh)는 SLURM 제출용 래퍼입니다.

```bash
sbatch run.sh
```

이 스크립트는 내부에서:

- `NUM_GPUS`를 SLURM 할당값 기준으로 설정하고
- `MASTER_PORT`를 `SLURM_JOB_ID` 기반으로 계산해서
- `scripts/run_qwen_lora_ablation.sh`를 호출합니다

그래서 기본 `29500` 포트 충돌을 줄이는 용도로 바로 사용할 수 있습니다.
