# scripts

프로젝트 실행 스크립트를 모아둔 폴더입니다.

이 폴더의 Python 파일들은 **실행 진입점(entry point)** 역할을 합니다.

## 구조

scripts/
├── setup/
├── data/
├── train/
├── inference/
├── eval/
└── utils/

---

## data

데이터 전처리

예

parse_eamt.py
extract_qids.py
build_kb_cache.py
build_sft_data.py
build_cerc_data.py

---

## train

모델 학습


train_lora.py
train_cerc.py

---

## inference

번역 생성

run_plain.py
run_entity_plan.py
run_bias.py
run_cerc.py

---

## eval

결과 평가

format_predictions.py
score_local.py
collect_results.py