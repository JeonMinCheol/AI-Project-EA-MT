# data

프로젝트에서 사용하는 모든 데이터를 저장하는 폴더입니다.

## 구조

data/
├── raw/
├── interim/
├── processed/
└── external/

---

## raw

원본 데이터

EA-MT 데이터셋

예
data/raw/train
data/raw/validation
data/raw/test

---

## interim

전처리 중간 데이터

예

parsed
qid_lists
kb_cache
prompt_ready

---

## processed
학습 및 추론에 바로 사용할 수 있는 데이터

예
sft/
inference_inputs/
cerc/

---

## external

외부 데이터

예

- Wikidata