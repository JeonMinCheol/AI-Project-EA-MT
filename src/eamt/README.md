# src

EA-MT 시스템 핵심 코드입니다.

## 구조

src/eamt/

data/
kb/
prompt/
model/
inference/
postprocess/
evaluation/
common/

---

## data

데이터 로딩 및 전처리

---

## kb

Wikidata 엔티티 처리

예

- QID 조회
- alias 선택

---

## prompt

프롬프트 생성 코드

예

- plain prompt
- entity plan prompt
- CERC prompt

---

## model

모델 관련 코드

예

- LoRA
- trainer
- generation

---

## inference

번역 파이프라인

---

## postprocess

출력 후처리

예

- regex 필터링
- 엔티티 일관성 체크

---

## evaluation

평가 및 분석 코드

---

## common

공통 유틸 코드