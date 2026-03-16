## 전체 파이프라인

EA-MT Pipeline

Raw Data
↓
Data Processing
↓
KB Cache (Wikidata)
↓
Supervised Fine-Tuning Dataset
↓
LoRA Training
↓
Inference
↓
Postprocessing
↓
Evaluation
↓
Report

---

## 단계별 설명

### 1️⃣ Raw Data

EA-MT 데이터셋 원본이 저장됩니다.

예시 데이터

```json
{
 "source": "What is the seventh tallest mountain in North America?",
 "target": "北アメリカで七番目に高い山は何ですか？",
 "entities": ["Q49"]
}

특징

- source: 번역할 문장

- target: 정답 번역

- entities: Wikidata QID

하지만 모델은 Q49가 무엇인지 알지 못하기 때문에 추가적인 지식이 필요합니다.

---

### 2️⃣ Data Processing

데이터를 모델 학습에 사용할 수 있도록 정리합니다.

주요 작업

- JSON 데이터 파싱

- 언어별 데이터 분리

- 엔티티(QID) 추출

- 데이터 포맷 통일

이 단계의 목적은 데이터를 깨끗하고 일관된 형태로 만드는 것입니다.

---

### 3️⃣ KB Cache (Wikidata)

엔티티 QID를 기반으로 Wikidata에서 엔티티 정보를 가져옵니다.

예시


Q49  
English: North America  
Japanese: 北アメリカ  
Type: Continent  
Aliases: [...]


이 정보를 로컬 캐시로 저장합니다.

---

### 4️⃣ Supervised Fine-Tuning Dataset

모델 학습을 위한 SFT(Supervised Fine-Tuning) 데이터를 생성합니다.

일반 번역 학습


source → target


EA-MT 학습


ENTITY PLAN + source → target


예시


\[ENTITY PLAN]  
North America → 北アメリカ

\[SOURCE]  
What is the seventh tallest mountain in North America?

\[TARGET]  
北アメリカで七番目に高い山は何ですか？

---

5️⃣ LoRA Training

Qwen2.5 모델을 LoRA(Low-Rank Adaptation) 방식으로 미세조정합니다.

- LoRA를 사용하는 이유

- 대형 모델 전체 학습 비용 감소

- 기존 언어 능력 유지

- 특정 태스크(EA-MT)에 빠르게 적응

이 단계에서 모델은

- 문장 번역 능력

- 엔티티 번역 능력

을 동시에 학습합니다.

---

### 6️⃣ Inference

학습된 모델을 사용하여 실제 번역을 수행합니다.

입력


source sentence  
wikidata entity id


추론 시에도 ENTITY PLAN을 사용합니다.

예시


\[ENTITY PLAN]  
Spirited Away → 千と千尋の神隠し

\[SOURCE]  
Did Spirited Away win an Academy Award?

---

### 7️⃣ Postprocessing

LLM 출력 결과를 정리합니다.

예를 들어 모델이 다음과 같은 출력을 할 수 있습니다.


The translation is:  
北アメリカで七番目に高い山は何ですか？


Postprocessing 단계에서

- 불필요한 설명 제거

- 출력 포맷 정리

- 엔티티 표기 확인

을 수행합니다.

---

### 8️⃣ Evaluation

모델의 성능을 평가합니다.

EA-MT 평가지표

Metric	설명
COMET	전체 번역 품질
M-ETA	엔티티 번역 정확도

최종 점수는 두 지표를 함께 고려합니다.

---

### 9️⃣ Report

실험 결과를 정리합니다.

- 주요 내용

- 성능 비교

- ablation study

- 오류 분석

- 사례 분석

이 결과는 보고서 및 발표 자료로 활용됩니다.