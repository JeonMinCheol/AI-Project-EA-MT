# configs

실험 설정 파일을 저장하는 폴더입니다.

코드에 하이퍼파라미터를 직접 작성하지 않고  
**YAML 설정 파일로 관리**하기 위해 사용합니다.

## 구조

configs/
├── data/
├── model/
├── inference/
└── experiment/

## 설명

### data/
데이터 관련 설정

예
- 학습 언어
- 데이터 경로

### model/
모델 설정

예
- base model
- LoRA rank
- learning rate

### inference/
추론 설정

예
- temperature
- top_p
- decoding 방식

### experiment/
전체 실험 설정

예
- baseline
- entity_plan
- bias
- CERC