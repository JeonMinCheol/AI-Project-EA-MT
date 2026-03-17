# EA-MT Project

Entity-Aware Machine Translation(EA-MT) 프로젝트 저장소입니다.

본 프로젝트는 **Qwen2.5 기반 EA-MT 시스템 구현**을 목표로 합니다.

일반적인 번역 모델은 문장 번역에는 강하지만 **엔티티 이름(인물, 장소, 영화 등)**을 정확하게 번역하지 못하는 경우가 많습니다.  
EA-MT는 이러한 문제를 해결하기 위해 **Wikidata 지식을 활용하여 엔티티를 정확하게 번역하는 시스템**을 구축하는 것을 목표로 합니다.

---

## 디렉터리 구조
ea-mt-project/
│
├── configs/ # 실험 설정
├── data/ # 데이터
├── scripts/ # 실행 스크립트
├── src/ # 핵심 코드
├── outputs/ # 실험 결과
├── reports/ # 보고서
├── notebooks/ # 분석 노트북
└── tests/ # 테스트 코드
