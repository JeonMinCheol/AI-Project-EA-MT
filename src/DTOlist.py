from typing import Any


class EAMTExample():
    """
    설명: 학습·검증·추론에서 공통으로 쓰는 기본 샘플 단위.
    비고: 공식 평가 포맷의 정답 wikidata_id를 포함한다.
    """
    def __init__(
        self,
        id: str,                         # 샘플 고유 ID
        source: str,                     # 원문(source) 문장
        target_lang: str,                # 목표 번역 언어 코드 (예: "ja", "ko")
        target: str = None,              # 정답 번역문; 학습/검증에는 존재 가능, 추론에서는 None 가능
        wikidata_id: str | None = None,  # 정답 entity의 정답 QID
        entity_qids: list[str] = None,   # 문장에 포함된 전체 엔티티 QID 목록
        meta: dict[str, Any] = None      # split, domain, 원본 레코드 정보 등 부가 메타데이터
    ):
        self.id = id
        self.source = source
        self.target_lang = target_lang
        self.target = target
        self.wikidata_id = wikidata_id
        self.entity_qids = entity_qids
        self.meta = meta
    
    def __call__(self, *args, **kwds):
        return self.id, self.source, self.target_lang, self.target, self.wikidata_id, self.entity_qids, self.meta


class KBEntityRecord():
    """
    설명: qid_index의 값 객체. QID 기준 direct lookup 결과를 표준화한 엔티티 레코드.
    비고: target_label이 없거나 language_available=False이면 기본 추론 후보에서 제외하거나 fallback 대상으로 표시한다.
    """
    def __init__(
        self,
        qid: str,                         # 엔티티의 Wikidata QID
        label_en: str | None,             # 영어 기준 대표 label
        aliases_en: list[str],            # 영어 기준 alias 목록
        target_lang: str | None,          # 이 레코드가 표현하는 목표 언어 코드
        target_label: str | None,         # 목표 언어에서의 엔티티의 label (이름)
        target_aliases: list[str],        # 목표 언어에서의 alias 목록
        entity_type: str | None,          # 엔티티 타입 (예: Person, Film, Food)
        description: str | None,          # 엔티티 설명 텍스트
        normalized_surfaces: list[str],   # 검색용으로 정규화된 surface 목록
        language_available: bool,         # 해당 target_lang에서 usable label/alias가 존재하는지 여부
        popularity_score: float           # 엔티티 우선순위 계산용 인기/빈도 점수
    ):
        self.qid = qid
        self.label_en = label_en
        self.aliases_en = aliases_en
        self.target_lang = target_lang
        self.target_label = target_label
        self.target_aliases = target_aliases
        self.entity_type = entity_type
        self.description = description
        self.normalized_surfaces = normalized_surfaces
        self.language_available = language_available
        self.popularity_score = popularity_score

    def __call__(self, *args, **kwds):
        return self.qid, self.label_en, self.aliases_en, self.target_label, self.target_aliases, self.entity_type, self.description, self.normalized_surfaces, self.language_available, self.popularity_score, self.target_lang


class SpanMatch():
    """
    설명: source 문장 내 정답 mention 위치를 나타내는 정렬 결과.
    비고: match_kind 예시: exact, alias_exact, normalized_exact, not_found
    """
    def __init__(
        self,
        source_span: str,                 # source 문장에서 매칭된 실제 span 문자열
        char_start: int,                  # 매칭 시작 문자 인덱스
        char_end: int,                    # 매칭 끝 문자 인덱스 (exclusive 권장)
        match_kind: str,                  # 매칭 방식/유형
        matched_surface: str | None = None,  # 실제로 매칭에 사용된 정규화/원형 surface
        match_score: float | None = None      # fuzzy match 등에서의 유사도 점수
    ):
        self.source_span = source_span
        self.char_start = char_start
        self.char_end = char_end
        self.match_kind = match_kind
        self.matched_surface = matched_surface
        self.match_score = match_score
    
    def __call__(self, *args, **kwds):
        return self.source_span, self.char_start, self.char_end, self.match_kind, self.matched_surface, self.match_score


class CandidateEntity():
    """
    설명: retrieval 계층이 reranker로 넘기는 후보 단위.
    비고: candidate_source 예시: anchored_qid, surface_search, auxiliary_entity
    """
    def __init__(
        self,
        qid: str,                         # 후보 엔티티의 QID
        candidate_source: str,            # 이 후보가 생성된 경로/출처
        target_label: str | None,         # 후보의 목표 언어 canonical label
        target_aliases: list[str],        # 후보의 목표 언어 alias 목록
        entity_type: str | None,          # 후보 엔티티 타입
        description: str | None,          # 후보 엔티티 설명
        popularity_score: float,          # 후보 prior 성격의 인기 점수
        alias_count: int,                 # target alias 개수
        ambiguity_count: int,             # 해당 surface에서 경쟁하는 QID 수
        source_span: str | None = None,   # source에서 이 후보와 연결된 span 문자열
        span_match: SpanMatch | None = None  # span 정렬 상세 결과
    ):
        self.qid = qid
        self.candidate_source = candidate_source
        self.target_label = target_label
        self.target_aliases = target_aliases
        self.entity_type = entity_type
        self.description = description
        self.popularity_score = popularity_score
        self.alias_count = alias_count
        self.ambiguity_count = ambiguity_count
        self.source_span = source_span
        self.span_match = span_match

    def __call__(self, *args, **kwds):
        return self.qid, self.candidate_source, self.target_label, self.target_aliases, self.entity_type, self.description, self.popularity_score, self.alias_count, self.ambiguity_count, self.source_span, self.span_match


class ScoredCandidate():
    """
    설명: reranker 점수 계산이 끝난 후보 객체. top-1 선택과 bias 생성에 사용된다.
    비고: 공식 평가 경로에서는 canonical QID의 prior_bonus를 강하게 둘 수 있다.
    """
    def __init__(
        self,
        candidate: CandidateEntity,                  # 원본 후보 객체
        reranker_score: float,                       # reranker 모델/규칙이 계산한 본 점수
        prior_bonus: float,                          # anchored QID 등 prior 기반 가산점
        final_score: float,                          # 최종 선택에 사용되는 총합 점수
        margin_to_next: float | None = None,         # 2등 후보와의 점수 차
        feature_dump: dict[str, float | int | str] = None  # 디버깅용 feature 값 기록
    ):
        self.candidate = candidate
        self.reranker_score = reranker_score
        self.prior_bonus = prior_bonus
        self.final_score = final_score
        self.margin_to_next = margin_to_next
        self.feature_dump = feature_dump

    def __call__(self, *args, **kwds):
        return self.candidate, self.reranker_score, self.prior_bonus, self.final_score, self.margin_to_next, self.feature_dump


class EntityMemoryBlock():
    """
    설명: 모델 입력에 삽입하는 명시적 지식 블록.
    비고: rendered_text는 prompt builder가 바로 붙일 수 있는 최종 문자열이다.
    """
    def __init__(
        self,
        entries: list[dict],              # 메모리 엔트리 원본 목록
        memory_modes: str,                # memory 구성 모드 (예: canonical_only, canonical_plus_alias)
        rendered_text: str,               # 프롬프트에 바로 삽입할 문자열
        source_span: str | None = None,   # 대응되는 source mention span
        qid: str = None,                  # 정답 entity QID
        canonical_target: str = None,     # canonical target label
        alias_candidates: list[str] = None,  # 메모리에 포함된 alias 후보 목록
        entity_type: str | None = None,   # 정답 entity 타입
        description: str | None = None    # 정답 entity 설명
    ):
        self.entries = entries
        self.memory_modes = memory_modes
        self.rendered_text = rendered_text
        self.source_span = source_span
        self.qid = qid
        self.canonical_target = canonical_target
        self.alias_candidates = alias_candidates
        self.entity_type = entity_type
        self.description = description

    def __call__(self, *args, **kwds):
        return self.entries, self.memory_modes, self.rendered_text, self.source_span, self.qid, self.canonical_target, self.alias_candidates


class TranslationRequest():
    """
    설명: 추론 파이프라인에 들어가는 요청 객체.
    비고: 요청 단위에서 공식 평가 모드와 일반화 모드를 함께 제어한다.
    """
    def __init__(
        self,
        target_lang: str,                     # 목표 번역 언어 코드
        wikidata_id: str | None = None,       # 정답 entity QID; 평가 환경에서는 주어질 수 있음
        allow_auxiliary_search: bool = None,  # 보조 surface/entity 검색 허용 여부
        apply_constrained_decoding: bool = None,  # constrained decoding 적용 여부
        apply_ercm: bool = None,              # ERCM 후처리 적용 여부
        max_candidates: int = None,           # retrieval 단계 최대 후보 수
        alias_limit: int = None               # 메모리/bias에 포함할 alias 최대 개수
    ):
        self.target_lang = target_lang
        self.wikidata_id = wikidata_id
        self.allow_auxiliary_search = allow_auxiliary_search
        self.apply_constrained_decoding = apply_constrained_decoding
        self.apply_ercm = apply_ercm
        self.max_candidates = max_candidates
        self.alias_limit = alias_limit

    def __call__(self, *args, **kwds):
        return self.target_lang, self.wikidata_id, self.allow_auxiliary_search, self.apply_constrained_decoding, self.apply_ercm, self.max_candidates, self.alias_limit


class TranslationDraft():
    """
    설명: 1차 번역 초안과 생성 중간산출을 보관하는 객체.
    비고: ERCM 입력으로 반드시 draft_text를 전달한다.
    """
    def __init__(
        self,
        prompt_text: str,                         # 모델에 실제 입력된 최종 프롬프트
        draft_text: str,                          # 1차 생성된 번역 초안
        raw_generation: str,                      # 디코더 원출력 또는 후처리 전 텍스트
        used_memory: EntityMemoryBlock | None,    # 생성에 사용된 entity memory
        used_logit_bias: dict[str, float] | None = None,  # 적용된 bias 토큰/문자열과 가중치
        generation_trace: dict[str, Any] = None   # 생성 디버그 정보 (beam, score, step log 등)
    ):
        self.prompt_text = prompt_text
        self.draft_text = draft_text
        self.raw_generation = raw_generation
        self.used_memory = used_memory
        self.used_logit_bias = used_logit_bias
        self.generation_trace = generation_trace

    def __call__(self, *args, **kwds):
        return self.prompt_text, self.draft_text, self.raw_generation, self.used_memory, self.used_logit_bias, self.generation_trace


class ERCMDecision():
    """
    설명: ERCM 발동 여부와 사유를 표현하는 결정 객체.
    비고: reasons 예시: missing_canonical, english_residue, wrong_alias, low_margin
    """
    def __init__(
        self,
        should_run: bool,                 # ERCM을 실제 실행할지 여부
        reasons: list[str],               # ERCM 발동 사유 목록
        confidence: float | None = None,  # 이 결정의 신뢰도
        error_types: list[str] = None,    # 감지된 오류 유형 목록
        verifier_score: float | None = None  # 검증기/verifier 점수
    ):
        self.should_run = should_run
        self.reasons = reasons
        self.confidence = confidence
        self.error_types = error_types
        self.verifier_score = verifier_score

    def __call__(self, *args, **kwds):
        return self.should_run, self.reasons, self.confidence, self.error_types, self.verifier_score


class ERCMCorrectionResult():
    """
    설명: ERCM 수행 결과. 수정본 선택 전 단계의 산출물.
    비고: status 예시: skipped, corrected, rejected
    """
    def __init__(
        self,
        status: str,                          # ERCM 처리 상태
        corrected_text: str | None,           # 수정된 번역문
        applied_edits: list[str] = None,      # 적용된 수정 내역 요약
        verifier_score: float | None = None,  # 수정본 검증 점수
        selected: bool | None = None          # 최종 결과로 채택되었는지 여부
    ):
        self.status = status
        self.corrected_text = corrected_text
        self.applied_edits = applied_edits
        self.verifier_score = verifier_score
        self.selected = selected

    def __call__(self, *args, **kwds):
        return self.status, self.corrected_text, self.applied_edits, self.verifier_score, self.selected


class TranslationResult():
    """
    설명: 최종 응답과 디버그 메타데이터를 함께 담는 결과 객체.
    비고: 평가 저장 시 M-ETA/COMET 계산에 필요한 중간 정보도 debug에 넣을 수 있다.
    """
    def __init__(
        self,
        final_translation: str,               # 최종 반환 번역문
        draft_translation: str,               # ERCM 전 초안 번역문
        used_ercm: bool,                      # ERCM 사용 여부
        top_candidate: ScoredCandidate,       # 최종 선택된 top-1 후보
        memory: EntityMemoryBlock | None = None,   # 사용된 entity memory
        ercm_result: ERCMCorrectionResult | None = None,  # ERCM 결과 상세
        debug: dict[str, Any] = None          # 전체 파이프라인 디버그 정보
    ):
        self.final_translation = final_translation
        self.draft_translation = draft_translation
        self.used_ercm = used_ercm
        self.top_candidate = top_candidate
        self.memory = memory
        self.ercm_result = ercm_result
        self.debug = debug

    def __call__(self, *args, **kwds):
        return self.final_translation, self.draft_translation, self.used_ercm, self.top_candidate, self.memory, self.ercm_result, self.debug


class RuntimeResources():
    """
    설명: 실행 시점에 주입되는 리소스와 설정 묶음.
    비고: 파이프라인 함수는 모델과 인덱스를 전역으로 참조하지 않고 명시적으로 주입받는 것을 원칙으로 한다.
    """
    def __init__(
        self,
        qid_index: int,          # QID direct lookup용 인덱스
        surface_index: int,      # surface search용 역색인
        reranker_mode,           # reranker 객체 또는 reranking 전략 식별자
        translator_model,        # 번역 생성 모델
        tokenizer,               # 번역 모델 tokenizer
        ercm_model = None        # ERCM용 보조 모델
    ):
        self.qid_index = qid_index
        self.surface_index = surface_index
        self.reranker_mode = reranker_mode
        self.translator_model = translator_model
        self.tokenizer = tokenizer
        self.ercm_model = ercm_model

    def __call__(self, *args, **kwds):
        return self.qid_index, self.surface_index, self.reranker_mode, self.translator_model, self.tokenizer, self.ercm_model


class InferenceConfig():
    """
    설명: 추론 시 사용되는 세부 설정 묶음.
    """
    def __init__(
        self,
        bias_strength: float,        # constrained decoding에서 bias 강도
        ercm_threshold: float,       # ERCM 발동 임계값
        auxilary_ngram_max: int,     # auxiliary surface search에서 사용할 최대 n-gram 길이
        max_alias_bias_terms: int    # bias에 반영할 alias 최대 개수
    ):
        self.bias_strength = bias_strength
        self.ercm_threshold = ercm_threshold
        self.auxilary_ngram_max = auxilary_ngram_max
        self.max_alias_bias_terms = max_alias_bias_terms

    def __call__(self, *args, **kwds):
        return self.bias_strength, self.ercm_threshold, self.auxilary_ngram_max, self.max_alias_bias_terms