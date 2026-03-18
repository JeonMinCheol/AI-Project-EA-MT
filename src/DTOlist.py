from typing import Any

class EAMTExample():
    """
    설명: 학습·검증·추론에서 공통으로 쓰는 기본 샘플 단위. 공식 평가 포맷의 focal wikidata_id를 포함한다.
    비고: target은 학습/검증에는 존재할 수 있고, 실서비스 추론에서는 None일 수 있다.
    """
    def __init__(self, id:str, source:str, target_lang:str, target:str=None, wikidata_id:str|None=None, entity_qids:list[str]=None, meta:dict[str, Any]=None):
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
    비고: target_label이 없거나 language_available=False이면 기본 추론 후보에서 제외하거나 fallback 대상로 표시한다.
    """
    def __init__(self, qid:str, label_en:str|None, aliases_en:list[str], target_lang:str|None, target_label:str|None, target_aliases:list[str], entity_type:str|None, description:str|None,normalized_surfaces:list[str], language_avaliable:bool, popularity_score:float):
        self.qid = qid
        self.label_en = label_en
        self.aliases_en = aliases_en
        self.target_lang = target_lang
        self.target_label = target_label
        self.target_aliases = target_aliases
        self.entity_type = entity_type
        self.description = description
        self.normalized_surfaces = normalized_surfaces
        self.language_avaliable = language_avaliable
        self.popularity_score = popularity_score

    def __call__(self, *args, **kwds):
        return self.qid, self.label_en, self.aliases_en, self.target_label, self.target_aliases, self.entity_type, self.description, self.normalized_surfaces, self.language_avaliable, self.popularity_score, self.target_lang

class SpanMatch():
    """
    설명: source 문장 내 focal mention 위치를 나타내는 정렬 결과.
    비고: match_kind 예시: exact, alias_exact, normalized_exact, not_found
    """
    def __init__(self, source_span:str, char_start:int, char_end:int, match_kind:str, matched_surface:str|None=None, match_score:float|None=None):
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
    def __init__(self, qid:str, candidate_source:str, target_label:str|None, target_aliases:list[str], entity_type:str|None, description:str|None, popularity_score:float, alias_count:int, ambiguity_count:int, source_span:str|None=None, span_match:SpanMatch|None=None):
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
    def __init__(self, candidate:CandidateEntity, reranker_score:float, prior_bonus:float, final_score:float, margin_to_next:float|None=None, feature_dump:dict[str,float|int|str]=None):
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
    def __init__(self, entries:list[dict], memory_modes:str, rendered_text:str, source_span:str|None=None, qid:str=None, canonical_target:str=None, alias_candidates:list[str]=None, entity_type:str|None=None, description:str|None=None):
        self.entries = entries
        self.memory_modes = memory_modes
        self.rendered_text = rendered_text
        self.source_span = source_span
        self.qid = qid
        self.canonical_target = canonical_target
        self.alias_candidates = alias_candidates

    def __call__(self, *args, **kwds):
        return self.entries, self.memory_modes, self.rendered_text, self.source_span, self.qid, self.canonical_target, self.alias_candidates
    
class TranslationRequest():
    """
    설명: 추론 파이프라인에 들어가는 요청 객체.
    비고: 요청 단위에서 공식 평가 모드와 일반화 모드를 함께 제어한다.
    """
    def __init__(self, target_lang:str, wikidata_id:str|None=None, allow_auxiliary_search:bool=None, apply_constrained_decoding:bool=None, apply_ercm:bool=None, max_candidates:int=None, alias_limit:int=None):
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
    def __init__(self, prompt_text:str, draft_text:str, raw_generation:str, used_memory:EntityMemoryBlock|None, used_logit_bias:dict[str,float]|None=None, generation_trace:dict[str,Any]=None):
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
    def __init__(self, should_run:bool, reasons:list[str], confidence:float|None=None, error_types:list[str]=None, verifier_score:float|None=None):
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
    def __init__(self, status:str, corrected_text:str|None, applied_edits:list[str]=None, verifier_score:float|None=None, selected:bool|None=None):
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
    def __init__(self, final_translation:str, draft_translation:str, used_ercm:bool, top_candidate:ScoredCandidate, memory:EntityMemoryBlock|None=None, ercm_result:ERCMCorrectionResult|None=None, debug:dict[str,Any]=None):
        self.final_translation = final_translation
        self.draft_translation = draft_translation
        self.used_ercm = used_ercm
        self.top_candidate = top_candidate
        self.memory = memory
        self.ercm_result = ercm_result
        self.debug = debug

    def __call__(self, *args, **kwds):
        return self.final_translation, self.draft_translation, self.used_ercm, self.top_candidate, self.memory, self.ercm_result, self.debug

"""
설명: 실행 시점에 주입되는 리소스와 설정 묶음.
비고: 파이프라인 함수는 모델과 인덱스를 전역으로 참조하지 않고 명시적으로 주입받는 것을 원칙으로 한다.
"""
class RuntimeResources():
    def __init__(self, qid_index:int, surface_index:int, reranker_mode, translator_model, tokenizer, ercm_model=None):
        self.qid_index = qid_index
        self.surface_index = surface_index
        self.reranker_mode = reranker_mode
        self.translator_model = translator_model
        self.tokenizer = tokenizer
        self.ercm_model = ercm_model

    def __call__(self, *args, **kwds):
        return self.qid_index, self.surface_index, self.reranker_mode, self.translator_model, self.tokenizer, self.ercm_model
    
class InferenceConfig():
    def __init__(self, bias_strength:float, ercm_threshold:float, auxilary_ngram_max:int, max_alias_bias_terms:int):
        self.bias_strength = bias_strength
        self.ercm_threshold = ercm_threshold
        self.auxilary_ngram_max = auxilary_ngram_max
        self.max_alias_bias_terms = max_alias_bias_terms

    def __call__(self, *args, **kwds):
        return self.bias_strength, self.ercm_threshold, self.auxilary_ngram_max, self.max_alias_bias_terms