from DTOlist import *
from memory.builder import build_entity_memory_block, render_entity_memory_text
from translation.prompting import build_translation_prompt
import re


"""
train_builders.py 역할:
학습용 샘플 생성 함수들을 모아두는 파일.

현재 구현:
- build_plain_translation_sample
- build_entity_memory_sample
- build_noisy_entity_memory_sample
- tokenize_train_sample

목적:
원본 학습 샘플(EAMTExample)과 resources를 받아
plain / entity-memory / noisy-memory SFT 학습 샘플을 생성한다.

추가 정책:
- alias_limit 적용
- description 길이 제한
- source sentence / canonical target 우선 보존
- tokenizer(max_length / truncation) 적용 가능
"""


DEFAULT_MAX_LENGTH = 2048
DEFAULT_ALIAS_LIMIT = 1
DEFAULT_DESCRIPTION_MAX_CHARS = 80

VALID_NOISE_TYPES = {
    "drop_alias",
    "drop_description",
    "replace_canonical_with_alias",
}


def _safe_str(value):
    """
    None이면 빈 문자열, 아니면 문자열로 바꿔 공백 제거.
    """
    if value is None:
        return ""
    return str(value).strip()


def _safe_list(values):
    """
    None / 문자열 / 리스트 입력을 모두 안전한 문자열 리스트로 변환.
    """
    if values is None:
        return []

    if isinstance(values, str):
        value = values.strip()
        return [value] if value else []

    if isinstance(values, list):
        cleaned = []
        for value in values:
            value_str = _safe_str(value)
            if value_str:
                cleaned.append(value_str)
        return cleaned

    return []


def _get_field(obj, field_name, default=None):
    """
    dict / object 둘 다 지원.
    """
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def _extract_primary_qid(example):
    """
    우선순위:
    1) example.wikidata_id
    2) example.entity_qids의 첫 번째 값
    """
    wikidata_id = _safe_str(_get_field(example, "wikidata_id"))
    if wikidata_id:
        return wikidata_id

    entity_qids = _get_field(example, "entity_qids", None)
    if entity_qids and isinstance(entity_qids, list) and len(entity_qids) > 0:
        return _safe_str(entity_qids[0])

    return ""


def _lookup_entity_record_by_qid(resources, qid):
    """
    resources.qid_index 또는 dict 형태의 resources["qid_index"]에서
    qid에 해당하는 엔티티 레코드를 가져온다.
    """
    if not qid:
        return None

    qid_index = _get_field(resources, "qid_index", None)
    if qid_index is None and isinstance(resources, dict):
        qid_index = resources.get("qid_index")

    if qid_index is None:
        return None

    if isinstance(qid_index, dict):
        return qid_index.get(qid)

    return None


def truncate_description(text, max_chars=DEFAULT_DESCRIPTION_MAX_CHARS):
    """
    description이 너무 길면 앞부분만 남기고 자른다.
    """
    text = _safe_str(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def _apply_memory_budget_to_record(
    record,
    alias_limit=DEFAULT_ALIAS_LIMIT,
    description_max_chars=DEFAULT_DESCRIPTION_MAX_CHARS,
):
    """
    record에 memory budget 정책을 적용한다.
    - alias 수 제한
    - description 길이 제한
    """
    if record is None:
        return None

    if isinstance(record, dict):
        new_record = dict(record)
    else:
        new_record = {
            "qid": _get_field(record, "qid"),
            "label_en": _get_field(record, "label_en"),
            "aliases_en": _get_field(record, "aliases_en", []),
            "target_lang": _get_field(record, "target_lang"),
            "target_label": _get_field(record, "target_label"),
            "target_aliases": _get_field(record, "target_aliases", []),
            "entity_type": _get_field(record, "entity_type"),
            "description": _get_field(record, "description"),
            "normalized_surfaces": _get_field(record, "normalized_surfaces", []),
            "language_available": _get_field(record, "language_available", True),
            "popularity_score": _get_field(record, "popularity_score", 0.0),
        }

    aliases = _safe_list(new_record.get("target_aliases", []))
    new_record["target_aliases"] = aliases[:alias_limit] if alias_limit >= 0 else aliases

    new_record["description"] = truncate_description(
        new_record.get("description", ""),
        max_chars=description_max_chars,
    )

    return new_record


def _find_span_in_source(source, candidates):
    """
    source 문장에서 candidate 문자열(label/alias)을 찾아
    원문 기준 span 문자열을 반환한다.
    못 찾으면 빈 문자열 반환.
    """
    source = _safe_str(source)
    if not source:
        return ""

    for cand in candidates:
        cand = _safe_str(cand)
        if not cand:
            continue

        pattern = re.compile(re.escape(cand), re.IGNORECASE)
        match = pattern.search(source)
        if match:
            return source[match.start():match.end()]

    return ""


def _resolve_source_span(source, record):
    """
    KBEntityRecord에서 label_en, aliases_en를 꺼내
    source 문장에서 가장 적절한 source_span을 찾는다.

    우선순위:
    1) label_en
    2) aliases_en
    """
    label_en = _safe_str(_get_field(record, "label_en"))
    aliases_en = _safe_list(_get_field(record, "aliases_en", []))

    candidates = []

    if label_en:
        candidates.append(label_en)

    for alias in aliases_en:
        if alias:
            candidates.append(alias)

    candidates = sorted(set(candidates), key=len, reverse=True)

    return _find_span_in_source(source, candidates)


def _record_to_candidate(record, source_sentence, qid):
    """
    KBEntityRecord 또는 dict 레코드를 CandidateEntity 형태로 변환한다.
    span alignment를 이용해 source_span을 채운다.
    """
    if record is None:
        return None

    target_label = _safe_str(_get_field(record, "target_label"))
    target_aliases = _safe_list(_get_field(record, "target_aliases", []))
    entity_type = _safe_str(_get_field(record, "entity_type"))
    description = _safe_str(_get_field(record, "description"))

    source_span = _resolve_source_span(source_sentence, record)

    candidate = CandidateEntity(
        qid=qid,
        candidate_source="gold_qid_lookup",
        target_label=target_label,
        target_aliases=target_aliases,
        entity_type=entity_type,
        description=description,
        popularity_score=0.0,
        alias_count=len(target_aliases),
        ambiguity_count=1,
        source_span=source_span,
        span_match=None,
    )

    return candidate


def _build_fallback_candidate(qid, source_sentence):
    """
    record lookup 실패 시 최소한의 정보로 candidate를 만든다.
    """
    return CandidateEntity(
        qid=qid,
        candidate_source="fallback_qid_only",
        target_label="",
        target_aliases=[],
        entity_type="",
        description="",
        popularity_score=0.0,
        alias_count=0,
        ambiguity_count=1,
        source_span="",
        span_match=None,
    )


def _clone_memory_block(memory):
    """
    EntityMemoryBlock을 얕은 복사하여 noisy 버전을 만들기 위한 헬퍼.
    entries 안의 dict도 복사한다.
    """
    cloned_entries = []

    for entry in memory.entries:
        cloned_entries.append(dict(entry))

    cloned = EntityMemoryBlock(
        entries=cloned_entries,
        memory_modes=memory.memory_modes,
        rendered_text="",
        source_span=memory.source_span,
        qid=memory.qid,
        canonical_target=memory.canonical_target,
        alias_candidates=list(memory.alias_candidates) if memory.alias_candidates else [],
        entity_type=memory.entity_type,
        description=memory.description,
    )

    if hasattr(memory, "source_sentence"):
        cloned.source_sentence = memory.source_sentence

    return cloned


def _validate_noise_type(noise_type):
    """
    지원하지 않는 noise_type이면 명확한 에러를 낸다.
    """
    if noise_type not in VALID_NOISE_TYPES:
        raise ValueError(
            f"Unsupported noise_type: {noise_type}. "
            f"Valid options: {sorted(VALID_NOISE_TYPES)}"
        )


def _apply_memory_noise(memory, noise_type="drop_alias"):
    """
    memory에 간단한 rule-based noise를 적용한다.

    지원 noise_type:
    - drop_alias
    - drop_description
    - replace_canonical_with_alias
    """
    _validate_noise_type(noise_type)

    noisy_memory = _clone_memory_block(memory)

    if not noisy_memory.entries:
        return noisy_memory

    entry = noisy_memory.entries[0]

    if noise_type == "drop_alias":
        entry["alias_candidates"] = []
        noisy_memory.alias_candidates = []

    elif noise_type == "drop_description":
        entry["description"] = ""
        noisy_memory.description = ""

    elif noise_type == "replace_canonical_with_alias":
        aliases = entry.get("alias_candidates", [])
        if aliases:
            entry["canonical_target"] = aliases[0]
            noisy_memory.canonical_target = aliases[0]

    return noisy_memory


def _prepare_candidate_and_memory(
    example,
    resources,
    alias_limit=DEFAULT_ALIAS_LIMIT,
    description_max_chars=DEFAULT_DESCRIPTION_MAX_CHARS,
):
    """
    example/resources로부터 candidate와 clean memory를 준비한다.
    lookup 실패 시 fallback candidate를 사용한다.
    """
    source = _safe_str(_get_field(example, "source"))
    target_lang = _safe_str(_get_field(example, "target_lang"))
    qid = _extract_primary_qid(example)

    record = _lookup_entity_record_by_qid(resources, qid)
    record = _apply_memory_budget_to_record(
        record,
        alias_limit=alias_limit,
        description_max_chars=description_max_chars,
    )

    candidate = _record_to_candidate(record, source_sentence=source, qid=qid)

    if candidate is None:
        candidate = _build_fallback_candidate(qid=qid, source_sentence=source)

    memory = build_entity_memory_block(
        [candidate],
        source_sentence=source,
        alias_limit=alias_limit,
    )
    render_entity_memory_text(memory, target_lang=target_lang)

    return candidate, memory


def build_plain_translation_sample(example):
    """
    plain translation SFT 학습 샘플 1개 생성.

    출력:
        {
            "prompt": str,
            "target": str,
            "mode": "plain"
        }
    """
    source = _safe_str(_get_field(example, "source"))
    target = _safe_str(_get_field(example, "target"))
    target_lang = _safe_str(_get_field(example, "target_lang"))

    prompt = build_translation_prompt(
        source=source,
        target_lang=target_lang,
        memory=None,
        mode="plain",
    )

    sample = {
        "prompt": prompt,
        "target": target,
        "mode": "plain",
    }

    return sample


def build_entity_memory_sample(
    example,
    resources,
    alias_limit=DEFAULT_ALIAS_LIMIT,
    description_max_chars=DEFAULT_DESCRIPTION_MAX_CHARS,
):
    """
    entity-memory SFT 학습 샘플 1개 생성.

    출력:
        {
            "prompt": str,
            "target": str,
            "mode": "entity-memory"
        }
    """
    source = _safe_str(_get_field(example, "source"))
    target = _safe_str(_get_field(example, "target"))
    target_lang = _safe_str(_get_field(example, "target_lang"))

    _, memory = _prepare_candidate_and_memory(
        example=example,
        resources=resources,
        alias_limit=alias_limit,
        description_max_chars=description_max_chars,
    )

    prompt = build_translation_prompt(
        source=source,
        target_lang=target_lang,
        memory=memory,
        mode="entity-aware",
    )

    sample = {
        "prompt": prompt,
        "target": target,
        "mode": "entity-memory",
    }

    return sample


def build_noisy_entity_memory_sample(
    example,
    resources,
    alias_limit=DEFAULT_ALIAS_LIMIT,
    description_max_chars=DEFAULT_DESCRIPTION_MAX_CHARS,
    noise_type="drop_alias",
):
    """
    noisy-memory SFT 학습 샘플 1개 생성.

    지원 noise_type:
    - drop_alias
    - drop_description
    - replace_canonical_with_alias

    출력:
        {
            "prompt": str,
            "target": str,
            "mode": "noisy-memory"
        }
    """
    source = _safe_str(_get_field(example, "source"))
    target = _safe_str(_get_field(example, "target"))
    target_lang = _safe_str(_get_field(example, "target_lang"))

    _, clean_memory = _prepare_candidate_and_memory(
        example=example,
        resources=resources,
        alias_limit=alias_limit,
        description_max_chars=description_max_chars,
    )

    noisy_memory = _apply_memory_noise(clean_memory, noise_type=noise_type)
    render_entity_memory_text(noisy_memory, target_lang=target_lang)

    prompt = build_translation_prompt(
        source=source,
        target_lang=target_lang,
        memory=noisy_memory,
        mode="entity-aware",
    )

    sample = {
        "prompt": prompt,
        "target": target,
        "mode": "noisy-memory",
    }

    return sample


def tokenize_train_sample(sample, tokenizer, max_length=DEFAULT_MAX_LENGTH):
    """
    학습 샘플 하나를 tokenizer로 인코딩한다.

    입력:
        sample: {"prompt": ..., "target": ..., "mode": ...}
        tokenizer: Qwen tokenizer
        max_length: 최대 입력 길이

    출력:
        dict with input_ids, attention_mask
    """
    prompt = _safe_str(sample.get("prompt"))
    target = _safe_str(sample.get("target"))

    full_text = prompt + "\n" + target if target else prompt

    encoded = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    return encoded