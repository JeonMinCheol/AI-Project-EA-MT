from DTOlist import *


"""
builder.py 역할:
앞 단계(index / retrieval / reranker)에서 넘어온 CandidateEntity 또는
ScoredCandidate를 받아 EntityMemoryBlock을 생성하고,
prompt에 넣을 문자열로 렌더링한다.
"""


def _safe_str(value):
    """
    None이면 빈 문자열, 아니면 문자열로 바꿔서 양쪽 공백 제거.
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


def _unique_keep_order(values):
    """
    중복 제거하되 원래 순서는 유지.
    """
    seen = set()
    results = []

    for value in values:
        if value not in seen:
            seen.add(value)
            results.append(value)

    return results


def _unwrap_candidate(candidate_like):
    """
    CandidateEntity 또는 ScoredCandidate를 받아
    실제 CandidateEntity를 반환한다.

    - CandidateEntity면 그대로 반환
    - ScoredCandidate면 .candidate 반환
    """
    if isinstance(candidate_like, ScoredCandidate):
        return candidate_like.candidate
    return candidate_like


def _extract_source_span(candidate):
    """
    source_span 추출.
    우선순위:
    1) candidate.source_span
    2) candidate.span_match.source_span
    3) 없으면 빈 문자열
    """
    if candidate.source_span:
        return _safe_str(candidate.source_span)

    if candidate.span_match is not None and candidate.span_match.source_span:
        return _safe_str(candidate.span_match.source_span)

    return ""


def build_entity_memory_block(top_candidates, alias_limit=1):
    """
    앞 단계에서 넘어온 후보 리스트를 받아 EntityMemoryBlock 생성.

    입력:
        top_candidates: CandidateEntity 또는 ScoredCandidate 리스트
        alias_limit: alias 최대 포함 개수

    출력:
        EntityMemoryBlock 객체
    """
    entries = []

    memory_source_span = None
    memory_qid = None
    memory_canonical_target = None
    memory_alias_candidates = []
    memory_entity_type = None
    memory_description = None

    for idx, candidate_like in enumerate(top_candidates):
        candidate = _unwrap_candidate(candidate_like)

        source_span = _extract_source_span(candidate)
        qid = _safe_str(candidate.qid)
        canonical_target = _safe_str(candidate.target_label)
        alias_candidates = _safe_list(candidate.target_aliases)
        entity_type = _safe_str(candidate.entity_type)
        description = _safe_str(candidate.description)

        # canonical_target과 같은 alias 제거
        alias_candidates = [
            alias for alias in alias_candidates
            if alias and alias != canonical_target
        ]

        alias_candidates = _unique_keep_order(alias_candidates)

        if alias_limit >= 0:
            alias_candidates = alias_candidates[:alias_limit]

        entry = {
            "source_span": source_span,
            "qid": qid,
            "canonical_target": canonical_target,
            "alias_candidates": alias_candidates,
            "entity_type": entity_type,
            "description": description,
        }
        entries.append(entry)

        # 첫 번째(top-1) 후보 기준으로 block-level 요약 필드 채움
        if idx == 0:
            memory_source_span = source_span
            memory_qid = qid
            memory_canonical_target = canonical_target
            memory_alias_candidates = alias_candidates
            memory_entity_type = entity_type
            memory_description = description

    memory_block = EntityMemoryBlock(
        entries=entries,
        memory_modes="canonical_plus_alias" if alias_limit != 0 else "canonical_only",
        rendered_text="",
        source_span=memory_source_span,
        qid=memory_qid,
        canonical_target=memory_canonical_target,
        alias_candidates=memory_alias_candidates,
        entity_type=memory_entity_type,
        description=memory_description,
    )

    return memory_block


def render_entity_memory_text(memory, target_lang):
    """
    EntityMemoryBlock을 prompt에 삽입할 문자열로 렌더링한다.

    입력:
        memory: EntityMemoryBlock
        target_lang: 목표 언어 코드 (예: ko, ja, zh)

    출력:
        prompt용 문자열
    """
    target_lang = _safe_str(target_lang)

    if memory is None or not memory.entries:
        rendered_text = "[ENTITY MEMORY]\n- None"
        if memory is not None:
            memory.rendered_text = rendered_text
        return rendered_text

    lines = ["[ENTITY MEMORY]"]

    for entry in memory.entries:
        source_span = _safe_str(entry.get("source_span"))
        qid = _safe_str(entry.get("qid"))
        canonical_target = _safe_str(entry.get("canonical_target"))
        alias_candidates = _safe_list(entry.get("alias_candidates"))
        entity_type = _safe_str(entry.get("entity_type"))
        description = _safe_str(entry.get("description"))

        lines.append(f"- Source Span: {source_span}")

        if qid:
            lines.append(f"  QID: {qid}")

        if canonical_target:
            lines.append(f"  Target ({target_lang}): {canonical_target}")

        if alias_candidates:
            lines.append(f"  Aliases: {', '.join(alias_candidates)}")

        if entity_type:
            lines.append(f"  Type: {entity_type}")

        if description:
            lines.append(f"  Description: {description}")

    rendered_text = "\n".join(lines)
    memory.rendered_text = rendered_text
    return rendered_text