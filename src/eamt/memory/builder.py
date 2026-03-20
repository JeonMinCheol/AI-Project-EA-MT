from DTOlist import *


"""
builder.py 역할:
앞 단계(index / retrieval / reranker)에서 넘어온 CandidateEntity 또는
ScoredCandidate를 받아 top-1 후보 중심의 EntityMemoryBlock을 생성하고,
prompt에 넣을 문자열로 렌더링한다.

현재 버전:
- source_sentence 전체를 추가로 받을 수 있음
- DTOlist는 아직 수정하지 않으므로, source_sentence는
  entry 내부와 memory 객체의 동적 속성으로 저장
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
    CandidateEntity 또는 ScoredCandidate를 받아 실제 CandidateEntity를 반환한다.
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


def build_entity_memory_block(top_candidates, source_sentence="", alias_limit=1):
    """
    top-1 후보를 중심으로 EntityMemoryBlock 생성.

    입력:
        top_candidates: CandidateEntity 또는 ScoredCandidate 리스트
        source_sentence: 원문 전체 문장
        alias_limit: alias 최대 포함 개수

    출력:
        EntityMemoryBlock 객체

    정책:
        - top-1 후보만 사용
        - canonical target 중심
        - alias는 소수만 포함
        - 필요 시 qid, entity_type, description 유지
        - source_sentence 전체와 source_span 둘 다 보존
    """
    source_sentence = _safe_str(source_sentence)

    if not top_candidates:
        memory_block = EntityMemoryBlock(
            entries=[],
            memory_modes="canonical_only",
            rendered_text="",
            source_span="",
            qid="",
            canonical_target="",
            alias_candidates=[],
            entity_type="",
            description="",
        )
        # DTO에 아직 없으므로 동적으로 붙임
        memory_block.source_sentence = source_sentence
        return memory_block

    top_candidate_like = top_candidates[0]
    candidate = _unwrap_candidate(top_candidate_like)

    source_span = _extract_source_span(candidate)
    qid = _safe_str(candidate.qid)
    canonical_target = _safe_str(candidate.target_label)
    alias_candidates = _safe_list(candidate.target_aliases)
    entity_type = _safe_str(candidate.entity_type)
    description = _safe_str(candidate.description)

    # canonical target과 같은 alias 제거
    alias_candidates = [
        alias for alias in alias_candidates
        if alias and alias != canonical_target
    ]

    # 중복 제거 + 순서 유지
    alias_candidates = _unique_keep_order(alias_candidates)

    # alias 개수 제한
    if alias_limit >= 0:
        alias_candidates = alias_candidates[:alias_limit]

    entry = {
        "source_sentence": source_sentence,
        "source_span": source_span,
        "qid": qid,
        "canonical_target": canonical_target,
        "alias_candidates": alias_candidates,
        "entity_type": entity_type,
        "description": description,
    }

    memory_block = EntityMemoryBlock(
        entries=[entry],
        memory_modes="canonical_plus_alias" if alias_limit != 0 else "canonical_only",
        rendered_text="",
        source_span=source_span,
        qid=qid,
        canonical_target=canonical_target,
        alias_candidates=alias_candidates,
        entity_type=entity_type,
        description=description,
    )

    # DTO에 아직 source_sentence 필드가 없으므로 동적으로 붙임
    memory_block.source_sentence = source_sentence

    return memory_block


def render_entity_memory_text(memory, target_lang):
    """
    EntityMemoryBlock을 prompt에 삽입 가능한 문자열로 렌더링한다.

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

    # memory block 레벨 source_sentence가 있으면 같이 보여줌
    source_sentence = _safe_str(getattr(memory, "source_sentence", ""))
    if source_sentence:
        lines.append(f"- Source Sentence: {source_sentence}")

    for entry in memory.entries:
        source_span = _safe_str(entry.get("source_span"))
        qid = _safe_str(entry.get("qid"))
        canonical_target = _safe_str(entry.get("canonical_target"))
        alias_candidates = _safe_list(entry.get("alias_candidates"))
        entity_type = _safe_str(entry.get("entity_type"))
        description = _safe_str(entry.get("description"))

        if source_span:
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