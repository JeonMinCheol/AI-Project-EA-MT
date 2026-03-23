from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..kb.index import lookup_entity_by_qid, search_surface_candidates
from .align import SourceSpanMatch, align_source_span, extract_candidate_spans_from_source


@dataclass
class RetrievedCandidate:
    qid: str
    target_label: str
    target_aliases: List[str]
    entity_type: str
    description: str
    popularity_score: float
    alias_count: int
    ambiguity_count: int
    candidate_source: str
    span_match: Optional[SourceSpanMatch] = None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [_safe_text(v) for v in value if _safe_text(v)]
    return []


def _get_resource_attr(resources: Any, names: Sequence[str]) -> Any:
    if resources is None:
        return None

    if isinstance(resources, dict):
        for name in names:
            if name in resources:
                return resources[name]

    for name in names:
        if hasattr(resources, name):
            return getattr(resources, name)

    return None


def _get_qid_index(resources: Any) -> Any:
    return _get_resource_attr(resources, ["qid_index", "kb_qid_index"])


def _get_surface_index(resources: Any) -> Any:
    return _get_resource_attr(resources, ["surface_index", "kb_surface_index"])


def _extract_example_source(example: Any) -> str:
    return (
        _safe_text(getattr(example, "source", None))
        or _safe_text(getattr(example, "source_text", None))
        or _safe_text(getattr(example, "sentence", None))
        or ""
    )


def _extract_example_qid(example: Any) -> Optional[str]:
    qid = (
        _safe_text(getattr(example, "wikidata_id", None))
        or _safe_text(getattr(example, "qid", None))
        or ""
    )
    return qid or None


def _extract_source_label(record: Any) -> str:
    return (
        _safe_text(getattr(record, "label_en", None))
        or _safe_text(getattr(record, "source_label", None))
        or _safe_text(getattr(record, "label", None))
        or _safe_text(getattr(record, "target_label", None))
        or ""
    )


def _extract_source_aliases(record: Any) -> List[str]:
    for field_name in ("aliases_en", "source_aliases", "aliases", "target_aliases"):
        value = getattr(record, field_name, None)
        aliases = _safe_list(value)
        if aliases:
            return aliases
    return []


def _extract_target_label(record: Any) -> str:
    return (
        _safe_text(getattr(record, "target_label", None))
        or _safe_text(getattr(record, "label_ko", None))
        or _safe_text(getattr(record, "label", None))
        or _safe_text(getattr(record, "label_en", None))
        or ""
    )


def _extract_target_aliases(record: Any) -> List[str]:
    for field_name in ("target_aliases", "aliases_ko", "aliases", "aliases_en"):
        value = getattr(record, field_name, None)
        aliases = _safe_list(value)
        if aliases:
            return aliases
    return []


def _extract_entity_type(record: Any) -> str:
    return (
        _safe_text(getattr(record, "entity_type", None))
        or _safe_text(getattr(record, "type", None))
        or ""
    )


def _extract_description(record: Any) -> str:
    return (
        _safe_text(getattr(record, "description", None))
        or _safe_text(getattr(record, "desc", None))
        or ""
    )


def _extract_qid(record: Any) -> str:
    return (
        _safe_text(getattr(record, "qid", None))
        or _safe_text(getattr(record, "wikidata_id", None))
        or ""
    )


def _extract_popularity(record: Any) -> float:
    for field_name in ("popularity_score", "popularity", "prior_popularity"):
        value = getattr(record, field_name, None)
        try:
            if value is not None:
                return float(value)
        except Exception:
            pass
    return 0.0


def _extract_alias_count(record: Any, aliases: List[str]) -> int:
    value = getattr(record, "alias_count", None)
    try:
        if value is not None:
            return int(value)
    except Exception:
        pass
    return len(aliases)


def _extract_ambiguity_count(record: Any) -> int:
    for field_name in ("ambiguity_count", "ambiguity", "ambiguity_score"):
        value = getattr(record, field_name, None)
        try:
            if value is not None:
                return int(float(value))
        except Exception:
            pass
    return 0


def _make_candidate_from_record(
    record: Any,
    candidate_source: str,
    span_match: Optional[SourceSpanMatch] = None,
) -> RetrievedCandidate:
    target_aliases = _extract_target_aliases(record)

    return RetrievedCandidate(
        qid=_extract_qid(record),
        target_label=_extract_target_label(record),
        target_aliases=target_aliases,
        entity_type=_extract_entity_type(record),
        description=_extract_description(record),
        popularity_score=_extract_popularity(record),
        alias_count=_extract_alias_count(record, target_aliases),
        ambiguity_count=_extract_ambiguity_count(record),
        candidate_source=candidate_source,
        span_match=span_match,
    )


def _attach_span_and_source(candidate: Any, span_text: str, source: str) -> Any:
    """
    kb/index.py 에서 이미 CandidateEntity 류 객체가 반환되는 경우, span_match / candidate_source만 덧붙입니다.
    """
    if candidate is None:
        return None

    label = (
        _safe_text(getattr(candidate, "target_label", None))
        or _safe_text(getattr(candidate, "label", None))
        or ""
    )
    aliases = _safe_list(
        getattr(candidate, "target_aliases", None) or getattr(candidate, "aliases", None)
    )

    span_match = align_source_span(source=source, label=span_text or label, aliases=aliases)

    try:
        candidate.candidate_source = "surface_retrieval"
    except Exception:
        pass

    try:
        candidate.span_match = span_match
        return candidate
    except Exception:
        return _make_candidate_from_record(
            record=candidate,
            candidate_source="surface_retrieval",
            span_match=span_match,
        )


def _dedupe_by_qid(candidates: Iterable[Any]) -> List[Any]:
    results: List[Any] = []
    seen = set()

    for candidate in candidates:
        qid = _safe_text(getattr(candidate, "qid", None) or getattr(candidate, "wikidata_id", None))
        if not qid or qid in seen:
            continue
        seen.add(qid)
        results.append(candidate)

    return results


def collect_entity_candidates(
    example: Any,
    resources: Any,
    top_k: int = 10,
    per_surface_k: int = 5,
    min_char_len: int = 2,
    max_n: int = 5,
) -> List[Any]:
    """
    retrieval 메인 진입점
    - QID가 있으면 direct lookup 우선
    - source span 후보를 SALT 스타일로 생성
    - surface 기반 candidate top-K를 추가 수집
    """
    source = _extract_example_source(example)
    gold_qid = _extract_example_qid(example)

    qid_index = _get_qid_index(resources)
    surface_index = _get_surface_index(resources)

    collected: List[Any] = []

    # 1) QID anchored lookup 우선
    if gold_qid and qid_index is not None:
        record = lookup_entity_by_qid(gold_qid, qid_index)
        if record is not None:
            source_label = _extract_source_label(record)
            source_aliases = _extract_source_aliases(record)
            span_match = align_source_span(
                source=source,
                label=source_label,
                aliases=source_aliases,
                min_char_len=min_char_len,
            )
            collected.append(
                _make_candidate_from_record(
                    record=record,
                    candidate_source="qid_lookup",
                    span_match=span_match,
                )
            )

    # 2) SALT 스타일 surface candidate retrieval
    if surface_index is not None and source:
        candidate_spans = extract_candidate_spans_from_source(
            source=source,
            min_n=1,
            max_n=max_n,
            min_char_len=min_char_len,
        )

        for span in candidate_spans:
            candidates = search_surface_candidates(
                surface=span,
                surface_index=surface_index,
                max_candidates=per_surface_k,
            )
            for candidate in candidates:
                attached = _attach_span_and_source(candidate, span_text=span, source=source)
                if attached is not None:
                    collected.append(attached)

    collected = _dedupe_by_qid(collected)
    return collected[:top_k]


__all__ = [
    "RetrievedCandidate",
    "collect_entity_candidates",
]