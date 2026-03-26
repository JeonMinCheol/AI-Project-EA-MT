from __future__ import annotations

<<<<<<< HEAD
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


=======
from typing import Any, Iterable, List, Optional, Sequence

from src.DTOlist import CandidateEntity, KBEntityRecord, SpanMatch
from src.eamt.kb.index import (
    lookup_entity_by_qid,
    normalize_surface,
    search_surface_candidates,
)
from .align import SourceSpanMatch, align_source_span, extract_candidate_spans_from_source


>>>>>>> upstream/main
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


<<<<<<< HEAD
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
=======
def _extract_source(example: Any) -> str:
>>>>>>> upstream/main
    return (
        _safe_text(getattr(example, "source", None))
        or _safe_text(getattr(example, "source_text", None))
        or _safe_text(getattr(example, "sentence", None))
        or ""
    )


<<<<<<< HEAD
def _extract_example_qid(example: Any) -> Optional[str]:
    qid = (
        _safe_text(getattr(example, "wikidata_id", None))
        or _safe_text(getattr(example, "qid", None))
        or ""
=======
def _extract_target_lang(example: Any, resources: Any) -> Optional[str]:
    lang = (
        _safe_text(getattr(example, "target_lang", None))
        or _safe_text(getattr(example, "target_locale", None))
    )
    if lang:
        return lang

    if isinstance(resources, dict):
        return _safe_text(resources.get("target_lang") or resources.get("default_target_lang")) or None

    return (
        _safe_text(getattr(resources, "target_lang", None))
        or _safe_text(getattr(resources, "default_target_lang", None))
        or None
    )


def _extract_gold_qid(example: Any) -> Optional[str]:
    qid = (
        _safe_text(getattr(example, "wikidata_id", None))
        or _safe_text(getattr(example, "qid", None))
>>>>>>> upstream/main
    )
    return qid or None


<<<<<<< HEAD
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
=======
def _get_qid_index(resources: Any) -> Any:
    if isinstance(resources, dict):
        return resources.get("qid_index")
    return getattr(resources, "qid_index", None)


def _get_surface_index(resources: Any) -> Any:
    if isinstance(resources, dict):
        return resources.get("surface_index")
    return getattr(resources, "surface_index", None)


def _to_span_match(match: Optional[SourceSpanMatch]) -> Optional[SpanMatch]:
    if match is None:
        return None
    return SpanMatch(
        source_span=match.matched_text,
        char_start=match.start,
        char_end=match.end,
        match_kind=match.match_kind,
        matched_surface=match.normalized_text,
        match_score=match.match_score,
    )


def _ambiguity_count(surface_index: Any, span_text: str) -> int:
    if not surface_index or not span_text:
        return 0
    normalized = normalize_surface(span_text)
    records = surface_index.get(normalized, [])
    return len(records) if isinstance(records, list) else 0


def _best_record(records: Optional[list[KBEntityRecord]]) -> Optional[KBEntityRecord]:
    if not records:
        return None
    return sorted(records, key=lambda r: getattr(r, "popularity_score", 0.0), reverse=True)[0]


def _record_to_candidate(
    record: KBEntityRecord,
    candidate_source: str,
    source_span: Optional[str],
    span_match: Optional[SpanMatch],
    ambiguity_count: int,
) -> CandidateEntity:
    target_aliases = _safe_list(getattr(record, "target_aliases", []))
    return CandidateEntity(
        qid=_safe_text(record.qid),
        candidate_source=candidate_source,
        target_label=_safe_text(getattr(record, "target_label", None)) or None,
        target_aliases=target_aliases,
        entity_type=_safe_text(getattr(record, "entity_type", None)) or None,
        description=_safe_text(getattr(record, "description", None)) or None,
        popularity_score=float(getattr(record, "popularity_score", 0.0) or 0.0),
        alias_count=len(target_aliases),
        ambiguity_count=ambiguity_count,
        source_span=source_span,
>>>>>>> upstream/main
        span_match=span_match,
    )


<<<<<<< HEAD
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
=======
def _dedupe_by_qid(candidates: Iterable[CandidateEntity]) -> List[CandidateEntity]:
    results: List[CandidateEntity] = []
    seen = set()

    for candidate in candidates:
        qid = _safe_text(getattr(candidate, "qid", None))
>>>>>>> upstream/main
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
<<<<<<< HEAD
) -> List[Any]:
    """
    retrieval 메인 진입점
    - QID가 있으면 direct lookup 우선
    - source span 후보를 SALT 스타일로 생성
    - surface 기반 candidate top-K를 추가 수집
    """
    source = _extract_example_source(example)
    gold_qid = _extract_example_qid(example)
=======
) -> List[CandidateEntity]:
    """
    - QID가 있으면 direct lookup 우선
    - source에서 SALT 스타일 span 후보 생성
    - surface 기반 candidate top-K 추가 수집
    """
    source = _extract_source(example)
    target_lang = _extract_target_lang(example, resources)
    gold_qid = _extract_gold_qid(example)

    if not source or not target_lang:
        return []
>>>>>>> upstream/main

    qid_index = _get_qid_index(resources)
    surface_index = _get_surface_index(resources)

<<<<<<< HEAD
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
=======
    collected: List[CandidateEntity] = []

    # 1) anchored QID lookup
    if gold_qid and qid_index is not None:
        records = lookup_entity_by_qid(qid_index, gold_qid, target_lang)
        record = _best_record(records)
        if record is not None:
            raw_match = align_source_span(
                source=source,
                label=_safe_text(getattr(record, "label_en", None)),
                aliases=_safe_list(getattr(record, "aliases_en", [])),
                min_char_len=min_char_len,
            )
            span_match = _to_span_match(raw_match)
            source_span = span_match.source_span if span_match else None
            collected.append(
                _record_to_candidate(
                    record=record,
                    candidate_source="anchored_qid",
                    source_span=source_span,
                    span_match=span_match,
                    ambiguity_count=_ambiguity_count(surface_index, source_span or ""),
                )
            )

    # 2) SALT-style surface search
    if surface_index is not None:
        spans = extract_candidate_spans_from_source(
>>>>>>> upstream/main
            source=source,
            min_n=1,
            max_n=max_n,
            min_char_len=min_char_len,
        )

<<<<<<< HEAD
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
=======
        for span in spans:
            candidates = search_surface_candidates(
                surface_index=surface_index,
                surfaces=[span],
                target_lang=target_lang,
                max_candidates=per_surface_k,
            )

            raw_match = align_source_span(source=source, label=span, aliases=None, min_char_len=min_char_len)
            span_match = _to_span_match(raw_match)

            for candidate in candidates:
                candidate.source_span = span
                candidate.span_match = span_match
                candidate.candidate_source = "surface_search"
                if not getattr(candidate, "ambiguity_count", None):
                    candidate.ambiguity_count = _ambiguity_count(surface_index, span)
                collected.append(candidate)
>>>>>>> upstream/main

    collected = _dedupe_by_qid(collected)
    return collected[:top_k]


__all__ = [
<<<<<<< HEAD
    "RetrievedCandidate",
=======
>>>>>>> upstream/main
    "collect_entity_candidates",
]