from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Union

FeatureValue = Union[float, int, str]

# 모델 입력에 사용하는 숫자형 feature의 고정 순서입니다.
# 주의:
# - is_canonical_prior / prior_bonus 는 학습 데이터 누수를 막기 위해
#   numeric vector에서 제외합니다.
NUMERIC_FEATURE_KEYS: List[str] = [
    "context_suitability",
    "has_source_span",
    "span_match_score",
    "surface_overlap_score",
    "description_overlap_score",
    "target_label_len",
    "alias_count",
    "has_entity_type",
    "has_description",
    "popularity_score",
    "ambiguity_count",
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]+")


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_text(text: str) -> str:
    text = _safe_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(_normalize_text(text))


def _token_overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(_tokenize(a))
    b_tokens = set(_tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _substring_hit_score(source: str, target: str) -> float:
    source_n = _normalize_text(source)
    target_n = _normalize_text(target)
    if not source_n or not target_n:
        return 0.0
    if target_n in source_n or source_n in target_n:
        return 1.0
    return 0.0


def _first_non_empty_text(obj: Any, field_names: Sequence[str], default: str = "") -> str:
    for name in field_names:
        value = _safe_getattr(obj, name, None)
        if _safe_text(value):
            return _safe_text(value)
    return default


def _extract_target_label(candidate: Any) -> str:
    return _first_non_empty_text(
        candidate,
        ["target_label", "label", "canonical_label", "name", "title"],
        default="",
    )


def _extract_entity_type(candidate: Any) -> str:
    return _first_non_empty_text(
        candidate,
        ["entity_type", "type", "candidate_type"],
        default="",
    )


def _extract_description(candidate: Any) -> str:
    return _first_non_empty_text(
        candidate,
        ["description", "desc", "summary"],
        default="",
    )


def _extract_qid(candidate: Any) -> str:
    return _first_non_empty_text(
        candidate,
        ["qid", "wikidata_id", "candidate_qid", "id"],
        default="",
    )


def _extract_aliases(candidate: Any) -> List[str]:
    raw_aliases = (
        _safe_getattr(candidate, "aliases", None)
        or _safe_getattr(candidate, "alias_list", None)
        or _safe_getattr(candidate, "target_aliases", None)
        or _safe_getattr(candidate, "surface_forms", None)
        or []
    )

    if isinstance(raw_aliases, str):
        aliases = [part.strip() for part in raw_aliases.split(",") if part.strip()]
    elif isinstance(raw_aliases, (list, tuple, set)):
        aliases = [_safe_text(v) for v in raw_aliases if _safe_text(v)]
    else:
        aliases = []

    # 중복 제거 + 순서 보존
    deduped: List[str] = []
    seen = set()
    for alias in aliases:
        norm = _normalize_text(alias)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(alias)
    return deduped


def _extract_span_text(candidate: Any) -> str:
    span_match = _safe_getattr(candidate, "span_match", None)
    if span_match is None:
        return ""

    return _first_non_empty_text(
        span_match,
        ["matched_text", "surface", "span_text", "text", "matched_surface"],
        default="",
    )


def _extract_alias_count(candidate: Any, aliases: List[str]) -> int:
    explicit_count = _safe_getattr(candidate, "alias_count", None)
    if isinstance(explicit_count, (int, float)) and explicit_count >= 0:
        return int(explicit_count)
    return len(aliases)


def _extract_float(candidate: Any, name: str, default: float = 0.0) -> float:
    value = _safe_getattr(candidate, name, default)
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _extract_popularity(candidate: Any) -> float:
    for name in ("popularity_score", "popularity", "prior_popularity"):
        value = _safe_getattr(candidate, name, None)
        try:
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return max(0.0, float(value))
        except Exception:
            pass
    return 0.0


def _extract_ambiguity(candidate: Any) -> float:
    for name in ("ambiguity_count", "ambiguity", "ambiguity_score"):
        value = _safe_getattr(candidate, name, None)
        try:
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return max(0.0, float(value))
        except Exception:
            pass
    return 0.0


def _extract_span_match_score(candidate: Any) -> float:
    span_match = _safe_getattr(candidate, "span_match", None)
    if span_match is None:
        return 0.0

    value = _safe_getattr(span_match, "match_score", 0.0)
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _extract_span_match_kind(candidate: Any) -> str:
    span_match = _safe_getattr(candidate, "span_match", None)
    if span_match is None:
        return "none"
    return _safe_text(_safe_getattr(span_match, "match_kind", "none")) or "none"


def _compute_surface_overlap(
    comparison_source: str,
    candidate: Any,
    target_label: str,
    aliases: List[str],
) -> float:
    span_text = _extract_span_text(candidate)
    comparison_pool = [target_label, span_text] + aliases
    scores: List[float] = []

    for text in comparison_pool:
        if not text:
            continue
        scores.append(_token_overlap_ratio(comparison_source, text))
        scores.append(_substring_hit_score(comparison_source, text))

    return max(scores) if scores else 0.0


def _compute_description_overlap(source: str, description: str) -> float:
    if not description:
        return 0.0
    return _token_overlap_ratio(source, description)


def _compute_context_suitability(
    source: str,
    comparison_source: str,
    candidate: Any,
    target_label: str,
    description: str,
    aliases: List[str],
) -> float:
    surface_overlap = _compute_surface_overlap(comparison_source, candidate, target_label, aliases)
    description_overlap = _compute_description_overlap(source, description)
    span_match_score = _extract_span_match_score(candidate)
    return round((0.50 * surface_overlap) + (0.20 * description_overlap) + (0.30 * span_match_score), 6)


def build_candidate_feature_vector(
    source: str,
    candidate: Any,
    canonical_qid: Optional[str] = None,
    source_span: Optional[str] = None,
) -> Dict[str, FeatureValue]:
    """
    후보 엔티티의 특징을 추출합니다.

    반환값은 다음 두 용도를 동시에 만족하도록 구성했습니다.
    1) 모델 입력용 숫자형 feature
    2) 디버깅/로그/ablation 용 메타데이터

    주의:
    - canonical_qid는 추론 시 prior로만 약하게 활용하는 용도입니다.
    - 학습 데이터 생성 시에는 canonical_qid=None 으로 호출하세요.
    """
    target_label = _extract_target_label(candidate)
    entity_type = _extract_entity_type(candidate)
    description = _extract_description(candidate)
    qid = _extract_qid(candidate)
    candidate_source = _safe_text(_safe_getattr(candidate, "candidate_source", "unknown")) or "unknown"
    aliases = _extract_aliases(candidate)

    span_text_from_candidate = _extract_span_text(candidate)
    source_span_text = _safe_text(source_span)

    if source_span_text:
        comparison_source = source_span_text
        comparison_source_origin = "source_span"
    elif span_text_from_candidate:
        comparison_source = span_text_from_candidate
        comparison_source_origin = "candidate_span_match"
    else:
        comparison_source = _safe_text(source)
        comparison_source_origin = "full_source"

    has_span = 1 if (source_span_text or _safe_getattr(candidate, "span_match", None) is not None) else 0
    span_match_kind = _extract_span_match_kind(candidate)
    span_match_score = _extract_span_match_score(candidate)

    raw_alias_count = _extract_alias_count(candidate, aliases)
    raw_popularity_score = _extract_popularity(candidate)
    raw_ambiguity_count = _extract_ambiguity(candidate)

    has_entity_type = 1 if entity_type else 0
    has_description = 1 if description else 0
    is_canonical = 1 if canonical_qid and qid == canonical_qid else 0

    surface_overlap_score = _compute_surface_overlap(comparison_source, candidate, target_label, aliases)
    description_overlap_score = _compute_description_overlap(source, description)
    context_suitability = _compute_context_suitability(
        source=source,
        comparison_source=comparison_source,
        candidate=candidate,
        target_label=target_label,
        description=description,
        aliases=aliases,
    )

    # 스케일 안정화를 위해 일부 카운트성 feature는 log1p를 적용합니다.
    scaled_target_label_len = round(math.log1p(len(target_label)), 6)
    scaled_alias_count = round(math.log1p(raw_alias_count), 6)
    scaled_ambiguity_count = round(math.log1p(raw_ambiguity_count), 6)

    features: Dict[str, FeatureValue] = {
        # 디버깅/로그 용 텍스트 메타데이터
        "source_text": _safe_text(source),
        "source_span_text": source_span_text,
        "comparison_source_text": comparison_source,
        "comparison_source_origin": comparison_source_origin,
        "target_label": target_label,
        "entity_type": entity_type,
        "description": description,
        "qid": qid,
        "candidate_source": candidate_source,
        "span_match_kind": span_match_kind,
        # 숫자형 모델 feature
        "context_suitability": context_suitability,
        "has_source_span": has_span,
        "span_match_score": round(span_match_score, 6),
        "surface_overlap_score": round(surface_overlap_score, 6),
        "description_overlap_score": round(description_overlap_score, 6),
        "target_label_len": scaled_target_label_len,
        "alias_count": scaled_alias_count,
        "has_entity_type": has_entity_type,
        "has_description": has_description,
        "popularity_score": round(raw_popularity_score, 6),
        "ambiguity_count": scaled_ambiguity_count,
        # prior는 numeric vector에는 넣지 않고 메타데이터로만 유지
        "is_canonical_prior": is_canonical,
        "prior_bonus": 1.0 if is_canonical else 0.0,
        # raw 값도 로그/분석용으로 유지
        "raw_target_label_len": len(target_label),
        "raw_alias_count": raw_alias_count,
        "raw_ambiguity_count": raw_ambiguity_count,
    }

    return features


def feature_dict_to_numeric_vector(feature_dict: Dict[str, FeatureValue]) -> List[float]:
    """고정된 순서로 숫자형 feature만 추출합니다."""
    vector: List[float] = []
    for key in NUMERIC_FEATURE_KEYS:
        value = feature_dict.get(key, 0.0)
        try:
            vector.append(float(value))
        except Exception:
            vector.append(0.0)
    return vector


__all__ = [
    "NUMERIC_FEATURE_KEYS",
    "build_candidate_feature_vector",
    "feature_dict_to_numeric_vector",
]