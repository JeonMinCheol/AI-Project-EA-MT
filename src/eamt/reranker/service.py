from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .features import build_candidate_feature_vector, feature_dict_to_numeric_vector

try:
    from ..models import ScoredCandidate  # type: ignore
except Exception:
    @dataclass
    class ScoredCandidate:  # type: ignore[override]
        candidate: Any
        reranker_score: float
        final_score: float
        feature_dump: Dict[str, Any] = field(default_factory=dict)
        prior_bonus: float = 0.0
        margin_to_next: Optional[float] = None


def _heuristic_reranker_score(feature_dump: Dict[str, Any]) -> float:
    """모델이 없을 때도 순위가 완전히 무너지지 않도록 기본 점수를 계산합니다."""
    return (
        0.45 * float(feature_dump.get("context_suitability", 0.0))
        + 0.20 * float(feature_dump.get("surface_overlap_score", 0.0))
        + 0.10 * float(feature_dump.get("description_overlap_score", 0.0))
        + 0.15 * float(feature_dump.get("span_match_score", 0.0))
        + 0.10 * float(feature_dump.get("has_source_span", 0.0))
    )


def _coerce_scalar(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            return _coerce_scalar(value[0], default=default)
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return default


def _predict_with_model(reranker_model: Any, feature_dump: Dict[str, Any]) -> Tuple[float, bool]:
    """
    지원 우선순위:
      1) predict_feature_dict(feature_dump)
      2) predict([vector]) / predict(vector)
      3) callable(vector)
      4) predict_proba([vector])
      5) fallback heuristic

    반환:
      (score, used_fallback)
    """
    if reranker_model is None:
        return _heuristic_reranker_score(feature_dump), True

    vector = feature_dict_to_numeric_vector(feature_dump)

    try:
        if hasattr(reranker_model, "predict_feature_dict"):
            score = reranker_model.predict_feature_dict(feature_dump)
            return _coerce_scalar(score), False

        if hasattr(reranker_model, "predict"):
            try:
                score = reranker_model.predict([vector])
            except Exception:
                score = reranker_model.predict(vector)
            return _coerce_scalar(score), False

        if hasattr(reranker_model, "predict_proba"):
            proba = reranker_model.predict_proba([vector])
            if proba is not None:
                if hasattr(proba, "tolist"):
                    proba = proba.tolist()
                if isinstance(proba, list) and proba:
                    row = proba[0]
                    if isinstance(row, list) and row:
                        return _coerce_scalar(row[-1]), False
                    return _coerce_scalar(row), False

        if callable(reranker_model):
            try:
                score = reranker_model(vector)
            except Exception:
                score = reranker_model([vector])
            return _coerce_scalar(score), False
    except Exception:
        return _heuristic_reranker_score(feature_dump), True

    return _heuristic_reranker_score(feature_dump), True


def score_candidates(
    source: str,
    candidates: List[Any],
    canonical_qid: Optional[str] = None,
    reranker_model: Any = None,
    source_span: Optional[str] = None,
    prior_bonus_weight: float = 0.1,
) -> List[ScoredCandidate]:
    """
    후보군을 점수화하고 final_score 기준 내림차순으로 정렬합니다.
    """
    if not candidates:
        return []

    scored_candidates: List[ScoredCandidate] = []

    for cand in candidates:
        feature_dump = build_candidate_feature_vector(
            source=source,
            candidate=cand,
            canonical_qid=canonical_qid,
            source_span=source_span,
        )

        reranker_score, used_fallback = _predict_with_model(reranker_model, feature_dump)
        base_prior = float(feature_dump.get("prior_bonus", 0.0))
        applied_prior_bonus = base_prior * float(prior_bonus_weight)
        final_score = float(reranker_score) + applied_prior_bonus

        feature_dump["used_fallback"] = 1 if used_fallback else 0
        feature_dump["applied_prior_bonus"] = applied_prior_bonus

        scored_candidates.append(
            ScoredCandidate(
                candidate=cand,
                reranker_score=float(reranker_score),
                prior_bonus=float(applied_prior_bonus),
                final_score=float(final_score),
                feature_dump=feature_dump,
            )
        )

    scored_candidates.sort(key=lambda x: float(getattr(x, "final_score", 0.0)), reverse=True)
    return scored_candidates


def select_top_candidate(scored_candidates: List[ScoredCandidate]) -> Optional[ScoredCandidate]:
    """
    1위 후보를 반환하고, 2위와의 점수 차이 margin_to_next를 계산합니다.
    """
    if not scored_candidates:
        return None

    top_candidate = scored_candidates[0]

    if len(scored_candidates) > 1:
        margin = float(getattr(top_candidate, "final_score", 0.0)) - float(
            getattr(scored_candidates[1], "final_score", 0.0)
        )
    else:
        margin = None

    try:
        top_candidate.margin_to_next = margin
    except Exception:
        pass

    return top_candidate


__all__ = ["score_candidates", "select_top_candidate", "ScoredCandidate"]