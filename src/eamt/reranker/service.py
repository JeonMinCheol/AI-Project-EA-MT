from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.eamt.reranker.features import (
    build_candidate_feature_vector,
    feature_dict_to_numeric_vector,
)

try:
    from src.eamt.models import ScoredCandidate  # type: ignore
except Exception:
    @dataclass
    class ScoredCandidate:  # type: ignore[override]
        candidate: Any
        reranker_score: float
        final_score: float
        feature_dump: Dict[str, Any] = field(default_factory=dict)
        prior_bonus: float = 0.0
        margin_to_next: float = 0.0


def _heuristic_reranker_score(feature_dump: Dict[str, Any]) -> float:
    """모델이 없을 때도 순위가 완전히 무너지지 않도록 기본 점수를 계산합니다."""
    return (
        0.45 * float(feature_dump.get("context_suitability", 0.0))
        + 0.20 * float(feature_dump.get("surface_overlap_score", 0.0))
        + 0.10 * float(feature_dump.get("description_overlap_score", 0.0))
        + 0.15 * float(feature_dump.get("span_match_score", 0.0))
        + 0.10 * float(feature_dump.get("has_source_span", 0.0))
    )


def _predict_with_model(reranker_model: Any, feature_dump: Dict[str, Any]) -> float:
    """
    팀원 구현체가 조금씩 달라도 merge 후 최대한 안 깨지도록 여러 인터페이스를 허용합니다.
    지원 우선순위:
      1) predict_feature_dict(feature_dump)
      2) predict(vector)
      3) callable(vector)
      4) predict_proba([vector])
      5) fallback heuristic
    """
    if reranker_model is None:
        return _heuristic_reranker_score(feature_dump)

    vector = feature_dict_to_numeric_vector(feature_dump)

    try:
        if hasattr(reranker_model, "predict_feature_dict"):
            score = reranker_model.predict_feature_dict(feature_dump)
            return float(score)

        if hasattr(reranker_model, "predict"):
            score = reranker_model.predict(vector)
            if isinstance(score, (list, tuple)):
                return float(score[0]) if score else 0.0
            return float(score)

        if hasattr(reranker_model, "predict_proba"):
            proba = reranker_model.predict_proba([vector])
            # sklearn류 분류기 대응: positive class probability 사용
            if proba is None:
                return _heuristic_reranker_score(feature_dump)
            if hasattr(proba, "tolist"):
                proba = proba.tolist()
            if isinstance(proba, list) and proba:
                row = proba[0]
                if isinstance(row, list) and row:
                    return float(row[-1])
                return float(row)

        if callable(reranker_model):
            score = reranker_model(vector)
            return float(score)
    except Exception:
        return _heuristic_reranker_score(feature_dump)

    return _heuristic_reranker_score(feature_dump)


def score_candidates(
    source: str,
    candidates: List[Any],
    canonical_qid: Optional[str],
    reranker_model: Any,
) -> List[Any]:
    """후보군을 점수화하고 final_score 기준 내림차순으로 정렬합니다."""
    scored_candidates: List[Any] = []

    for cand in candidates:
        feature_dump = build_candidate_feature_vector(source, cand, canonical_qid)
        reranker_score = _predict_with_model(reranker_model, feature_dump)
        prior_bonus = float(feature_dump.get("prior_bonus", 0.0))
        final_score = float(reranker_score) + prior_bonus

        scored_candidates.append(
            ScoredCandidate(
                candidate=cand,
                reranker_score=float(reranker_score),
                prior_bonus=prior_bonus,
                final_score=float(final_score),
                feature_dump=feature_dump,
            )
        )

    scored_candidates.sort(key=lambda x: float(getattr(x, "final_score", 0.0)), reverse=True)
    return scored_candidates


def select_top_candidate(scored_candidates: List[Any]) -> Optional[Any]:
    """
    1위 후보를 반환하고, 2위와의 점수 차이 margin_to_next를 계산합니다.
    ERCM trigger의 입력 신호로 바로 사용할 수 있습니다.
    """
    if not scored_candidates:
        return None

    top_candidate = scored_candidates[0]
    if len(scored_candidates) > 1:
        margin = float(getattr(top_candidate, "final_score", 0.0)) - float(
            getattr(scored_candidates[1], "final_score", 0.0)
        )
    else:
        margin = float("inf")

    try:
        top_candidate.margin_to_next = margin
    except Exception:
        # dataclass frozen 등 특수 케이스에 대비
        pass
    return top_candidate


__all__ = ["score_candidates", "select_top_candidate", "ScoredCandidate"]