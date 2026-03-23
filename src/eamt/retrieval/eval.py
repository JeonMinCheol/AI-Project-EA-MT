from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from .service import collect_entity_candidates


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_gold_qid(example: Any) -> str:
    return (
        _safe_text(getattr(example, "wikidata_id", None))
        or _safe_text(getattr(example, "qid", None))
        or ""
    )


def _extract_candidate_qid(candidate: Any) -> str:
    return (
        _safe_text(getattr(candidate, "qid", None))
        or _safe_text(getattr(candidate, "wikidata_id", None))
        or ""
    )


def is_gold_in_top_k(gold_qid: str, candidates: Sequence[Any], k: int) -> bool:
    if not gold_qid or not candidates:
        return False

    top_candidates = list(candidates[:k])
    candidate_qids = [_extract_candidate_qid(c) for c in top_candidates]
    return gold_qid in candidate_qids


def compute_retrieval_recall_at_k(
    examples: Sequence[Any],
    retrieved_lists: Sequence[Sequence[Any]],
    ks: Iterable[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    pairs = list(zip(examples, retrieved_lists))

    for k in ks:
        total = 0
        hit = 0

        for example, candidates in pairs:
            gold_qid = _extract_gold_qid(example)
            if not gold_qid:
                continue

            total += 1
            if is_gold_in_top_k(gold_qid, candidates, k):
                hit += 1

        metrics[f"recall@{k}"] = float(hit / total) if total else 0.0

    return metrics


def evaluate_retrieval_service(
    examples: Sequence[Any],
    resources: Any,
    *,
    top_k: int = 10,
    per_surface_k: int = 5,
    ks: Iterable[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    retrieved_lists: List[List[Any]] = []

    for example in examples:
        candidates = collect_entity_candidates(
            example=example,
            resources=resources,
            top_k=top_k,
            per_surface_k=per_surface_k,
        )
        retrieved_lists.append(candidates)

    return compute_retrieval_recall_at_k(
        examples=examples,
        retrieved_lists=retrieved_lists,
        ks=ks,
    )


__all__ = [
    "is_gold_in_top_k",
    "compute_retrieval_recall_at_k",
    "evaluate_retrieval_service",
]