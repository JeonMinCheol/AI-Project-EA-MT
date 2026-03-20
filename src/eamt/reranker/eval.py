from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _rank_indices_desc(scores: Sequence[float]) -> List[int]:
    return sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)


def _gold_index_from_labels(labels: Sequence[int]) -> int:
    for idx, label in enumerate(labels):
        if int(label) == 1:
            return idx
    return -1


def compute_top1_accuracy(group_scores: Sequence[Sequence[float]], group_labels: Sequence[Sequence[int]]) -> float:
    total = 0
    correct = 0

    for scores, labels in zip(group_scores, group_labels):
        if not scores or not labels:
            continue
        gold_idx = _gold_index_from_labels(labels)
        if gold_idx < 0:
            continue
        pred_idx = _rank_indices_desc(scores)[0]
        total += 1
        if pred_idx == gold_idx:
            correct += 1

    return float(correct / total) if total else 0.0


def compute_recall_at_k(
    group_scores: Sequence[Sequence[float]],
    group_labels: Sequence[Sequence[int]],
    k: int,
) -> float:
    total = 0
    hit = 0

    for scores, labels in zip(group_scores, group_labels):
        if not scores or not labels:
            continue
        gold_idx = _gold_index_from_labels(labels)
        if gold_idx < 0:
            continue

        ranked = _rank_indices_desc(scores)[:k]
        total += 1
        if gold_idx in ranked:
            hit += 1

    return float(hit / total) if total else 0.0


def compute_mrr(group_scores: Sequence[Sequence[float]], group_labels: Sequence[Sequence[int]]) -> float:
    total = 0
    rr_sum = 0.0

    for scores, labels in zip(group_scores, group_labels):
        if not scores or not labels:
            continue
        gold_idx = _gold_index_from_labels(labels)
        if gold_idx < 0:
            continue

        ranked = _rank_indices_desc(scores)
        total += 1
        for rank, idx in enumerate(ranked, start=1):
            if idx == gold_idx:
                rr_sum += 1.0 / rank
                break

    return float(rr_sum / total) if total else 0.0


def summarize_reranker_metrics(
    group_scores: Sequence[Sequence[float]],
    group_labels: Sequence[Sequence[int]],
    ks: Iterable[int] = (1, 3, 5),
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "top1_accuracy": compute_top1_accuracy(group_scores, group_labels),
        "mrr": compute_mrr(group_scores, group_labels),
    }

    for k in ks:
        metrics[f"recall@{k}"] = compute_recall_at_k(group_scores, group_labels, k=k)

    return metrics


def _score_single_group(model: Any, numeric_features: Sequence[Sequence[float]], device: str | None = None) -> List[float]:
    if not numeric_features:
        return []

    if torch is not None:
        try:
            x = torch.tensor(numeric_features, dtype=torch.float32)
            if device:
                x = x.to(device)
            x = x.unsqueeze(0)  # [1, num_candidates, feature_dim]

            model.eval()
            with torch.no_grad():
                scores = model(x).squeeze(0)

            if hasattr(scores, "tolist"):
                return [float(v) for v in scores.tolist()]
            return [float(v) for v in list(scores)]
        except Exception:
            pass

    scores: List[float] = []
    for vector in numeric_features:
        try:
            if hasattr(model, "predict"):
                value = model.predict(vector)
            else:
                value = model(vector)
            if isinstance(value, (list, tuple)):
                scores.append(float(value[0]) if value else 0.0)
            else:
                scores.append(float(value))
        except Exception:
            scores.append(0.0)
    return scores


def evaluate_grouped_examples(
    model: Any,
    grouped_examples: Sequence[Dict[str, Any]],
    ks: Iterable[int] = (1, 3, 5),
    device: str | None = None,
) -> Dict[str, float]:
    all_scores: List[List[float]] = []
    all_labels: List[List[int]] = []

    for example in grouped_examples:
        numeric_features = example.get("numeric_features", [])
        labels = example.get("labels", [])
        if not numeric_features or not labels:
            continue

        scores = _score_single_group(model=model, numeric_features=numeric_features, device=device)
        all_scores.append(scores)
        all_labels.append([int(v) for v in labels])

    return summarize_reranker_metrics(all_scores, all_labels, ks=ks)


__all__ = [
    "compute_top1_accuracy",
    "compute_recall_at_k",
    "compute_mrr",
    "summarize_reranker_metrics",
    "evaluate_grouped_examples",
]