from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None
    F = None
    DataLoader = None
    Dataset = object  # type: ignore

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

from .eval import evaluate_grouped_examples
from .model import CandidateReranker


if torch is None or F is None or DataLoader is None:
    class GroupedRerankerDataset:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch가 설치되어 있어야 GroupedRerankerDataset을 사용할 수 있습니다.")

    def grouped_collate_fn(*args, **kwargs):  # type: ignore[override]
        raise ImportError("PyTorch가 설치되어 있어야 grouped_collate_fn을 사용할 수 있습니다.")

    def compute_listwise_loss(*args, **kwargs):  # type: ignore[override]
        raise ImportError("PyTorch가 설치되어 있어야 compute_listwise_loss를 사용할 수 있습니다.")

    def train_reranker_model(*args, **kwargs):  # type: ignore[override]
        raise ImportError("PyTorch가 설치되어 있어야 train_reranker_model을 사용할 수 있습니다.")

else:
    @dataclass
    class GroupedBatch:
        features: Tensor
        mask: Tensor
        gold_indices: Tensor
        labels: List[List[int]]
        qids: List[List[Any]]
        example_ids: List[str]

    class GroupedRerankerDataset(Dataset):
        """
        grouped example 형식:
        {
            "example_id": ...,
            "numeric_features": [[...], [...], ...],
            "labels": [1, 0, 0, ...],
            "qids": [...]
        }
        """

        def __init__(self, grouped_examples: Sequence[Dict[str, Any]]):
            filtered: List[Dict[str, Any]] = []
            for ex in grouped_examples:
                numeric_features = ex.get("numeric_features", [])
                labels = ex.get("labels", [])
                if not numeric_features or not labels:
                    continue
                filtered.append(ex)
            self.examples = filtered

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            return self.examples[index]

    def _gold_index(labels: Sequence[int]) -> int:
        for idx, label in enumerate(labels):
            if int(label) == 1:
                return idx
        return -100

    def grouped_collate_fn(batch: Sequence[Dict[str, Any]]) -> GroupedBatch:
        batch = list(batch)
        batch_size = len(batch)
        max_candidates = max(len(item["numeric_features"]) for item in batch)
        feature_dim = len(batch[0]["numeric_features"][0])

        features = torch.zeros((batch_size, max_candidates, feature_dim), dtype=torch.float32)
        mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
        gold_indices = torch.full((batch_size,), -100, dtype=torch.long)

        labels_out: List[List[int]] = []
        qids_out: List[List[Any]] = []
        example_ids: List[str] = []

        for i, item in enumerate(batch):
            numeric_features = item["numeric_features"]
            labels = [int(v) for v in item.get("labels", [])]
            qids = item.get("qids", [])
            example_id = str(item.get("example_id", f"example_{i}"))

            for j, vec in enumerate(numeric_features):
                features[i, j] = torch.tensor(vec, dtype=torch.float32)
                mask[i, j] = True

            gold_indices[i] = _gold_index(labels)
            labels_out.append(labels)
            qids_out.append(list(qids))
            example_ids.append(example_id)

        return GroupedBatch(
            features=features,
            mask=mask,
            gold_indices=gold_indices,
            labels=labels_out,
            qids=qids_out,
            example_ids=example_ids,
        )

    def compute_listwise_loss(
        scores: Tensor,
        gold_indices: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """
        scores shape: [batch_size, max_candidates]
        """
        masked_scores = scores.masked_fill(~mask, -1e9)
        valid = gold_indices >= 0

        if not bool(valid.any()):
            return masked_scores.sum() * 0.0

        return F.cross_entropy(masked_scores[valid], gold_indices[valid])

    def train_reranker_model(
        train_examples: Sequence[Dict[str, Any]],
        valid_examples: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        model: Optional[CandidateReranker] = None,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
        ks: Tuple[int, ...] = (1, 3, 5),
        save_path: Optional[str] = None,
    ) -> Tuple[CandidateReranker, Dict[str, List[float]]]:
        if model is None:
            model = CandidateReranker()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = model.to(device)

        train_dataset = GroupedRerankerDataset(train_examples)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=grouped_collate_fn,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "valid_top1_accuracy": [],
            "valid_mrr": [],
        }
        for k in ks:
            history[f"valid_recall@{k}"] = []

        best_top1 = -1.0

        for _epoch in range(epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                features = batch.features.to(device)
                mask = batch.mask.to(device)
                gold_indices = batch.gold_indices.to(device)

                optimizer.zero_grad()

                scores = model(features)
                loss = compute_listwise_loss(
                    scores=scores,
                    gold_indices=gold_indices,
                    mask=mask,
                )

                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                num_batches += 1

            mean_train_loss = total_loss / max(1, num_batches)
            history["train_loss"].append(mean_train_loss)

            if valid_examples:
                valid_metrics = evaluate_grouped_examples(
                    model=model,
                    grouped_examples=valid_examples,
                    ks=ks,
                    device=device,
                )

                top1 = float(valid_metrics.get("top1_accuracy", 0.0))
                mrr = float(valid_metrics.get("mrr", 0.0))

                history["valid_top1_accuracy"].append(top1)
                history["valid_mrr"].append(mrr)

                for k in ks:
                    history[f"valid_recall@{k}"].append(
                        float(valid_metrics.get(f"recall@{k}", 0.0))
                    )

                if save_path and top1 > best_top1:
                    best_top1 = top1
                    torch.save(model.state_dict(), save_path)
            else:
                history["valid_top1_accuracy"].append(0.0)
                history["valid_mrr"].append(0.0)
                for k in ks:
                    history[f"valid_recall@{k}"].append(0.0)

        return model, history


__all__ = [
    "GroupedRerankerDataset",
    "GroupedBatch",
    "grouped_collate_fn",
    "compute_listwise_loss",
    "train_reranker_model",
]