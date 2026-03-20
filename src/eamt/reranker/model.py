from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from .features import NUMERIC_FEATURE_KEYS, feature_dict_to_numeric_vector


if nn is None:
    class CandidateReranker:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch가 설치되어 있어야 CandidateReranker를 사용할 수 있습니다.")
else:
    class CandidateReranker(nn.Module):
        """MLP 기반 후보군 재정렬 모델."""

        def __init__(self, input_dim: int | None = None, hidden_dim: int = 128, dropout: float = 0.1):
            super().__init__()
            self.input_dim = input_dim or len(NUMERIC_FEATURE_KEYS)
            self.hidden_dim = hidden_dim
            self.dropout = dropout

            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1),
            )

        def forward(self, x):
            return self.encoder(x).squeeze(-1)

        def encode_feature_dict(self, feature_dict: Dict[str, Any]):
            vector = feature_dict_to_numeric_vector(feature_dict)
            return torch.tensor(vector, dtype=torch.float32)

        def encode_feature_dicts(self, feature_dicts: Sequence[Dict[str, Any]]):
            vectors = [feature_dict_to_numeric_vector(fd) for fd in feature_dicts]
            if not vectors:
                return torch.empty((0, self.input_dim), dtype=torch.float32)
            return torch.tensor(vectors, dtype=torch.float32)

        @torch.no_grad()
        def predict_feature_dict(self, feature_dict: Dict[str, Any]) -> float:
            self.eval()
            x = self.encode_feature_dict(feature_dict).unsqueeze(0)
            score = self.forward(x)
            return float(score.squeeze().item())

        @torch.no_grad()
        def predict(self, vector: Iterable[float]) -> float:
            self.eval()
            x = torch.tensor(list(vector), dtype=torch.float32).unsqueeze(0)
            score = self.forward(x)
            return float(score.squeeze().item())


__all__ = ["CandidateReranker", "NUMERIC_FEATURE_KEYS"]