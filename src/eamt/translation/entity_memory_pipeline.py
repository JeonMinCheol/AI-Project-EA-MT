from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from DTOlist import CandidateEntity, EAMTExample, EntityMemoryBlock, RuntimeResources
except Exception:  # pragma: no cover
    from src.DTOlist import CandidateEntity, EAMTExample, EntityMemoryBlock, RuntimeResources

from eamt.kb.index import lookup_entity_by_qid
from eamt.kb.resources import build_runtime_resources_from_db
from eamt.memory.builder import build_entity_memory_block, render_entity_memory_text
from eamt.retrieval.service import collect_entity_candidates
from eamt.reranker.model import CandidateReranker
from eamt.reranker.service import score_candidates, select_top_candidate

try:
    from eamt.retrieval.align import align_source_span
except Exception:  # pragma: no cover
    from ..retrieval.align import align_source_span


SUPPORTED_ENTITY_PIPELINE_MODES = (
    "anchored",
    "surface",
    "retrieve",
    "rerank",
)


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        value = values.strip()
        return [value] if value else []
    if isinstance(values, Sequence):
        cleaned: List[str] = []
        for item in values:
            item_text = _safe_str(item)
            if item_text:
                cleaned.append(item_text)
        return cleaned
    return []


def _normalize_target_lang(value: str) -> str:
    text = _safe_str(value)
    if "_" in text:
        return text.split("_", 1)[0]
    return text


def _extract_source(example: EAMTExample | Mapping[str, Any]) -> str:
    if isinstance(example, Mapping):
        return _safe_str(example.get("source") or example.get("source_text") or example.get("text"))
    return _safe_str(
        getattr(example, "source", None)
        or getattr(example, "source_text", None)
        or getattr(example, "text", None)
    )


def _extract_target_lang(example: EAMTExample | Mapping[str, Any]) -> str:
    if isinstance(example, Mapping):
        return _normalize_target_lang(
            _safe_str(example.get("target_lang") or example.get("target_locale") or example.get("target_language"))
        )
    return _normalize_target_lang(
        _safe_str(
            getattr(example, "target_lang", None)
            or getattr(example, "target_locale", None)
            or getattr(example, "target_language", None)
        )
    )


def _extract_gold_qid(example: EAMTExample | Mapping[str, Any]) -> str:
    if isinstance(example, Mapping):
        return _safe_str(example.get("wikidata_id") or example.get("qid"))
    wikidata_id = _safe_str(getattr(example, "wikidata_id", None))
    if wikidata_id:
        return wikidata_id

    entity_qids = getattr(example, "entity_qids", None)
    if isinstance(entity_qids, Sequence):
        for qid in entity_qids:
            qid_text = _safe_str(qid)
            if qid_text:
                return qid_text
    return ""


def _truncate_text(text: str, max_chars: int) -> str:
    value = _safe_str(text)
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def _apply_candidate_memory_budget(candidate: Any, *, alias_limit: int, description_max_chars: int) -> Any:
    cloned = copy.copy(candidate)
    aliases = _safe_list(getattr(cloned, "target_aliases", []))
    if alias_limit >= 0:
        aliases = aliases[:alias_limit]

    try:
        cloned.target_aliases = aliases
    except Exception:
        pass

    try:
        cloned.alias_count = len(aliases)
    except Exception:
        pass

    description = _truncate_text(_safe_str(getattr(cloned, "description", None)), description_max_chars)
    try:
        cloned.description = description
    except Exception:
        pass

    return cloned


def _build_anchored_candidate(
    example: EAMTExample | Mapping[str, Any],
    resources: RuntimeResources,
    *,
    min_char_len: int,
) -> CandidateEntity | None:
    gold_qid = _extract_gold_qid(example)
    target_lang = _extract_target_lang(example)
    source = _extract_source(example)

    if not gold_qid or not target_lang or not source:
        return None

    records = lookup_entity_by_qid(resources.qid_index, gold_qid, target_lang)
    if not records:
        return None

    record = sorted(records, key=lambda item: float(getattr(item, "popularity_score", 0.0)), reverse=True)[0]
    raw_match = align_source_span(
        source=source,
        label=_safe_str(getattr(record, "label_en", None)),
        aliases=_safe_list(getattr(record, "aliases_en", [])),
        min_char_len=min_char_len,
    )

    source_span = raw_match.matched_text if raw_match is not None else ""
    ambiguity_count = 0
    if resources.surface_index is not None and source_span:
        normalized = _safe_str(source_span).lower()
        try:
            from eamt.kb.index import normalize_surface
        except Exception:  # pragma: no cover
            from ..kb.index import normalize_surface
        ambiguity_count = len(resources.surface_index.get(normalize_surface(normalized), []))

    target_aliases = _safe_list(getattr(record, "target_aliases", []))
    return CandidateEntity(
        qid=_safe_str(getattr(record, "qid", None)),
        candidate_source="anchored_qid",
        target_label=_safe_str(getattr(record, "target_label", None)) or None,
        target_aliases=target_aliases,
        entity_type=_safe_str(getattr(record, "entity_type", None)) or None,
        description=_safe_str(getattr(record, "description", None)) or None,
        popularity_score=float(getattr(record, "popularity_score", 0.0) or 0.0),
        alias_count=len(target_aliases),
        ambiguity_count=ambiguity_count,
        source_span=source_span or None,
        span_match=raw_match,
    )


def _collect_pipeline_candidates(
    example: EAMTExample | Mapping[str, Any],
    resources: RuntimeResources,
    *,
    entity_pipeline_mode: str,
    retrieval_top_k: int,
    retrieval_per_surface_k: int,
    retrieval_min_char_len: int,
    retrieval_max_n: int,
) -> List[Any]:
    if entity_pipeline_mode == "anchored":
        anchored = _build_anchored_candidate(
            example,
            resources,
            min_char_len=retrieval_min_char_len,
        )
        return [anchored] if anchored is not None else []

    use_anchored_lookup = entity_pipeline_mode in {"retrieve", "rerank"}
    use_surface_search = entity_pipeline_mode in {"surface", "retrieve", "rerank"}

    return collect_entity_candidates(
        example=example,
        resources=resources,
        top_k=retrieval_top_k,
        per_surface_k=retrieval_per_surface_k,
        min_char_len=retrieval_min_char_len,
        max_n=retrieval_max_n,
        use_anchored_lookup=use_anchored_lookup,
        use_surface_search=use_surface_search,
    )


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(state_dict)
    if normalized and all(key.startswith("module.") for key in normalized.keys()):
        return {key[len("module."):]: value for key, value in normalized.items()}
    return normalized


def load_reranker_model(
    model_path: str | None,
    *,
    device: str | None = None,
    hidden_dim: int = 128,
    dropout: float = 0.1,
) -> Any:
    if not _safe_str(model_path):
        return None
    if torch is None:
        raise ImportError("reranker 모델 로딩에는 PyTorch가 필요합니다.")

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateReranker(hidden_dim=hidden_dim, dropout=dropout)
    state_dict = torch.load(_safe_str(model_path), map_location=resolved_device)
    if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"지원하지 않는 reranker checkpoint 형식입니다: {model_path}")

    model.load_state_dict(_strip_module_prefix(state_dict))
    model = model.to(resolved_device)
    model.eval()
    return model


@dataclass
class EntityPipelineArtifacts:
    resources: RuntimeResources
    reranker_model: Any | None = None


def load_entity_pipeline_artifacts(
    *,
    target_lang: str,
    entity_pipeline_mode: str,
    reranker_model_path: str | None = None,
    reranker_device: str | None = None,
    reranker_hidden_dim: int = 128,
    reranker_dropout: float = 0.1,
) -> EntityPipelineArtifacts:
    if entity_pipeline_mode not in SUPPORTED_ENTITY_PIPELINE_MODES:
        raise ValueError(
            f"지원하지 않는 entity pipeline mode입니다: {entity_pipeline_mode}. "
            f"지원 목록: {SUPPORTED_ENTITY_PIPELINE_MODES}"
        )

    resources = build_runtime_resources_from_db(target_lang=target_lang)
    reranker_model = None
    if entity_pipeline_mode == "rerank":
        reranker_model = load_reranker_model(
            reranker_model_path,
            device=reranker_device,
            hidden_dim=reranker_hidden_dim,
            dropout=reranker_dropout,
        )

    return EntityPipelineArtifacts(resources=resources, reranker_model=reranker_model)


def build_entity_memory_from_pipeline(
    example: EAMTExample | Mapping[str, Any],
    *,
    artifacts: EntityPipelineArtifacts,
    entity_pipeline_mode: str,
    alias_limit: int,
    description_max_chars: int,
    retrieval_top_k: int = 10,
    retrieval_per_surface_k: int = 5,
    retrieval_min_char_len: int = 2,
    retrieval_max_n: int = 5,
    reranker_prior_bonus_weight: float = 0.1,
) -> EntityMemoryBlock | None:
    if entity_pipeline_mode not in SUPPORTED_ENTITY_PIPELINE_MODES:
        raise ValueError(
            f"지원하지 않는 entity pipeline mode입니다: {entity_pipeline_mode}. "
            f"지원 목록: {SUPPORTED_ENTITY_PIPELINE_MODES}"
        )

    source = _extract_source(example)
    target_lang = _extract_target_lang(example)
    if not source or not target_lang:
        return None

    candidates = _collect_pipeline_candidates(
        example,
        artifacts.resources,
        entity_pipeline_mode=entity_pipeline_mode,
        retrieval_top_k=retrieval_top_k,
        retrieval_per_surface_k=retrieval_per_surface_k,
        retrieval_min_char_len=retrieval_min_char_len,
        retrieval_max_n=retrieval_max_n,
    )
    if not candidates:
        return None

    top_candidate = candidates[0]
    if entity_pipeline_mode == "rerank":
        scored_candidates = score_candidates(
            source=source,
            candidates=list(candidates),
            canonical_qid=_extract_gold_qid(example) or None,
            reranker_model=artifacts.reranker_model,
            source_span=getattr(candidates[0], "source_span", None),
            prior_bonus_weight=reranker_prior_bonus_weight,
        )
        selected = select_top_candidate(scored_candidates)
        if selected is None:
            return None
        top_candidate = selected.candidate

    memory_candidate = _apply_candidate_memory_budget(
        top_candidate,
        alias_limit=alias_limit,
        description_max_chars=description_max_chars,
    )
    memory = build_entity_memory_block(
        [memory_candidate],
        source_sentence=source,
        alias_limit=alias_limit,
    )
    render_entity_memory_text(memory, target_lang=target_lang)
    return memory


__all__ = [
    "EntityPipelineArtifacts",
    "SUPPORTED_ENTITY_PIPELINE_MODES",
    "build_entity_memory_from_pipeline",
    "load_entity_pipeline_artifacts",
    "load_reranker_model",
]
