from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import comet as comet_module
except Exception as comet_import_error:  # pragma: no cover
    comet_module = None
    COMET_IMPORT_ERROR_DETAIL = str(comet_import_error)
    download_model = None
    load_from_checkpoint = None
else:  # pragma: no cover
    download_model = getattr(comet_module, "download_model", None)
    load_from_checkpoint = getattr(comet_module, "load_from_checkpoint", None)
    if download_model is None or load_from_checkpoint is None:
        COMET_IMPORT_ERROR_DETAIL = (
            f"`import comet` resolved to {getattr(comet_module, '__file__', 'unknown location')}, "
            "but that module does not provide `download_model` and `load_from_checkpoint`. "
            "This usually means a different `comet` package is installed instead of `unbabel-comet`."
        )
    else:
        COMET_IMPORT_ERROR_DETAIL = None

try:
    from DTOlist import EAMTExample, EntityMemoryBlock
except Exception:  # pragma: no cover
    from src.DTOlist import EAMTExample, EntityMemoryBlock

try:
    from .inference import (
        DEFAULT_QWEN_MODEL_NAME,
        DEFAULT_SYSTEM_PROMPT,
        load_qwen2_5_instruct,
        predict_eamt_dataset,
    )
except Exception:  # pragma: no cover
    from eamt.translation.inference import (
        DEFAULT_QWEN_MODEL_NAME,
        DEFAULT_SYSTEM_PROMPT,
        load_qwen2_5_instruct,
        predict_eamt_dataset,
    )


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _get_value(record: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(record, Mapping) and key in record and record[key] is not None:
            return record[key]
        if hasattr(record, key):
            value = getattr(record, key)
            if value is not None:
                return value
    return default


def _prediction_text_from_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return _safe_str(value.get("prediction"))
    if hasattr(value, "final_translation"):
        return _safe_str(getattr(value, "final_translation"))
    return _safe_str(value)


def _normalize_prediction_lookup(predictions: Mapping[str, Any] | Sequence[Any]) -> Dict[str, str]:
    if isinstance(predictions, Mapping):
        return {str(key): _prediction_text_from_value(value) for key, value in predictions.items()}

    normalized: Dict[str, str] = {}
    for item in predictions:
        instance_id = _safe_str(_get_value(item, "id"))
        if not instance_id:
            continue
        prediction = _safe_str(
            _get_value(item, "prediction", "final_translation", "translation")
        )
        normalized[instance_id] = prediction
    return normalized


def _normalize_reference_records(
    references: Mapping[str, Any] | Sequence[Any],
    entity_types: Sequence[str] | None = None,
) -> List[Any]:
    records = list(references.values()) if isinstance(references, Mapping) else list(references)
    allowed_types = {_safe_str(entity_type).casefold() for entity_type in (entity_types or [])}

    filtered: List[Any] = []
    for record in records:
        targets = _get_value(record, "targets", default=[]) or []
        if not targets:
            continue

        if allowed_types:
            record_types = _get_value(record, "entity_types", default=[]) or []
            record_type_set = {_safe_str(record_type).casefold() for record_type in record_types}
            if not record_type_set.intersection(allowed_types):
                continue

        filtered.append(record)

    return filtered


def _harmonic_mean(left: float, right: float) -> float:
    if left <= 0.0 or right <= 0.0:
        return 0.0
    return float(2.0 * left * right / (left + right))


def _require_comet() -> None:
    if download_model is None or load_from_checkpoint is None:
        detail = (
            f" 현재 감지된 상태: {COMET_IMPORT_ERROR_DETAIL}"
            if COMET_IMPORT_ERROR_DETAIL
            else ""
        )
        raise ImportError(
            "COMET 점수를 계산하려면 `unbabel-comet` 패키지가 현재 실행 중인 Python 환경에 "
            "정상 설치되어 있어야 합니다."
            + detail
        )


def _with_optional_tqdm(
    values: Sequence[Any],
    *,
    enabled: bool,
    desc: str,
    unit: str,
):
    if not enabled or tqdm is None:
        return values
    return tqdm(values, desc=desc, unit=unit)


def _timestamp_text() -> str:
    return time.strftime("%F %T")


def _format_seconds(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "unknown"

    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _log_stage_event(stage_name: str, status: str, **details: Any) -> None:
    suffix_parts = []
    for key, value in details.items():
        if value is None or value == "":
            continue
        suffix_parts.append(f"{key}={value}")

    suffix = f" | {' | '.join(suffix_parts)}" if suffix_parts else ""
    print(f"[{_timestamp_text()}] [Stage {status}] {stage_name}{suffix}", flush=True)


def save_predictions_jsonl(
    predictions: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file_obj:
        for prediction in predictions:
            file_obj.write(json.dumps(dict(prediction), ensure_ascii=False) + "\n")

    return output_path


def get_mentions_from_references(data: Sequence[Any]) -> Dict[str, Set[str]]:
    mentions: Dict[str, Set[str]] = {}

    for instance in data:
        instance_id = _safe_str(_get_value(instance, "id"))
        instance_mentions: Set[str] = set()

        for target in _get_value(instance, "targets", default=[]) or []:
            mention = _safe_str(_get_value(target, "mention"))
            if mention:
                instance_mentions.add(mention)

            mention_candidates = _get_value(target, "mention_candidates", default=[]) or []
            for candidate in mention_candidates:
                candidate_text = _safe_str(candidate)
                if candidate_text:
                    instance_mentions.add(candidate_text)

        mentions[instance_id] = instance_mentions

    return mentions


def compute_entity_name_translation_accuracy(
    predictions: Mapping[str, Any] | Sequence[Any],
    mentions: Mapping[str, Iterable[str]],
    verbose: bool = False,
    show_progress: bool = False,
) -> Dict[str, float | int]:
    prediction_lookup = _normalize_prediction_lookup(predictions)

    correct = 0
    total = 0
    missing_predictions = 0

    for instance_id, instance_mentions_iter in _with_optional_tqdm(
        list(mentions.items()),
        enabled=show_progress,
        desc="Computing M-ETA",
        unit="example",
    ):
        instance_mentions = {_safe_str(mention) for mention in instance_mentions_iter if _safe_str(mention)}
        assert instance_mentions, f"No mentions for instance {instance_id}"

        total += 1

        if instance_id not in prediction_lookup:
            missing_predictions += 1
            if verbose:
                print(
                    f"No prediction for instance {instance_id}. "
                    "Check that this is expected behavior, as it may affect the evaluation."
                )
            continue

        prediction = prediction_lookup[instance_id]
        normalized_translation = prediction.casefold()
        entity_match = False

        for mention in instance_mentions:
            if mention.casefold() in normalized_translation:
                correct += 1
                entity_match = True
                break

        if not entity_match and verbose:
            print(f"Prediction: {prediction}")
            print(f"Ground truth mentions: {instance_mentions}")
            print("")

    return {
        "correct": correct,
        "total": total,
        "missing_predictions": missing_predictions,
        "accuracy": float(correct / total) if total > 0 else 0.0,
    }


def compute_comet_score(
    predictions: Mapping[str, Any] | Sequence[Any],
    references: Mapping[str, Any] | Sequence[Any],
    *,
    entity_types: Sequence[str] | None = None,
    comet_model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    num_gpus: int = 1,
    show_progress: bool = False,
) -> Dict[str, float | int | str]:
    _require_comet()

    prediction_lookup = _normalize_prediction_lookup(predictions)
    reference_records = _normalize_reference_records(references, entity_types=entity_types)

    instances: List[Dict[str, str]] = []
    instance_ranges: Dict[str, tuple[int, int]] = {}
    current_index = 0

    for record in sorted(reference_records, key=lambda item: _safe_str(_get_value(item, "id"))):
        instance_id = _safe_str(_get_value(record, "id"))
        if instance_id not in prediction_lookup:
            continue

        source = _safe_str(_get_value(record, "source", "text"))
        prediction = prediction_lookup[instance_id]
        start_index = current_index

        for target in _get_value(record, "targets", default=[]) or []:
            reference_translation = _safe_str(_get_value(target, "translation"))
            if not reference_translation:
                continue

            instances.append(
                {
                    "src": source,
                    "ref": reference_translation,
                    "mt": prediction,
                }
            )
            current_index += 1

        if current_index > start_index:
            instance_ranges[instance_id] = (start_index, current_index)

    num_missing_predictions = sum(
        1
        for record in reference_records
        if _safe_str(_get_value(record, "id")) not in prediction_lookup
    )

    if not instances:
        return {
            "model_name": comet_model_name,
            "score": 0.0,
            "score_percent": 0.0,
            "matched_predictions": 0,
            "missing_predictions": num_missing_predictions,
        }

    _log_stage_event(
        "COMET Resolve Checkpoint",
        "Info",
        model=comet_model_name,
    )
    model_path = download_model(comet_model_name)
    _log_stage_event(
        "COMET Load Model",
        "Info",
        checkpoint=model_path,
    )
    model = load_from_checkpoint(model_path)
    _log_stage_event(
        "COMET Predict",
        "Info",
        sentence_pairs=len(instances),
        batch_size=batch_size,
        num_gpus=num_gpus,
        missing_predictions=num_missing_predictions,
    )
    try:
        outputs = model.predict(
            instances,
            batch_size=batch_size,
            gpus=num_gpus,
            progress_bar=show_progress,
        )
    except TypeError:
        outputs = model.predict(instances, batch_size=batch_size, gpus=num_gpus)
    scores = list(outputs.scores)

    max_scores: List[float] = []
    for _, (start_index, end_index) in instance_ranges.items():
        max_scores.append(float(max(scores[start_index:end_index])))

    denominator = len(max_scores) + num_missing_predictions
    score = float(sum(max_scores) / denominator) if denominator > 0 else 0.0

    return {
        "model_name": comet_model_name,
        "score": score,
        "score_percent": 100.0 * score,
        "matched_predictions": len(max_scores),
        "missing_predictions": num_missing_predictions,
    }


def evaluate_eamt_predictions(
    predictions: Mapping[str, Any] | Sequence[Any],
    references: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    mentions_by_id: Mapping[str, Iterable[str]] | None = None,
    entity_types: Sequence[str] | None = None,
    comet_model_name: str = "Unbabel/wmt22-comet-da",
    comet_batch_size: int = 8,
    comet_num_gpus: int = 1,
    verbose: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    metrics_start_time = time.monotonic()
    reference_records = None
    if references is not None:
        reference_records = _normalize_reference_records(references, entity_types=entity_types)

    if mentions_by_id is None:
        if reference_records is None:
            raise ValueError("references 또는 mentions_by_id 중 하나는 반드시 필요합니다.")
        mentions_by_id = get_mentions_from_references(reference_records)
        if not mentions_by_id:
            raise ValueError(
                "references에서 평가용 mention을 찾지 못했습니다. "
                "`targets[].mention` 또는 `mentions_by_id`를 제공해주세요."
            )

    m_eta_start_time = time.monotonic()
    _log_stage_event(
        "M-ETA",
        "Start",
        examples=len(mentions_by_id),
    )
    m_eta_result = compute_entity_name_translation_accuracy(
        predictions=predictions,
        mentions=mentions_by_id,
        verbose=verbose,
        show_progress=show_progress,
    )
    _log_stage_event(
        "M-ETA",
        "End",
        accuracy=f"{float(m_eta_result['accuracy']):.4f}",
        correct=f"{int(m_eta_result['correct'])}/{int(m_eta_result['total'])}",
        missing_predictions=int(m_eta_result["missing_predictions"]),
        elapsed=_format_seconds(time.monotonic() - m_eta_start_time),
    )

    if reference_records is None:
        raise ValueError("COMET 점수를 계산하려면 references가 필요합니다.")

    comet_start_time = time.monotonic()
    _log_stage_event(
        "COMET",
        "Start",
        model=comet_model_name,
        reference_examples=len(reference_records),
        batch_size=comet_batch_size,
        num_gpus=comet_num_gpus,
    )
    comet_result = compute_comet_score(
        predictions=predictions,
        references=reference_records,
        entity_types=entity_types,
        comet_model_name=comet_model_name,
        batch_size=comet_batch_size,
        num_gpus=comet_num_gpus,
        show_progress=show_progress,
    )
    _log_stage_event(
        "COMET",
        "End",
        score=f"{float(comet_result['score']):.4f}",
        matched_predictions=int(comet_result["matched_predictions"]),
        missing_predictions=int(comet_result["missing_predictions"]),
        elapsed=_format_seconds(time.monotonic() - comet_start_time),
    )
    comet_score = float(comet_result["score"])
    final_score = _harmonic_mean(float(m_eta_result["accuracy"]), comet_score)
    _log_stage_event(
        "Evaluation Metrics",
        "End",
        final_score=f"{final_score:.4f}",
        elapsed=_format_seconds(time.monotonic() - metrics_start_time),
    )

    return {
        "m_eta": float(m_eta_result["accuracy"]),
        "m_eta_percent": 100.0 * float(m_eta_result["accuracy"]),
        "correct": int(m_eta_result["correct"]),
        "total": int(m_eta_result["total"]),
        "missing_predictions": int(m_eta_result["missing_predictions"]),
        "comet": comet_score,
        "comet_percent": float(comet_result["score_percent"]),
        "final_score": final_score,
        "final_score_percent": 100.0 * final_score,
        "comet_model_name": comet_result["model_name"],
    }


def evaluate_qwen_on_eamt(
    dataset: Sequence[EAMTExample | Mapping[str, Any]],
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_name: str = DEFAULT_QWEN_MODEL_NAME,
    references: Mapping[str, Any] | Sequence[Any] | None = None,
    mentions_by_id: Mapping[str, Iterable[str]] | None = None,
    memory_provider: Callable[[EAMTExample | Mapping[str, Any]], EntityMemoryBlock | None] | None = None,
    mode: str = "plain",
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool | None = None,
    generation_kwargs: Mapping[str, Any] | None = None,
    entity_types: Sequence[str] | None = None,
    comet_model_name: str = "Unbabel/wmt22-comet-da",
    comet_batch_size: int = 8,
    comet_num_gpus: int = 1,
    load_model_kwargs: Mapping[str, Any] | None = None,
    release_model_before_metrics: bool = True,
    generation_batch_size: int = 8,
    show_progress: bool = True,
    progress_log_interval_seconds: float = 30.0,
) -> Dict[str, Any]:
    pipeline_start_time = time.monotonic()
    _log_stage_event(
        "EA-MT Evaluation Pipeline",
        "Start",
        examples=len(dataset),
        model=model_name,
        generation_batch_size=max(1, int(generation_batch_size)),
    )
    loaded_model_in_function = model is None or tokenizer is None

    if loaded_model_in_function:
        model, tokenizer = load_qwen2_5_instruct(
            model_name=model_name,
            **dict(load_model_kwargs or {}),
        )

    predictions = predict_eamt_dataset(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        memory_provider=memory_provider,
        mode=mode,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        generation_kwargs=generation_kwargs,
        batch_size=generation_batch_size,
        show_progress=show_progress,
        progress_log_interval_seconds=progress_log_interval_seconds,
    )

    if loaded_model_in_function and release_model_before_metrics:
        _log_stage_event("Qwen Model Release", "Info", action="release_before_metrics")
        del model
        del tokenizer
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = evaluate_eamt_predictions(
        predictions=predictions,
        references=references if references is not None else dataset,
        mentions_by_id=mentions_by_id,
        entity_types=entity_types,
        comet_model_name=comet_model_name,
        comet_batch_size=comet_batch_size,
        comet_num_gpus=comet_num_gpus,
        show_progress=show_progress,
    )
    total_elapsed = time.monotonic() - pipeline_start_time
    _log_stage_event(
        "EA-MT Evaluation Pipeline",
        "End",
        elapsed=_format_seconds(total_elapsed),
        m_eta=f"{float(metrics['m_eta']):.4f}",
        comet=f"{float(metrics['comet']):.4f}",
        final_score=f"{float(metrics['final_score']):.4f}",
    )

    return {
        "predictions": predictions,
        "metrics": metrics,
        "model_name": model_name,
        "runtime": {
            "elapsed_seconds": float(total_elapsed),
        },
    }


__all__ = [
    "compute_comet_score",
    "compute_entity_name_translation_accuracy",
    "evaluate_eamt_predictions",
    "evaluate_qwen_on_eamt",
    "get_mentions_from_references",
    "save_predictions_jsonl",
]
