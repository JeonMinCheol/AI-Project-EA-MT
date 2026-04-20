from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as transformers_import_error:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TRANSFORMERS_IMPORT_ERROR_DETAIL = str(transformers_import_error)
else:  # pragma: no cover
    TRANSFORMERS_IMPORT_ERROR_DETAIL = None

try:
    from peft import PeftModel
except Exception as peft_import_error:  # pragma: no cover
    PeftModel = None
    PEFT_IMPORT_ERROR_DETAIL = str(peft_import_error)
else:  # pragma: no cover
    PEFT_IMPORT_ERROR_DETAIL = None

from DTOlist import EAMTExample, EntityMemoryBlock, RuntimeResources, TranslationDraft

try:
    from .prompting import build_translation_prompt
except Exception:  # pragma: no cover
    try:
        from eamt.translation.prompting import build_translation_prompt
    except Exception:  # pragma: no cover
        from prompting import build_translation_prompt


DEFAULT_QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional translator for the SemEval EA-MT task. "
    "Return only the final translation in the target language."
)
TARGET_LOCALE_MAP = {
    "ar": "ar_AE",
    "de": "de_DE",
    "es": "es_ES",
    "fr": "fr_FR",
    "it": "it_IT",
    "ja": "ja_JP",
    "ko": "ko_KR",
    "th": "th_TH",
    "tr": "tr_TR",
    "zh": "zh_TW",
    "en": "en_US",
}


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


def _normalize_target_locale(value: Any) -> str:
    text = _safe_str(value)
    if not text:
        return ""
    if "_" in text:
        return text
    return TARGET_LOCALE_MAP.get(text.lower(), text)


def _normalize_source_locale(value: Any) -> str:
    text = _safe_str(value)
    if not text:
        return "en_US"
    if "_" in text:
        return text
    if text.lower() == "en":
        return "en_US"
    return text


def _normalize_prompt_target_lang(value: Any) -> str:
    text = _safe_str(value)
    if not text:
        return ""
    if "_" in text:
        return text.split("_", 1)[0]
    return text


def _extract_example_target_lang(example: EAMTExample | Mapping[str, Any]) -> str:
    return _normalize_target_locale(
        _get_value(example, "target_locale", "target_lang", "target_language")
    )


def _extract_prediction_text_from_ercm_output(value: Any) -> str:
    if value is None:
        return ""

    candidate_keys = (
        "final_text",
        "revised_text",
        "corrected_text",
        "prediction",
        "draft_text",
        "raw_generation",
    )

    if isinstance(value, Mapping):
        for key in candidate_keys:
            parsed = _safe_str(value.get(key))
            if parsed:
                return parsed
        return ""

    for key in candidate_keys:
        parsed = _safe_str(getattr(value, key, None))
        if parsed:
            return parsed

    return _safe_str(value)


def _resolve_torch_dtype(dtype: Any) -> Any:
    if dtype in (None, "auto"):
        return dtype
    if torch is None:
        return dtype
    if isinstance(dtype, str):
        if not hasattr(torch, dtype):
            raise ValueError(f"지원하지 않는 torch dtype입니다: {dtype}")
        return getattr(torch, dtype)
    return dtype


def _parse_gpu_ids(gpu_ids: Sequence[int] | str | None) -> List[int]:
    if gpu_ids is None:
        if torch is not None and torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []

    if isinstance(gpu_ids, str):
        parts = [part.strip() for part in gpu_ids.split(",")]
        parsed = [int(part) for part in parts if part]
    else:
        parsed = [int(gpu_id) for gpu_id in gpu_ids]

    unique_gpu_ids: List[int] = []
    for gpu_id in parsed:
        if gpu_id not in unique_gpu_ids:
            unique_gpu_ids.append(gpu_id)

    if torch is not None and torch.cuda.is_available():
        available = torch.cuda.device_count()
        invalid = [gpu_id for gpu_id in unique_gpu_ids if gpu_id < 0 or gpu_id >= available]
        if invalid:
            raise ValueError(
                f"사용 가능한 GPU 인덱스 범위를 벗어난 값이 있습니다: {invalid}. "
                f"현재 감지된 GPU 수는 {available}개입니다."
            )

    return unique_gpu_ids


def _resolve_device_map(
    device_map: str | Mapping[str, Any] | None,
    gpu_ids: Sequence[int],
) -> str | Mapping[str, Any] | None:
    if device_map is None:
        return None
    if isinstance(device_map, Mapping):
        return device_map

    text = _safe_str(device_map)
    if not text or text.lower() == "none":
        return None
    if text == "auto" and len(gpu_ids) > 1:
        return "balanced"
    return text


def _build_max_memory_map(
    gpu_ids: Sequence[int],
    *,
    per_gpu_max_memory: str | None = None,
    cpu_max_memory: str | None = None,
) -> Dict[Any, str] | None:
    if torch is None or not torch.cuda.is_available() or not gpu_ids:
        return None

    max_memory: Dict[Any, str] = {}
    for gpu_id in gpu_ids:
        if per_gpu_max_memory:
            max_memory[int(gpu_id)] = str(per_gpu_max_memory)
            continue

        total_bytes = int(torch.cuda.get_device_properties(int(gpu_id)).total_memory)
        usable_gib = max(1, int((total_bytes / (1024 ** 3)) * 0.9))
        max_memory[int(gpu_id)] = f"{usable_gib}GiB"

    if cpu_max_memory:
        max_memory["cpu"] = str(cpu_max_memory)

    return max_memory


def _collect_model_gpu_ids(model: Any) -> List[int]:
    gpu_ids: List[int] = []

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        for value in device_map.values():
            gpu_id: int | None = None
            if isinstance(value, int):
                gpu_id = int(value)
            elif isinstance(value, str) and value.startswith("cuda:"):
                try:
                    gpu_id = int(value.split(":", 1)[1])
                except ValueError:
                    gpu_id = None
            if gpu_id is not None and gpu_id not in gpu_ids:
                gpu_ids.append(gpu_id)

    if gpu_ids:
        return sorted(gpu_ids)

    device = _get_model_device(model)
    if device is not None and getattr(device, "type", None) == "cuda":
        device_index = getattr(device, "index", None)
        if device_index is None and torch is not None and torch.cuda.is_available():
            device_index = int(torch.cuda.current_device())
        if device_index is not None:
            gpu_ids.append(int(device_index))

    return sorted(gpu_ids)


def _describe_gpu(gpu_id: int) -> str:
    if torch is None or not torch.cuda.is_available():
        return str(gpu_id)
    try:
        name = torch.cuda.get_device_name(int(gpu_id))
    except Exception:
        return str(gpu_id)
    return f"{gpu_id}({name})"


def _log_inference_gpu_usage(
    model: Any,
    *,
    model_name: str,
    requested_gpu_ids: Sequence[int],
    resolved_device_map: str | Mapping[str, Any] | None,
) -> None:
    actual_gpu_ids = _collect_model_gpu_ids(model)
    visible_gpu_count = int(torch.cuda.device_count()) if torch is not None and torch.cuda.is_available() else 0

    requested_text = ",".join(str(gpu_id) for gpu_id in requested_gpu_ids) if requested_gpu_ids else "auto/none"
    if actual_gpu_ids:
        actual_text = ", ".join(_describe_gpu(gpu_id) for gpu_id in actual_gpu_ids)
    else:
        actual_text = "cpu"

    print(
        "[Qwen Inference] "
        f"model={model_name} | "
        f"visible_gpus={visible_gpu_count} | "
        f"requested_gpu_ids={requested_text} | "
        f"device_map={resolved_device_map} | "
        f"num_inference_gpus={len(actual_gpu_ids)} | "
        f"actual_devices={actual_text}"
    )


def _require_transformers() -> None:
    if torch is None:
        raise ImportError("PyTorch가 설치되어 있어야 Qwen 추론을 실행할 수 있습니다.")
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        detail = (
            f" 현재 감지된 상태: {TRANSFORMERS_IMPORT_ERROR_DETAIL}"
            if TRANSFORMERS_IMPORT_ERROR_DETAIL
            else ""
        )
        raise ImportError(
            "Qwen 모델을 불러오려면 `transformers`가 현재 실행 중인 Python 환경에서 "
            "정상 import 되어야 합니다."
            + detail
        )


def _require_peft() -> None:
    if PeftModel is None:
        detail = (
            f" 현재 감지된 상태: {PEFT_IMPORT_ERROR_DETAIL}"
            if PEFT_IMPORT_ERROR_DETAIL
            else ""
        )
        raise ImportError(
            "LoRA adapter를 불러오려면 `peft`가 현재 실행 중인 Python 환경에서 "
            "정상 import 되어야 합니다."
            + detail
        )


def _get_model_device(model: Any) -> Any:
    if torch is None:
        return None

    try:
        return next(model.parameters()).device
    except Exception:
        pass

    try:
        return next(model.buffers()).device
    except Exception:
        return None


def _move_batch_to_device(batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to") and device is not None:
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _build_messages(prompt_text: str, system_prompt: str | None) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if _safe_str(system_prompt):
        messages.append({"role": "system", "content": _safe_str(system_prompt)})
    messages.append({"role": "user", "content": prompt_text})
    return messages


def _render_model_input(tokenizer: Any, prompt_text: str, system_prompt: str | None) -> str:
    messages = _build_messages(prompt_text, system_prompt)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt_text


def _with_optional_tqdm(
    values: Sequence[Any],
    *,
    enabled: bool,
    desc: str,
):
    if not enabled or tqdm is None:
        return values
    return tqdm(
        values,
        desc=desc,
        unit="example",
        dynamic_ncols=True,
        mininterval=1.0,
        smoothing=0.1,
    )


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


def _log_progress_summary(
    *,
    stage_name: str,
    processed: int,
    total: int,
    start_time: float,
    last_batch_size: int,
) -> None:
    elapsed = max(time.monotonic() - start_time, 1e-9)
    rate = processed / elapsed
    remaining = max(total - processed, 0)
    eta = remaining / rate if rate > 0 else float("inf")
    percentage = 100.0 * processed / total if total > 0 else 100.0

    print(
        f"[{_timestamp_text()}] [Progress] {stage_name} | "
        f"processed={processed}/{total} ({percentage:.1f}%) | "
        f"elapsed={_format_seconds(elapsed)} | "
        f"eta={_format_seconds(eta)} | "
        f"avg_rate={rate:.2f} ex/s | "
        f"last_batch={last_batch_size}",
        flush=True,
    )


def _extract_source_and_target_lang(example: EAMTExample | Mapping[str, Any]) -> tuple[str, str]:
    source = _safe_str(_get_value(example, "source", "text"))
    if not source:
        raise ValueError("example에 source 또는 text 필드가 필요합니다.")

    target_lang = _safe_str(
        _get_value(example, "target_lang", "target_locale", "target_language")
    )
    if not target_lang:
        raise ValueError("example에 target_lang 또는 target_locale 필드가 필요합니다.")

    return source, target_lang


def _build_prompt_payload(
    example: EAMTExample | Mapping[str, Any],
    tokenizer: Any,
    memory: EntityMemoryBlock | None,
    *,
    mode: str,
    system_prompt: str | None,
) -> Dict[str, str]:
    source, target_lang = _extract_source_and_target_lang(example)

    prompt_text = build_translation_prompt(
        source=source,
        target_lang=_normalize_prompt_target_lang(target_lang),
        memory=memory,
        mode=mode,
    )
    rendered_prompt = _render_model_input(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        system_prompt=system_prompt,
    )

    return {
        "prompt_text": prompt_text,
        "rendered_prompt": rendered_prompt,
    }


def _build_generate_options(
    tokenizer: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool | None,
    generation_kwargs: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    should_sample = bool(do_sample) if do_sample is not None else temperature > 0.0
    generate_options: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": should_sample,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    if should_sample:
        generate_options["temperature"] = temperature
        generate_options["top_p"] = top_p
    if generation_kwargs:
        generate_options.update(dict(generation_kwargs))
    return generate_options


def _generate_texts_from_rendered_prompts(
    rendered_prompts: Sequence[str],
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool | None,
    generation_kwargs: Mapping[str, Any] | None,
) -> tuple[List[str], List[int], List[int]]:
    encoded = tokenizer(
        list(rendered_prompts),
        return_tensors="pt",
        padding=True,
    )
    device = _get_model_device(model)
    encoded = _move_batch_to_device(encoded, device)

    generate_options = _build_generate_options(
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        generation_kwargs=generation_kwargs,
    )

    inference_context = torch.inference_mode if torch is not None else nullcontext
    with inference_context():
        generated = model.generate(**encoded, **generate_options)

    input_width = int(encoded["input_ids"].shape[-1])
    generated_ids = generated[:, input_width:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    texts = [_safe_str(text) for text in decoded]

    if "attention_mask" in encoded:
        input_token_counts = [int(value) for value in encoded["attention_mask"].sum(dim=-1).tolist()]
    else:
        input_token_counts = [input_width] * len(texts)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        output_token_counts = [
            int((row != pad_token_id).sum().item())
            for row in generated_ids
        ]
    else:
        output_token_counts = [int(generated_ids.shape[-1])] * len(texts)

    return texts, input_token_counts, output_token_counts


def _is_cuda_oom_error(error: BaseException) -> bool:
    if torch is not None and hasattr(torch.cuda, "OutOfMemoryError") and isinstance(
        error,
        torch.cuda.OutOfMemoryError,
    ):
        return True

    message = str(error).casefold()
    return "out of memory" in message and "cuda" in message


def _predict_batch_records(
    batch_examples: Sequence[EAMTExample | Mapping[str, Any]],
    batch_memories: Sequence[EntityMemoryBlock | None],
    *,
    model: Any,
    tokenizer: Any,
    mode: str,
    system_prompt: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool | None,
    generation_kwargs: Mapping[str, Any] | None,
) -> List[Dict[str, str]]:
    prompt_payloads = [
        _build_prompt_payload(
            example,
            tokenizer,
            memory,
            mode=mode,
            system_prompt=system_prompt,
        )
        for example, memory in zip(batch_examples, batch_memories)
    ]

    rendered_prompts = [payload["rendered_prompt"] for payload in prompt_payloads]
    generated_texts, _, _ = _generate_texts_from_rendered_prompts(
        rendered_prompts,
        model,
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        generation_kwargs=generation_kwargs,
    )

    return [
        make_prediction_record(example, generated_text)
        for example, generated_text in zip(batch_examples, generated_texts)
    ]


def _predict_batch_records_with_fallback(
    batch_examples: Sequence[EAMTExample | Mapping[str, Any]],
    batch_memories: Sequence[EntityMemoryBlock | None],
    *,
    model: Any,
    tokenizer: Any,
    mode: str,
    system_prompt: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool | None,
    generation_kwargs: Mapping[str, Any] | None,
) -> List[Dict[str, str]]:
    try:
        return _predict_batch_records(
            batch_examples,
            batch_memories,
            model=model,
            tokenizer=tokenizer,
            mode=mode,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            generation_kwargs=generation_kwargs,
        )
    except RuntimeError as error:
        if not _is_cuda_oom_error(error) or len(batch_examples) <= 1:
            raise

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        split_index = max(1, len(batch_examples) // 2)
        _log_stage_event(
            "Translation Batch Fallback",
            "Info",
            reason="cuda_oom",
            failed_batch_size=len(batch_examples),
            retry_split=f"{split_index}+{len(batch_examples) - split_index}",
        )

        left_predictions = _predict_batch_records_with_fallback(
            batch_examples[:split_index],
            batch_memories[:split_index],
            model=model,
            tokenizer=tokenizer,
            mode=mode,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            generation_kwargs=generation_kwargs,
        )
        right_predictions = _predict_batch_records_with_fallback(
            batch_examples[split_index:],
            batch_memories[split_index:],
            model=model,
            tokenizer=tokenizer,
            mode=mode,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            generation_kwargs=generation_kwargs,
        )
        return left_predictions + right_predictions


def load_qwen2_5_instruct(
    model_name: str = DEFAULT_QWEN_MODEL_NAME,
    *,
    device_map: str | Mapping[str, Any] | None = "auto",
    gpu_ids: Sequence[int] | str | None = None,
    per_gpu_max_memory: str | None = None,
    cpu_max_memory: str | None = None,
    torch_dtype: Any = "auto",
    trust_remote_code: bool = True,
    local_files_only: bool = False,
    peft_adapter_path: str | None = None,
    merge_peft_adapter: bool = False,
    tokenizer_kwargs: Mapping[str, Any] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    """
    Hugging Face에서 Qwen2.5 Instruct 모델과 tokenizer를 불러온다.
    """
    _require_transformers()
    load_start_time = time.monotonic()
    _log_stage_event(
        "Qwen Model Load",
        "Start",
        model=model_name,
        local_files_only=local_files_only,
    )

    tokenizer_options = dict(tokenizer_kwargs or {})
    model_options = dict(model_kwargs or {})
    resolved_gpu_ids = _parse_gpu_ids(gpu_ids)
    resolved_device_map = _resolve_device_map(device_map, resolved_gpu_ids)

    if resolved_device_map is not None and "device_map" not in model_options:
        model_options["device_map"] = resolved_device_map

    if "max_memory" not in model_options:
        max_memory = _build_max_memory_map(
            resolved_gpu_ids,
            per_gpu_max_memory=per_gpu_max_memory,
            cpu_max_memory=cpu_max_memory,
        )
        if max_memory and resolved_device_map is not None:
            model_options["max_memory"] = max_memory

    if "low_cpu_mem_usage" not in model_options:
        model_options["low_cpu_mem_usage"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        **tokenizer_options,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_resolve_torch_dtype(torch_dtype),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        **model_options,
    )

    adapter_path = _safe_str(peft_adapter_path)
    if adapter_path:
        _require_peft()
        _log_stage_event(
            "LoRA Adapter Load",
            "Start",
            adapter_path=adapter_path,
            merge=merge_peft_adapter,
        )
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
        )
        if merge_peft_adapter:
            model = model.merge_and_unload()
        _log_stage_event(
            "LoRA Adapter Load",
            "End",
            adapter_path=adapter_path,
            merge=merge_peft_adapter,
        )

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    if getattr(model, "generation_config", None) is not None and getattr(
        model.generation_config, "pad_token_id", None
    ) is None:
        model.generation_config.pad_token_id = getattr(tokenizer, "pad_token_id", None)

    if hasattr(model, "eval"):
        model.eval()

    _log_inference_gpu_usage(
        model,
        model_name=model_name,
        requested_gpu_ids=resolved_gpu_ids,
        resolved_device_map=resolved_device_map,
    )
    _log_stage_event(
        "Qwen Model Load",
        "End",
        model=model_name,
        elapsed=_format_seconds(time.monotonic() - load_start_time),
    )

    return model, tokenizer


def load_qwen2_5_runtime_resources(
    model_name: str = DEFAULT_QWEN_MODEL_NAME,
    **kwargs: Any,
) -> RuntimeResources:
    model, tokenizer = load_qwen2_5_instruct(model_name=model_name, **kwargs)
    return RuntimeResources(
        qid_index=None,
        surface_index=None,
        reranker_mode=None,
        translator_model=model,
        tokenizer=tokenizer,
        ercm_model=None,
    )


def generate_draft_translation(
    example: EAMTExample | Mapping[str, Any],
    model: Any,
    tokenizer: Any,
    memory: EntityMemoryBlock | None = None,
    *,
    mode: str = "plain",
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool | None = None,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> TranslationDraft:
    """
    단일 EA-MT 샘플에 대해 Qwen 초안 번역을 생성한다.
    """
    prompt_payload = _build_prompt_payload(
        example,
        tokenizer,
        memory,
        mode=mode,
        system_prompt=system_prompt,
    )
    generated_texts, input_token_counts, output_token_counts = _generate_texts_from_rendered_prompts(
        [prompt_payload["rendered_prompt"]],
        model,
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        generation_kwargs=generation_kwargs,
    )
    raw_generation = generated_texts[0]

    generation_trace = {
        "mode": mode,
        "system_prompt": _safe_str(system_prompt),
        "translation_prompt": prompt_payload["prompt_text"],
        "input_token_count": int(input_token_counts[0]),
        "output_token_count": int(output_token_counts[0]),
    }

    return TranslationDraft(
        prompt_text=prompt_payload["rendered_prompt"],
        draft_text=raw_generation,
        raw_generation=raw_generation,
        used_memory=memory,
        used_logit_bias=None,
        generation_trace=generation_trace,
        target_lang=_extract_example_target_lang(example),
    )


def make_prediction_record(example: EAMTExample | Mapping[str, Any], prediction: str) -> Dict[str, str]:
    source = _safe_str(_get_value(example, "source", "text"))
    return {
        "id": _safe_str(_get_value(example, "id")),
        "source_language": _normalize_source_locale(
            _get_value(example, "source_locale", "source_language", default="en_US")
        ),
        "target_language": _normalize_target_locale(
            _get_value(example, "target_locale", "target_lang", "target_language")
        ),
        "text": source,
        "prediction": _safe_str(prediction),
    }


def predict_eamt_dataset(
    dataset: Sequence[EAMTExample | Mapping[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    memory_provider: Callable[[EAMTExample | Mapping[str, Any]], EntityMemoryBlock | None] | None = None,
    mode: str = "plain",
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool | None = None,
    generation_kwargs: Mapping[str, Any] | None = None,
    batch_size: int = 8,
    show_progress: bool = True,
    progress_desc: str = "Generating translations",
    progress_log_interval_seconds: float = 30.0,
    ercm_runner: Callable[
        [EAMTExample | Mapping[str, Any], TranslationDraft, EntityMemoryBlock | None],
        Any,
    ] | None = None,
) -> List[Dict[str, str]]:
    """
    EA-MT 데이터셋 전체에 대해 prediction JSONL 호환 레코드를 만든다.
    draft 생성 -> ERCM(optional) -> 최종 prediction record 생성
    """
    total_examples = len(dataset)
    effective_batch_size = max(1, int(batch_size))
    predictions: List[Dict[str, str]] = []

    _log_stage_event(
        "Translation Generation",
        "Start",
        examples=total_examples,
        batch_size=effective_batch_size,
        mode=mode,
        max_new_tokens=max_new_tokens,
    )

    if total_examples == 0:
        _log_stage_event("Translation Generation", "End", processed=0, elapsed="00:00")
        return predictions

    generation_start_time = time.monotonic()
    last_progress_log_time = generation_start_time
    progress_bar = None
    if show_progress and tqdm is not None:
        progress_bar = tqdm(
            total=total_examples,
            desc=progress_desc,
            unit="example",
            dynamic_ncols=True,
            mininterval=1.0,
            smoothing=0.1,
        )

    for batch_start in range(0, total_examples, effective_batch_size):
        batch_examples = list(dataset[batch_start : batch_start + effective_batch_size])
        batch_memories = [
            memory_provider(example) if memory_provider is not None else None
            for example in batch_examples
        ]

        batch_predictions: List[Dict[str, str]] = []

        for example, memory in zip(batch_examples, batch_memories):
            draft = generate_draft_translation(
                example,
                model,
                tokenizer,
                memory=memory,
                mode=mode,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                generation_kwargs=generation_kwargs,
            )

            prediction_text = draft.draft_text

            if ercm_runner is not None:
                ercm_output = ercm_runner(example, draft, memory)
                parsed_text = _extract_prediction_text_from_ercm_output(ercm_output)
                if parsed_text:
                    prediction_text = parsed_text

            batch_predictions.append(make_prediction_record(example, prediction_text))

        predictions.extend(batch_predictions)

        processed = len(predictions)
        now = time.monotonic()
        elapsed = max(now - generation_start_time, 1e-9)
        rate = processed / elapsed
        remaining = max(total_examples - processed, 0)
        eta = remaining / rate if rate > 0 else float("inf")

        if progress_bar is not None:
            progress_bar.update(len(batch_predictions))
            progress_bar.set_postfix_str(
                f"bs={len(batch_predictions)} avg={rate:.2f}/s eta={_format_seconds(eta)}"
            )

        if (
            now - last_progress_log_time >= float(progress_log_interval_seconds)
            or processed == total_examples
        ):
            _log_progress_summary(
                stage_name="Translation Generation",
                processed=processed,
                total=total_examples,
                start_time=generation_start_time,
                last_batch_size=len(batch_predictions),
            )
            last_progress_log_time = now

    if progress_bar is not None:
        progress_bar.close()

    total_elapsed = max(time.monotonic() - generation_start_time, 1e-9)
    _log_stage_event(
        "Translation Generation",
        "End",
        processed=len(predictions),
        elapsed=_format_seconds(total_elapsed),
        avg_rate=f"{len(predictions) / total_elapsed:.2f} ex/s",
    )

    return predictions


__all__ = [
    "DEFAULT_QWEN_MODEL_NAME",
    "DEFAULT_SYSTEM_PROMPT",
    "generate_draft_translation",
    "load_qwen2_5_instruct",
    "load_qwen2_5_runtime_resources",
    "make_prediction_record",
    "predict_eamt_dataset",
]
