from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate_path in (PROJECT_ROOT, SRC_ROOT):
    candidate_text = str(candidate_path)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    Dataset = object  # type: ignore[assignment]

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except Exception as peft_import_error:  # pragma: no cover
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    PEFT_IMPORT_ERROR_DETAIL = str(peft_import_error)
else:  # pragma: no cover
    PEFT_IMPORT_ERROR_DETAIL = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except Exception as transformers_import_error:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    Trainer = None
    TrainingArguments = None
    set_seed = None
    TRANSFORMERS_IMPORT_ERROR_DETAIL = str(transformers_import_error)
else:  # pragma: no cover
    TRANSFORMERS_IMPORT_ERROR_DETAIL = None

try:
    from DTOlist import CandidateEntity, EAMTExample, RuntimeResources
except Exception:  # pragma: no cover
    from src.DTOlist import CandidateEntity, EAMTExample, RuntimeResources

from eamt.data.db_loader import DEFAULT_SYSTEM_PROMPT, load_eamt_dataset_from_db
from eamt.kb.index import lookup_entity_by_qid
from eamt.kb.resources import build_runtime_resources_from_db
from eamt.memory.builder import build_entity_memory_block, render_entity_memory_text
from eamt.translation.entity_memory_pipeline import (
    SUPPORTED_ENTITY_PIPELINE_MODES,
    build_entity_memory_from_pipeline,
    load_entity_pipeline_artifacts,
)
from eamt.translation.prompting import build_translation_prompt


DEFAULT_QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
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


def _unique_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _normalize_target_lang(value: str) -> str:
    text = _safe_str(value)
    if "_" in text:
        return text.split("_", 1)[0]
    return text


def _extract_primary_qid(example: EAMTExample) -> str:
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


def _resolve_torch_dtype(dtype: str | None) -> Any:
    if dtype in (None, "", "auto"):
        return dtype or "auto"
    if torch is None:
        return dtype

    normalized = _safe_str(dtype)
    alias_map = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    normalized = alias_map.get(normalized, normalized)
    if not hasattr(torch, normalized):
        raise ValueError(f"지원하지 않는 torch dtype입니다: {dtype}")
    return getattr(torch, normalized)


def _get_record_value(record: Any, field_name: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def _truncate_description(text: str, max_chars: int) -> str:
    text = _safe_str(text)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _apply_memory_budget_to_record(
    record: Any,
    *,
    alias_limit: int,
    description_max_chars: int,
) -> Dict[str, Any] | None:
    if record is None:
        return None

    aliases = _safe_list(_get_record_value(record, "target_aliases", []))
    alias_values = aliases if alias_limit < 0 else aliases[:alias_limit]

    return {
        "qid": _safe_str(_get_record_value(record, "qid")),
        "label_en": _safe_str(_get_record_value(record, "label_en")),
        "aliases_en": _safe_list(_get_record_value(record, "aliases_en", [])),
        "target_lang": _safe_str(_get_record_value(record, "target_lang")),
        "target_label": _safe_str(_get_record_value(record, "target_label")),
        "target_aliases": alias_values,
        "entity_type": _safe_str(_get_record_value(record, "entity_type")),
        "description": _truncate_description(
            _safe_str(_get_record_value(record, "description")),
            max_chars=description_max_chars,
        ),
        "language_available": bool(_get_record_value(record, "language_available", True)),
        "popularity_score": float(_get_record_value(record, "popularity_score", 0.0) or 0.0),
    }


def _find_source_span(source: str, candidates: Sequence[str]) -> str:
    source_text = _safe_str(source)
    source_lower = source_text.casefold()

    for candidate in candidates:
        candidate_text = _safe_str(candidate)
        if not candidate_text:
            continue
        candidate_lower = candidate_text.casefold()
        start_index = source_lower.find(candidate_lower)
        if start_index >= 0:
            end_index = start_index + len(candidate_text)
            return source_text[start_index:end_index]
    return ""


def _record_to_candidate(record: Dict[str, Any] | None, *, source: str, qid: str) -> CandidateEntity:
    if record is None:
        return CandidateEntity(
            qid=qid,
            candidate_source="fallback_qid_only",
            target_label="",
            target_aliases=[],
            entity_type="",
            description="",
            popularity_score=0.0,
            alias_count=0,
            ambiguity_count=1,
            source_span="",
            span_match=None,
        )

    source_candidates = [record.get("label_en", "")] + _safe_list(record.get("aliases_en", []))
    source_span = _find_source_span(source, source_candidates)
    target_aliases = [
        alias
        for alias in _safe_list(record.get("target_aliases", []))
        if alias and alias != _safe_str(record.get("target_label"))
    ]
    target_aliases = _unique_keep_order(target_aliases)

    return CandidateEntity(
        qid=qid,
        candidate_source="gold_qid_lookup",
        target_label=_safe_str(record.get("target_label")),
        target_aliases=target_aliases,
        entity_type=_safe_str(record.get("entity_type")),
        description=_safe_str(record.get("description")),
        popularity_score=float(record.get("popularity_score") or 0.0),
        alias_count=len(target_aliases),
        ambiguity_count=1,
        source_span=source_span,
        span_match=None,
    )


def _build_memory_for_example(
    example: EAMTExample,
    *,
    resources: RuntimeResources,
    alias_limit: int,
    description_max_chars: int,
) -> Any:
    source = _safe_str(getattr(example, "source", ""))
    target_lang = _normalize_target_lang(_safe_str(getattr(example, "target_lang", "")))
    qid = _extract_primary_qid(example)

    records = lookup_entity_by_qid(resources.qid_index, qid, target_lang) if qid else None
    record = _apply_memory_budget_to_record(
        records[0] if records else None,
        alias_limit=alias_limit,
        description_max_chars=description_max_chars,
    )
    candidate = _record_to_candidate(record, source=source, qid=qid)
    memory = build_entity_memory_block(
        [candidate],
        source_sentence=source,
        alias_limit=alias_limit,
    )
    render_entity_memory_text(memory, target_lang=target_lang)
    return memory


def build_sft_sample(
    example: EAMTExample,
    *,
    mode: str,
    entity_pipeline_artifacts: Any | None,
    entity_pipeline_mode: str,
    retrieval_top_k: int,
    retrieval_per_surface_k: int,
    retrieval_min_char_len: int,
    retrieval_max_n: int,
    reranker_prior_bonus_weight: float,
    alias_limit: int,
    description_max_chars: int,
) -> Dict[str, str] | None:
    source = _safe_str(getattr(example, "source", ""))
    target = _safe_str(getattr(example, "target", ""))
    target_lang = _normalize_target_lang(_safe_str(getattr(example, "target_lang", "")))

    if not source or not target or not target_lang:
        return None

    prompt_mode = "plain"
    memory = None
    if mode == "entity-aware":
        if entity_pipeline_artifacts is None:
            raise ValueError("entity-aware 모드에는 entity pipeline artifacts가 필요합니다.")
        prompt_mode = "entity-aware"
        memory = build_entity_memory_from_pipeline(
            example,
            artifacts=entity_pipeline_artifacts,
            entity_pipeline_mode=entity_pipeline_mode,
            alias_limit=alias_limit,
            description_max_chars=description_max_chars,
            retrieval_top_k=retrieval_top_k,
            retrieval_per_surface_k=retrieval_per_surface_k,
            retrieval_min_char_len=retrieval_min_char_len,
            retrieval_max_n=retrieval_max_n,
            reranker_prior_bonus_weight=reranker_prior_bonus_weight,
        )

    prompt = build_translation_prompt(
        source=source,
        target_lang=target_lang,
        memory=memory,
        mode=prompt_mode,
    )
    return {
        "id": _safe_str(getattr(example, "id", "")),
        "prompt": prompt,
        "target": target,
        "mode": mode,
    }


def load_sft_samples_from_db(
    *,
    split: str,
    target_locale: str,
    source_locale: str,
    limit: int | None,
    mode: str,
    entity_pipeline_mode: str,
    retrieval_top_k: int,
    retrieval_per_surface_k: int,
    retrieval_min_char_len: int,
    retrieval_max_n: int,
    reranker_model_path: str | None,
    reranker_hidden_dim: int,
    reranker_dropout: float,
    reranker_device: str | None,
    reranker_prior_bonus_weight: float,
    alias_limit: int,
    description_max_chars: int,
) -> List[Dict[str, str]]:
    bundle = load_eamt_dataset_from_db(
        split=split,
        target_locale=target_locale,
        source_locale=source_locale,
        limit=limit,
        require_references=True,
        require_mentions=False,
    )

    entity_pipeline_artifacts = None
    if mode == "entity-aware":
        entity_pipeline_artifacts = load_entity_pipeline_artifacts(
            target_lang=_normalize_target_lang(target_locale),
            entity_pipeline_mode=entity_pipeline_mode,
            reranker_model_path=reranker_model_path,
            reranker_device=reranker_device,
            reranker_hidden_dim=reranker_hidden_dim,
            reranker_dropout=reranker_dropout,
        )

    samples: List[Dict[str, str]] = []
    for example in bundle.examples:
        sample = build_sft_sample(
            example,
            mode=mode,
            entity_pipeline_artifacts=entity_pipeline_artifacts,
            entity_pipeline_mode=entity_pipeline_mode,
            retrieval_top_k=retrieval_top_k,
            retrieval_per_surface_k=retrieval_per_surface_k,
            retrieval_min_char_len=retrieval_min_char_len,
            retrieval_max_n=retrieval_max_n,
            reranker_prior_bonus_weight=reranker_prior_bonus_weight,
            alias_limit=alias_limit,
            description_max_chars=description_max_chars,
        )
        if sample is not None:
            samples.append(sample)
    return samples


def _build_messages(
    *,
    prompt_text: str,
    target_text: str | None,
    system_prompt: str | None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if _safe_str(system_prompt):
        messages.append({"role": "system", "content": _safe_str(system_prompt)})
    messages.append({"role": "user", "content": _safe_str(prompt_text)})
    if target_text is not None:
        messages.append({"role": "assistant", "content": _safe_str(target_text)})
    return messages


def _render_messages_without_template(
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    rendered_lines: List[str] = []
    for message in messages:
        role = _safe_str(message.get("role")).upper() or "USER"
        content = _safe_str(message.get("content"))
        rendered_lines.append(f"[{role}]")
        rendered_lines.append(content)
    if add_generation_prompt:
        rendered_lines.append("[ASSISTANT]")
    return "\n".join(rendered_lines)


def _tokenize_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> List[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        token_ids = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        return [int(token_id) for token_id in token_ids]

    rendered_text = _render_messages_without_template(
        messages,
        add_generation_prompt=add_generation_prompt,
    )
    encoded = tokenizer(rendered_text, add_special_tokens=True)
    return [int(token_id) for token_id in encoded["input_ids"]]


def _shared_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    prefix_length = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        prefix_length += 1
    return prefix_length


class QwenCompletionOnlyDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Mapping[str, str]],
        tokenizer: Any,
        *,
        system_prompt: str | None,
        max_length: int,
    ) -> None:
        self.features: List[Dict[str, List[int]]] = []
        self.skipped_examples = 0

        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        for sample in samples:
            prompt_text = _safe_str(sample.get("prompt"))
            target_text = _safe_str(sample.get("target"))
            if not prompt_text or not target_text:
                self.skipped_examples += 1
                continue

            prompt_messages = _build_messages(
                prompt_text=prompt_text,
                target_text=None,
                system_prompt=system_prompt,
            )
            full_messages = _build_messages(
                prompt_text=prompt_text,
                target_text=target_text,
                system_prompt=system_prompt,
            )

            prompt_ids = _tokenize_messages(
                tokenizer,
                prompt_messages,
                add_generation_prompt=True,
            )
            full_ids = _tokenize_messages(
                tokenizer,
                full_messages,
                add_generation_prompt=False,
            )

            if eos_token_id is not None and (not full_ids or full_ids[-1] != eos_token_id):
                full_ids = list(full_ids) + [int(eos_token_id)]

            if max_length > 0 and len(full_ids) > max_length:
                full_ids = list(full_ids[:max_length])

            prefix_length = _shared_prefix_length(prompt_ids, full_ids)
            if prefix_length >= len(full_ids):
                self.skipped_examples += 1
                continue

            labels = list(full_ids)
            for index in range(prefix_length):
                labels[index] = -100

            if all(label == -100 for label in labels):
                self.skipped_examples += 1
                continue

            self.features.append(
                {
                    "input_ids": list(full_ids),
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.features[index]


class CompletionOnlyDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.pad_token_id = int(pad_token_id if pad_token_id is not None else eos_token_id or 0)

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> Dict[str, Any]:
        if torch is None:
            raise ImportError("PyTorch가 설치되어 있어야 data collator를 사용할 수 있습니다.")

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []

        for feature in features:
            pad_width = max_length - len(feature["input_ids"])
            input_ids.append(list(feature["input_ids"]) + [self.pad_token_id] * pad_width)
            attention_mask.append(list(feature["attention_mask"]) + [0] * pad_width)
            labels.append(list(feature["labels"]) + [-100] * pad_width)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _require_training_stack() -> None:
    if torch is None:
        raise ImportError("LoRA 학습에는 PyTorch가 필요합니다.")
    if AutoModelForCausalLM is None or AutoTokenizer is None or Trainer is None or TrainingArguments is None:
        detail = (
            f" 현재 감지된 상태: {TRANSFORMERS_IMPORT_ERROR_DETAIL}"
            if TRANSFORMERS_IMPORT_ERROR_DETAIL
            else ""
        )
        raise ImportError(
            "`transformers`가 정상 import 되어야 Qwen LoRA 학습을 실행할 수 있습니다."
            + detail
        )
    if LoraConfig is None or TaskType is None or get_peft_model is None:
        detail = f" 현재 감지된 상태: {PEFT_IMPORT_ERROR_DETAIL}" if PEFT_IMPORT_ERROR_DETAIL else ""
        raise ImportError(
            "`peft`가 설치되어 있어야 LoRA 어댑터를 생성할 수 있습니다."
            + detail
        )


def _parse_target_modules(raw_value: str, model: Any) -> List[str]:
    if _safe_str(raw_value).lower() != "auto":
        return [module_name.strip() for module_name in raw_value.split(",") if module_name.strip()]

    available = {
        module_name.rsplit(".", 1)[-1]
        for module_name, _module in model.named_modules()
    }
    resolved = [name for name in DEFAULT_LORA_TARGET_MODULES if name in available]
    return resolved or list(DEFAULT_LORA_TARGET_MODULES)


def _count_parameters(model: Any) -> Dict[str, float]:
    total_parameters = 0
    trainable_parameters = 0

    for parameter in model.parameters():
        parameter_count = int(parameter.numel())
        total_parameters += parameter_count
        if parameter.requires_grad:
            trainable_parameters += parameter_count

    ratio = 0.0
    if total_parameters > 0:
        ratio = 100.0 * trainable_parameters / total_parameters

    return {
        "trainable_parameters": float(trainable_parameters),
        "total_parameters": float(total_parameters),
        "trainable_ratio_percent": ratio,
    }


def _supports_training_argument(name: str) -> bool:
    if TrainingArguments is None:
        return False
    return name in inspect.signature(TrainingArguments.__init__).parameters


def _build_training_arguments(
    *,
    output_dir: str,
    num_train_epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    lr_scheduler_type: str,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    save_total_limit: int,
    dataloader_num_workers: int,
    seed: int,
    has_eval_dataset: bool,
    bf16: bool,
    fp16: bool,
    gradient_checkpointing: bool,
) -> Any:
    if TrainingArguments is None:
        raise ImportError("transformers.TrainingArguments를 import할 수 없습니다.")

    kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "dataloader_num_workers": dataloader_num_workers,
        "seed": seed,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": gradient_checkpointing,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "optim": "adamw_torch",
        "logging_strategy": "steps",
        "save_strategy": "steps",
        "report_to": [],
    }

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        kwargs["ddp_find_unused_parameters"] = False

    evaluation_key = "eval_strategy" if _supports_training_argument("eval_strategy") else "evaluation_strategy"
    kwargs[evaluation_key] = "steps" if has_eval_dataset else "no"

    if has_eval_dataset:
        kwargs["eval_steps"] = eval_steps
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False

    if _supports_training_argument("save_safetensors"):
        kwargs["save_safetensors"] = True

    return TrainingArguments(**kwargs)


def _prepare_tokenizer(
    *,
    model_name: str,
    local_files_only: bool,
    trust_remote_code: bool,
    use_fast_tokenizer: bool,
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast_tokenizer,
    )
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "right":
        tokenizer.padding_side = "right"
    return tokenizer


def _prepare_model(
    *,
    model_name: str,
    torch_dtype: str,
    local_files_only: bool,
    trust_remote_code: bool,
    device_map: str | None,
    attn_implementation: str | None,
    load_in_4bit: bool,
    load_in_8bit: bool,
    gradient_checkpointing: bool,
) -> Any:
    model_kwargs: Dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": _resolve_torch_dtype(torch_dtype),
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if load_in_4bit or load_in_8bit:
        if BitsAndBytesConfig is None:
            raise ImportError(
                "4-bit/8-bit 로딩에는 bitsandbytes 지원이 포함된 transformers가 필요합니다."
            )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        model_kwargs["device_map"] = device_map or "auto"
        model_kwargs["low_cpu_mem_usage"] = True
    elif device_map:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    if load_in_4bit or load_in_8bit:
        if prepare_model_for_kbit_training is None:
            raise ImportError("prepare_model_for_kbit_training을 사용할 수 없습니다.")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    return model


def train_qwen_lora(
    *,
    model_name: str,
    output_dir: str,
    target_locale: str,
    source_locale: str,
    train_split: str,
    eval_split: str | None,
    train_limit: int | None,
    eval_limit: int | None,
    mode: str,
    entity_pipeline_mode: str,
    retrieval_top_k: int,
    retrieval_per_surface_k: int,
    retrieval_min_char_len: int,
    retrieval_max_n: int,
    reranker_model_path: str | None,
    reranker_hidden_dim: int,
    reranker_dropout: float,
    reranker_device: str | None,
    reranker_prior_bonus_weight: float,
    max_length: int,
    system_prompt: str | None,
    num_train_epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    lr_scheduler_type: str,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    save_total_limit: int,
    dataloader_num_workers: int,
    seed: int,
    bf16: bool,
    fp16: bool,
    gradient_checkpointing: bool,
    torch_dtype: str,
    device_map: str | None,
    attn_implementation: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
    use_fast_tokenizer: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
    lora_target_modules: str,
    alias_limit: int,
    description_max_chars: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    resume_from_checkpoint: str | None,
) -> Dict[str, Any]:
    _require_training_stack()

    if load_in_4bit and load_in_8bit:
        raise ValueError("4-bit와 8-bit 양자화 옵션은 동시에 사용할 수 없습니다.")
    if bf16 and fp16:
        raise ValueError("bf16과 fp16은 동시에 활성화할 수 없습니다.")
    if mode not in {"plain", "entity-aware"}:
        raise ValueError(f"지원하지 않는 학습 모드입니다: {mode}")
    if entity_pipeline_mode not in SUPPORTED_ENTITY_PIPELINE_MODES:
        raise ValueError(
            f"지원하지 않는 entity pipeline mode입니다: {entity_pipeline_mode}. "
            f"지원 목록: {SUPPORTED_ENTITY_PIPELINE_MODES}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if set_seed is not None:
        set_seed(seed)

    load_start = time.monotonic()
    print(
        "[Qwen LoRA] 데이터셋 로딩 시작 "
        f"| train_split={train_split} | eval_split={eval_split or 'none'} | mode={mode}",
        flush=True,
    )

    train_samples = load_sft_samples_from_db(
        split=train_split,
        target_locale=target_locale,
        source_locale=source_locale,
        limit=train_limit,
        mode=mode,
        entity_pipeline_mode=entity_pipeline_mode,
        retrieval_top_k=retrieval_top_k,
        retrieval_per_surface_k=retrieval_per_surface_k,
        retrieval_min_char_len=retrieval_min_char_len,
        retrieval_max_n=retrieval_max_n,
        reranker_model_path=reranker_model_path,
        reranker_hidden_dim=reranker_hidden_dim,
        reranker_dropout=reranker_dropout,
        reranker_device=reranker_device,
        reranker_prior_bonus_weight=reranker_prior_bonus_weight,
        alias_limit=alias_limit,
        description_max_chars=description_max_chars,
    )

    eval_samples: List[Dict[str, str]] = []
    normalized_eval_split = _safe_str(eval_split).lower()
    if normalized_eval_split and normalized_eval_split != "none":
        eval_samples = load_sft_samples_from_db(
            split=eval_split or "validation",
            target_locale=target_locale,
            source_locale=source_locale,
            limit=eval_limit,
            mode=mode,
            entity_pipeline_mode=entity_pipeline_mode,
            retrieval_top_k=retrieval_top_k,
            retrieval_per_surface_k=retrieval_per_surface_k,
            retrieval_min_char_len=retrieval_min_char_len,
            retrieval_max_n=retrieval_max_n,
            reranker_model_path=reranker_model_path,
            reranker_hidden_dim=reranker_hidden_dim,
            reranker_dropout=reranker_dropout,
            reranker_device=reranker_device,
            reranker_prior_bonus_weight=reranker_prior_bonus_weight,
            alias_limit=alias_limit,
            description_max_chars=description_max_chars,
        )

    print(
        "[Qwen LoRA] 데이터셋 로딩 완료 "
        f"| train_samples={len(train_samples)} | eval_samples={len(eval_samples)} "
        f"| elapsed={time.monotonic() - load_start:.2f}s",
        flush=True,
    )

    tokenizer = _prepare_tokenizer(
        model_name=model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        use_fast_tokenizer=use_fast_tokenizer,
    )

    train_dataset = QwenCompletionOnlyDataset(
        train_samples,
        tokenizer,
        system_prompt=system_prompt,
        max_length=max_length,
    )
    eval_dataset = QwenCompletionOnlyDataset(
        eval_samples,
        tokenizer,
        system_prompt=system_prompt,
        max_length=max_length,
    ) if eval_samples else None

    if len(train_dataset) == 0:
        raise ValueError("학습에 사용할 수 있는 샘플이 없습니다. DB 데이터와 max_length 설정을 확인하세요.")

    model = _prepare_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        attn_implementation=attn_implementation,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        gradient_checkpointing=gradient_checkpointing,
    )

    target_modules = _parse_target_modules(lora_target_modules, model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    parameter_stats = _count_parameters(model)
    print(
        "[Qwen LoRA] PEFT 적용 완료 "
        f"| target_modules={','.join(target_modules)} "
        f"| trainable={int(parameter_stats['trainable_parameters'])} "
        f"| total={int(parameter_stats['total_parameters'])} "
        f"| ratio={parameter_stats['trainable_ratio_percent']:.4f}%",
        flush=True,
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    training_args = _build_training_arguments(
        output_dir=str(output_path),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
        has_eval_dataset=eval_dataset is not None and len(eval_dataset) > 0,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionOnlyDataCollator(tokenizer),
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    train_metrics = dict(train_result.metrics)

    eval_metrics = None
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_metrics = dict(trainer.evaluate())

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    effective_entity_pipeline_mode = entity_pipeline_mode if mode == "entity-aware" else None
    candidate_pipeline_summary = {
        "mode": effective_entity_pipeline_mode,
        "retrieval_top_k": retrieval_top_k if mode == "entity-aware" else None,
        "retrieval_per_surface_k": retrieval_per_surface_k if mode == "entity-aware" else None,
        "retrieval_min_char_len": retrieval_min_char_len if mode == "entity-aware" else None,
        "retrieval_max_n": retrieval_max_n if mode == "entity-aware" else None,
        "reranker_model_path": reranker_model_path if mode == "entity-aware" else None,
        "reranker_hidden_dim": reranker_hidden_dim if mode == "entity-aware" else None,
        "reranker_dropout": reranker_dropout if mode == "entity-aware" else None,
        "reranker_device": reranker_device if mode == "entity-aware" else None,
        "reranker_prior_bonus_weight": reranker_prior_bonus_weight if mode == "entity-aware" else None,
        "alias_limit": alias_limit if mode == "entity-aware" else None,
        "description_max_chars": description_max_chars if mode == "entity-aware" else None,
    }

    summary = {
        "model_name": model_name,
        "output_dir": str(output_path),
        "target_locale": target_locale,
        "source_locale": source_locale,
        "train_split": train_split,
        "eval_split": eval_split,
        "mode": mode,
        "entity_pipeline_mode": effective_entity_pipeline_mode,
        "max_length": max_length,
        "system_prompt": system_prompt,
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "train_features": len(train_dataset),
        "eval_features": len(eval_dataset) if eval_dataset is not None else 0,
        "skipped_train_samples": train_dataset.skipped_examples,
        "skipped_eval_samples": eval_dataset.skipped_examples if eval_dataset is not None else 0,
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "bias": lora_bias,
            "target_modules": target_modules,
        },
        "candidate_pipeline": candidate_pipeline_summary,
        "quantization": {
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
        },
        "parameter_stats": parameter_stats,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }

    summary_path = output_path / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Qwen LoRA] 어댑터 저장 완료 | output_dir={output_path}", flush=True)
    print(f"[Qwen LoRA] 요약 저장 완료 | summary={summary_path}", flush=True)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Hugging Face Qwen model with PEFT LoRA using EA-MT data from MySQL.",
    )
    parser.add_argument("--model-name", default=DEFAULT_QWEN_MODEL_NAME, help="Hugging Face model id")
    parser.add_argument("--output-dir", required=True, help="Directory to save the LoRA adapter")
    parser.add_argument("--target-locale", required=True, help="DB target locale, e.g. ko, fr, ja")
    parser.add_argument("--source-locale", default="en", help="DB source locale filter. Default: en")
    parser.add_argument("--train-split", default="train", help="Training split. Default: train")
    parser.add_argument(
        "--eval-split",
        default="validation",
        help="Evaluation split. Use `none` to disable eval. Default: validation",
    )
    parser.add_argument("--train-limit", type=int, default=None, help="Optional row limit for train split")
    parser.add_argument("--eval-limit", type=int, default=None, help="Optional row limit for eval split")
    parser.add_argument(
        "--mode",
        default="plain",
        choices=["plain", "entity-aware"],
        help="Prompt style for SFT. Default: plain",
    )
    parser.add_argument(
        "--entity-pipeline-mode",
        default="anchored",
        choices=list(SUPPORTED_ENTITY_PIPELINE_MODES),
        help=(
            "How entity-aware memory is built before LoRA training. "
            "`anchored`=gold QID lookup only, `surface`=surface retrieval only, "
            "`retrieve`=anchored+surface top-1, `rerank`=anchored+surface+reranker."
        ),
    )
    parser.add_argument("--retrieval-top-k", type=int, default=10, help="Candidate pool size before selection")
    parser.add_argument(
        "--retrieval-per-surface-k",
        type=int,
        default=5,
        help="Max candidates collected per surface span",
    )
    parser.add_argument(
        "--retrieval-min-char-len",
        type=int,
        default=2,
        help="Minimum normalized span length for retrieval",
    )
    parser.add_argument(
        "--retrieval-max-n",
        type=int,
        default=5,
        help="Maximum n-gram length used for surface retrieval",
    )
    parser.add_argument(
        "--reranker-model-path",
        default=None,
        help="Optional trained reranker checkpoint path for `--entity-pipeline-mode rerank`",
    )
    parser.add_argument(
        "--reranker-hidden-dim",
        type=int,
        default=128,
        help="Hidden size used when loading the reranker checkpoint",
    )
    parser.add_argument(
        "--reranker-dropout",
        type=float,
        default=0.1,
        help="Dropout used when loading the reranker checkpoint",
    )
    parser.add_argument(
        "--reranker-device",
        default=None,
        help="Optional device for reranker inference, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--reranker-prior-bonus-weight",
        type=float,
        default=0.1,
        help="Prior bonus weight applied during reranking",
    )
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum tokenized sequence length")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt inserted before the user translation prompt",
    )
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 mixed precision")
    parser.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing. Default: enabled",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Model loading dtype. Example: auto, bfloat16, float16, float32",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Optional transformers device_map value. Usually leave unset for Trainer/DDP runs.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional attention backend such as flash_attention_2 or sdpa",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from the local Hugging Face cache",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model/tokenizer",
    )
    parser.add_argument(
        "--use-fast-tokenizer",
        dest="use_fast_tokenizer",
        action="store_true",
        default=True,
        help="Use the fast tokenizer backend when available. Default: enabled",
    )
    parser.add_argument(
        "--no-fast-tokenizer",
        dest="use_fast_tokenizer",
        action="store_false",
        help="Disable the fast tokenizer backend",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-bias",
        default="none",
        choices=["none", "all", "lora_only"],
        help="PEFT bias handling. Default: none",
    )
    parser.add_argument(
        "--lora-target-modules",
        default="auto",
        help=(
            "Comma-separated LoRA target modules or `auto`. "
            "Auto resolves Qwen defaults such as q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj."
        ),
    )
    parser.add_argument("--alias-limit", type=int, default=1, help="Entity-aware alias budget")
    parser.add_argument(
        "--description-max-chars",
        type=int,
        default=80,
        help="Entity-aware description length budget",
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load the base model in 8-bit")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Optional checkpoint path for Trainer resume",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = train_qwen_lora(
            model_name=args.model_name,
            output_dir=args.output_dir,
            target_locale=args.target_locale,
            source_locale=args.source_locale,
            train_split=args.train_split,
            eval_split=args.eval_split,
            train_limit=args.train_limit,
            eval_limit=args.eval_limit,
            mode=args.mode,
            entity_pipeline_mode=args.entity_pipeline_mode,
            retrieval_top_k=args.retrieval_top_k,
            retrieval_per_surface_k=args.retrieval_per_surface_k,
            retrieval_min_char_len=args.retrieval_min_char_len,
            retrieval_max_n=args.retrieval_max_n,
            reranker_model_path=args.reranker_model_path,
            reranker_hidden_dim=args.reranker_hidden_dim,
            reranker_dropout=args.reranker_dropout,
            reranker_device=args.reranker_device,
            reranker_prior_bonus_weight=args.reranker_prior_bonus_weight,
            max_length=args.max_length,
            system_prompt=args.system_prompt,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            dataloader_num_workers=args.dataloader_num_workers,
            seed=args.seed,
            bf16=bool(args.bf16),
            fp16=bool(args.fp16),
            gradient_checkpointing=bool(args.gradient_checkpointing),
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            use_fast_tokenizer=bool(args.use_fast_tokenizer),
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_bias=args.lora_bias,
            lora_target_modules=args.lora_target_modules,
            alias_limit=args.alias_limit,
            description_max_chars=args.description_max_chars,
            load_in_4bit=bool(args.load_in_4bit),
            load_in_8bit=bool(args.load_in_8bit),
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except Exception as error:
        print(f"[Qwen LoRA] 실행 실패 | {error}", file=sys.stderr, flush=True)
        return 1

    print(json.dumps(summary, ensure_ascii=False))
    return 0


__all__ = [
    "QwenCompletionOnlyDataset",
    "CompletionOnlyDataCollator",
    "build_parser",
    "build_sft_sample",
    "load_sft_samples_from_db",
    "main",
    "train_qwen_lora",
]
