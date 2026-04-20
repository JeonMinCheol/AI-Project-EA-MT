from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

try:
    import pymysql
    from pymysql.cursors import DictCursor
except Exception:  # pragma: no cover
    pymysql = None
    DictCursor = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from DTOlist import EAMTExample
except Exception:  # pragma: no cover
    from src.DTOlist import EAMTExample


DEFAULT_QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional translator for the SemEval EA-MT task. "
    "Return only the final translation in the target language."
)


@dataclass
class EAMTDBDatasetBundle:
    examples: List[EAMTExample]
    references: List[Dict[str, Any]]
    by_dataset_id: Dict[str, Dict[str, Any]]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _require_pymysql() -> None:
    if pymysql is None or DictCursor is None:
        raise ImportError("DB 로더를 사용하려면 `pymysql` 패키지가 설치되어 있어야 합니다.")


def _load_dotenv_if_available() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _build_in_placeholders(size: int) -> str:
    if size <= 0:
        raise ValueError("IN 절 placeholder 개수는 1 이상이어야 합니다.")
    return ", ".join(["%s"] * size)


def _unique_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _timestamp_text() -> str:
    return time.strftime("%F %T")


def _format_seconds(seconds: float) -> str:
    whole_seconds = max(0, int(seconds))
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


def load_db_config_from_env() -> Dict[str, Any]:
    _load_dotenv_if_available()

    return {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "db": os.getenv("DB_NAME"),
        "charset": "utf8mb4",
        "cursorclass": DictCursor,
    }


def connect_mysql_from_env(**overrides: Any) -> Any:
    _require_pymysql()
    config = load_db_config_from_env()
    config.update(overrides)
    return pymysql.connect(**config)


def _fetch_examples(
    conn: Any,
    *,
    split: str,
    target_locale: str | None = None,
    source_locale: str | None = None,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    where_clauses = ["split = %s"]
    params: List[Any] = [split]

    if target_locale:
        where_clauses.append("target_locale = %s")
        params.append(target_locale)
    if source_locale:
        where_clauses.append("source_locale = %s")
        params.append(source_locale)

    query = f"""
        SELECT
            example_id,
            dataset_id,
            split,
            source_locale,
            target_locale,
            source_text
        FROM eamt_example
        WHERE {" AND ".join(where_clauses)}
        ORDER BY example_id
    """
    if limit is not None:
        query += " LIMIT %s"
        params.append(int(limit))

    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        return list(cur.fetchall())


def _fetch_references(conn: Any, example_ids: Sequence[int]) -> List[Dict[str, Any]]:
    if not example_ids:
        return []

    placeholders = _build_in_placeholders(len(example_ids))
    query = f"""
        SELECT
            example_id,
            translation_text,
            mention_text,
            reference_order
        FROM example_reference
        WHERE example_id IN ({placeholders})
        ORDER BY example_id, reference_order, reference_id
    """
    with conn.cursor() as cur:
        cur.execute(query, tuple(example_ids))
        return list(cur.fetchall())


def _fetch_entities_and_types(conn: Any, example_ids: Sequence[int]) -> List[Dict[str, Any]]:
    if not example_ids:
        return []

    placeholders = _build_in_placeholders(len(example_ids))
    query = f"""
        SELECT
            ee.example_id,
            ee.qid,
            ee.entity_order,
            et.type_text
        FROM example_entity AS ee
        LEFT JOIN entity_type AS et
            ON ee.qid = et.qid
        WHERE ee.example_id IN ({placeholders})
        ORDER BY ee.example_id, ee.entity_order, et.type_text
    """
    with conn.cursor() as cur:
        cur.execute(query, tuple(example_ids))
        return list(cur.fetchall())


def _fetch_entity_terms(
    conn: Any,
    qids: Sequence[str],
    lang_codes: Sequence[str],
) -> List[Dict[str, Any]]:
    if not qids or not lang_codes:
        return []

    qid_placeholders = _build_in_placeholders(len(qids))
    lang_placeholders = _build_in_placeholders(len(lang_codes))
    query = f"""
        SELECT
            qid,
            lang_code,
            term_type,
            term_text
        FROM entity_term
        WHERE qid IN ({qid_placeholders})
          AND lang_code IN ({lang_placeholders})
          AND term_type IN ('label', 'alias')
        ORDER BY qid, lang_code, FIELD(term_type, 'label', 'alias'), term_id
    """
    params = tuple(qids) + tuple(lang_codes)
    with conn.cursor() as cur:
        cur.execute(query, params)
        return list(cur.fetchall())


def load_eamt_dataset_from_db(
    *,
    conn: Any | None = None,
    split: str = "validation",
    target_locale: str | None = None,
    source_locale: str | None = None,
    limit: int | None = None,
    require_references: bool = True,
    require_mentions: bool = True,
) -> EAMTDBDatasetBundle:
    """
    DB의 EA-MT 샘플을 읽어서 모델 입력용 examples와 평가용 references를 함께 반환한다.
    """
    _require_pymysql()

    created_connection = conn is None
    if created_connection:
        conn = connect_mysql_from_env()

    try:
        example_rows = _fetch_examples(
            conn,
            split=split,
            target_locale=target_locale,
            source_locale=source_locale,
            limit=limit,
        )

        if not example_rows:
            return EAMTDBDatasetBundle(examples=[], references=[], by_dataset_id={})

        example_ids = [int(row["example_id"]) for row in example_rows]
        reference_rows = _fetch_references(conn, example_ids)
        entity_rows = _fetch_entities_and_types(conn, example_ids)

        references_by_example: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in reference_rows:
            references_by_example[int(row["example_id"])].append(
                {
                    "translation": _safe_str(row.get("translation_text")),
                    "mention": _safe_str(row.get("mention_text")),
                    "reference_order": row.get("reference_order"),
                }
            )

        qids_by_example: Dict[int, List[tuple[int, str]]] = defaultdict(list)
        types_by_example: Dict[int, List[str]] = defaultdict(list)

        for row in entity_rows:
            example_id = int(row["example_id"])
            qid = _safe_str(row.get("qid"))
            entity_order = int(row.get("entity_order") or 0)
            type_text = _safe_str(row.get("type_text"))

            if qid:
                qids_by_example[example_id].append((entity_order, qid))
            if type_text:
                types_by_example[example_id].append(type_text)

        unique_qids = _unique_keep_order(
            [
                qid
                for qid_list in qids_by_example.values()
                for _, qid in sorted(qid_list, key=lambda item: item[0])
            ]
        )
        unique_target_locales = _unique_keep_order(
            [_safe_str(row.get("target_locale")) for row in example_rows if _safe_str(row.get("target_locale"))]
        )
        term_rows = _fetch_entity_terms(conn, unique_qids, unique_target_locales)
        terms_by_qid_lang: Dict[tuple[str, str], List[str]] = defaultdict(list)
        for row in term_rows:
            key = (_safe_str(row.get("qid")), _safe_str(row.get("lang_code")))
            term_text = _safe_str(row.get("term_text"))
            if term_text:
                terms_by_qid_lang[key].append(term_text)

        examples: List[EAMTExample] = []
        references: List[Dict[str, Any]] = []
        by_dataset_id: Dict[str, Dict[str, Any]] = {}
        missing_reference_dataset_ids: List[str] = []
        missing_mention_dataset_ids: List[str] = []

        for row in example_rows:
            example_id = int(row["example_id"])
            dataset_id = _safe_str(row.get("dataset_id"))
            source_text = _safe_str(row.get("source_text"))
            source_locale_value = _safe_str(row.get("source_locale"))
            target_locale_value = _safe_str(row.get("target_locale"))
            split_value = _safe_str(row.get("split"))

            ordered_qids = [
                qid
                for _, qid in sorted(qids_by_example.get(example_id, []), key=lambda item: item[0])
            ]
            ordered_qids = _unique_keep_order(ordered_qids)
            entity_types = _unique_keep_order(types_by_example.get(example_id, []))
            recovered_mentions = _unique_keep_order(
                [
                    term_text
                    for qid in ordered_qids
                    for term_text in terms_by_qid_lang.get((qid, target_locale_value), [])
                ]
            )

            target_items = [
                {
                    "translation": item["translation"],
                    "mention": item["mention"] or (recovered_mentions[0] if recovered_mentions else ""),
                    "mention_candidates": _unique_keep_order(
                        ([item["mention"]] if item["mention"] else []) + recovered_mentions
                    ),
                }
                for item in references_by_example.get(example_id, [])
                if item["translation"]
            ]

            if require_references and not target_items:
                missing_reference_dataset_ids.append(dataset_id)
                continue

            has_any_mentions = any(
                bool(item.get("mention")) or bool(item.get("mention_candidates"))
                for item in target_items
            )
            if require_references and require_mentions and target_items and not has_any_mentions:
                missing_mention_dataset_ids.append(dataset_id)
                continue

            example = EAMTExample(
                id=dataset_id,
                source=source_text,
                target_lang=target_locale_value,
                target=target_items[0]["translation"] if target_items else None,
                wikidata_id=ordered_qids[0] if ordered_qids else None,
                entity_qids=ordered_qids,
                meta={
                    "example_id": example_id,
                    "split": split_value,
                    "source_locale": source_locale_value,
                    "target_locale": target_locale_value,
                    "targets": target_items,
                    "entity_types": entity_types,
                },
            )
            examples.append(example)

            reference_record = {
                "id": dataset_id,
                "wikidata_id": ordered_qids[0] if ordered_qids else None,
                "entity_types": entity_types,
                "source": source_text,
                "targets": target_items,
                "source_locale": source_locale_value,
                "target_locale": target_locale_value,
            }
            references.append(reference_record)
            by_dataset_id[dataset_id] = {
                "example": example,
                "reference": reference_record,
            }

        if require_references and missing_reference_dataset_ids:
            sample_ids = ", ".join(missing_reference_dataset_ids[:5])
            raise ValueError(
                f"`{split}` split에서 reference가 없는 샘플이 {len(missing_reference_dataset_ids)}개 있습니다. "
                f"예시: {sample_ids}. M-ETA/COMET/조화 평균 계산에는 example_reference가 필요합니다. "
                "baseline 평가는 보통 validation split을 사용하세요."
            )

        if require_references and require_mentions and missing_mention_dataset_ids:
            sample_ids = ", ".join(missing_mention_dataset_ids[:5])
            raise ValueError(
                f"`{split}` split에서 mention 후보를 만들 수 없는 샘플이 {len(missing_mention_dataset_ids)}개 있습니다. "
                f"예시: {sample_ids}. example_reference.mention_text 또는 "
                "entity_term(label/alias, target_locale)이 필요합니다."
            )

        return EAMTDBDatasetBundle(
            examples=examples,
            references=references,
            by_dataset_id=by_dataset_id,
        )
    finally:
        if created_connection and conn is not None:
            conn.close()


def load_test_eamt_dataset_from_db(
    *,
    conn: Any | None = None,
    target_locale: str | None = None,
    source_locale: str | None = None,
    limit: int | None = None,
    require_references: bool = True,
) -> EAMTDBDatasetBundle:
    return load_eamt_dataset_from_db(
        conn=conn,
        split="test",
        target_locale=target_locale,
        source_locale=source_locale,
        limit=limit,
        require_references=require_references,
    )


def load_validation_eamt_dataset_from_db(
    *,
    conn: Any | None = None,
    target_locale: str | None = None,
    source_locale: str | None = None,
    limit: int | None = None,
    require_references: bool = True,
) -> EAMTDBDatasetBundle:
    return load_eamt_dataset_from_db(
        conn=conn,
        split="validation",
        target_locale=target_locale,
        source_locale=source_locale,
        limit=limit,
        require_references=require_references,
    )


def evaluate_qwen_baseline_from_db(
    *,
    conn: Any | None = None,
    split: str = "validation",
    target_locale: str | None = None,
    source_locale: str | None = None,
    limit: int | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_name: str = DEFAULT_QWEN_MODEL_NAME,
    mode: str = "plain",
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool | None = None,
    generation_kwargs: Mapping[str, Any] | None = None,
    comet_model_name: str = "Unbabel/wmt22-comet-da",
    comet_batch_size: int = 8,
    comet_num_gpus: int = 1,
    load_model_kwargs: Mapping[str, Any] | None = None,
    prediction_output_path: str | None = None,
    generation_batch_size: int = 8,
    show_progress: bool = True,
    progress_log_interval_seconds: float = 30.0,
    entity_pipeline_mode: str = "anchored",
    retrieval_top_k: int = 10,
    retrieval_per_surface_k: int = 5,
    retrieval_min_char_len: int = 2,
    retrieval_max_n: int = 5,
    reranker_model_path: str | None = None,
    reranker_hidden_dim: int = 128,
    reranker_dropout: float = 0.1,
    reranker_device: str | None = None,
    reranker_prior_bonus_weight: float = 0.1,
    alias_limit: int = 1,
    description_max_chars: int = 80,
) -> Dict[str, Any]:
    """
    DB에서 split을 읽어 Qwen baseline prediction과 M-ETA/COMET/final score를 계산한다.
    """
    try:
        from eamt.translation.evaluation import (
            evaluate_qwen_on_eamt,
            save_predictions_jsonl,
        )
    except Exception:  # pragma: no cover
        from ..translation.evaluation import (
            evaluate_qwen_on_eamt,
            save_predictions_jsonl,
        )

    dataset_load_start_time = time.monotonic()
    _log_stage_event(
        "EA-MT Dataset Load",
        "Start",
        split=split,
        target_locale=target_locale,
        source_locale=source_locale,
        limit=limit,
    )
    bundle = load_eamt_dataset_from_db(
        conn=conn,
        split=split,
        target_locale=target_locale,
        source_locale=source_locale,
        limit=limit,
        require_references=True,
    )
    _log_stage_event(
        "EA-MT Dataset Load",
        "End",
        examples=len(bundle.examples),
        elapsed=_format_seconds(time.monotonic() - dataset_load_start_time),
    )

    memory_provider = None
    if mode == "entity-aware":
        try:
            from eamt.translation.entity_memory_pipeline import (
                build_entity_memory_from_pipeline,
                load_entity_pipeline_artifacts,
            )
        except Exception:  # pragma: no cover
            from ..translation.entity_memory_pipeline import (
                build_entity_memory_from_pipeline,
                load_entity_pipeline_artifacts,
            )

        artifacts = load_entity_pipeline_artifacts(
            target_lang=target_locale or "",
            entity_pipeline_mode=entity_pipeline_mode,
            reranker_model_path=reranker_model_path,
            reranker_device=reranker_device,
            reranker_hidden_dim=reranker_hidden_dim,
            reranker_dropout=reranker_dropout,
        )

        def memory_provider(example: Any) -> Any:
            return build_entity_memory_from_pipeline(
                example,
                artifacts=artifacts,
                entity_pipeline_mode=entity_pipeline_mode,
                alias_limit=alias_limit,
                description_max_chars=description_max_chars,
                retrieval_top_k=retrieval_top_k,
                retrieval_per_surface_k=retrieval_per_surface_k,
                retrieval_min_char_len=retrieval_min_char_len,
                retrieval_max_n=retrieval_max_n,
                reranker_prior_bonus_weight=reranker_prior_bonus_weight,
            )

    results = evaluate_qwen_on_eamt(
        dataset=bundle.examples,
        references=bundle.references,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        memory_provider=memory_provider,
        mode=mode,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        generation_kwargs=generation_kwargs,
        comet_model_name=comet_model_name,
        comet_batch_size=comet_batch_size,
        comet_num_gpus=comet_num_gpus,
        load_model_kwargs=load_model_kwargs,
        generation_batch_size=generation_batch_size,
        show_progress=show_progress,
        progress_log_interval_seconds=progress_log_interval_seconds,
    )

    if prediction_output_path:
        save_predictions_jsonl(results["predictions"], prediction_output_path)

    results["dataset"] = {
        "split": split,
        "target_locale": target_locale,
        "source_locale": source_locale,
        "num_examples": len(bundle.examples),
    }
    return results


__all__ = [
    "EAMTDBDatasetBundle",
    "connect_mysql_from_env",
    "evaluate_qwen_baseline_from_db",
    "load_db_config_from_env",
    "load_eamt_dataset_from_db",
    "load_test_eamt_dataset_from_db",
    "load_validation_eamt_dataset_from_db",
]
