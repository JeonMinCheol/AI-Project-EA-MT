from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Set

try:
    from DTOlist import EAMTExample
except Exception:  # pragma: no cover
    from src.DTOlist import EAMTExample

DEFAULT_SYSTEM_PROMPT = (
    "You are a professional translator for the SemEval EA-MT task. "
    "Return only the final translation in the target language."
)


@dataclass
class EAMTDatasetBundle:
    examples: List[EAMTExample]
    references: List[Dict[str, Any]]
    references_by_id: Dict[str, Dict[str, Any]]
    mentions_by_id: Dict[str, Set[str]]
    entity_types_by_id: Dict[str, List[str]]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _db_config() -> dict[str, Any]:
    try:
        import pymysql
    except Exception as import_error:  # pragma: no cover
        raise ImportError("pymysql가 설치되어 있어야 DB 로더를 사용할 수 있습니다.") from import_error

    return {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "db": os.getenv("DB_NAME"),
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
    }


def load_eamt_dataset_from_db(
    *,
    split: str,
    target_locale: str | None = None,
    source_locale: str = "en",
    limit: int | None = None,
    require_references: bool = True,
    require_mentions: bool = False,
) -> EAMTDatasetBundle:
    config = _db_config()
    import pymysql
    conn = pymysql.connect(**config)

    try:
        with conn.cursor() as cursor:
            query = [
                "SELECT example_id, dataset_id, split, source_locale, target_locale, source_text",
                "FROM eamt_example",
                "WHERE split = %s",
            ]
            params: List[Any] = [split]

            normalized_source = _safe_str(source_locale)
            if normalized_source:
                query.append("AND source_locale = %s")
                params.append(normalized_source)

            normalized_target = _safe_str(target_locale)
            if normalized_target:
                query.append("AND target_locale = %s")
                params.append(normalized_target)

            query.append("ORDER BY example_id ASC")
            if limit is not None:
                query.append("LIMIT %s")
                params.append(int(limit))

            cursor.execute("\n".join(query), tuple(params))
            rows = list(cursor.fetchall())

            if not rows:
                return EAMTDatasetBundle([], [], {}, {}, {})

            example_ids = [int(row["example_id"]) for row in rows]
            placeholders = ", ".join(["%s"] * len(example_ids))

            qid_map: Dict[int, List[str]] = {example_id: [] for example_id in example_ids}
            cursor.execute(
                f"""
                SELECT example_id, qid
                FROM example_entity
                WHERE example_id IN ({placeholders})
                ORDER BY example_id ASC, entity_order ASC
                """,
                tuple(example_ids),
            )
            for record in cursor.fetchall():
                qid = _safe_str(record.get("qid"))
                if qid:
                    qid_map[int(record["example_id"])].append(qid)

            reference_targets_by_example: Dict[int, List[Dict[str, str]]] = {
                example_id: [] for example_id in example_ids
            }
            if require_references or require_mentions:
                cursor.execute(
                    f"""
                    SELECT example_id, translation_text, mention_text
                    FROM example_reference
                    WHERE example_id IN ({placeholders})
                    ORDER BY example_id ASC, reference_order ASC
                    """,
                    tuple(example_ids),
                )
                for record in cursor.fetchall():
                    translation = _safe_str(record.get("translation_text"))
                    mention = _safe_str(record.get("mention_text"))
                    reference_targets_by_example[int(record["example_id"])].append(
                        {
                            "translation": translation,
                            "mention": mention,
                        }
                    )

        examples: List[EAMTExample] = []
        references: List[Dict[str, Any]] = []
        references_by_id: Dict[str, Dict[str, Any]] = {}
        mentions_by_id: Dict[str, Set[str]] = {}
        entity_types_by_id: Dict[str, List[str]] = {}

        for row in rows:
            dataset_id = _safe_str(row.get("dataset_id"))
            source_text = _safe_str(row.get("source_text"))
            target_lang = _safe_str(row.get("target_locale"))
            example_id = int(row["example_id"])
            entity_qids = list(qid_map.get(example_id, []))

            targets = reference_targets_by_example.get(example_id, [])
            target_text = _safe_str(targets[0].get("translation")) if targets else None
            wikidata_id = entity_qids[0] if entity_qids else None

            examples.append(
                EAMTExample(
                    id=dataset_id,
                    source=source_text,
                    target_lang=target_lang,
                    target=target_text,
                    wikidata_id=wikidata_id,
                    entity_qids=entity_qids,
                    meta={
                        "split": _safe_str(row.get("split")),
                        "source_locale": _safe_str(row.get("source_locale")),
                        "target_locale": target_lang,
                    },
                )
            )

            reference_record = {
                "id": dataset_id,
                "targets": targets,
                "entity_types": [],
            }
            references.append(reference_record)
            references_by_id[dataset_id] = reference_record
            mentions_by_id[dataset_id] = {
                _safe_str(target.get("mention"))
                for target in targets
                if _safe_str(target.get("mention"))
            }
            entity_types_by_id[dataset_id] = []

        return EAMTDatasetBundle(
            examples=examples,
            references=references,
            references_by_id=references_by_id,
            mentions_by_id=mentions_by_id,
            entity_types_by_id=entity_types_by_id,
        )
    finally:
        conn.close()


def evaluate_qwen_baseline_from_db(
    *,
    split: str,
    target_locale: str | None,
    source_locale: str,
    limit: int | None,
    model_name: str,
    mode: str,
    system_prompt: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_batch_size: int,
    comet_model_name: str,
    comet_batch_size: int,
    comet_num_gpus: int,
    load_model_kwargs: Mapping[str, Any] | None,
    prediction_output_path: str | None,
    show_progress: bool,
    progress_log_interval_seconds: float,
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
) -> Dict[str, Any]:
    from eamt.translation.entity_memory_pipeline import (
        build_entity_memory_from_pipeline,
        load_entity_pipeline_artifacts,
    )
    from eamt.translation.evaluation import evaluate_qwen_on_eamt, save_predictions_jsonl

    bundle = load_eamt_dataset_from_db(
        split=split,
        target_locale=target_locale,
        source_locale=source_locale,
        limit=limit,
        require_references=True,
        require_mentions=True,
    )

    memory_provider = None
    if mode == "entity-aware":
        normalized_target = _safe_str(target_locale)
        if not normalized_target:
            raise ValueError("mode=entity-aware 일 때는 target_locale이 필요합니다.")

        artifacts = load_entity_pipeline_artifacts(
            target_lang=normalized_target,
            entity_pipeline_mode=entity_pipeline_mode,
            reranker_model_path=reranker_model_path,
            reranker_device=reranker_device,
            reranker_hidden_dim=reranker_hidden_dim,
            reranker_dropout=reranker_dropout,
        )

        def memory_provider(example: EAMTExample):
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

    resolved_system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt

    results = evaluate_qwen_on_eamt(
        dataset=bundle.examples,
        model_name=model_name,
        references=bundle.references,
        mentions_by_id=bundle.mentions_by_id,
        memory_provider=memory_provider,
        mode=mode,
        system_prompt=resolved_system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        comet_model_name=comet_model_name,
        comet_batch_size=comet_batch_size,
        comet_num_gpus=comet_num_gpus,
        load_model_kwargs=load_model_kwargs,
        generation_batch_size=generation_batch_size,
        show_progress=show_progress,
        progress_log_interval_seconds=progress_log_interval_seconds,
    )

    if prediction_output_path:
        save_predictions_jsonl(results.get("predictions", []), prediction_output_path)

    results["dataset"] = {
        "split": split,
        "target_locale": target_locale,
        "source_locale": source_locale,
        "num_examples": len(bundle.examples),
    }
    return results


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "EAMTDatasetBundle",
    "evaluate_qwen_baseline_from_db",
    "load_eamt_dataset_from_db",
]
