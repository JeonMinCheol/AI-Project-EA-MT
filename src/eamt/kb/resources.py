from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False

from src.DTOlist import KBEntityRecord, RuntimeResources
from .index import build_qid_index, build_surface_index, normalize_surface

load_dotenv()


def _db_config() -> Dict[str, Any]:
    try:
        import pymysql
    except Exception as import_error:  # pragma: no cover
        raise ImportError("pymysql가 설치되어 있어야 KB 리소스를 DB에서 불러올 수 있습니다.") from import_error

    return {
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "db": os.getenv("DB_NAME"),
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
    }


def load_kb_records_from_db(target_lang: str) -> List[KBEntityRecord]:
    import pymysql
    conn = pymysql.connect(**_db_config())
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT qid, popularity_score FROM entity")
            entity_rows = cur.fetchall()

            cur.execute("""
                SELECT qid, lang_code, term_type, term_text
                FROM entity_term
            """)
            term_rows = cur.fetchall()

            cur.execute("""
                SELECT qid, type_text
                FROM entity_type
            """)
            type_rows = cur.fetchall()

        popularity_map: Dict[str, float] = {
            row["qid"]: float(row.get("popularity_score") or 0.0)
            for row in entity_rows
        }

        type_map: Dict[str, List[str]] = defaultdict(list)
        for row in type_rows:
            type_map[row["qid"]].append(row["type_text"])

        terms_by_qid_lang_type: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for row in term_rows:
            qid = row["qid"]
            lang = row["lang_code"]
            term_type = row["term_type"]
            text = (row["term_text"] or "").strip()
            if text:
                terms_by_qid_lang_type[qid][lang][term_type].append(text)

        records: List[KBEntityRecord] = []

        for qid, _pop in popularity_map.items():
            en_labels = terms_by_qid_lang_type[qid]["en"].get("label", [])
            en_aliases = terms_by_qid_lang_type[qid]["en"].get("alias", [])

            target_labels = terms_by_qid_lang_type[qid][target_lang].get("label", [])
            target_aliases = terms_by_qid_lang_type[qid][target_lang].get("alias", [])
            target_descs = terms_by_qid_lang_type[qid][target_lang].get("description", [])

            en_descs = terms_by_qid_lang_type[qid]["en"].get("description", [])

            target_label = target_labels[0] if target_labels else None
            description = (target_descs[0] if target_descs else (en_descs[0] if en_descs else None))
            language_available = bool(target_label or target_aliases)

            normalized_surfaces = []
            for surface in [target_label, *target_aliases, *(en_labels[:1]), *en_aliases]:
                if surface:
                    normalized_surfaces.append(normalize_surface(surface))
            normalized_surfaces = sorted(set(s for s in normalized_surfaces if s))

            record = KBEntityRecord(
                qid=qid,
                label_en=en_labels[0] if en_labels else None,
                aliases_en=en_aliases,
                target_lang=target_lang,
                target_label=target_label,
                target_aliases=target_aliases,
                entity_type=type_map[qid][0] if type_map[qid] else None,
                description=description,
                normalized_surfaces=normalized_surfaces,
                language_available=language_available,
                popularity_score=float(popularity_map.get(qid, 0.0)),
            )
            records.append(record)

        return records
    finally:
        conn.close()


def build_runtime_resources_from_db(
    target_lang: str,
    reranker_mode: Any = None,
    translator_model: Any = None,
    tokenizer: Any = None,
    ercm_model: Any = None,
) -> RuntimeResources:
    records = load_kb_records_from_db(target_lang=target_lang)
    qid_index = build_qid_index(records)
    surface_index = build_surface_index(records)

    return RuntimeResources(
        qid_index=qid_index,
        surface_index=surface_index,
        reranker_mode=reranker_mode,
        translator_model=translator_model,
        tokenizer=tokenizer,
        ercm_model=ercm_model,
    )