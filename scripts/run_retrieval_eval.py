from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.eamt.kb.resources import build_runtime_resources_from_db
from src.eamt.retrieval.eval import evaluate_retrieval_service


@dataclass
class Example:
    id: str
    source: str
    target_lang: str
    wikidata_id: str | None = None


def get_jsonl_files(root_path: str) -> List[str]:
    files: List[str] = []
    for root, _dirs, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith(".jsonl"):
                files.append(os.path.join(root, filename))
    return sorted(files)


def load_validation_examples(base_dir: str, target_lang: str) -> List[Example]:
    val_dir = os.path.join(base_dir, "data", "raw", "validation")
    examples: List[Example] = []
    for file_path in get_jsonl_files(val_dir):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("target_locale") != target_lang:
                    continue
                examples.append(
                    Example(
                        id=row["id"],
                        source=row["source"],
                        target_lang=row["target_locale"],
                        wikidata_id=row.get("wikidata_id"),
                    )
                )
    return examples


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_lang = "ko"

    resources = build_runtime_resources_from_db(target_lang=target_lang)
    examples = load_validation_examples(base_dir, target_lang=target_lang)

    metrics = evaluate_retrieval_service(
        examples=examples,
        resources=resources,
        top_k=10,
        per_surface_k=5,
        ks=(1, 3, 5, 10),
    )

    out_dir = Path(base_dir) / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()