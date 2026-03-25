from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.eamt.kb.resources import build_runtime_resources_from_db
from src.eamt.reranker.eval import evaluate_grouped_examples
from src.eamt.reranker.train import train_reranker_model
from src.eamt.reranker.train_builders import build_grouped_reranker_train_example


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


def load_train_examples(base_dir: str, target_lang: str) -> List[Example]:
    train_dir = os.path.join(base_dir, "data", "raw", "train")
    examples: List[Example] = []

    for file_path in get_jsonl_files(train_dir):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("target_locale") != target_lang:
                    continue

                entity_qids = row.get("entities", [])
                for idx, qid in enumerate(entity_qids):
                    examples.append(
                        Example(
                            id=f"{row['id']}::{idx}",
                            source=row["source"],
                            target_lang=row["target_locale"],
                            wikidata_id=qid,
                        )
                    )
    return examples


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

    train_examples_raw = load_train_examples(base_dir, target_lang=target_lang)
    valid_examples_raw = load_validation_examples(base_dir, target_lang=target_lang)

    train_grouped = [
        build_grouped_reranker_train_example(example, resources, max_negatives=8)
        for example in train_examples_raw
    ]
    valid_grouped = [
        build_grouped_reranker_train_example(example, resources, max_negatives=8)
        for example in valid_examples_raw
    ]

    model, history = train_reranker_model(
        train_examples=train_grouped,
        valid_examples=valid_grouped,
        epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        save_path=os.path.join(base_dir, "artifacts", "reranker.pt"),
        ks=(1, 3, 5),
    )

    final_metrics = evaluate_grouped_examples(
        model=model,
        grouped_examples=valid_grouped,
        ks=(1, 3, 5),
    )

    out_dir = Path(base_dir) / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "reranker_train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(out_dir / "reranker_valid_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print("=== final reranker metrics ===")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()