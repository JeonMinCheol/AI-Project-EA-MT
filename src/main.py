from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from eamt.data.db_loader import evaluate_qwen_baseline_from_db


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load an EA-MT split from MySQL and evaluate a Qwen model with optional LoRA adapters."
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to evaluate. Default: validation",
    )
    parser.add_argument("--target-locale", default=None, help="Filter by target locale, e.g. ko")
    parser.add_argument("--source-locale", default="en", help="Filter by source locale, e.g. en")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--mode",
        default="plain",
        choices=["plain", "entity-aware"],
        help="Prompting mode. Baseline uses plain by default.",
    )
    parser.add_argument(
        "--entity-pipeline-mode",
        default="anchored",
        choices=["anchored", "surface", "retrieve", "rerank"],
        help="How to build entity memory when mode=entity-aware. Default: anchored",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--retrieval-per-surface-k", type=int, default=5)
    parser.add_argument("--retrieval-min-char-len", type=int, default=2)
    parser.add_argument("--retrieval-max-n", type=int, default=5)
    parser.add_argument("--reranker-model-path", default=None)
    parser.add_argument("--reranker-hidden-dim", type=int, default=128)
    parser.add_argument("--reranker-dropout", type=float, default=0.1)
    parser.add_argument("--reranker-device", default=None)
    parser.add_argument("--reranker-prior-bonus-weight", type=float, default=0.1)
    parser.add_argument("--alias-limit", type=int, default=1)
    parser.add_argument("--description-max-chars", type=int, default=80)
    parser.add_argument(
        "--peft-adapter-path",
        default=None,
        help="Optional LoRA adapter directory saved by PEFT",
    )
    parser.add_argument(
        "--use-training-summary",
        action="store_true",
        help="Load missing evaluation settings from <peft-adapter-path>/training_summary.json",
    )
    parser.add_argument(
        "--training-summary-path",
        default=None,
        help="Optional explicit path to training_summary.json used with --use-training-summary",
    )
    parser.add_argument(
        "--merge-peft-adapter",
        action="store_true",
        help="Merge the PEFT adapter into the base model before evaluation",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt override for generation",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=8,
        help="Batch size for translation generation. Default: 8",
    )
    parser.add_argument(
        "--progress-log-interval-seconds",
        type=float,
        default=30.0,
        help="How often to emit explicit progress summary logs. Default: 30",
    )
    parser.add_argument("--comet-model-name", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--comet-batch-size", type=int, default=8)
    parser.add_argument("--comet-num-gpus", type=int, default=1)
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPU ids for model sharding")
    parser.add_argument(
        "--per-gpu-max-memory",
        default=None,
        help="Optional max memory per GPU, e.g. 40GiB",
    )
    parser.add_argument(
        "--cpu-max-memory",
        default=None,
        help="Optional CPU offload memory budget, e.g. 64GiB",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Hugging Face files from local cache only",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="torch dtype for model loading, e.g. auto, float16",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="transformers device_map argument. With multiple GPUs, auto is promoted to balanced.",
    )
    parser.add_argument(
        "--prediction-output-path",
        default=None,
        help="Optional path to save EA-MT prediction JSONL",
    )
    parser.add_argument(
        "--metrics-output-path",
        default=None,
        help=(
            "Optional path to save evaluation metrics JSON. "
            "If omitted, it is auto-derived from --prediction-output-path or --peft-adapter-path."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during generation and evaluation",
    )
    return parser


def _extract_explicit_option_names(argv: Sequence[str] | None) -> set[str]:
    explicit_options: set[str] = set()
    for token in list(argv or []):
        if not token.startswith("--"):
            continue
        normalized = token[2:]
        if not normalized:
            continue
        if "=" in normalized:
            normalized = normalized.split("=", 1)[0]
        explicit_options.add(normalized.replace("-", "_"))
    return explicit_options


def _load_training_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"training summary를 찾을 수 없습니다: {path}")

    with path.open("r", encoding="utf-8") as file_obj:
        summary = json.load(file_obj)

    if not isinstance(summary, dict):
        raise SystemExit(f"지원하지 않는 training summary 형식입니다: {path}")

    return summary


def _apply_training_summary_overrides(
    args: argparse.Namespace,
    argv: Sequence[str] | None,
) -> argparse.Namespace:
    if not args.use_training_summary:
        return args

    if not args.training_summary_path and not args.peft_adapter_path:
        raise SystemExit(
            "`--use-training-summary`를 쓰려면 `--peft-adapter-path` 또는 "
            "`--training-summary-path`가 필요합니다."
        )

    explicit_options = _extract_explicit_option_names(argv)
    summary_path = (
        Path(args.training_summary_path)
        if args.training_summary_path
        else Path(str(args.peft_adapter_path)) / "training_summary.json"
    )
    summary = _load_training_summary(summary_path)
    candidate_pipeline = summary.get("candidate_pipeline", {}) or {}

    summary_defaults: dict[str, Any] = {
        "model_name": summary.get("model_name"),
        "split": summary.get("eval_split"),
        "target_locale": summary.get("target_locale"),
        "source_locale": summary.get("source_locale"),
        "mode": summary.get("mode"),
        "entity_pipeline_mode": summary.get("entity_pipeline_mode") or candidate_pipeline.get("mode"),
        "retrieval_top_k": candidate_pipeline.get("retrieval_top_k"),
        "retrieval_per_surface_k": candidate_pipeline.get("retrieval_per_surface_k"),
        "retrieval_min_char_len": candidate_pipeline.get("retrieval_min_char_len"),
        "retrieval_max_n": candidate_pipeline.get("retrieval_max_n"),
        "reranker_model_path": candidate_pipeline.get("reranker_model_path"),
        "reranker_hidden_dim": candidate_pipeline.get("reranker_hidden_dim"),
        "reranker_dropout": candidate_pipeline.get("reranker_dropout"),
        "reranker_device": candidate_pipeline.get("reranker_device"),
        "reranker_prior_bonus_weight": candidate_pipeline.get("reranker_prior_bonus_weight"),
        "alias_limit": candidate_pipeline.get("alias_limit", summary.get("alias_limit")),
        "description_max_chars": candidate_pipeline.get(
            "description_max_chars",
            summary.get("description_max_chars"),
        ),
        "system_prompt": summary.get("system_prompt"),
    }

    for field_name, summary_value in summary_defaults.items():
        if field_name in explicit_options:
            continue
        if summary_value is None or summary_value == "":
            continue
        if field_name == "split" and str(summary_value).strip().lower() == "none":
            continue
        setattr(args, field_name, summary_value)

    display_entity_pipeline_mode = (
        args.entity_pipeline_mode if args.mode == "entity-aware" else "unused"
    )
    print(
        "[Qwen Eval] training summary 적용 "
        f"| path={summary_path} | mode={args.mode} "
        f"| entity_pipeline_mode={display_entity_pipeline_mode}",
        flush=True,
    )
    return args


def _build_load_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "local_files_only": bool(args.local_files_only),
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "gpu_ids": args.gpu_ids,
        "per_gpu_max_memory": args.per_gpu_max_memory,
        "cpu_max_memory": args.cpu_max_memory,
        "peft_adapter_path": args.peft_adapter_path,
        "merge_peft_adapter": bool(args.merge_peft_adapter),
    }


def _compact_results(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": results.get("dataset", {}),
        "metrics": results.get("metrics", {}),
        "runtime": results.get("runtime", {}),
    }


def _resolve_metrics_output_path(args: argparse.Namespace) -> Path | None:
    if args.metrics_output_path:
        return Path(args.metrics_output_path)

    if args.prediction_output_path:
        prediction_path = Path(args.prediction_output_path)
        return prediction_path.with_name(f"{prediction_path.stem}_metrics.json")

    if args.peft_adapter_path:
        return Path(str(args.peft_adapter_path)) / "evaluation_metrics.json"

    return None


def _save_metrics_summary(
    results: dict[str, Any],
    args: argparse.Namespace,
    prediction_output_path: str | None,
    output_path: str | Path,
) -> Path:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _compact_results(results)
    summary["evaluation"] = {
        "model_name": results.get("model_name") or args.model_name,
        "split": args.split,
        "target_locale": args.target_locale,
        "source_locale": args.source_locale,
        "mode": args.mode,
        "entity_pipeline_mode": args.entity_pipeline_mode if args.mode == "entity-aware" else None,
        "peft_adapter_path": args.peft_adapter_path,
        "merge_peft_adapter": bool(args.merge_peft_adapter),
        "prediction_output_path": prediction_output_path,
    }

    with resolved_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")

    return resolved_path


def _print_results(
    results: dict[str, Any],
    prediction_output_path: str | None,
    metrics_output_path: str | Path | None,
) -> None:
    metrics = results["metrics"]
    dataset_info = results.get("dataset", {})
    runtime_info = results.get("runtime", {})

    print("=============================================")
    print(f"Split               : {dataset_info.get('split')}")
    print(f"Target Locale       : {dataset_info.get('target_locale')}")
    print(f"Source Locale       : {dataset_info.get('source_locale')}")
    print(f"Num Examples        : {dataset_info.get('num_examples')}")
    print(f"Model               : {results.get('model_name')}")
    print("---------------------------------------------")
    print(f"M-ETA               : {metrics['m_eta_percent']:.2f}")
    print(f"COMET               : {metrics['comet_percent']:.2f}")
    print(f"Final Harmonic      : {metrics['final_score_percent']:.2f}")
    if runtime_info.get("elapsed_seconds") is not None:
        print(f"Elapsed Seconds     : {runtime_info.get('elapsed_seconds'):.2f}")
    print("---------------------------------------------")
    print(f"Correct / Total     : {metrics['correct']} / {metrics['total']}")
    print(f"Missing Predictions : {metrics['missing_predictions']}")
    print("=============================================")

    if prediction_output_path:
        print(f"Predictions saved to: {prediction_output_path}")
    if metrics_output_path:
        print(f"Metrics saved to    : {metrics_output_path}")

    compact_results = _compact_results(results)
    print(json.dumps(compact_results, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    normalized_argv = list(argv) if argv is not None else None
    args = build_parser().parse_args(normalized_argv)
    args = _apply_training_summary_overrides(args, normalized_argv)

    if args.mode == "entity-aware" and not args.target_locale:
        raise SystemExit("`--mode entity-aware`에서는 `--target-locale`가 필요합니다.")

    results = evaluate_qwen_baseline_from_db(
        split=args.split,
        target_locale=args.target_locale,
        source_locale=args.source_locale,
        limit=args.limit,
        model_name=args.model_name,
        mode=args.mode,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generation_batch_size=args.generation_batch_size,
        comet_model_name=args.comet_model_name,
        comet_batch_size=args.comet_batch_size,
        comet_num_gpus=args.comet_num_gpus,
        load_model_kwargs=_build_load_model_kwargs(args),
        prediction_output_path=args.prediction_output_path,
        show_progress=not args.no_progress,
        progress_log_interval_seconds=args.progress_log_interval_seconds,
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
        alias_limit=args.alias_limit,
        description_max_chars=args.description_max_chars,
    )

    metrics_output_path = _resolve_metrics_output_path(args)
    if metrics_output_path is not None:
        _save_metrics_summary(
            results,
            args,
            prediction_output_path=args.prediction_output_path,
            output_path=metrics_output_path,
        )

    _print_results(results, args.prediction_output_path, metrics_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
