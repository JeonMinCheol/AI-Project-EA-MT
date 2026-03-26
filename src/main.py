from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from eamt.data.db_loader import evaluate_qwen_baseline_from_db


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load an EA-MT split from MySQL and evaluate the Qwen baseline."
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
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during generation and evaluation",
    )
    return parser


def _build_load_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "local_files_only": bool(args.local_files_only),
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "gpu_ids": args.gpu_ids,
        "per_gpu_max_memory": args.per_gpu_max_memory,
        "cpu_max_memory": args.cpu_max_memory,
    }


def _print_results(results: dict[str, Any], prediction_output_path: str | None) -> None:
    metrics = results["metrics"]
    dataset_info = results.get("dataset", {})
    runtime_info = results.get("runtime", {})

    print("=============================================")
    print(f"Split               : {dataset_info.get('split')}")
    print(f"Target Locale       : {dataset_info.get('target_locale')}")
    print(f"Source Locale       : en")
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

    compact_results = {
        "dataset": dataset_info,
        "metrics": metrics,
        "runtime": runtime_info,
    }
    print(json.dumps(compact_results, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    results = evaluate_qwen_baseline_from_db(
        split=args.split,
        target_locale=args.target_locale,
        source_locale=args.source_locale,
        limit=args.limit,
        model_name=args.model_name,
        mode=args.mode,
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
    )

    _print_results(results, args.prediction_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
