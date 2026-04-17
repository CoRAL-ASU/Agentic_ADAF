"""Entry point for running the adaptive table QA pipeline."""

from __future__ import annotations

import argparse
from typing import Any, Dict

from agents.coordinator import AdaptiveOrchestrator
from utils.data_loader import load_benchmark
from utils.settings import (
    get_dealog_summarizer_model,
    get_primary_model,
    get_visual_caption_model,
    get_visual_caption_model_path,
    get_visual_model,
    get_visual_ocr_engine,
    get_visual_ocr_model_dir,
)


def _format_result(sample: Dict[str, Any], result: Dict[str, Any]) -> str:
    status = "verified" if result.get("verified") else "unverified"
    return (
        f"Q: {sample.get('question', '')}\n"
        f"A ({status}): {result.get('answer', '')}\n"
    )


def main() -> None:
    """Run the orchestrator over a benchmark dataset."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tatqa")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--llm", type=str, default=get_primary_model())
    parser.add_argument(
        "--summarizer-llm",
        type=str,
        default=get_dealog_summarizer_model(),
        help="Model used by the summarizer/verifier LLM client.",
    )
    parser.add_argument(
        "--summarizer-temperature",
        type=float,
        default=0.2,
        help="Temperature used for summarizer completions.",
    )
    parser.add_argument(
        "--summarizer-max-tokens",
        type=int,
        default=256,
        help="Max completion tokens used by the summarizer.",
    )
    parser.add_argument(
        "--visual-llm",
        type=str,
        default=get_visual_model(),
        help="Legacy alias for --visual-caption-model",
    )
    parser.add_argument(
        "--visual-caption-model",
        type=str,
        default=get_visual_caption_model(),
    )
    parser.add_argument(
        "--visual-caption-path",
        type=str,
        default=get_visual_caption_model_path(),
    )
    parser.add_argument(
        "--visual-ocr-engine",
        type=str,
        default=get_visual_ocr_engine(),
    )
    parser.add_argument(
        "--visual-ocr-model-dir",
        type=str,
        default=get_visual_ocr_model_dir(),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples to load from the dataset.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=6,
        help="Max coordination rounds (R) before forcing a summary.",
    )
    args = parser.parse_args()

    data = load_benchmark(args.dataset, split=args.split, limit=args.limit)
    orchestrator = AdaptiveOrchestrator(
        model_name=args.llm,
        summarizer_model_name=args.summarizer_llm,
        summarizer_temperature=args.summarizer_temperature,
        summarizer_max_tokens=args.summarizer_max_tokens,
        visual_model_name=args.visual_llm or args.visual_caption_model,
        visual_caption_model=args.visual_caption_model or args.visual_llm,
        visual_caption_model_path=args.visual_caption_path,
        visual_ocr_engine=args.visual_ocr_engine,
        visual_ocr_model_dir=args.visual_ocr_model_dir,
        max_rounds=args.max_rounds,
    )
    for sample in data:
        result = orchestrator.run(sample)
        print(_format_result(sample, result))


if __name__ == "__main__":
    main()
