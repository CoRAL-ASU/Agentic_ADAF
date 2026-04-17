#!/usr/bin/env python3
"""Run the DeALoG orchestrator over a dataset and write benchmark metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.coordinator import AdaptiveOrchestrator
from utils.data_loader import load_benchmark


def _normalise_answer(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("−", "-")
    text = re.sub(r"\s+", " ", text)
    # Tolerate common punctuation-only formatting drift (e.g., "ferrari.")
    return text.strip(string.whitespace + string.punctuation)


def _numbers_match(ref: str, cand: str) -> bool:
    def extract_numbers(text: str) -> List[float]:
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
        nums = []
        for match in matches:
            try:
                nums.append(float(match))
            except ValueError:
                continue
        return nums

    ref_nums = extract_numbers(ref)
    cand_nums = extract_numbers(cand)
    if not ref_nums or not cand_nums:
        return False
    tolerance = max(1e-3, 0.01 * abs(ref_nums[0]))
    return any(abs(ref_nums[0] - num) <= tolerance for num in cand_nums)


def run_dataset(
    dataset: str,
    split: str,
    llm: str,
    *,
    summarizer_llm: str | None,
    summarizer_temperature: float,
    summarizer_max_tokens: int,
    visual_caption_model: str | None,
    visual_caption_path: str | None,
    visual_ocr_engine: str | None,
    visual_ocr_model_dir: str | None,
    limit: int | None,
    min_chain_len: int | None,
    max_chain_len: int | None,
    max_rounds: int,
    scheduler_model: str | None,
    scheduler_threshold: float,
    parallel_retrieval: bool,
) -> Dict[str, Any]:
    # Apply limit after optional chain-length filtering so sliced subsets
    # (e.g., 7-8 hops) don't get emptied by early truncation.
    data = load_benchmark(dataset, split=split, limit=None)
    if min_chain_len is not None or max_chain_len is not None:
        filtered: List[Dict[str, Any]] = []
        for sample in data:
            chain = sample.get("operator_chain")
            chain_len = len(chain) if isinstance(chain, list) else None
            if chain_len is None:
                continue
            if min_chain_len is not None and chain_len < min_chain_len:
                continue
            if max_chain_len is not None and chain_len > max_chain_len:
                continue
            filtered.append(sample)
        data = filtered
    if limit:
        data = data[:limit]

    orchestrator = AdaptiveOrchestrator(
        model_name=llm,
        summarizer_model_name=summarizer_llm,
        summarizer_temperature=summarizer_temperature,
        summarizer_max_tokens=summarizer_max_tokens,
        visual_model_name=visual_caption_model,
        visual_caption_model=visual_caption_model,
        visual_caption_model_path=visual_caption_path,
        visual_ocr_engine=visual_ocr_engine,
        visual_ocr_model_dir=visual_ocr_model_dir,
        max_rounds=max_rounds,
        scheduler_model_path=scheduler_model,
        scheduler_threshold=scheduler_threshold,
        parallel_retrieval=parallel_retrieval,
    )

    per_example: List[Dict[str, Any]] = []
    total_correct = 0
    total_latency = 0.0

    for idx, sample in enumerate(data):
        start = time.perf_counter()
        result = orchestrator.run(sample)
        latency = time.perf_counter() - start
        total_latency += latency

        gold = _normalise_answer(sample.get("answer"))
        pred = _normalise_answer(result.get("answer"))
        correct = bool(gold) and (gold == pred or _numbers_match(gold, pred))
        total_correct += int(correct)
        per_example.append(
            {
                "id": sample.get("id", f"{dataset}-{split}-{idx}"),
                "question": sample.get("question"),
                "prediction": result.get("answer"),
                "reference": sample.get("answer"),
                "rationale": result.get("rationale"),
                "log": result.get("log"),
                "correct": correct,
                "latency_sec": latency,
            }
        )

    n = len(per_example)
    accuracy = total_correct / n if n else 0.0
    avg_latency = total_latency / n if n else 0.0
    return {
        "dataset": dataset,
        "split": split,
        "model": llm,
        "accuracy": accuracy,
        "num_examples": n,
        "latency_sec": round(avg_latency, 3),
        "calls": None,
        "tokens": None,
        "api_cost": None,
        "per_example": per_example,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeALoG orchestrator and emit metrics.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--llm",
        default=os.getenv("PRIMARY_MODEL_NAME"),
        help="Primary model ID (defaults to PRIMARY_MODEL_NAME).",
    )
    parser.add_argument(
        "--summarizer-llm",
        default=os.getenv("DEALOG_SUMMARIZER_MODEL"),
        help="Model used by summarizer/verifier client (defaults to DEALOG_SUMMARIZER_MODEL).",
    )
    parser.add_argument(
        "--summarizer-temperature",
        type=float,
        default=float(os.getenv("DEALOG_SUMMARIZER_TEMPERATURE", "0.2")),
        help="Temperature for summarizer completions.",
    )
    parser.add_argument(
        "--summarizer-max-tokens",
        type=int,
        default=int(os.getenv("DEALOG_SUMMARIZER_MAX_TOKENS", "256")),
        help="Max completion tokens for summarizer completions.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of samples.")
    parser.add_argument("--min-chain-len", type=int, default=None, help="Filter samples by minimum operator-chain length.")
    parser.add_argument("--max-chain-len", type=int, default=None, help="Filter samples by maximum operator-chain length.")
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--visual-caption-model", default=None)
    parser.add_argument("--visual-caption-path", default=None)
    parser.add_argument("--visual-ocr-engine", default=None)
    parser.add_argument("--visual-ocr-model-dir", default=None)
    parser.add_argument("--max-rounds", type=int, default=6, help="Max coordination rounds (R).")
    parser.add_argument("--scheduler", default=None, help="Path to a joblib logistic gate.")
    parser.add_argument("--scheduler-threshold", type=float, default=0.4, help="p(continue) threshold.")
    parser.add_argument("--parallel-retrieval", action="store_true", help="Enable parallel retrieval micro-benchmark mode.")
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.getenv("DEALOG_CUDA_VISIBLE_DEVICES"),
        help="Optional CUDA_VISIBLE_DEVICES value for this run (defaults to DEALOG_CUDA_VISIBLE_DEVICES).",
    )
    args = parser.parse_args()

    # Shell expansion may pass an empty string for --llm/--summarizer-llm; recover from .env.
    args.llm = str(args.llm or "").strip() or str(os.getenv("PRIMARY_MODEL_NAME") or "").strip()
    if not args.llm:
        parser.error("Missing model ID. Pass --llm or set PRIMARY_MODEL_NAME in .env.")

    args.summarizer_llm = (
        str(args.summarizer_llm or "").strip() or str(os.getenv("DEALOG_SUMMARIZER_MODEL") or "").strip() or None
    )

    if args.cuda_visible_devices is not None and str(args.cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices).strip()
        print(f"[DeALoG] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    metrics = run_dataset(
        dataset=args.dataset,
        split=args.split,
        llm=args.llm,
        summarizer_llm=args.summarizer_llm,
        summarizer_temperature=args.summarizer_temperature,
        summarizer_max_tokens=args.summarizer_max_tokens,
        visual_caption_model=args.visual_caption_model,
        visual_caption_path=args.visual_caption_path,
        visual_ocr_engine=args.visual_ocr_engine,
        visual_ocr_model_dir=args.visual_ocr_model_dir,
        limit=args.limit,
        min_chain_len=args.min_chain_len,
        max_chain_len=args.max_chain_len,
        max_rounds=args.max_rounds,
        scheduler_model=args.scheduler,
        scheduler_threshold=args.scheduler_threshold,
        parallel_retrieval=args.parallel_retrieval,
    )

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    args.results_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"[DeALoG] dataset={args.dataset} split={args.split} model={args.llm} "
        f"accuracy={metrics['accuracy']:.3f} latency={metrics['latency_sec']}"
    )


if __name__ == "__main__":
    main()
