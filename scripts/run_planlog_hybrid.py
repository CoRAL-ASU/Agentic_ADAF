#!/usr/bin/env python3
"""Run a plan→log hybrid: seed a planner sketch then execute via the shared-log agents."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.coordinator import AdaptiveOrchestrator
from utils.data_loader import load_benchmark


def _build_needs(sample: Dict[str, Any]) -> List[str]:
    needs: List[str] = []
    if sample.get("paragraphs"):
        needs.append("context")
    if sample.get("table"):
        needs.append("table")
    if sample.get("calculation") or sample.get("operator_chain"):
        needs.append("calculation")
    if sample.get("visuals"):
        needs.append("visual")
    # Fallback to default if nothing detected.
    return needs or ["context", "table", "calculation"]


def _build_plan(sample: Dict[str, Any]) -> str:
    steps: List[str] = []
    if sample.get("table"):
        steps.append("Inspect table and extract needed rows/columns.")
    if sample.get("paragraphs"):
        steps.append("Retrieve supporting paragraphs for entities/years.")
    if sample.get("calculation") or sample.get("operator_chain"):
        steps.append("Compute requested aggregation or operator chain.")
    if sample.get("visuals"):
        steps.append("Interpret visuals via caption/OCR if present.")
    steps.append("Synthesize answer and verify against evidence.")
    return "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))


def run_hybrid(
    dataset: str,
    split: str,
    limit: Optional[int],
    llm: Optional[str],
    summarizer_llm: Optional[str],
    summarizer_max_tokens: int,
    visual_caption_model: Optional[str],
    visual_caption_path: Optional[str],
    visual_ocr_engine: Optional[str],
    visual_ocr_dir: Optional[str],
    max_rounds: int,
    scheduler: Optional[str],
    scheduler_threshold: float,
    output: Path,
) -> Dict[str, Any]:
    data = load_benchmark(dataset, split=split, limit=limit)
    per_example: List[Dict[str, Any]] = []
    total_correct = 0
    total_latency = 0.0

    orchestrator = AdaptiveOrchestrator(
        model_name=llm,
        summarizer_model_name=summarizer_llm,
        summarizer_max_tokens=summarizer_max_tokens,
        visual_caption_model=visual_caption_model,
        visual_caption_model_path=visual_caption_path,
        visual_ocr_engine=visual_ocr_engine,
        visual_ocr_model_dir=visual_ocr_dir,
        max_rounds=max_rounds,
        scheduler_model_path=scheduler,
        scheduler_threshold=scheduler_threshold,
    )

    for idx, sample in enumerate(data):
        plan_text = _build_plan(sample)
        needs = _build_needs(sample)
        start = time.perf_counter()
        result = orchestrator.run(
            sample,
            initial_plan=plan_text,
            initial_needs=needs,
        )
        latency = time.perf_counter() - start
        total_latency += latency

        prediction = result.get("answer", "")
        gold = str(sample.get("answer") or "").strip().lower()
        correct = bool(gold) and gold == str(prediction).strip().lower()
        total_correct += int(correct)

        per_example.append(
            {
                "id": sample.get("id", f"{dataset}-{split}-{idx}"),
                "question": sample.get("question"),
                "plan": plan_text,
                "prediction": prediction,
                "reference": sample.get("answer"),
                "verified": bool(result.get("verified")),
                "correct": bool(correct),
                "latency_sec": round(latency, 3),
            }
        )

    total = len(per_example)
    metrics = {
        "dataset": dataset,
        "split": split,
        "strategy": "planlog_hybrid",
        "model": llm,
        "accuracy": total_correct / total if total else 0.0,
        "num_examples": total,
        "latency_sec": round(total_latency / total, 3) if total else 0.0,
        "per_example": per_example,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Plan→Log hybrid.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm", required=True)
    ap.add_argument(
        "--summarizer-llm",
        default=os.getenv("DEALOG_SUMMARIZER_MODEL"),
        help="Model used by summarizer/verifier client (defaults to DEALOG_SUMMARIZER_MODEL).",
    )
    ap.add_argument(
        "--summarizer-max-tokens",
        type=int,
        default=int(os.getenv("DEALOG_SUMMARIZER_MAX_TOKENS", "256")),
        help="Max completion tokens for summarizer completions.",
    )
    ap.add_argument("--visual-caption-model", default=None)
    ap.add_argument("--visual-caption-path", default=None)
    ap.add_argument("--visual-ocr-engine", default=None)
    ap.add_argument("--visual-ocr-model-dir", default=None)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--scheduler", default=None, help="Path to a joblib logistic gate.")
    ap.add_argument("--scheduler-threshold", type=float, default=0.4)
    ap.add_argument("--results-file", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_hybrid(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        llm=args.llm,
        summarizer_llm=args.summarizer_llm,
        summarizer_max_tokens=args.summarizer_max_tokens,
        visual_caption_model=args.visual_caption_model,
        visual_caption_path=args.visual_caption_path,
        visual_ocr_engine=args.visual_ocr_engine,
        visual_ocr_dir=args.visual_ocr_model_dir,
        max_rounds=args.max_rounds,
        scheduler=args.scheduler,
        scheduler_threshold=args.scheduler_threshold,
        output=args.results_file,
    )
    print(f"[Plan→Log] results saved to {args.results_file}")


if __name__ == "__main__":
    main()
