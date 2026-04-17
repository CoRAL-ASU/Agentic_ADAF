#!/usr/bin/env python3
"""Compute BERTScore metrics for FeTaQA predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from bert_score import score as bert_score


def compute_bertscore(results_file: Path, lang: str, model_type: str, output_file: Path) -> None:
    metrics = json.loads(results_file.read_text())
    predictions: List[str] = []
    references: List[str] = []
    ids: List[str] = []
    for example in metrics.get("per_example", []):
        pred = example.get("prediction") or ""
        ref = example.get("reference") or ""
        predictions.append(pred)
        references.append(ref)
        ids.append(example.get("id"))

    precision, recall, f1 = bert_score(predictions, references, lang=lang, model_type=model_type)
    per_example = [
        {
            "id": idx,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }
        for idx, p, r, f in zip(ids, precision, recall, f1)
    ]
    summary = {
        "lang": lang,
        "model_type": model_type,
        "mean_precision": float(precision.mean()),
        "mean_recall": float(recall.mean()),
        "mean_f1": float(f1.mean()),
        "num_examples": len(per_example),
        "per_example": per_example,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BERTScore for FeTaQA results.")
    parser.add_argument("--results-file", type=Path, required=True, help="Path to run_dealog output JSON.")
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--model-type", default="microsoft/deberta-large-mnli")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_bertscore(
        results_file=args.results_file,
        lang=args.lang,
        model_type=args.model_type,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
