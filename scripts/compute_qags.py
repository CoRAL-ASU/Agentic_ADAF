#!/usr/bin/env python3
"""Compute QAGS-style faithfulness scores over DeALoG outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys
import sys
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline, T5Tokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_loader import load_benchmark


def _normalise(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalise(prediction).split()
    ref_tokens = _normalise(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = sum(min(pred_tokens.count(tok), ref_tokens.count(tok)) for tok in set(pred_tokens))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _flatten_table(table: Iterable[Iterable[str]]) -> str:
    if not table:
        return ""
    rows = []
    header = None
    for idx, row in enumerate(table):
        if idx == 0:
            header = row
            continue
        cells = []
        for col_idx, value in enumerate(row):
            key = header[col_idx] if header and col_idx < len(header) else f"col_{col_idx}"
            cells.append(f"{key}={value}")
        rows.append("; ".join(cells))
    return " | ".join(rows)


def _build_context(sample: Dict) -> str:
    bits = [sample.get("question", "")]
    flattened = _flatten_table(sample.get("table") or [])
    if flattened:
        bits.append(f"Table: {flattened}")
    for paragraph in sample.get("paragraphs") or []:
        if paragraph:
            bits.append(paragraph)
    return "\n".join(bits)


def _parse_pairs(text: str) -> List[Tuple[str, str]]:
    """Extract Q/A pairs even if the generator deviates from the requested format."""

    pairs: List[Tuple[str, str]] = []
    current_q = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("q:") or lower.startswith("question:"):
            current_q = stripped.split(":", 1)[1].strip()
            continue
        if (lower.startswith("a:") or lower.startswith("answer:")) and current_q:
            pairs.append((current_q, stripped.split(":", 1)[1].strip()))
            current_q = None
            continue
        # handle enumerated formats like "1) question" / "Answer: ..."
        if lower.startswith("answer") and ":" in stripped and current_q:
            pairs.append((current_q, stripped.split(":", 1)[1].strip()))
            current_q = None

    if pairs:
        return pairs

    # fallback: treat alternating non-empty lines as Q/A when prefixes are missing
    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    for idx in range(0, len(cleaned_lines) - 1, 2):
        q_line = cleaned_lines[idx]
        a_line = cleaned_lines[idx + 1]
        pairs.append((q_line, a_line))
    return pairs


def compute_qags(
    dataset: str,
    split: str,
    results_file: Path,
    question_model: str,
    qa_model: str,
    max_questions: int,
    output_file: Path,
) -> None:
    data = load_benchmark(dataset, split=split)
    sample_map = {sample.get("id"): sample for sample in data}
    metrics = json.loads(results_file.read_text())
    reader = pipeline("question-answering", model=qa_model, tokenizer=qa_model)
    qg_tokenizer = T5Tokenizer.from_pretrained(question_model, use_fast=False)
    qg_pipeline = pipeline("text2text-generation", model=question_model, tokenizer=qg_tokenizer)

    scores: List[Dict[str, float]] = []
    for example in metrics.get("per_example", []):
        sample = sample_map.get(example["id"])
        if not sample:
            continue
        rationale = example.get("rationale") or example.get("prediction") or ""
        if not rationale:
            continue
        prompt = (
            f"Read the explanation below and produce up to {max_questions} fact-check question/answer pairs.\n"
            "Each line MUST follow the exact format:\n"
            "Q: <question>\nA: <answer>\n"
            "Do NOT output booleans like 'True' or 'False'. Only emit factual questions and answers.\n"
            "Stop after the last pair with no extra text.\n"
            f"Explanation:\n{rationale}\n"
        )
        output = qg_pipeline(prompt, max_length=256, num_return_sequences=1)[0]
        generated = output["generated_text"]
        print(f"[QG DEBUG] id={example['id']} raw={generated!r}")
        qa_pairs = _parse_pairs(generated)[:max_questions]
        if not qa_pairs:
            continue
        context = _build_context(sample)
        per_pair_scores = []
        for question, answer in qa_pairs:
            reader_answer = reader(question=question, context=context)["answer"]
            per_pair_scores.append(_f1(answer, reader_answer))
        qags_score = sum(per_pair_scores) / len(per_pair_scores)
        scores.append({"id": example["id"], "qags": qags_score, "num_pairs": len(per_pair_scores)})

    mean_score = sum(item["qags"] for item in scores) / len(scores) if scores else 0.0
    summary = {
        "dataset": dataset,
        "split": split,
        "question_model": question_model,
        "qa_model": qa_model,
        "mean_qags": mean_score,
        "num_examples": len(scores),
        "per_example": scores,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute QAGS-style faithfulness for FeTaQA runs.")
    parser.add_argument("--dataset", default="fetaqa")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--results-file", type=Path, required=True, help="Path to run_dealog output JSON.")
    parser.add_argument("--question-model", default="iarfmoose/t5-base-question-generator")
    parser.add_argument("--qa-model", default="deepset/roberta-large-squad2")
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_qags(
        dataset=args.dataset,
        split=args.split,
        results_file=args.results_file,
        question_model=args.question_model,
        qa_model=args.qa_model,
        max_questions=args.max_questions,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
