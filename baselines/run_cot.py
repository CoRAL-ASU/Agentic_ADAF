#!/usr/bin/env python3
"""Chain-of-thought baseline powered by an LLM."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_loader import load_benchmark
from utils.llm_client import LLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an LLM-backed CoT baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-chain-len", type=int, default=None)
    parser.add_argument("--max-chain-len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def _build_prompt(sample: Dict[str, Any]) -> str:
    question = sample.get("question", "")
    table = sample.get("table")
    paragraphs = sample.get("paragraphs") or sample.get("context") or []
    table_str = ""
    if table:
        rows = [" | ".join(map(str, row)) for row in table[:10]]
        table_str = "\n".join(rows)
    context_str = "\n".join(paragraphs[:3]) if paragraphs else ""

    instructions = (
        "You are a financial QA assistant. Use the table and context to answer the question.\n"
        "Reason step by step, and finish with a line of the form 'Answer: <final answer>'."
    )
    return (
        f"{instructions}\nQuestion: {question}\n"
        f"Table:\n{table_str or '(no table)'}\n"
        f"Context:\n{context_str or '(no context)'}\n"
        "Reasoning:"
    )


def _extract_answer(response: Optional[Any]) -> str:
    if not response:
        return ""
    if isinstance(response, dict):
        response = response.get("content") or ""
    text = str(response)
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    for line in lines[::-1]:
        if line.lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return lines[-1] if lines else ""


def _normalise_answer(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


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


def run_cot(args: argparse.Namespace) -> Dict[str, Any]:
    data = load_benchmark(args.dataset, split=args.split, limit=None)
    if args.min_chain_len is not None or args.max_chain_len is not None:
        filtered: List[Dict[str, Any]] = []
        for sample in data:
            chain = sample.get("operator_chain")
            chain_len = len(chain) if isinstance(chain, list) else None
            if chain_len is None:
                continue
            if args.min_chain_len is not None and chain_len < args.min_chain_len:
                continue
            if args.max_chain_len is not None and chain_len > args.max_chain_len:
                continue
            filtered.append(sample)
        data = filtered
    if args.limit is not None and args.limit > 0:
        data = data[: args.limit]

    client = LLMClient(default_model=args.model)
    per_example: List[Dict[str, Any]] = []
    total_correct = 0

    for idx, sample in enumerate(data):
        prompt = _build_prompt(sample)
        response = client.complete(
            prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
        answer = _extract_answer(response)
        gold = _normalise_answer(sample.get("answer"))
        pred = _normalise_answer(answer)
        is_correct = bool(gold) and (gold == pred or _numbers_match(gold, pred))
        total_correct += int(is_correct)

        per_example.append(
            {
                "id": sample.get("id", f"{args.dataset}-{args.split}-{idx}"),
                "question": sample.get("question"),
                "prediction": answer,
                "reference": sample.get("answer"),
                "llm_response": response,
                "correct": is_correct,
            }
        )

    n = len(per_example)
    accuracy = total_correct / n if n else 0.0
    return {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "accuracy": accuracy,
        "num_examples": n,
        "latency_sec": None,
        "calls": None,
        "tokens": None,
        "api_cost": None,
        "min_chain_len": args.min_chain_len,
        "max_chain_len": args.max_chain_len,
        "per_example": per_example,
    }


def main() -> None:
    args = parse_args()
    metrics = run_cot(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"[CoT] dataset={args.dataset} split={args.split} model={args.model} "
        f"accuracy={metrics['accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
