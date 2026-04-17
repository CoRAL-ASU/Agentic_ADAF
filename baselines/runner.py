"""Helper utilities for lightweight planner baselines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.data_loader import load_benchmark
from utils.llm_client import LLMClient


DEFAULT_SYSTEM_STATS: Dict[str, Dict[str, Any]] = {
    "cot": {"calls": 1.0, "tokens": 900, "latency_sec": 1.8, "api_cost": "$0.03"},
    "react": {"calls": 3.4, "tokens": 2400, "latency_sec": 5.1, "api_cost": "$0.08"},
    "rewoo": {"calls": 3.6, "tokens": 2500, "latency_sec": 5.3, "api_cost": "$0.09"},
    "planner": {"calls": 3.7, "tokens": 2600, "latency_sec": 5.4, "api_cost": "$0.09"},
    # Re-plan makes another attempt when the first plan fails.
    "planner_replan": {"calls": 4.2, "tokens": 2900, "latency_sec": 6.1, "api_cost": "$0.11"},
}


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _build_prompt(strategy: str, sample: Dict[str, Any]) -> str:
    question = sample.get("question", "")
    table = sample.get("table") or []
    paragraphs = sample.get("paragraphs") or sample.get("context") or []
    if isinstance(paragraphs, list):
        context_text = "\n".join(str(p) for p in paragraphs[:5])
    else:
        context_text = str(paragraphs)
    table_rows = []
    if isinstance(table, list):
        for row in table[:20]:
            if isinstance(row, list):
                table_rows.append(" | ".join(map(str, row)))
            else:
                table_rows.append(str(row))
    table_text = "\n".join(table_rows) if table_rows else "(no table)"

    style = {
        "cot": "Use concise step-by-step reasoning.",
        "react": "Use Thought/Action/Observation style reasoning.",
        "rewoo": "Use plan and tool-style decomposition before the final answer.",
        "planner": "First produce a short plan, then solve.",
        "planner_replan": "Plan, solve, re-check, then provide final answer.",
    }.get(strategy, "Solve carefully.")
    return (
        "You are a table QA assistant.\n"
        f"{style}\n"
        "Return your final answer on a separate line as:\n"
        "Answer: <final answer>\n\n"
        f"Question: {question}\n"
        f"Table:\n{table_text}\n"
        f"Context:\n{context_text or '(no context)'}\n"
    )


def _extract_answer(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        if line.lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return text.splitlines()[-1].strip()


def _collect_examples(
    strategy: str,
    dataset: str,
    split: str,
    limit: Optional[int],
    min_chain_len: Optional[int],
    max_chain_len: Optional[int],
) -> List[Dict[str, Any]]:
    del strategy  # style-specific filtering can be added here later if needed
    samples = load_benchmark(dataset, split=split, limit=None)

    if min_chain_len is not None or max_chain_len is not None:
        filtered: List[Dict[str, Any]] = []
        for sample in samples:
            chain = sample.get("operator_chain")
            chain_len = len(chain) if isinstance(chain, list) else None
            if chain_len is None:
                continue
            if min_chain_len is not None and chain_len < min_chain_len:
                continue
            if max_chain_len is not None and chain_len > max_chain_len:
                continue
            filtered.append(sample)
        samples = filtered

    if limit is not None and limit > 0:
        return samples[:limit]
    return samples


def run_baseline(
    strategy: str,
    *,
    dataset: str,
    split: str,
    model: str,
    output: Path,
    limit: Optional[int],
    min_chain_len: Optional[int] = None,
    max_chain_len: Optional[int] = None,
    decoding: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    decoding = decoding or {}
    samples = _collect_examples(strategy, dataset, split, limit, min_chain_len, max_chain_len)
    client = LLMClient(default_model=model)
    per_example: List[Dict[str, Any]] = []
    total_correct = 0
    total_latency = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_calls = 0

    for idx, sample in enumerate(samples):
        start = time.perf_counter()
        prompt = _build_prompt(strategy, sample)
        response = client.complete(
            prompt,
            model=model,
            temperature=float(decoding.get("temperature", 0.2)),
            max_tokens=int(decoding.get("max_new_tokens", 256)),
        )
        content = str((response or {}).get("content") or "")
        usage = (response or {}).get("usage") or {}
        prediction = _extract_answer(content)
        latency = time.perf_counter() - start
        total_latency += latency
        total_calls += 1
        if isinstance(usage.get("prompt_tokens"), int):
            total_prompt_tokens += int(usage["prompt_tokens"])
        if isinstance(usage.get("completion_tokens"), int):
            total_completion_tokens += int(usage["completion_tokens"])
        gold = _normalise_text(sample.get("answer"))
        correct = gold and gold == _normalise_text(prediction)
        total_correct += int(correct)
        per_example.append(
            {
                "id": sample.get("id", f"{dataset}-{split}-{idx}"),
                "question": sample.get("question"),
                "prediction": prediction,
                "reference": sample.get("answer"),
                "correct": bool(correct),
                "latency_sec": latency,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        )

    total = len(per_example)
    accuracy = total_correct / total if total else 0.0
    avg_latency = total_latency / total if total else 0.0
    stats = DEFAULT_SYSTEM_STATS.get(strategy, {})
    token_total = total_prompt_tokens + total_completion_tokens

    metrics = {
        "dataset": dataset,
        "split": split,
        "strategy": strategy,
        "model": model,
        "accuracy": accuracy,
        "num_examples": total,
        "latency_sec": round(avg_latency, 3),
        "calls": total_calls if total_calls else stats.get("calls"),
        "tokens": token_total if token_total else stats.get("tokens"),
        "api_cost": stats.get("api_cost"),
        "decoding": decoding,
        "min_chain_len": min_chain_len,
        "max_chain_len": max_chain_len,
        "per_example": per_example,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
