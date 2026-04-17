#!/usr/bin/env python3
"""Collect LLM-as-a-rater annotations for FeTaQA explanations via OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL") or os.environ.get("OPENROUTER_SITE")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME") or os.environ.get("OPENROUTER_SITE_NAME")

SYSTEM_PROMPT = (
    "You are a meticulous fact-checker. For each example, read the question, "
    "table, optional context, and the model’s explanation. Return exactly one "
    "label from {Supported, Partially Supported, Unsupported}, followed by a concise "
    "justification. “Supported” means every number/entity in the explanation is backed "
    "by the provided evidence; “Partially Supported” means some claims are grounded but "
    "at least one detail is missing or ambiguous; “Unsupported” means the explanation "
    "contradicts or lacks evidence. Do not invent new facts or cite outside knowledge."
)


def build_user_message(example: Dict[str, Any]) -> str:
    question = example.get("question", "").strip()
    table_text = example.get("table_text") or example.get("table") or ""
    context = example.get("context") or example.get("paragraphs") or ""
    if isinstance(context, list):
        context = "\n".join(context)
    rationale = example.get("rationale") or example.get("candidate_explanation") or ""
    return (
        f"Question:\n{question}\n\n"
        f"Table (linearized):\n{table_text}\n\n"
        f"Context:\n{context}\n\n"
        f"Model explanation:\n{rationale}\n\n"
        "Respond in the format:\n"
        "Label: <Supported|Partially Supported|Unsupported>\n"
        "Reason: <short justification referencing evidence or the missing piece>\n"
    )


def call_openrouter(model: str, payload: Dict[str, Any], *, api_key: str, max_retries: int = 5, backoff: float = 2.0) -> Dict[str, Any]:
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-Title"] = OPENROUTER_APP_NAME
    for attempt in range(1, max_retries + 1):
        response = requests.post(url, headers=headers, json={"model": model, **payload}, timeout=60)
        if response.status_code == 200:
            return response.json()
        time.sleep(backoff * attempt)
    raise RuntimeError(f"OpenRouter request failed after {max_retries} attempts: {response.text}")


def parse_label(text: str) -> Dict[str, str]:
    label = "Unsupported"
    reason = text.strip()
    lower = text.lower()
    if "supported" in lower and "partially" not in lower:
        label = "Supported"
    elif "partially" in lower:
        label = "Partially Supported"
    return {"label": label, "reason": reason}


def load_examples(data_path: Path) -> List[Dict[str, Any]]:
    text = data_path.read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    return payload.get("per_example", [])


def run_annotations(
    data_path: Path,
    output_path: Path,
    models: List[str],
    *,
    api_key: str,
) -> None:
    items = load_examples(data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for example in items:
            user_message = build_user_message(example)
            for model in models:
                payload = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ]
                }
                response = call_openrouter(model, payload, api_key=api_key)
                text = response["choices"][0]["message"]["content"]
                parsed = parse_label(text)
                out_f.write(
                    json.dumps(
                        {
                            "id": example.get("id"),
                            "model": model,
                            "label": parsed["label"],
                            "reason": parsed["reason"],
                            "raw_response": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out_f.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenRouter LLM raters over FeTaQA examples.")
    parser.add_argument("--data", type=Path, required=True, help="Path to JSON or JSONL file with examples.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the annotation JSONL.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            os.environ.get("LLM_RATER_A", "meta-llama/llama-3.1-70b-instruct"),
            os.environ.get("LLM_RATER_B", "openai/gpt-4.1-mini"),
            os.environ.get("LLM_RATER_C", "qwen/qwen3-4b"),
        ],
        help="List of OpenRouter model IDs to act as raters.",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"), help="OpenRouter API key.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Export it or pass --api-key.")
    run_annotations(args.data, args.output, [m for m in args.models if m], api_key=args.api_key)


if __name__ == "__main__":
    main()
