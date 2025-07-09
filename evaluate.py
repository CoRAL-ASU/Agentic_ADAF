"""Utilities for evaluating a fine-tuned model on the TAT-QA dataset."""

from __future__ import annotations

import argparse
import json
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.metrics import compute_f1
from utils.data_loader import load_benchmark


def _build_prompt(question: str, table, paragraphs: List[str]) -> str:
    table_str = json.dumps(table) if table else ""
    ctx = " ".join(paragraphs) if paragraphs else ""
    return f"Question: {question}\nTable: {table_str}\nContext: {ctx}\nAnswer:"


def _generate_answer(model, tokenizer, sample: dict, max_new_tokens: int = 32) -> str:
    prompt = _build_prompt(sample.get("question", ""), sample.get("table"), sample.get("paragraphs", []))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt) :]
    return text.strip()


def evaluate_model(model_path: str, split: str = "dev") -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    dataset = load_benchmark("tatqa", split=split)

    scores = []
    for sample in dataset:
        pred = _generate_answer(model, tokenizer, sample)
        scores.append(compute_f1(pred, str(sample.get("answer", ""))))
    return sum(scores) / len(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to a fine-tuned model directory")
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    score = evaluate_model(args.model, args.split)
    print(f"F1: {score:.4f}")
