#!/usr/bin/env python3
"""Compute Log-Groundedness over saved DeALoG logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import spacy
except Exception:  # pragma: no cover - spaCy is optional at runtime
    spacy = None


def _load_spacy() -> object | None:
    if not spacy:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?%?", text)


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _extract_mentions(text: str, nlp_model) -> List[Tuple[str, str]]:
    mentions: List[Tuple[str, str]] = []
    for num in _extract_numbers(text):
        mentions.append((num, "number"))
    if nlp_model:
        doc = nlp_model(text)
        for ent in doc.ents:
            if ent.text.strip():
                mentions.append((ent.text.strip(), ent.label_))
    else:
        candidates = re.findall(r"\b[A-Z][A-Za-z0-9]+\b(?:\s+[A-Z][A-Za-z0-9]+)*", text)
        for cand in candidates:
            mentions.append((cand.strip(), "entity"))
    return mentions


def _build_evidence(log_entries: Iterable[Dict]) -> List[str]:
    evidence_types = {"lookup", "quote", "visual", "table", "context"}
    evidence: List[str] = []
    for entry in log_entries or []:
        entry_type = (entry.get("type") or "").lower()
        if entry_type not in evidence_types:
            continue
        content = entry.get("content")
        if isinstance(content, dict):
            evidence.append(json.dumps(content))
        elif isinstance(content, list):
            evidence.extend(map(str, content))
        elif content:
            evidence.append(str(content))
        metadata = entry.get("metadata") or {}
        for value in metadata.values():
            if value:
                evidence.append(str(value))
    return evidence


def _numeric_match(value: str, evidence: Iterable[str]) -> bool:
    try:
        clean = value.replace("%", "")
        target = float(clean)
    except ValueError:
        return False
    tolerance = max(1e-3, 0.01 * abs(target))
    for snippet in evidence:
        for candidate in _extract_numbers(snippet):
            try:
                cand_val = float(candidate.replace("%", ""))
            except ValueError:
                continue
            if abs(cand_val - target) <= tolerance:
                return True
    return False


def _textual_match(value: str, evidence: Iterable[str]) -> bool:
    norm_value = _normalise_text(value)
    for snippet in evidence:
        if norm_value and norm_value in _normalise_text(snippet):
            return True
    return False


def compute_log_groundedness(results_file: Path, output_file: Path) -> None:
    metrics = json.loads(results_file.read_text())
    nlp_model = _load_spacy()
    per_example = []
    perfect = 0
    total_score = 0.0

    for example in metrics.get("per_example", []):
        rationale = example.get("rationale") or ""
        if not rationale:
            continue
        mentions = _extract_mentions(rationale, nlp_model)
        if not mentions:
            per_example.append({"id": example["id"], "score": 1.0, "total_mentions": 0})
            perfect += 1
            total_score += 1.0
            continue
        evidence = _build_evidence(example.get("log") or [])
        supported = 0
        for mention, label in mentions:
            if label == "number":
                supported += int(_numeric_match(mention, evidence))
            else:
                supported += int(_textual_match(mention, evidence))
        score = supported / len(mentions)
        total_score += score
        if math.isclose(score, 1.0):
            perfect += 1
        per_example.append(
            {
                "id": example["id"],
                "score": score,
                "supported_mentions": supported,
                "total_mentions": len(mentions),
            }
        )

    num_examples = len(per_example)
    summary = {
        "mean_log_groundedness": total_score / num_examples if num_examples else 0.0,
        "perfect_fraction": perfect / num_examples if num_examples else 0.0,
        "num_examples": num_examples,
        "per_example": per_example,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Log-Groundedness for FeTaQA logs.")
    parser.add_argument("--results-file", type=Path, required=True, help="Path to run_dealog JSON output.")
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_log_groundedness(results_file=args.results_file, output_file=args.output_file)


if __name__ == "__main__":
    main()
