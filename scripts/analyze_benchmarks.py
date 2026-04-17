#!/usr/bin/env python3
"""Aggregate benchmark logs into accuracy deltas, CIs, and permutation tests."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _load_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _accuracy_from_examples(examples: Iterable[Dict]) -> float:
    total = 0
    correct = 0
    for row in examples:
        total += 1
        if row.get("correct"):
            correct += 1
    return correct / total if total else 0.0


def _bootstrap_ci(values: List[float], *, num_samples: int = 1000, confidence: float = 0.95) -> Optional[Tuple[float, float]]:
    if not values:
        return None
    rng = random.Random(0)
    n = len(values)
    samples = sorted(
        sum(rng.choice(values) for _ in range(n)) / n for _ in range(num_samples)
    )
    lower_idx = int(((1 - confidence) / 2) * num_samples)
    upper_idx = int((confidence + (1 - confidence) / 2) * num_samples)
    lower = samples[max(0, min(lower_idx, num_samples - 1))]
    upper = samples[max(0, min(upper_idx, num_samples - 1))]
    return (lower, upper)


def _bootstrap_diff_ci(
    baseline: List[int],
    system: List[int],
    *,
    num_samples: int = 1000,
    confidence: float = 0.95,
) -> Optional[Tuple[float, float]]:
    if not baseline or len(baseline) != len(system):
        return None
    rng = random.Random(1)
    n = len(baseline)
    samples = []
    diffs = [system[i] - baseline[i] for i in range(n)]
    for _ in range(num_samples):
        idxs = [rng.randrange(n) for _ in range(n)]
        samples.append(sum(diffs[i] for i in idxs) / n)
    samples.sort()
    lower_idx = int(((1 - confidence) / 2) * num_samples)
    upper_idx = int((confidence + (1 - confidence) / 2) * num_samples)
    lower = samples[max(0, min(lower_idx, num_samples - 1))]
    upper = samples[max(0, min(upper_idx, num_samples - 1))]
    return (lower, upper)


def _paired_permutation_test(
    baseline: List[int], system: List[int], *, num_perms: int = 2000
) -> Optional[float]:
    if not baseline or len(baseline) != len(system):
        return None
    rng = random.Random(2)
    diffs = [s - b for s, b in zip(system, baseline)]
    observed = sum(diffs) / len(diffs)
    if math.isclose(observed, 0.0):
        return 1.0

    extreme = 0
    for _ in range(num_perms):
        permuted = [
            diff if rng.random() < 0.5 else -diff for diff in diffs
        ]
        stat = sum(permuted) / len(permuted)
        if abs(stat) >= abs(observed):
            extreme += 1
    return (extreme + 1) / (num_perms + 1)


def _per_example_map(examples: Iterable[Dict]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, row in enumerate(examples):
        key = str(row.get("id", idx))
        mapping[key] = int(bool(row.get("correct")))
    return mapping


def _gather_runs(records: List[Dict]) -> Dict[Tuple[str, str, str], Dict[str, Dict]]:
    runs: Dict[Tuple[str, str, str], Dict[str, Dict]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        metrics = record.get("metrics") or {}
        key = (
            record.get("dataset"),
            record.get("variant") or "",
            record.get("backbone"),
        )
        runs.setdefault(key, {})[record["system"]] = {
            "record": record,
            "metrics": metrics,
            "per_example": _per_example_map(metrics.get("per_example", [])),
        }
    return runs


def _format_ci(ci: Optional[Tuple[float, float]]) -> str:
    if not ci:
        return "N/A"
    return f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]"


def _format_optional(value: Optional[float], *, scale: float = 1.0, precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value*scale:.{precision}f}"


def analyse(results_path: Path, *, baseline_system: str, output_path: Optional[Path] = None) -> Path:
    records = _load_records(results_path)
    grouped = _gather_runs(records)
    rows: List[str] = []
    rows.append(
        "| Dataset | Backbone | System | Accuracy (%) | Δ vs. baseline (pp) | 95% CI (pp) | "
        "Paired p-value | Calls | Tokens | Latency (s) | API Cost |"
    )
    rows.append("|---|---|---|---|---|---|---|---|---|---|---|")

    for key, systems in sorted(grouped.items()):
        dataset, variant, backbone = key
        dataset_label = systems[list(systems.keys())[0]]["record"].get("dataset_label", dataset)
        backbone_label = systems[list(systems.keys())[0]]["record"].get("backbone_label", backbone)
        prefix = dataset_label if not variant else f"{dataset_label} ({variant})"
        if baseline_system not in systems:
            continue
        baseline = systems[baseline_system]
        base_examples = baseline["per_example"]
        base_vector = list(base_examples.values())
        base_accuracy = baseline["metrics"].get("accuracy")
        if base_accuracy is None and base_vector:
            base_accuracy = sum(base_vector) / len(base_vector)

        base_ci = _bootstrap_ci(base_vector)
        row_base = (
            f"| {prefix} | {backbone_label} | {systems[baseline_system]['record'].get('system_label', baseline_system)} | "
            f"{_format_optional(base_accuracy, scale=100)} | 0.0 | {_format_ci(base_ci)} | 1.000 | "
            f"{baseline['metrics'].get('calls', 'N/A')} | "
            f"{baseline['metrics'].get('tokens', 'N/A')} | "
            f"{baseline['metrics'].get('latency_sec', 'N/A')} | "
            f"{baseline['metrics'].get('api_cost', 'N/A')} |"
        )
        rows.append(row_base)

        for system_name, payload in systems.items():
            if system_name == baseline_system:
                continue
            per_example = payload["per_example"]
            overlap_keys = sorted(set(base_examples.keys()) & set(per_example.keys()))
            baseline_vector = [base_examples[k] for k in overlap_keys]
            system_vector = [per_example[k] for k in overlap_keys]
            accuracy = payload["metrics"].get("accuracy")
            if accuracy is None and system_vector:
                accuracy = sum(system_vector) / len(system_vector)
            delta = None
            if accuracy is not None and base_accuracy is not None:
                delta = accuracy - base_accuracy
            ci = _bootstrap_diff_ci(baseline_vector, system_vector)
            p_value = _paired_permutation_test(baseline_vector, system_vector)
            rows.append(
                f"| {prefix} | {backbone_label} | {payload['record'].get('system_label', system_name)} | "
                f"{_format_optional(accuracy, scale=100)} | {_format_optional(delta, scale=100)} | "
                f"{_format_ci(ci)} | {_format_optional(p_value, precision=4)} | "
                f"{payload['metrics'].get('calls', 'N/A')} | "
                f"{payload['metrics'].get('tokens', 'N/A')} | "
                f"{payload['metrics'].get('latency_sec', 'N/A')} | "
                f"{payload['metrics'].get('api_cost', 'N/A')} |"
            )

    output_path = output_path or results_path.with_name(results_path.stem + "_analysis.md")
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse benchmark JSONL output.")
    parser.add_argument("results", type=Path, help="Path to results.jsonl produced by run_benchmark_matrix.py")
    parser.add_argument("--baseline", default="cot", help="System name used as baseline (default: cot)")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for the markdown table.")
    args = parser.parse_args()

    output_path = analyse(args.results, baseline_system=args.baseline, output_path=args.output)
    print(f"Analysis written to {output_path}")


if __name__ == "__main__":
    main()
