"""Fault-injection harness for bad-agent robustness experiments.

For each corruption rate (% of table/context entries), we:
- Corrupt a random subset of table/context log entries (value swap, unit flip, off-by-one)
- Run the pipeline, record verifier catch-rate, repair-rate after one re-engagement, and final accuracy.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.coordinator import AdaptiveOrchestrator
from utils.data_loader import load_benchmark


Numeric = float | int


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _gather_numeric_positions(obj: Any, path: tuple[int, ...] = ()) -> List[tuple[tuple[int, ...], float]]:
    """Return (path, value) pairs for numeric items inside nested lists/tuples."""
    positions: List[tuple[tuple[int, ...], float]] = []
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            positions.extend(_gather_numeric_positions(v, path + (i,)))
    elif _is_number(obj):
        positions.append((path, float(obj)))
    return positions


def _set_at_path(obj: Any, path: Sequence[int], new_value: Numeric) -> Any:
    """Mutate obj in-place at path (list/tuple indices)."""
    if not path:
        return new_value
    cursor = obj
    for idx in path[:-1]:
        cursor = cursor[idx]
    cursor[path[-1]] = new_value
    return obj


def _corrupt_number(value: float, mode: str, rng: random.Random) -> float:
    if mode == "off_by_one":
        delta = 1 if rng.random() < 0.5 else -1
        return value + delta
    if mode == "unit_flip":
        factor = rng.choice([0.1, 0.01, 10.0, 100.0])
        return value * factor
    return value


def _corrupt_text_numbers(text: str, mode: str, rng: random.Random) -> tuple[str, bool]:
    matches = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", text))
    if not matches:
        return text, False
    target = rng.choice(matches)
    original = target.group(0)
    try:
        original_val = float(original)
    except ValueError:
        return text, False
    corrupted_val = _corrupt_number(original_val, mode, rng)
    corrupted_text = text[: target.start()] + str(corrupted_val) + text[target.end() :]
    return corrupted_text, True


class FaultInjector:
    """Callable transform passed into SharedLog to corrupt entries on write."""

    def __init__(self, corruption_rate: float, rng: random.Random) -> None:
        self.corruption_rate = corruption_rate
        self.rng = rng
        self.total_targets = 0
        self.total_corrupted = 0
        self.sample_corrupted = 0
        self.enabled = True

    def start_sample(self) -> None:
        self.sample_corrupted = 0

    def disable(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True

    def __call__(self, entry):
        if not self.enabled or entry.type not in {"table", "context"}:
            return entry

        self.total_targets += 1
        if self.rng.random() >= self.corruption_rate:
            return entry

        new_entry = copy.deepcopy(entry)
        corruption_kind = self._apply_corruption(new_entry)
        if corruption_kind is None:
            return entry

        new_entry.metadata = dict(new_entry.metadata)
        new_entry.metadata.update(
            {
                "fault_injected": True,
                "corruption_kind": corruption_kind,
                "corruption_rate": self.corruption_rate,
            }
        )
        self.total_corrupted += 1
        self.sample_corrupted += 1
        return new_entry

    def _apply_corruption(self, entry) -> str | None:
        modes = ["swap", "unit_flip", "off_by_one"]
        mode = self.rng.choice(modes)

        # Corrupt context strings
        if entry.type == "context" and isinstance(entry.content, str):
            mode_for_text = "unit_flip" if mode == "swap" else mode
            new_text, changed = _corrupt_text_numbers(entry.content, mode_for_text, self.rng)
            if changed:
                entry.content = new_text
                return mode_for_text
            return None

        # Corrupt tabular/numeric payloads
        positions = _gather_numeric_positions(entry.content)
        if not positions:
            return None

        if mode == "swap" and len(positions) >= 2:
            (path_a, val_a), (path_b, val_b) = self.rng.sample(positions, 2)
            entry.content = _set_at_path(entry.content, path_a, val_b)
            entry.content = _set_at_path(entry.content, path_b, val_a)
            return "swap"

        # Otherwise mutate one value
        target_path, target_val = self.rng.choice(positions)
        corrupted_val = _corrupt_number(target_val, "unit_flip" if mode == "unit_flip" else "off_by_one", self.rng)
        entry.content = _set_at_path(entry.content, target_path, corrupted_val)
        return "unit_flip" if mode == "unit_flip" else "off_by_one"


def run_experiment(
    dataset: str,
    split: str,
    limit: int | None,
    llm: str | None,
    corruption_rates: List[float],
    seeds: List[int],
    results_path: Path,
) -> None:
    data = load_benchmark(dataset, split=split, limit=limit)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate: List[Dict[str, Any]] = []

    for rate in corruption_rates:
        for seed in seeds:
            rng = random.Random(seed)
            injector = FaultInjector(rate, rng)
            orchestrator = AdaptiveOrchestrator(model_name=llm, entry_transform=injector)

            caught = 0
            repaired = 0
            final_correct = 0
            total_samples = 0
            total_corrupted_entries = 0

            for sample in data:
                total_samples += 1
                injector.start_sample()

                first_result = orchestrator.run(sample)
                corrupted = injector.sample_corrupted > 0
                total_corrupted_entries += injector.sample_corrupted

                flagged = bool(corrupted and not first_result.get("verified"))
                final_verified = bool(first_result.get("verified"))

                if flagged:
                    caught += 1
                    # One re-engagement pass with corruption disabled
                    injector.disable()
                    second_result = orchestrator.run(sample)
                    injector.enable()
                    if second_result.get("verified"):
                        repaired += 1
                        final_verified = True

                if final_verified:
                    final_correct += 1

                record = {
                    "dataset": dataset,
                    "split": split,
                    "sample_id": sample.get("id"),
                    "corruption_rate": rate,
                    "seed": seed,
                    "corrupted_entries": injector.sample_corrupted,
                    "flagged": flagged,
                    "repaired": flagged and final_verified,
                    "verified": final_verified,
                    "answer": first_result.get("answer"),
                }
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

            aggregate.append(
                {
                    "corruption_rate": rate,
                    "seed": seed,
                    "total_samples": total_samples,
                    "total_corrupted_entries": total_corrupted_entries,
                    "catch_rate": caught / total_samples if total_samples else 0.0,
                    "repair_rate": repaired / caught if caught else 0.0,
                    "final_accuracy": final_correct / total_samples if total_samples else 0.0,
                }
            )

    summary_path = results_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"Wrote per-sample records to {results_path}")
    print(f"Wrote aggregate summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fault-injection experiment harness.")
    parser.add_argument("--dataset", type=str, default="tatqa")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm", type=str, default=None)
    parser.add_argument(
        "--corruption-rate",
        type=str,
        default="0.1,0.2,0.3",
        help="Comma-delimited rates (0-1) for corruption.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3",
        help="Comma-delimited integer seeds.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("benchmarks/results/fault_injection.jsonl"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rates = [float(r) for r in args.corruption_rate.split(",") if r]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    run_experiment(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        llm=args.llm,
        corruption_rates=rates,
        seeds=seeds,
        results_path=args.results_file,
    )
