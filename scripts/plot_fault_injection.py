"""Plot corruption level vs repair/final accuracy from the fault-injection experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def plot(summary_path: Path, out_path: Path) -> None:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    # collapse across seeds by average
    by_rate = {}
    for row in summary:
        rate = row["corruption_rate"]
        by_rate.setdefault(rate, []).append(row)

    rates: List[float] = []
    catch: List[float] = []
    repair: List[float] = []
    accuracy: List[float] = []
    for rate, rows in sorted(by_rate.items()):
        rates.append(rate)
        catch.append(sum(r["catch_rate"] for r in rows) / len(rows))
        repair.append(sum(r["repair_rate"] for r in rows) / len(rows))
        accuracy.append(sum(r["final_accuracy"] for r in rows) / len(rows))

    plt.figure(figsize=(6, 4))
    plt.plot(rates, catch, marker="o", label="catch rate")
    plt.plot(rates, repair, marker="o", label="repair rate")
    plt.plot(rates, accuracy, marker="o", label="final accuracy")
    plt.xlabel("corruption rate")
    plt.ylabel("score")
    plt.ylim(0, 1)
    plt.title("Fault injection: corruption vs repair/accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fault-injection results.")
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("benchmarks/results/fault_injection.summary.json"),
        help="Summary JSON from fault_injection_experiment.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results/fault_injection.png"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.summary_file, args.out)
