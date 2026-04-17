#!/usr/bin/env python3
"""Compute latency-quality Pareto frontier from DeALoG result JSON files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path
from typing import Dict, List


def _load_rows(pattern: str) -> List[Dict]:
    rows: List[Dict] = []
    for file_path in sorted(glob.glob(pattern)):
        path = Path(file_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_example = payload.get("per_example") or []
        lats = [
            float(example.get("latency_sec", 0.0))
            for example in per_example
            if isinstance(example.get("latency_sec"), (int, float))
        ]
        mode = "parallel" if "_par_" in path.name else "sequential"
        max_rounds = None
        if "_r" in path.stem:
            try:
                max_rounds = int(path.stem.split("_r")[1].split("_")[0])
            except ValueError:
                max_rounds = None
        mean_lat = sum(lats) / len(lats) if lats else 0.0
        p50_lat = statistics.median(lats) if lats else 0.0
        p90_lat = sorted(lats)[int(0.9 * (len(lats) - 1))] if lats else 0.0
        rows.append(
            {
                "file": path.name,
                "mode": mode,
                "max_rounds": max_rounds,
                "accuracy": float(payload.get("accuracy", 0.0)),
                "latency_mean_sec": mean_lat,
                "latency_p50_sec": p50_lat,
                "latency_p90_sec": p90_lat,
            }
        )
    return rows


def _pareto_frontier(rows: List[Dict]) -> List[Dict]:
    frontier: List[Dict] = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            no_worse = (
                b["latency_mean_sec"] <= a["latency_mean_sec"]
                and b["accuracy"] >= a["accuracy"]
            )
            strictly_better = (
                b["latency_mean_sec"] < a["latency_mean_sec"]
                or b["accuracy"] > a["accuracy"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(a)
    return sorted(frontier, key=lambda row: (row["latency_mean_sec"], -row["accuracy"]))


def _write_csv(rows: List[Dict], frontier: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frontier_files = {row["file"] for row in frontier}
    fieldnames = list(rows[0].keys()) + ["pareto"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["latency_mean_sec"], -r["accuracy"])):
            record = dict(row)
            record["pareto"] = "yes" if row["file"] in frontier_files else "no"
            writer.writerow(record)


def _write_markdown(rows: List[Dict], frontier: List[Dict], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    frontier_files = {row["file"] for row in frontier}
    ordered = sorted(rows, key=lambda r: (r["latency_mean_sec"], -r["accuracy"]))

    lines = []
    lines.append("### Latency-Quality Sweep\n")
    lines.append("| config | mode | max_rounds | accuracy | mean latency (s) | p50 (s) | p90 (s) | pareto |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in ordered:
        lines.append(
            f"| {row['file']} | {row['mode']} | {row['max_rounds']} | "
            f"{row['accuracy']:.3f} | {row['latency_mean_sec']:.6f} | "
            f"{row['latency_p50_sec']:.6f} | {row['latency_p90_sec']:.6f} | "
            f"{'yes' if row['file'] in frontier_files else 'no'} |"
        )

    lines.append("\n### Pareto Frontier\n")
    lines.append("| config | mode | max_rounds | accuracy | mean latency (s) |")
    lines.append("|---|---|---:|---:|---:|")
    for row in frontier:
        lines.append(
            f"| {row['file']} | {row['mode']} | {row['max_rounds']} | "
            f"{row['accuracy']:.3f} | {row['latency_mean_sec']:.6f} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a latency-quality Pareto frontier.")
    parser.add_argument(
        "--input-glob",
        default="benchmarks/results/e7/*.json",
        help="Glob pattern matching result JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmarks/results/e7/pareto_latency_quality.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-md",
        default="benchmarks/results/e7/pareto_latency_quality.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.input_glob)
    if not rows:
        raise SystemExit(f"No files matched: {args.input_glob}")

    frontier = _pareto_frontier(rows)
    _write_csv(rows, frontier, Path(args.output_csv))
    _write_markdown(rows, frontier, Path(args.output_md))
    print(f"Loaded {len(rows)} runs; Pareto points={len(frontier)}")
    for row in frontier:
        print(
            f"- {row['file']}: accuracy={row['accuracy']:.3f}, "
            f"mean_latency={row['latency_mean_sec']:.6f}s"
        )


if __name__ == "__main__":
    main()
