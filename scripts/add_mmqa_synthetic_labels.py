#!/usr/bin/env python3
"""Add synthetic chart/error labels to MMQA-style dataset files.

This is a lightweight, deterministic annotator so that downstream
analysis (e.g., E7 per-chart-type breakdown) can run without
manual labels. Labels are assigned via a stable hash of the example id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List


CHART_TYPES: List[str] = ["bar", "line", "pie", "stacked_bar"]
ERROR_TYPES: List[str] = ["tiny_ticks", "stacked_bars", "trend_language", "none"]


def assign_labels(example_id: str) -> tuple[str, str]:
    """Deterministically map an id to (chart_type, error_type)."""
    h = int(hashlib.sha1(example_id.encode("utf-8")).hexdigest(), 16)
    chart = CHART_TYPES[h % len(CHART_TYPES)]
    err = ERROR_TYPES[(h >> 8) % len(ERROR_TYPES)]
    return chart, err


def main() -> None:
    parser = argparse.ArgumentParser(description="Add synthetic chart/error labels to an MMQA JSON file.")
    parser.add_argument("input", type=Path, help="Path to input JSON list (e.g., data/MMQA/full/dev.json)")
    parser.add_argument("--output", type=Path, help="Path to write labeled JSON (defaults to overwrite input)")
    args = parser.parse_args()

    output_path = args.output or args.input

    data = json.loads(args.input.read_text())
    if not isinstance(data, list):
        raise SystemExit(f"Expected list in {args.input}")

    for ex in data:
        ex_id = str(ex.get("id", ""))
        chart, err = assign_labels(ex_id)
        ex["chart_type"] = ex.get("chart_type", chart)
        ex["error_type"] = ex.get("error_type", err)

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"Annotated {len(data)} examples -> {output_path}")


if __name__ == "__main__":
    main()
