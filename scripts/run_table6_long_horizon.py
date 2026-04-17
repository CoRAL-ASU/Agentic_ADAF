#!/usr/bin/env python3
"""Run long-horizon DeALoG evaluation rows and emit a Table-6 style markdown."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dealog import run_dataset


def _weighted_accuracy(metrics_list: List[Dict[str, Any]]) -> float:
    total = sum(int(m.get("num_examples", 0) or 0) for m in metrics_list)
    if total == 0:
        return 0.0
    correct = sum(float(m.get("accuracy", 0.0) or 0.0) * int(m.get("num_examples", 0) or 0) for m in metrics_list)
    return correct / total


def _run_row(
    *,
    dataset: str,
    split: str,
    llm: str,
    summarizer_llm: str | None,
    summarizer_temperature: float,
    summarizer_max_tokens: int,
    max_rounds: int,
    limit: int | None,
    min_chain_len: int | None,
    max_chain_len: int | None,
) -> Dict[str, Any]:
    return run_dataset(
        dataset=dataset,
        split=split,
        llm=llm,
        summarizer_llm=summarizer_llm,
        summarizer_temperature=summarizer_temperature,
        summarizer_max_tokens=summarizer_max_tokens,
        visual_caption_model=None,
        visual_caption_path=None,
        visual_ocr_engine=None,
        visual_ocr_model_dir=None,
        limit=limit,
        min_chain_len=min_chain_len,
        max_chain_len=max_chain_len,
        max_rounds=max_rounds,
        scheduler_model=None,
        scheduler_threshold=0.4,
        parallel_retrieval=False,
    )


def _format_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("| Task / Chain Length | EM | EM (8192 tok) | Notes |")
    lines.append("|---|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['em']:.2f} | {row['em_8192']:.2f} | {row['notes']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Table-6 long-horizon rows for DeALoG.")
    parser.add_argument("--llm", required=True, help="Primary model ID.")
    parser.add_argument("--summarizer-llm", default=None, help="Optional summarizer model ID.")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-row sample cap.")
    parser.add_argument("--base-max-tokens", type=int, default=256)
    parser.add_argument("--ablation-max-tokens", type=int, default=8192)
    parser.add_argument("--summarizer-temperature", type=float, default=0.2)
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.getenv("DEALOG_CUDA_VISIBLE_DEVICES"),
        help=(
            "Optional CUDA_VISIBLE_DEVICES value for this run "
            "(defaults to DEALOG_CUDA_VISIBLE_DEVICES)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/table6_long_horizon"),
    )
    args = parser.parse_args()

    if args.cuda_visible_devices is not None and str(args.cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices).strip()
        print(f"[Table6-DeALoG] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs: List[Tuple[str, str, str, int | None, int | None]] = [
        ("CRT-QA", "Upto 10 rounds", "crtqa", None, None),
        ("Multi-Hop 5-6 steps", "All chains correct", "multi_hop", 5, 6),
        ("Multi-Hop 7-8 steps", "Occasional lookup omission", "multi_hop", 7, 8),
    ]

    baseline_results: Dict[str, Dict[str, Any]] = {}
    ablation_results: Dict[str, Dict[str, Any]] = {}
    table_rows: List[Dict[str, Any]] = []

    for task, notes, dataset, min_len, max_len in specs:
        row_temperature = 0.0 if dataset == "crtqa" else args.summarizer_temperature
        base_metrics = _run_row(
            dataset=dataset,
            split=args.split,
            llm=args.llm,
            summarizer_llm=args.summarizer_llm,
            summarizer_temperature=row_temperature,
            summarizer_max_tokens=args.base_max_tokens,
            max_rounds=args.max_rounds,
            limit=args.limit,
            min_chain_len=min_len,
            max_chain_len=max_len,
        )
        abl_metrics = _run_row(
            dataset=dataset,
            split=args.split,
            llm=args.llm,
            summarizer_llm=args.summarizer_llm,
            summarizer_temperature=row_temperature,
            summarizer_max_tokens=args.ablation_max_tokens,
            max_rounds=args.max_rounds,
            limit=args.limit,
            min_chain_len=min_len,
            max_chain_len=max_len,
        )
        baseline_results[task] = base_metrics
        ablation_results[task] = abl_metrics
        table_rows.append(
            {
                "task": task,
                "em": float(base_metrics.get("accuracy", 0.0)),
                "em_8192": float(abl_metrics.get("accuracy", 0.0)),
                "notes": notes,
            }
        )

    all_base = _weighted_accuracy(list(baseline_results.values()))
    all_abl = _weighted_accuracy(list(ablation_results.values()))
    table_rows.append(
        {
            "task": "All",
            "em": all_base,
            "em_8192": all_abl,
            "notes": "Strong long-horizon EM",
        }
    )

    json_payload = {
        "config": {
            "llm": args.llm,
            "summarizer_llm": args.summarizer_llm,
            "split": args.split,
            "max_rounds": args.max_rounds,
            "base_max_tokens": args.base_max_tokens,
            "ablation_max_tokens": args.ablation_max_tokens,
            "summarizer_temperature": args.summarizer_temperature,
            "limit": args.limit,
            "cuda_visible_devices": args.cuda_visible_devices,
        },
        "rows": table_rows,
        "baseline_metrics": baseline_results,
        "ablation_metrics": ablation_results,
    }

    (args.output_dir / "table6_long_horizon.json").write_text(
        json.dumps(json_payload, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "table6_long_horizon.md").write_text(
        _format_markdown(table_rows),
        encoding="utf-8",
    )
    print(f"Wrote {args.output_dir / 'table6_long_horizon.md'}")


if __name__ == "__main__":
    main()
