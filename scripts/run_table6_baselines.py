#!/usr/bin/env python3
"""Run baseline systems on CRT-QA and multi-hop slices aligned to Table 6."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.run_cot import run_cot
from baselines.runner import run_baseline
from scripts.run_dealog import run_dataset

SYSTEM_LABELS: Dict[str, str] = {
    "cot": "CoT",
    "react": "ReAct",
    "rewoo": "ReWOO",
    "planner": "Planner",
    "planner_replan": "Planner (Replan)",
    "dealog": "DeALoG",
}

TABLE_SPECS: List[Tuple[str, str, str, Optional[int], Optional[int]]] = [
    ("CRT-QA", "Upto 10 rounds", "crtqa", None, None),
    ("Multi-Hop 5-6 steps", "All chains correct", "multi_hop", 5, 6),
    ("Multi-Hop 7-8 steps", "Occasional lookup omission", "multi_hop", 7, 8),
]


def _parse_systems(raw: str) -> List[str]:
    systems = [item.strip() for item in raw.split(",") if item.strip()]
    if not systems:
        raise ValueError("At least one system must be provided.")
    unknown = [name for name in systems if name not in SYSTEM_LABELS]
    if unknown:
        raise ValueError(f"Unsupported systems: {unknown}. Supported: {sorted(SYSTEM_LABELS)}")
    return systems


def _weighted_accuracy(metrics_list: List[Dict[str, Any]]) -> float:
    total = sum(int(m.get("num_examples", 0) or 0) for m in metrics_list)
    if total == 0:
        return 0.0
    correct = sum(float(m.get("accuracy", 0.0) or 0.0) * int(m.get("num_examples", 0) or 0) for m in metrics_list)
    return correct / total


def _run_system_row(
    *,
    system: str,
    dataset: str,
    split: str,
    llm: str,
    raw_path: Path,
    limit: Optional[int],
    min_chain_len: Optional[int],
    max_chain_len: Optional[int],
    cot_temperature: float,
    cot_max_new_tokens: int,
    decoding: Dict[str, Any],
    max_rounds: int,
    summarizer_llm: Optional[str],
    summarizer_temperature: float,
    summarizer_max_tokens: int,
) -> Dict[str, Any]:
    if system == "cot":
        args = argparse.Namespace(
            dataset=dataset,
            split=split,
            model=llm,
            output=raw_path,
            limit=limit,
            min_chain_len=min_chain_len,
            max_chain_len=max_chain_len,
            temperature=cot_temperature,
            max_new_tokens=cot_max_new_tokens,
        )
        metrics = run_cot(args)
    elif system in {"react", "rewoo", "planner", "planner_replan"}:
        metrics = run_baseline(
            system,
            dataset=dataset,
            split=split,
            model=llm,
            output=raw_path,
            limit=limit,
            min_chain_len=min_chain_len,
            max_chain_len=max_chain_len,
            decoding=decoding,
        )
    elif system == "dealog":
        metrics = run_dataset(
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
    else:
        raise ValueError(f"Unsupported system: {system}")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _format_markdown(rows: List[Dict[str, Any]], systems: List[str]) -> str:
    headers = ["Task / Chain Length"] + [SYSTEM_LABELS[s] for s in systems] + ["Notes"]
    align = ["---"] + ["---:"] * len(systems) + ["---"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        cells = [row["task"]]
        for system in systems:
            cells.append(f"{float(row['scores'][system]):.2f}")
        cells.append(row["notes"])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Table-6 aligned baselines on CRT-QA and multi-hop slices.")
    parser.add_argument("--llm", required=True, help="Model used by baseline systems.")
    parser.add_argument(
        "--systems",
        default="cot,react,rewoo,planner,planner_replan",
        help="Comma-separated systems from: cot,react,rewoo,planner,planner_replan,dealog",
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-row sample cap.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/table6_baselines"))
    parser.add_argument("--decoding", default="{}", help="JSON decoding config for non-CoT baselines.")
    parser.add_argument("--cot-temperature", type=float, default=0.2)
    parser.add_argument("--cot-max-new-tokens", type=int, default=256)
    parser.add_argument("--max-rounds", type=int, default=10, help="Used by DeALoG only.")
    parser.add_argument("--summarizer-llm", default=None, help="Used by DeALoG only.")
    parser.add_argument("--summarizer-temperature", type=float, default=0.2, help="Used by DeALoG only.")
    parser.add_argument("--summarizer-max-tokens", type=int, default=256, help="Used by DeALoG only.")
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.getenv("DEALOG_CUDA_VISIBLE_DEVICES"),
        help=(
            "Optional CUDA_VISIBLE_DEVICES value for this run "
            "(defaults to DEALOG_CUDA_VISIBLE_DEVICES)."
        ),
    )
    args = parser.parse_args()

    if args.cuda_visible_devices is not None and str(args.cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices).strip()
        print(f"[Table6-Baselines] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    systems = _parse_systems(args.systems)
    decoding = json.loads(args.decoding) if args.decoding else {}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    system_row_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {system: {} for system in systems}

    for system in systems:
        for task, notes, dataset, min_len, max_len in TABLE_SPECS:
            slug = task.lower().replace(" ", "_").replace("/", "_")
            raw_path = raw_dir / f"{system}__{slug}.json"
            print(
                f"[run] system={system} task={task} dataset={dataset} "
                f"chain=[{min_len if min_len is not None else '-'}, {max_len if max_len is not None else '-'}]"
            )
            metrics = _run_system_row(
                system=system,
                dataset=dataset,
                split=args.split,
                llm=args.llm,
                raw_path=raw_path,
                limit=args.limit,
                min_chain_len=min_len,
                max_chain_len=max_len,
                cot_temperature=args.cot_temperature,
                cot_max_new_tokens=args.cot_max_new_tokens,
                decoding=decoding,
                max_rounds=args.max_rounds,
                summarizer_llm=args.summarizer_llm,
                summarizer_temperature=args.summarizer_temperature,
                summarizer_max_tokens=args.summarizer_max_tokens,
            )
            system_row_metrics[system][task] = {
                "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
                "num_examples": int(metrics.get("num_examples", 0) or 0),
                "raw_metrics_file": str(raw_path),
                "notes": notes,
            }

    rows: List[Dict[str, Any]] = []
    for task, notes, _, _, _ in TABLE_SPECS:
        rows.append(
            {
                "task": task,
                "notes": notes,
                "scores": {system: float(system_row_metrics[system][task]["accuracy"]) for system in systems},
                "num_examples": {
                    system: int(system_row_metrics[system][task]["num_examples"]) for system in systems
                },
            }
        )

    all_row = {"task": "All", "notes": "Strong long-horizon EM", "scores": {}, "num_examples": {}}
    for system in systems:
        weighted = _weighted_accuracy(
            [
                {
                    "accuracy": system_row_metrics[system][task]["accuracy"],
                    "num_examples": system_row_metrics[system][task]["num_examples"],
                }
                for task, _, _, _, _ in TABLE_SPECS
            ]
        )
        all_row["scores"][system] = weighted
        all_row["num_examples"][system] = sum(
            int(system_row_metrics[system][task]["num_examples"]) for task, _, _, _, _ in TABLE_SPECS
        )
    rows.append(all_row)

    payload = {
        "config": {
            "llm": args.llm,
            "systems": systems,
            "split": args.split,
            "limit": args.limit,
            "decoding": decoding,
            "cot_temperature": args.cot_temperature,
            "cot_max_new_tokens": args.cot_max_new_tokens,
            "max_rounds": args.max_rounds,
            "summarizer_llm": args.summarizer_llm,
            "summarizer_temperature": args.summarizer_temperature,
            "summarizer_max_tokens": args.summarizer_max_tokens,
            "cuda_visible_devices": args.cuda_visible_devices,
        },
        "rows": rows,
        "by_system_and_task": system_row_metrics,
    }

    json_path = args.output_dir / "table6_baselines.json"
    md_path = args.output_dir / "table6_baselines.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_format_markdown(rows, systems), encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
