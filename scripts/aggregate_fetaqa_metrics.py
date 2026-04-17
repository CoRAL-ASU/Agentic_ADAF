#!/usr/bin/env python3
"""Aggregate FeTaQA metrics (QAGS, BERTScore, Log-Groundedness, LLM raters) into one summary."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def _coerce_float(val: str) -> float:
    try:
        return float(val)
    except Exception:
        return float("nan")


def _read_qags(qags_dir: Path) -> Tuple[Dict[str, float], Set[str]]:
    """Read qags_scores.csv if present, otherwise summary.txt."""
    ids: Set[str] = set()
    info = {"mean_qags": float("nan"), "mean_qags_bs": float("nan"), "n": 0}
    scores_path = qags_dir / "qags_scores.csv"
    summary_path = qags_dir / "summary.txt"

    if scores_path.exists():
        qags_vals: List[float] = []
        qags_bs_vals: List[float] = []
        with scores_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.add(str(row.get("id", "")))
                qags_vals.append(_coerce_float(row.get("qags", "nan")))
                if "qags_bs" in row:
                    qags_bs_vals.append(_coerce_float(row.get("qags_bs", "nan")))
        clean_qags = [v for v in qags_vals if v == v]
        clean_qags_bs = [v for v in qags_bs_vals if v == v]
        if clean_qags:
            info["mean_qags"] = sum(clean_qags) / len(clean_qags)
            info["n"] = len(clean_qags)
        if clean_qags_bs:
            info["mean_qags_bs"] = sum(clean_qags_bs) / len(clean_qags_bs)
    elif summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            info["mean_qags"] = summary.get("QAGS_mean", float("nan"))
            info["mean_qags_bs"] = summary.get("QAGS_BS_mean", float("nan"))
            info["n"] = summary.get("N", 0)
        except Exception:
            pass
    return info, ids


def _read_groundedness(path: Path) -> Tuple[Dict[str, float], Set[str]]:
    ids: Set[str] = set()
    info = {"mean_log_groundedness": float("nan"), "perfect_fraction": float("nan"), "n": 0}
    if not path.exists():
        return info, ids
    data = json.loads(path.read_text())
    info["mean_log_groundedness"] = data.get("mean_log_groundedness", float("nan"))
    info["perfect_fraction"] = data.get("perfect_fraction", float("nan"))
    info["n"] = data.get("num_examples", 0)
    for ex in data.get("per_example", []):
        if "id" in ex:
            ids.add(str(ex["id"]))
    return info, ids


def _read_bertscore(path: Path) -> Tuple[Dict[str, float], Set[str]]:
    ids: Set[str] = set()
    info = {
        "mean_f1": float("nan"),
        "mean_precision": float("nan"),
        "mean_recall": float("nan"),
        "n": 0,
    }
    if not path.exists():
        return info, ids
    data = json.loads(path.read_text())
    info["mean_f1"] = data.get("mean_f1", float("nan"))
    info["mean_precision"] = data.get("mean_precision", float("nan"))
    info["mean_recall"] = data.get("mean_recall", float("nan"))
    info["n"] = data.get("num_examples", 0)
    for ex in data.get("per_example", []):
        if "id" in ex:
            ids.add(str(ex["id"]))
    return info, ids


def _read_human_eval(path: Path) -> Tuple[Dict[str, Dict[str, float]], Set[str]]:
    ids: Set[str] = set()
    model_counts: Dict[str, Counter] = defaultdict(Counter)
    if not path.exists():
        return {}, ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            model = rec.get("model", "unknown")
            label = (rec.get("label") or "unknown").strip() or "unknown"
            model_counts[model][label] += 1
            if rec.get("id"):
                ids.add(str(rec["id"]))
    fracs: Dict[str, Dict[str, float]] = {}
    for model, counter in model_counts.items():
        total = sum(counter.values()) or 1
        fracs[model] = {label: count / total for label, count in counter.items()}
    return fracs, ids


def aggregate(qags_dir: Path, grounded_path: Path, bertscore_path: Path, human_eval_path: Path, output_path: Path) -> Dict[str, object]:
    qags_info, qags_ids = _read_qags(qags_dir)
    grounded_info, grounded_ids = _read_groundedness(grounded_path)
    bert_info, bert_ids = _read_bertscore(bertscore_path)
    human_info, human_ids = _read_human_eval(human_eval_path)

    intersections = [s for s in [qags_ids, grounded_ids, bert_ids, human_ids] if s]
    common_ids = set.intersection(*intersections) if intersections else set()

    summary = {
        "qags": qags_info,
        "bertscore": bert_info,
        "log_groundedness": grounded_info,
        "human_eval": human_info,
        "id_alignment": {
            "qags_ids": len(qags_ids),
            "bertscore_ids": len(bert_ids),
            "log_groundedness_ids": len(grounded_ids),
            "human_eval_ids": len(human_ids),
            "common_ids_all": len(common_ids),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate FeTaQA metrics into one JSON.")
    ap.add_argument("--qags-dir", type=Path, default=Path("benchmarks/results/qags_run"))
    ap.add_argument("--groundedness", type=Path, default=Path("benchmarks/results/fetaqa_dev_grounded.json"))
    ap.add_argument("--bertscore", type=Path, default=Path("benchmarks/results/fetaqa_dev_bertscore.json"))
    ap.add_argument("--human-eval", type=Path, default=Path("benchmarks/results/human_eval/llm_raters.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("benchmarks/results/fetaqa_combined_metrics.json"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate(
        qags_dir=args.qags_dir,
        grounded_path=args.groundedness,
        bertscore_path=args.bertscore,
        human_eval_path=args.human_eval,
        output_path=args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
