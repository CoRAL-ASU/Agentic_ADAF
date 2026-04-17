#!/usr/bin/env python3
"""Summarize E5 scheduler results: turns saved, accuracy hold, token stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List


def _p90(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(0.9 * (len(values) - 1))
    return float(values[idx])


def _load(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    return data


def _turns(stats: Dict[str, Any]) -> List[int]:
    turns = []
    for ex in stats.get("per_example", []):
        log = ex.get("log") or []
        turns.append(len(log))
    return turns


def _summary_tokens(stats: Dict[str, Any]) -> List[int]:
    tokens = []
    for ex in stats.get("per_example", []):
        log = ex.get("log") or []
        for entry in log:
            if entry.get("type") == "summary":
                meta = entry.get("metadata") or {}
                for key in ("prompt_tokens", "completion_tokens"):
                    val = meta.get(key)
                    if isinstance(val, int):
                        tokens.append(val)
    return tokens


def summarize(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    base_turns = _turns(baseline)
    gate_turns = _turns(candidate)
    base_tokens = _summary_tokens(baseline)
    gate_tokens = _summary_tokens(candidate)

    base_turn_avg = sum(base_turns) / len(base_turns) if base_turns else 0.0
    gate_turn_avg = sum(gate_turns) / len(gate_turns) if gate_turns else 0.0
    turns_saved_pct = (
        (1 - gate_turn_avg / base_turn_avg) * 100 if base_turn_avg else 0.0
    )

    lines = [
        f"Dataset: {candidate.get('dataset')} split={candidate.get('split')} model={candidate.get('model')}",
        f"Accuracy baseline={baseline.get('accuracy'):.3f} gated={candidate.get('accuracy'):.3f} (hold Δ={candidate.get('accuracy') - baseline.get('accuracy'):.3f})",
        f"Turns (mean/median/P90) baseline={base_turn_avg:.2f}/{median(base_turns) if base_turns else 0}/{_p90(base_turns):.0f} "
        f"gated={gate_turn_avg:.2f}/{median(gate_turns) if gate_turns else 0}/{_p90(gate_turns):.0f}; % saved={turns_saved_pct:.1f}%",
        f"Summary tokens (median/P90) baseline={median(base_tokens) if base_tokens else 0}/{_p90(base_tokens)} "
        f"gated={median(gate_tokens) if gate_tokens else 0}/{_p90(gate_tokens)}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize E5 gated scheduler results.")
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline sequential JSON metrics.")
    parser.add_argument("--candidate", required=True, type=Path, help="Gated/parallel JSON metrics.")
    args = parser.parse_args()

    base = _load(args.baseline)
    cand = _load(args.candidate)
    print(summarize(base, cand))


if __name__ == "__main__":
    main()
