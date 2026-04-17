#!/usr/bin/env python3
"""ReWOO-style baseline driver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.runner import run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple ReWOO baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoding", default="{}")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-chain-len", type=int, default=None)
    parser.add_argument("--max-chain-len", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decoding = json.loads(args.decoding) if args.decoding else {}
    run_baseline(
        "rewoo",
        dataset=args.dataset,
        split=args.split,
        model=args.model,
        output=args.output,
        limit=args.limit,
        min_chain_len=args.min_chain_len,
        max_chain_len=args.max_chain_len,
        decoding=decoding,
    )
    print(f"[ReWOO] results saved to {args.output}")


if __name__ == "__main__":
    main()
