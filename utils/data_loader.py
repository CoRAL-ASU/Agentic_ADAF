"""Utility functions for loading benchmark datasets."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _load_tatqa(split: str = "dev") -> List[Dict[str, Any]]:
    """Return a list of QA examples from the TAT-QA dataset.

    Each example in the returned list is a dictionary with at least the
    following keys:

    ``question`` -- question string
    ``answer`` -- the annotated answer
    ``table`` -- table rows as loaded from the dataset
    ``paragraphs`` -- list of associated context paragraphs
    """

    base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "TATQA")
    file_map = {
        "train": "tatqa_dataset_train.json",
        "dev": "tatqa_dataset_dev.json",
        "test": "tatqa_dataset_test.json",
        "test_gold": "tatqa_dataset_test_gold.json",
    }

    if split not in file_map:
        raise ValueError(f"Unknown TAT-QA split: {split}")

    path = os.path.join(base_dir, file_map[split])
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed: List[Dict[str, Any]] = []
    for entry in raw_data:
        table = entry.get("table", {}).get("table")
        paragraphs = [p.get("text", "") for p in entry.get("paragraphs", [])]
        for q in entry.get("questions", []):
            processed.append(
                {
                    "question": q.get("question", ""),
                    "answer": q.get("answer"),
                    "table": table,
                    "paragraphs": paragraphs,
                }
            )

    return processed


def load_benchmark(name: str, *, split: str = "dev") -> List[Dict[str, Any]]:
    """Load a dataset by name.

    Only the ``tatqa`` benchmark is supported in this example repository.

    Parameters
    ----------
    name:
        The dataset name.  Currently ``"tatqa"`` is the only valid value.
    split:
        Which dataset split to load.  One of ``"train"``, ``"dev"``,
        ``"test"``, or ``"test_gold"``.
    """

    if name.lower() == "tatqa":
        return _load_tatqa(split)

    raise ValueError(f"Unsupported dataset: {name}")
