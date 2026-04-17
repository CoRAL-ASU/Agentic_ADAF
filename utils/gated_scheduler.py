"""Lightweight wrapper around a binary logistic classifier for scheduling decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _load_model(model_path: Path):
    try:
        import joblib  # type: ignore
    except ImportError:
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


class GatedScheduler:
    """Binary logistic gate to decide whether to continue another round."""

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.4) -> None:
        self.threshold = threshold
        self._model = None
        if model_path:
            self._model = _load_model(Path(model_path))

    def score(self, features) -> float:
        """Return p(continue). If no model, return 1.0 (always continue)."""
        if self._model is None:
            return 1.0
        try:
            prob = self._model.predict_proba([features])[0][1]
            return float(prob)
        except Exception:
            return 1.0

    def should_continue(self, features) -> bool:
        return self.score(features) >= self.threshold
