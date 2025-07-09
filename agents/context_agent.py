"""Agent for retrieving textual context snippets."""

from __future__ import annotations

from typing import Iterable, Optional


class ContextAgent:
    """Simple lexical search over a provided corpus."""

    def __init__(self, model: Optional[object] = None) -> None:  # pragma: no cover - trivial
        self.model = model
    def fetch_relevant_text(self, question: str, paragraphs: Optional[Iterable[str]] = None) -> str:
        """Return the most relevant paragraph for ``question``.

        Relevance is determined by simple token overlap between the question and
        each paragraph.  If no paragraphs are provided, an empty string is
        returned.
        """

        if not paragraphs:
            return ""

        q_tokens = set(question.lower().split())
        best_para = ""
        best_score = 0
        for para in paragraphs:
            tokens = set(para.lower().split())
            score = len(q_tokens & tokens)
            if score > best_score:
                best_score = score
                best_para = para
        return best_para
