"""Agent responsible for ingesting visual artefacts such as charts or images."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional


class VisualAgent:
    """Produce lightweight textual descriptions from visual inputs."""

    def __init__(
        self,
        model: Optional[Any] = None,
        *,
        caption_model_name: Optional[str] = None,
        caption_model_path: Optional[str] = None,
        ocr_engine: Optional[str] = None,
        ocr_model_dir: Optional[str] = None,
    ) -> None:  # pragma: no cover - trivial
        self.model = model
        self.caption_model_name = caption_model_name or (model if isinstance(model, str) else None)
        self.caption_model_path = caption_model_path
        self.ocr_engine = ocr_engine
        self.ocr_model_dir = ocr_model_dir
        if self.ocr_model_dir:
            abs_dir = os.path.abspath(self.ocr_model_dir)
            os.environ.setdefault("PADDLE_PDX_CACHE_HOME", abs_dir)
            os.environ.setdefault("PADDLEOCR_HOME", abs_dir)

    def describe(self, visuals: Optional[Iterable[Dict[str, Any]]]) -> str:
        """Return a textual description of ``visuals``.

        ``visuals`` is expected to be an iterable of dictionaries describing
        figures or images.  The repository's synthetic dataset does not provide
        real visual assets, so this implementation falls back to metadata-based
        descriptions when possible.
        """

        if not visuals:
            return self._format_response("No visual artefacts provided.", include_models=True)

        summaries = []
        for item in visuals:
            title = item.get("title") if isinstance(item, dict) else None
            caption = item.get("caption") if isinstance(item, dict) else None
            if title and caption:
                summaries.append(f"{title}: {caption}")
            elif title:
                summaries.append(f"{title} (no caption available)")
            elif caption:
                summaries.append(caption)

        if not summaries:
            return self._format_response("Visual metadata present but could not be interpreted.", include_models=True)

        return self._format_response(" \n".join(summaries), include_models=True)

    def _format_response(self, content: str, *, include_models: bool = False) -> str:
        """Append model metadata if requested."""

        if not include_models:
            return content

        meta_bits = []
        if self.caption_model_name:
            suffix = f" ({self.caption_model_path})" if self.caption_model_path else ""
            meta_bits.append(f"caption={self.caption_model_name}{suffix}")
        if self.ocr_engine:
            suffix = f" ({self.ocr_model_dir})" if self.ocr_model_dir else ""
            meta_bits.append(f"ocr={self.ocr_engine}{suffix}")
        if not meta_bits:
            return content
        return f"{content} \n[visual models: {', '.join(meta_bits)}]"
