"""Centralised configuration helpers sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _model_id_from_hf_cache_path(path_value: Optional[str]) -> Optional[str]:
    """Infer a Hugging Face model id from a cache path segment like models--org--name."""

    if not path_value:
        return None
    norm = os.path.expanduser(path_value)
    parts = [p for p in norm.split(os.sep) if p]
    for part in reversed(parts):
        if not part.startswith("models--"):
            continue
        chunks = part.split("--", 2)
        if len(chunks) < 3:
            continue
        org = chunks[1].strip()
        name = chunks[2].strip()
        if org and name:
            return f"{org}/{name}"
    return None


def _resolve_model_identifier(name_env: str, path_env: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve model identifier from explicit env name or HF-cache-style path env."""

    explicit = os.getenv(name_env)
    if explicit:
        return explicit
    derived = _model_id_from_hf_cache_path(os.getenv(path_env))
    if derived:
        return derived
    return default


@dataclass(frozen=True)
class ProviderSettings:
    """API credentials for external LLM providers."""

    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    mistral_api_token: Optional[str] = os.getenv("MISTRAL_API_TOKEN")
    huggingface_api_token: Optional[str] = os.getenv("HF_API_TOKEN")
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_site_url: Optional[str] = os.getenv("OPENROUTER_SITE_URL") or os.getenv("OPENROUTER_SITE")
    openrouter_app_name: Optional[str] = os.getenv("OPENROUTER_APP_NAME") or os.getenv("OPENROUTER_SITE_NAME")


@dataclass(frozen=True)
class ModelSettings:
    """Model choices for primary reasoning and visual agents."""

    primary_model_name: str = _resolve_model_identifier(
        "PRIMARY_MODEL_NAME",
        "PRIMARY_MODEL_PATH",
        "mistral-7b",
    ) or "mistral-7b"
    legacy_visual_model_name: Optional[str] = _resolve_model_identifier(
        "VISUAL_MODEL_NAME",
        "VISUAL_MODEL_PATH",
    )
    visual_caption_model: str = (
        _resolve_model_identifier("VISUAL_CAPTION_MODEL", "VISUAL_CAPTION_MODEL_PATH")
        or _resolve_model_identifier("VISUAL_MODEL_NAME", "VISUAL_MODEL_PATH", "blip-2")
        or "blip-2"
    )
    visual_caption_model_path: Optional[str] = os.getenv("VISUAL_CAPTION_MODEL_PATH")
    visual_ocr_engine: str = os.getenv("VISUAL_OCR_ENGINE", "PaddleOCR")
    visual_ocr_model_dir: Optional[str] = os.getenv("VISUAL_OCR_MODEL_DIR")
    dealog_summarizer_model: Optional[str] = _resolve_model_identifier(
        "DEALOG_SUMMARIZER_MODEL",
        "DEALOG_SUMMARIZER_MODEL_PATH",
    )


@dataclass(frozen=True)
class RuntimeSettings:
    """Miscellaneous runtime parameters for reproducibility."""

    data_dir: str = os.getenv("DATA_DIR", "./data")
    model_cache: str = os.getenv("MODEL_CACHE", "./models")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


provider_settings = ProviderSettings()
model_settings = ModelSettings()
runtime_settings = RuntimeSettings()


def get_primary_model(override: Optional[str] = None) -> str:
    """Return the requested primary LLM name."""

    return override or model_settings.primary_model_name


def get_visual_model(override: Optional[str] = None) -> str:
    """Return the requested visual model name."""

    return (
        override
        or model_settings.visual_caption_model
        or model_settings.legacy_visual_model_name
        or model_settings.primary_model_name
    )


def get_visual_caption_model(override: Optional[str] = None) -> str:
    """Return the configured visual captioning model identifier."""

    return override or model_settings.visual_caption_model


def get_visual_caption_model_path() -> Optional[str]:
    """Return the on-disk path to the captioning model if specified."""

    return model_settings.visual_caption_model_path


def get_visual_ocr_engine(override: Optional[str] = None) -> str:
    """Return the OCR engine identifier."""

    return override or model_settings.visual_ocr_engine


def get_visual_ocr_model_dir() -> Optional[str]:
    """Return the OCR model directory if present."""

    return model_settings.visual_ocr_model_dir


def get_dealog_summarizer_model(override: Optional[str] = None) -> Optional[str]:
    """Return the configured summarizer model for DeALoG."""

    return override or model_settings.dealog_summarizer_model
