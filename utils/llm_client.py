"""LLM client supporting OpenRouter-compatible APIs and local Transformers models."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


class LLMClient:
    """Best-effort wrapper around hosted and local chat/completions backends."""

    _local_model_cache: ClassVar[Dict[str, Tuple[Any, Any]]] = {}

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.5,
        backend: Optional[str] = None,
        local_model_path: Optional[str] = None,
        local_files_only: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        self.default_model = default_model or os.getenv("DEALOG_SUMMARIZER_MODEL") or os.getenv("PRIMARY_MODEL_NAME")
        self.site_url = os.getenv("OPENROUTER_SITE_URL") or os.getenv("OPENROUTER_SITE")
        self.app_name = os.getenv("OPENROUTER_APP_NAME") or os.getenv("OPENROUTER_SITE_NAME")
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_sec = float(retry_backoff_sec)
        self.backend = (backend or os.getenv("DEALOG_LLM_BACKEND") or "auto").strip().lower()
        self.local_model_path = local_model_path or os.getenv("LOCAL_LLM_MODEL_PATH")
        self.local_files_only = local_files_only

    def _fail_or_none(self, message: str) -> Optional[Dict[str, Any]]:
        raise RuntimeError(message)

    def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> Optional[Dict[str, Any]]:
        """
        Send ``prompt`` to the configured backend.

        Returns
        -------
        dict | None
            ``{"content": str, "usage": {"prompt_tokens": int, "completion_tokens": int}}``.
            Falls back to ``None`` when the client is unavailable or errors.
        """

        selected_backend = self._select_backend(model)
        if selected_backend == "local":
            return self._complete_local(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._complete_openrouter(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _select_backend(self, model: Optional[str]) -> str:
        if self.backend in {"openrouter", "local"}:
            return self.backend
        if self.api_key:
            return "openrouter"
        if self._resolve_local_model_source(model) is not None:
            return "local"
        return "openrouter"

    def _complete_openrouter(
        self,
        prompt: str,
        *,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return self._fail_or_none(
                "OPENROUTER_API_KEY is missing and no local model backend was resolved."
            )

        try:
            import requests  # type: ignore
        except ImportError:
            return self._fail_or_none("Missing dependency: requests")

        payload = {
            "model": model or self.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    return self._fail_or_none(
                        f"LLM request failed after {self.max_retries} attempts: {exc}"
                    )
                time.sleep(self.retry_backoff_sec * attempt)
        if last_exc is not None and "response" not in locals():
            return self._fail_or_none(f"LLM request failed: {last_exc}")

        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")
            usage = data.get("usage") or {}
            if content in (None, ""):
                return self._fail_or_none("LLM response did not include content.")
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                },
            }
        except Exception as exc:
            return self._fail_or_none(f"Failed to parse LLM response payload: {exc}")

    def _complete_local(
        self,
        prompt: str,
        *,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        model_source = self._resolve_local_model_source(model)
        if not model_source:
            return self._fail_or_none(
                "Local backend selected but no model source resolved. "
                "Set PRIMARY_MODEL_PATH / DEALOG_SUMMARIZER_MODEL_PATH / LOCAL_LLM_MODEL_PATH."
            )

        try:
            import torch
        except ImportError:
            return self._fail_or_none("Missing dependency: torch")

        tokenizer, local_model = self._load_local_model(model_source)

        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            device = next(local_model.parameters()).device
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        except Exception:
            pass

        prompt_tokens = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else None
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_tokens),
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        }
        if tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if temperature and temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": float(temperature)})
        else:
            generation_kwargs.update({"do_sample": False})

        try:
            with torch.no_grad():
                output_ids = local_model.generate(**inputs, **generation_kwargs)
        except Exception as exc:
            return self._fail_or_none(f"Local generation failed for model '{model_source}': {exc}")

        output_sequence = output_ids[0]
        if prompt_tokens is not None and prompt_tokens < output_sequence.shape[-1]:
            generated_ids = output_sequence[prompt_tokens:]
        else:
            generated_ids = output_sequence

        content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not content:
            full_text = tokenizer.decode(output_sequence, skip_special_tokens=True)
            if full_text.startswith(prompt):
                content = full_text[len(prompt) :].strip()
            else:
                content = full_text.strip()

        if not content:
            return self._fail_or_none("Local model generation returned empty content.")

        completion_tokens = int(generated_ids.shape[-1]) if hasattr(generated_ids, "shape") else None
        return {
            "content": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }

    def _load_local_model(self, model_source: str) -> Tuple[Any, Any]:
        cache_key = str(model_source)
        if cache_key in self._local_model_cache:
            return self._local_model_cache[cache_key]

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            self._fail_or_none("Missing dependency: transformers")

        model_kwargs: Dict[str, Any] = {
            "local_files_only": self.local_files_only,
            "trust_remote_code": True,
            "torch_dtype": "auto",
        }
        device_map = os.getenv("LOCAL_LLM_DEVICE_MAP", "auto")
        if device_map:
            model_kwargs["device_map"] = device_map

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            )
            local_model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        except Exception as exc:
            self._fail_or_none(
                f"Failed to load local model from '{model_source}'. "
                f"Check model path/cache and permissions. Original error: {exc}"
            )

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        self._local_model_cache[cache_key] = (tokenizer, local_model)
        return tokenizer, local_model

    def _resolve_local_model_source(self, model: Optional[str]) -> Optional[str]:
        requested_model = model or self.default_model
        if requested_model:
            as_path = Path(os.path.expanduser(requested_model))
            if as_path.exists():
                return self._normalise_local_path(as_path)

        explicit_path = self.local_model_path
        if explicit_path:
            path = Path(os.path.expanduser(explicit_path))
            if path.exists():
                return self._normalise_local_path(path)

        env_paths = [
            os.getenv("DEALOG_SUMMARIZER_MODEL_PATH"),
            os.getenv("PRIMARY_MODEL_PATH"),
        ]
        for candidate in env_paths:
            if not candidate:
                continue
            path = Path(os.path.expanduser(candidate))
            if not path.exists():
                continue
            if not requested_model:
                return self._normalise_local_path(path)
            if self._model_name_matches_path(requested_model, str(path)):
                return self._normalise_local_path(path)

        if requested_model:
            return requested_model
        return None

    def _normalise_local_path(self, path: Path) -> str:
        snapshots_dir = path / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = sorted(
                (entry for entry in snapshots_dir.iterdir() if entry.is_dir()),
                key=lambda entry: entry.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return str(snapshots[0])
        return str(path)

    def _model_name_matches_path(self, model_name: str, path_value: str) -> bool:
        expected = model_name.strip().lower()
        path_expanded = os.path.expanduser(path_value)
        derived = self._model_id_from_hf_cache_path(path_expanded)
        if derived and derived.lower() == expected:
            return True
        path_name = Path(path_expanded).name.lower()
        simplified = expected.replace("/", "--")
        return simplified in path_name or expected in path_name

    def _model_id_from_hf_cache_path(self, path_value: str) -> Optional[str]:
        parts = [segment for segment in path_value.split(os.sep) if segment]
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
