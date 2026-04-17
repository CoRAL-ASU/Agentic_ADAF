"""Shared utilities package bootstrap."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_DEFAULT_REPO_TMP = "/mnt/achakr40"


def configure_temp_dir() -> str:
    """Pin Python temp files to the repo temp mount."""

    target = Path(os.getenv("AGENTIC_TMPDIR", _DEFAULT_REPO_TMP)).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    temp_root = str(target)
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = temp_root
    tempfile.tempdir = temp_root
    return temp_root


configure_temp_dir()
