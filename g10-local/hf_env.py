"""Read HF_TOKEN from the capstone .env. The value is never printed."""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
CACHE_PATH = Path.home() / ".cache" / "huggingface" / "token"


def _from_file(path: Path) -> str:
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "HF_TOKEN":
            return value.strip().strip('"').strip("'")
    return ""


def hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip() or _from_file(ENV_PATH)
    if token:
        return token
    if CACHE_PATH.is_file():
        cached = CACHE_PATH.read_text(encoding="utf-8").strip()
        if cached:
            return cached
    raise SystemExit(
        "HF_TOKEN is empty. Put it in the .env file at the project root: HF_TOKEN=hf_..."
    )
