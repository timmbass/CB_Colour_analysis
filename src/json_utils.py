"""Shared JSON loading helpers."""

from __future__ import annotations

import json
import logging
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from src.paths import resolve_repo_path

logger = logging.getLogger(__name__)


def load_json_file(path: str | Path) -> Any:
    resolved = resolve_repo_path(path)
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        logger.error("JSON resource missing: %s", resolved)
        raise FileNotFoundError(f"JSON resource not found: {resolved}") from exc
    except JSONDecodeError as exc:
        logger.error("Malformed JSON resource: %s", resolved)
        raise ValueError(f"Malformed JSON resource: {resolved}") from exc


def load_json_object(path: str | Path, *, required_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    payload = load_json_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {resolve_repo_path(path)}")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"JSON resource {resolve_repo_path(path)} missing keys: {', '.join(missing)}")
    return payload
