"""Palette loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_palettes(path: str | Path) -> Dict[str, List[str]]:
    palette_path = Path(path)
    with palette_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized: Dict[str, List[str]] = {}
    for season, colors in raw.items():
        if isinstance(colors, list):
            normalized[season] = [str(c) for c in colors]

    return normalized


def palette_for_season(palettes: Dict[str, List[str]], season: str) -> List[str]:
    return palettes.get(season, [])
