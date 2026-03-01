"""Palette loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, TypedDict


class PaletteMeta(TypedDict, total=False):
    colors: List[str]
    description: str
    recommendations: str


class RecommendationBlock(TypedDict, total=False):
    title: str
    description: str
    best_colors: List[str]
    best_neutrals: List[str]
    metals: List[str]
    avoid: List[str]
    overall_effect: str
    style_note: str


class SeasonRecommendations(TypedDict, total=False):
    primary: RecommendationBlock
    strong: RecommendationBlock
    borderline: RecommendationBlock


class SeasonDescriptionBlock(TypedDict, total=False):
    title: str
    description: str
    best_colors: List[str]
    best_neutrals: List[str]
    metals: List[str]
    avoid: List[str]
    overall_effect: str
    style_note: str


def load_palettes(path: str | Path) -> Dict[str, List[str]]:
    palette_path = Path(path)
    with palette_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized: Dict[str, List[str]] = {}
    for season, colors in raw.items():
        if isinstance(colors, list):
            normalized[season] = [str(c) for c in colors]
        elif isinstance(colors, dict):
            palette = colors.get("colors")
            if isinstance(palette, list):
                normalized[season] = [str(c) for c in palette]

    return normalized


def palette_for_season(palettes: Dict[str, List[str]], season: str) -> List[str]:
    return palettes.get(season, [])


def load_palette_metadata(path: str | Path) -> Dict[str, PaletteMeta]:
    palette_path = Path(path)
    with palette_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized: Dict[str, PaletteMeta] = {}
    for season, payload in raw.items():
        entry: PaletteMeta = {}
        if isinstance(payload, list):
            entry["colors"] = [str(c) for c in payload]
        elif isinstance(payload, dict):
            palette = payload.get("colors")
            if isinstance(palette, list):
                entry["colors"] = [str(c) for c in palette]
            description = payload.get("description")
            if isinstance(description, str) and description.strip():
                entry["description"] = description
            recommendations = payload.get("recommendations")
            if isinstance(recommendations, str) and recommendations.strip():
                entry["recommendations"] = recommendations
        if entry:
            normalized[season] = entry

    return normalized


def load_recommendations(path: str | Path) -> Dict[str, SeasonRecommendations]:
    rec_path = Path(path)
    with rec_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized: Dict[str, SeasonRecommendations] = {}
    for season, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        entry: SeasonRecommendations = {}
        for key in ("primary", "strong", "borderline"):
            block = payload.get(key)
            if isinstance(block, dict):
                entry[key] = block
        if entry:
            normalized[season] = entry

    return normalized


def load_season_descriptions(path: str | Path) -> Dict[str, SeasonDescriptionBlock]:
    desc_path = Path(path)
    with desc_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized: Dict[str, SeasonDescriptionBlock] = {}
    if not isinstance(raw, dict):
        return normalized
    for key, payload in raw.items():
        if isinstance(payload, dict):
            normalized[str(key)] = payload
    return normalized
