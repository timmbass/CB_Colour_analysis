"""Palette loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, cast

from src.json_utils import load_json_file
from src.paths import resolve_repo_path


class PaletteMeta(TypedDict, total=False):
    colors: list[str]
    description: str
    recommendations: str


class RecommendationBlock(TypedDict, total=False):
    title: str
    description: str
    best_colors: list[str]
    best_neutrals: list[str]
    metals: list[str]
    avoid: list[str]
    overall_effect: str
    style_note: str


class SeasonRecommendations(TypedDict, total=False):
    primary: RecommendationBlock
    strong: RecommendationBlock
    borderline: RecommendationBlock


class SeasonDescriptionBlock(TypedDict, total=False):
    title: str
    description: str
    best_colors: list[str]
    best_neutrals: list[str]
    metals: list[str]
    avoid: list[str]
    overall_effect: str
    style_note: str


def load_palettes(path: str | Path) -> dict[str, list[str]]:
    raw = load_json_file(resolve_repo_path(path))

    normalized: dict[str, list[str]] = {}
    for season, colors in raw.items():
        if isinstance(colors, list):
            normalized[season] = [str(c) for c in colors]
        elif isinstance(colors, dict):
            palette = colors.get("colors")
            if isinstance(palette, list):
                normalized[season] = [str(c) for c in palette]

    return normalized


def palette_for_season(palettes: dict[str, list[str]], season: str) -> list[str]:
    return palettes.get(season, [])


def load_palette_metadata(path: str | Path) -> dict[str, PaletteMeta]:
    raw = load_json_file(resolve_repo_path(path))

    normalized: dict[str, PaletteMeta] = {}
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


def load_recommendations(path: str | Path) -> dict[str, SeasonRecommendations]:
    raw = load_json_file(resolve_repo_path(path))

    normalized: dict[str, SeasonRecommendations] = {}
    for season, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        entry: SeasonRecommendations = {}
        for key in ("primary", "strong", "borderline"):
            block = payload.get(key)
            if isinstance(block, dict):
                entry[key] = cast(RecommendationBlock, block)
        if entry:
            normalized[season] = entry

    return normalized


def load_season_descriptions(path: str | Path) -> dict[str, SeasonDescriptionBlock]:
    raw = load_json_file(resolve_repo_path(path))

    normalized: dict[str, SeasonDescriptionBlock] = {}
    if not isinstance(raw, dict):
        return normalized
    for key, payload in raw.items():
        if isinstance(payload, dict):
            normalized[str(key)] = cast(SeasonDescriptionBlock, payload)
    return normalized
