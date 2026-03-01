"""Variant selection within each base season."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VariantDecision:
    base_season: str
    variant: str
    palette_code: str
    variant_key: str


def choose_variant(
    base_season: str,
    temp_score: float,
    chroma_score: float,
    contrast_score: float,
) -> VariantDecision:
    season = str(base_season)

    if season == "Spring":
        if temp_score >= 0.30 and chroma_score >= 0.10:
            variant = "Warm Spring"
        else:
            variant = "Neutral Spring"
        code = "D"
    elif season == "Autumn":
        if temp_score >= 0.45 and chroma_score > -0.25:
            variant = "Warm Autumn"
        else:
            variant = "Soft Autumn"
        code = "A"
    elif season == "Summer":
        if chroma_score <= -0.35 or contrast_score <= 0.05:
            variant = "Soft Summer"
        else:
            variant = "Cool Summer"
        code = "B"
    elif season == "Winter":
        if contrast_score >= 0.45 and chroma_score >= 0.20:
            variant = "Bright Winter"
        else:
            variant = "Cool Winter"
        code = "C"
    else:
        variant = season
        code = "?"

    return VariantDecision(
        base_season=season,
        variant=variant,
        palette_code=code,
        variant_key=f"{variant} ({code})",
    )

