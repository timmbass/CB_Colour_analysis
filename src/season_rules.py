"""Rule-based seasonal color classifier."""

from __future__ import annotations

from dataclasses import dataclass

from src.color_features import ColorFeatures


@dataclass
class SeasonDecision:
    season: str
    confidence: float
    undertone: str
    clarity: str
    depth: str


def classify_season(
    features: ColorFeatures,
    warm_b_threshold: float = 3.0,
    bright_chroma_threshold: float = 18.0,
    light_l_threshold: float = 58.0,
) -> SeasonDecision:
    """Classify into Spring/Summer/Autumn/Winter using simple, explainable rules."""
    is_warm = features.b >= warm_b_threshold
    is_bright = features.chroma >= bright_chroma_threshold
    is_light = features.l >= light_l_threshold

    undertone = "warm" if is_warm else "cool"
    clarity = "bright" if is_bright else "muted"
    depth = "light" if is_light else "deep"

    # Base mapping from undertone + clarity.
    if is_warm and is_bright:
        season = "Spring"
    elif is_warm and not is_bright:
        season = "Autumn"
    elif (not is_warm) and is_bright:
        season = "Winter"
    else:
        season = "Summer"

    # Depth is used as a tie-break strength and confidence adjustment.
    # If the depth strongly conflicts with canonical season depth, lower confidence.
    canonical_light = season in {"Spring", "Summer"}
    depth_alignment = (canonical_light and is_light) or ((not canonical_light) and (not is_light))

    dist_b = min(abs(features.b - warm_b_threshold) / 12.0, 1.0)
    dist_c = min(abs(features.chroma - bright_chroma_threshold) / 20.0, 1.0)
    dist_l = min(abs(features.l - light_l_threshold) / 20.0, 1.0)

    confidence = 0.5 + 0.49 * ((dist_b + dist_c + dist_l) / 3.0)
    if not depth_alignment:
        confidence -= 0.08
    confidence = max(0.5, min(0.99, confidence))

    return SeasonDecision(
        season=season,
        confidence=round(confidence, 3),
        undertone=undertone,
        clarity=clarity,
        depth=depth,
    )
