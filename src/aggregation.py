"""Aggregation and formatting helpers used by the app."""

from __future__ import annotations

import numpy as np

from src.color_features import ColorFeatures
from src.season_index import SEASONS


def aggregate_features(feature_list: list[ColorFeatures]) -> ColorFeatures:
    weights = np.array([f.sample_count for f in feature_list], dtype=np.float32)
    l_vals = np.array([f.l for f in feature_list], dtype=np.float32)
    a_vals = np.array([f.a for f in feature_list], dtype=np.float32)
    b_vals = np.array([f.b for f in feature_list], dtype=np.float32)
    l_avg = float(np.average(l_vals, weights=weights))
    a_avg = float(np.average(a_vals, weights=weights))
    b_avg = float(np.average(b_vals, weights=weights))
    return ColorFeatures(
        l=l_avg,
        a=a_avg,
        b=b_avg,
        chroma=float(np.sqrt(a_avg**2 + b_avg**2)),
        sample_count=int(np.sum(weights)),
    )


def weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) != len(weights):
        raise ValueError("values and weights must have same length")
    arr = np.array(values, dtype=np.float64)
    w = np.maximum(np.array(weights, dtype=np.float64), 0.0)
    if float(np.sum(w)) <= 0.0:
        return float(np.median(arr))
    order = np.argsort(arr)
    sorted_vals = arr[order]
    sorted_w = w[order]
    cutoff = 0.5 * float(np.sum(sorted_w))
    idx = int(np.searchsorted(np.cumsum(sorted_w), cutoff, side="left"))
    return float(sorted_vals[min(max(idx, 0), len(sorted_vals) - 1)])


def aggregate_season_scores(
    per_image_scores: list[dict[str, float]],
    per_image_weights: list[float] | None = None,
    use_weights: bool = False,
) -> dict[str, float]:
    if not per_image_scores:
        return {season: 0.0 for season in SEASONS}
    if use_weights and per_image_weights is not None and len(per_image_weights) == len(per_image_scores):
        return {
            season: weighted_median([scores[season] for scores in per_image_scores], per_image_weights)
            for season in SEASONS
        }
    return {
        season: float(np.median(np.array([scores[season] for scores in per_image_scores], dtype=np.float64)))
        for season in SEASONS
    }


def format_feature(label: str, feat: ColorFeatures | None) -> str:
    if feat is None:
        return f"- {label}: n/a"
    return (
        f"- {label}: L*={feat.l:.2f}, a*={feat.a:.2f}, b*={feat.b:.2f}, "
        f"chroma={feat.chroma:.2f} (n={feat.sample_count})"
    )


def format_delta(label: str, value: float | None) -> str:
    if value is None:
        return f"- {label}: n/a"
    return f"- {label}: {value:.2f}"


def hue_proxy(feat: ColorFeatures | None) -> float | None:
    if feat is None:
        return None
    return float(np.arctan2(feat.b, feat.a))


def mutedness_proxy(feat: ColorFeatures | None) -> float | None:
    if feat is None:
        return None
    return float(feat.chroma / (feat.l + 1e-6))


def temp_band_from_b(b_value: float) -> str:
    if b_value < 10.0:
        return "cool"
    if b_value > 18.0:
        return "warm"
    return "neutral"


def palette_season_from_variant(base_season: str, variant_key: str, palette_code: str) -> str:
    code_to_base = {"A": "Autumn", "B": "Summer", "C": "Winter", "D": "Spring"}
    code_base = code_to_base.get(palette_code)
    variant_lower = variant_key.lower()
    for season in ("winter", "spring", "summer", "autumn"):
        if season in variant_lower:
            return season.capitalize()
    if base_season in SEASONS:
        return base_season
    return code_base or base_season


def normalize_axis(raw: float, axis_name: str) -> float:
    constants = {
        "temperature": (0.0, 0.75),
        "value": (0.0, 0.75),
        "chroma": (0.0, 0.75),
        "contrast": (0.0, 0.75),
    }
    if -1.0 <= raw <= 1.0:
        return float(raw)
    mean, std = constants.get(axis_name, (0.0, 1.0))
    return float(np.clip((float(raw) - mean) / max(std, 1e-6), -2.0, 2.0) / 2.0)


def season_accent_color(season: str) -> str:
    palette = {
        "Spring": "#f39c12",
        "Summer": "#4a90c2",
        "Autumn": "#b5653c",
        "Winter": "#2f4f8f",
    }
    return palette.get(season, "#1f77b4")
