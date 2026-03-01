"""Continuous, axis-based seasonal color classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.color_features import ColorFeatures
from src.season_index import SEASONS


@dataclass
class SeasonDecision:
    season: str
    confidence: float
    season_scores: dict[str, float]
    season_probabilities: dict[str, float]
    temp_score: float
    value_score: float
    chroma_score: float
    contrast_score: float
    definition_score: float


def _clip_axis(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _axis_temp_score(skin: ColorFeatures) -> float:
    # Requested formulation:
    # hue = atan2(b, a)
    # cool_score = sigmoid(-b + k1 * a)
    # warm_score = sigmoid( b - k2 * a)
    # axis score in [-1, 1] from warm-cool difference.
    _hue = float(np.arctan2(skin.b, skin.a))
    k1 = 0.20
    k2 = 0.20
    scale = 0.20
    cool_score = _sigmoid(scale * (-skin.b + k1 * skin.a))
    warm_score = _sigmoid(scale * (skin.b - k2 * skin.a))
    return _clip_axis(warm_score - cool_score)


def _axis_value_score(skin: ColorFeatures, hair: Optional[ColorFeatures]) -> float:
    # Lightness from skin L* plus relative skin-vs-hair lightness if available.
    skin_light = np.tanh((skin.l - 58.0) / 12.0)
    if hair is None:
        return _clip_axis(float(skin_light))
    rel = skin.l - hair.l  # larger => hair darker than skin => deeper
    rel_light = -np.tanh(rel / 18.0)
    return _clip_axis(float(0.7 * skin_light + 0.3 * rel_light))


def _axis_chroma_score(skin: ColorFeatures, skin_chroma_variance: float) -> float:
    # Higher skin chroma => brighter; higher variance penalizes clarity.
    chroma_base = np.tanh((skin.chroma - 18.0) / 8.0)
    chroma_std = np.sqrt(max(0.0, skin_chroma_variance))
    var_penalty = np.tanh((chroma_std - 6.0) / 4.0)
    return _clip_axis(float(0.8 * chroma_base - 0.2 * var_penalty))


def _axis_contrast_score(delta_l_hair_skin: Optional[float], delta_l_iris_skin: Optional[float]) -> float:
    # Contrast from hair/iris vs skin lightness separation.
    if delta_l_hair_skin is None and delta_l_iris_skin is None:
        return 0.0
    if delta_l_hair_skin is None:
        c_raw = float(delta_l_iris_skin)
    elif delta_l_iris_skin is None:
        c_raw = float(delta_l_hair_skin)
    else:
        c_raw = 0.6 * float(delta_l_hair_skin) + 0.4 * float(delta_l_iris_skin)
    return _clip_axis(float(np.tanh((c_raw - 16.0) / 8.0)))


def _season_compatibility(
    axes: dict[str, float],
    target: dict[str, float],
    weights: dict[str, float],
) -> float:
    # 1.0 is perfect match, 0.0 is opposite-end mismatch.
    score = 0.0
    for axis in ("temp", "value", "chroma", "contrast"):
        score += weights[axis] * (1.0 - 0.5 * abs(axes[axis] - target[axis]))
    return float(score)


def classify_season(
    skin_features: ColorFeatures,
    hair_features: Optional[ColorFeatures] = None,
    iris_features: Optional[ColorFeatures] = None,
    delta_l_hair_skin: Optional[float] = None,
    delta_l_iris_skin: Optional[float] = None,
    skin_chroma_variance: float = 0.0,
    definition_score: float = 0.0,
) -> SeasonDecision:
    """Classify seasons using continuous axis scoring and softmax-margin confidence."""
    _ = iris_features  # reserved for future direct axis terms

    temp_score = _axis_temp_score(skin_features)
    value_score = _axis_value_score(skin_features, hair_features)
    chroma_score = _axis_chroma_score(skin_features, skin_chroma_variance)
    contrast_score = _axis_contrast_score(delta_l_hair_skin, delta_l_iris_skin)
    definition_score = _clip_axis(definition_score)

    axes = {
        "temp": temp_score,
        "value": value_score,
        "chroma": chroma_score,
        "contrast": contrast_score,
    }
    weights = {"temp": 0.35, "value": 0.25, "chroma": 0.25, "contrast": 0.15}
    targets = {
        "Spring": {"temp": 0.95, "value": 0.75, "chroma": 0.85, "contrast": 0.45},
        "Summer": {"temp": -0.95, "value": 0.55, "chroma": -0.75, "contrast": -0.25},
        "Autumn": {"temp": 0.95, "value": -0.35, "chroma": -0.65, "contrast": 0.15},
        "Winter": {"temp": -0.95, "value": -0.35, "chroma": 0.85, "contrast": 0.90},
    }

    season_scores = {
        season: _season_compatibility(axes, target, weights) for season, target in targets.items()
    }

    # Edge-definition adjustment: subtle boosts on cool seasons.
    if definition_score > 0.0 and contrast_score > 0.0:
        season_scores["Winter"] += 0.08 * definition_score * contrast_score
    if definition_score < 0.0 and contrast_score < 0.0:
        season_scores["Summer"] += 0.08 * abs(definition_score) * abs(contrast_score)

    ordered = sorted(season_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_season, _ = ordered[0]

    # Confidence is softmax(top1) - softmax(top2), without clamping.
    raw = np.array([season_scores[s] for s in SEASONS], dtype=np.float64)
    shifted = raw - np.max(raw)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals)
    prob_map = {season: float(probs[i]) for i, season in enumerate(SEASONS)}
    sorted_probs = sorted(prob_map.values(), reverse=True)
    confidence = float(sorted_probs[0] - sorted_probs[1])

    return SeasonDecision(
        season=top_season,
        confidence=confidence,
        season_scores=season_scores,
        season_probabilities=prob_map,
        temp_score=temp_score,
        value_score=value_score,
        chroma_score=chroma_score,
        contrast_score=contrast_score,
        definition_score=definition_score,
    )
