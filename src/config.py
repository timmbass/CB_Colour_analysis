"""Versioned application config loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.json_utils import load_json_object
from src.paths import ANALYSIS_CONFIG_PATH, CALIBRATION_PARAMS_PATH, DYNAMIC_COLORS_CONFIG_PATH, resolve_repo_path

logger = logging.getLogger(__name__)


DEFAULT_CALIBRATION_CONFIG: dict[str, Any] = {
    "alpha": 0.5,
    "bias": [0.0, 0.0, 0.0, 0.0],
    "gamma": 3.0,
    "source": "default",
}

DEFAULT_ANALYSIS_CONFIG: dict[str, Any] = {
    "image_quality": {
        "tiny_face_ratio": 0.08,
        "low_cheek_sample": 450,
        "overexposed_pct": 35.0,
        "underexposed_pct": 35.0,
        "blur_laplacian_var": 70.0,
        "strong_color_cast_pre_wb": 0.38,
        "score_weights": {
            "face": 0.22,
            "cheek": 0.20,
            "over": 0.14,
            "under": 0.14,
            "blur": 0.20,
            "cast": 0.10,
        },
        "score_ranges": {
            "face": [0.03, 0.17],
            "cheek": [300.0, 1200.0],
            "over": [8.0, 40.0],
            "under": [8.0, 40.0],
            "blur": [35.0, 220.0],
            "cast": [0.08, 0.35],
        },
    },
    "season_rules": {
        "temp": {"k1": 0.20, "k2": 0.20, "scale": 0.20},
        "value": {"skin_center": 58.0, "skin_scale": 12.0, "hair_delta_scale": 18.0},
        "chroma": {"center": 18.0, "scale": 8.0, "variance_center": 6.0, "variance_scale": 4.0},
        "contrast": {"center": 16.0, "scale": 8.0},
        "weights": {"temp": 0.35, "value": 0.25, "chroma": 0.25, "contrast": 0.15},
        "targets": {
            "Spring": {"temp": 0.95, "value": 0.75, "chroma": 0.85, "contrast": 0.45},
            "Summer": {"temp": -0.95, "value": 0.55, "chroma": -0.75, "contrast": -0.25},
            "Autumn": {"temp": 0.95, "value": -0.35, "chroma": -0.65, "contrast": 0.15},
            "Winter": {"temp": -0.95, "value": -0.35, "chroma": 0.85, "contrast": 0.90},
        },
        "definition_adjustments": {"winter_boost": 0.08, "summer_boost": 0.08},
    },
    "drape_scoring": {
        "weights": {
            "delta_l": 0.20,
            "delta_chroma": 0.15,
            "delta_hue": 0.15,
            "grey_cast": 0.10,
            "var_l": 0.12,
            "var_ab": 0.10,
            "edge": 0.08,
            "sat": 0.06,
            "clarity": 0.04,
        },
        "normalization": {
            "delta_l": 12.0,
            "delta_chroma": 10.0,
            "delta_hue": 0.9,
            "var_l": 20.0,
            "var_ab": 50.0,
            "edge_min": 20.0,
            "edge_ratio": 0.45,
            "sat": 8.0,
        },
        "season_saturation_bias": {
            "default": [0.5, 0.5],
            "Summer": [0.75, 0.25],
            "Winter": [0.35, 0.65],
        },
        "clarity": {
            "center": 18.0,
            "scale": 12.0,
            "Winter": [0.45, 0.30, 0.25],
            "Summer": [0.55, 0.25, 0.20],
        },
    },
}


@dataclass(frozen=True)
class CalibrationConfig:
    alpha: float
    bias: list[float]
    gamma: float
    source: str


def load_analysis_config(path: str | Path = ANALYSIS_CONFIG_PATH) -> dict[str, Any]:
    payload = load_json_object(path)
    for section in ("image_quality", "season_rules", "drape_scoring"):
        if section not in payload or not isinstance(payload[section], dict):
            raise ValueError(f"analysis config missing object section: {section}")
    return payload


def load_calibration_config(path: str | Path = CALIBRATION_PARAMS_PATH) -> CalibrationConfig:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        return CalibrationConfig(**DEFAULT_CALIBRATION_CONFIG)
    try:
        payload = load_json_object(resolved)
    except ValueError:
        logger.warning("Falling back to default calibration config", exc_info=True)
        return CalibrationConfig(**DEFAULT_CALIBRATION_CONFIG)

    bias = payload.get("bias", DEFAULT_CALIBRATION_CONFIG["bias"])
    if not isinstance(bias, list) or len(bias) != 4:
        bias = list(DEFAULT_CALIBRATION_CONFIG["bias"])
    return CalibrationConfig(
        alpha=float(payload.get("alpha", DEFAULT_CALIBRATION_CONFIG["alpha"])),
        bias=[float(item) for item in bias],
        gamma=float(payload.get("gamma", DEFAULT_CALIBRATION_CONFIG["gamma"])),
        source="file",
    )


def load_dynamic_color_config(path: str | Path = DYNAMIC_COLORS_CONFIG_PATH) -> dict[str, Any]:
    return load_json_object(path)
