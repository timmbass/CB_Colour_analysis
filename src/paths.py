"""Repository-relative path helpers."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
ASSETS_MODELS_DIR = ASSETS_DIR / "models"
ASSETS_PALETTES_DIR = ASSETS_DIR / "palettes"
ASSETS_CONFIG_DIR = ASSETS_DIR / "config"
DOCS_DIR = ROOT_DIR / "docs"
TESTS_DIR = ROOT_DIR / "tests"
TOOLS_DIR = ROOT_DIR / "tools"

FACE_LANDMARKER_PATH = ASSETS_MODELS_DIR / "face_landmarker.task"
PALETTES_PATH = ASSETS_PALETTES_DIR / "seasonal_palettes.json"
RECOMMENDATIONS_PATH = ASSETS_PALETTES_DIR / "seasonal_recommendations.json"
SEASON_DESCRIPTIONS_PATH = ASSETS_PALETTES_DIR / "season_descriptions.json"
COPY_RULES_PATH = ASSETS_DIR / "copy_rules.v1.json"
DYNAMIC_COLORS_CONFIG_PATH = ROOT_DIR / "dynamic_colors_config.json"
CALIBRATION_PARAMS_PATH = ROOT_DIR / "calibration_params.json"
ANALYSIS_CONFIG_PATH = ASSETS_CONFIG_DIR / "analysis.v1.json"


def resolve_repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return ROOT_DIR / path_obj
