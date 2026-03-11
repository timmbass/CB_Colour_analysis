"""Small CLI smoke test for repository setup."""

from __future__ import annotations

from src.config import load_analysis_config, load_calibration_config, load_dynamic_color_config
from src.copy_renderer import load_copy_rules
from src.dynamic_colors import suggest_dynamic_colors
from src.palettes import load_palette_metadata, load_palettes
from src.paths import COPY_RULES_PATH, FACE_LANDMARKER_PATH, PALETTES_PATH


def main() -> int:
    load_analysis_config()
    load_calibration_config()
    dynamic_cfg = load_dynamic_color_config()
    load_palettes(PALETTES_PATH)
    load_palette_metadata(PALETTES_PATH)
    load_copy_rules(COPY_RULES_PATH)
    if not FACE_LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"Missing MediaPipe model asset: {FACE_LANDMARKER_PATH}")
    suggest_dynamic_colors(
        face_context=None,
        axes={"temp": 0.0, "value": 0.0, "chroma": 0.0, "contrast": 0.0},
        mode="simple",
        config=dynamic_cfg,
        diagnostics=False,
    )
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
