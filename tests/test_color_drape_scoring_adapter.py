import math

import cv2
import numpy as np

from src.color_features import compute_robust_lab_features
from src.drape_scoring import prepare_color_drape_context, score_color_drape
from src.face_regions import FaceRegionResult


def _build_synthetic_face_context() -> dict:
    h, w = 360, 360
    image = np.full((h, w, 3), 150, dtype=np.uint8)

    # Skin-like oval face region
    cv2.ellipse(image, (w // 2, 150), (90, 110), 0, 0, 360, (205, 165, 145), -1)

    left_mask = np.zeros((h, w), dtype=np.uint8)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(left_mask, (130, 160), (32, 20), 0, 0, 360, 255, -1)
    cv2.ellipse(right_mask, (230, 160), (32, 20), 0, 0, 360, 255, -1)

    t = np.linspace(0.0, 2.0 * np.pi, 468, endpoint=False)
    landmarks = np.stack([w / 2 + 95 * np.cos(t), 150 + 120 * np.sin(t)], axis=1).astype(np.int32)

    regions = FaceRegionResult(
        landmarks_px=landmarks,
        left_cheek=np.array([[120, 160], [140, 150], [150, 170]], dtype=np.int32),
        right_cheek=np.array([[220, 160], [240, 150], [250, 170]], dtype=np.int32),
        left_mask=left_mask,
        right_mask=right_mask,
    )
    baseline_skin = compute_robust_lab_features(image, left_mask, right_mask, min_samples=100)
    assert baseline_skin is not None
    return {
        "image_rgb": image,
        "regions": regions,
        "baseline_skin": baseline_skin,
        "season_hint": "Winter",
    }


def test_score_color_drape_returns_finite_float():
    ctx = prepare_color_drape_context(_build_synthetic_face_context())
    p = score_color_drape(ctx, (10, 20, 30))
    assert isinstance(p, float)
    assert math.isfinite(p)


def test_distinct_colours_produce_distinct_penalties():
    ctx = prepare_color_drape_context(_build_synthetic_face_context())
    p1 = score_color_drape(ctx, (40, 90, 180))
    p2 = score_color_drape(ctx, (210, 210, 210))
    assert not np.isclose(p1, p2, atol=1e-8)


def test_score_color_drape_is_deterministic():
    ctx = prepare_color_drape_context(_build_synthetic_face_context())
    p1 = score_color_drape(ctx, (30, 110, 170))
    p2 = score_color_drape(ctx, (30, 110, 170))
    assert np.isclose(p1, p2, atol=1e-12)
