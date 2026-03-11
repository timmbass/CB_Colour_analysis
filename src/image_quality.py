"""Image quality scoring for portrait color analysis."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import load_analysis_config


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def evaluate_image_quality(
    image_rgb: np.ndarray,
    image_rgb_pre_wb: np.ndarray,
    landmarks_px: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    cheek_kept_count: int | None = None,
) -> dict:
    cfg = load_analysis_config()["image_quality"]
    score_weights = cfg["score_weights"]
    score_ranges = cfg["score_ranges"]
    h, w = image_rgb.shape[:2]
    img_area = float(max(h * w, 1))

    xs = landmarks_px[:, 0]
    ys = landmarks_px[:, 1]
    x0 = int(np.clip(np.min(xs), 0, w - 1))
    x1 = int(np.clip(np.max(xs), 0, w - 1))
    y0 = int(np.clip(np.min(ys), 0, h - 1))
    y1 = int(np.clip(np.max(ys), 0, h - 1))
    if x1 <= x0 or y1 <= y0:
        face_ratio = 0.0
        face_crop = image_rgb
        pre_face_crop = image_rgb_pre_wb
    else:
        face_area = float((x1 - x0) * (y1 - y0))
        face_ratio = face_area / img_area
        face_crop = image_rgb[y0:y1, x0:x1]
        pre_face_crop = image_rgb_pre_wb[y0:y1, x0:x1]

    cheek_mask = ((left_mask > 0) | (right_mask > 0))
    cheek_pixels = image_rgb[cheek_mask]
    raw_cheek_count = int(cheek_pixels.shape[0])
    effective_cheek_count = int(cheek_kept_count) if cheek_kept_count is not None else raw_cheek_count

    over_pct = 100.0
    under_pct = 100.0
    if raw_cheek_count > 0:
        lab = cv2.cvtColor(cheek_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        l_vals = lab[:, 0] * (100.0 / 255.0)
        over_pct = float(np.mean(l_vals > 88.0) * 100.0)
        under_pct = float(np.mean(l_vals < 18.0) * 100.0)

    blur_var = 0.0
    if face_crop.size > 0:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    cast_imbalance = 0.0
    if pre_face_crop.size > 0:
        means = np.mean(pre_face_crop.reshape(-1, 3), axis=0).astype(np.float32)
        cast_imbalance = float((np.max(means) - np.min(means)) / (np.mean(means) + 1e-6))

    reasons: list[str] = []
    if face_ratio < float(cfg["tiny_face_ratio"]):
        reasons.append("tiny_face")
    if effective_cheek_count < int(cfg["low_cheek_sample"]):
        reasons.append("low_cheek_sample")
    if over_pct > float(cfg["overexposed_pct"]):
        reasons.append("overexposed_cheeks")
    if under_pct > float(cfg["underexposed_pct"]):
        reasons.append("underexposed_cheeks")
    if blur_var < float(cfg["blur_laplacian_var"]):
        reasons.append("blurry_face")
    if cast_imbalance > float(cfg["strong_color_cast_pre_wb"]):
        reasons.append("strong_color_cast_pre_wb")

    s_face = _clip01((face_ratio - float(score_ranges["face"][0])) / float(score_ranges["face"][1]))
    s_cheek = _clip01((effective_cheek_count - float(score_ranges["cheek"][0])) / float(score_ranges["cheek"][1]))
    s_over = 1.0 - _clip01((over_pct - float(score_ranges["over"][0])) / float(score_ranges["over"][1]))
    s_under = 1.0 - _clip01((under_pct - float(score_ranges["under"][0])) / float(score_ranges["under"][1]))
    s_blur = _clip01((blur_var - float(score_ranges["blur"][0])) / float(score_ranges["blur"][1]))
    s_cast = 1.0 - _clip01((cast_imbalance - float(score_ranges["cast"][0])) / float(score_ranges["cast"][1]))
    score = float(
        float(score_weights["face"]) * s_face
        + float(score_weights["cheek"]) * s_cheek
        + float(score_weights["over"]) * s_over
        + float(score_weights["under"]) * s_under
        + float(score_weights["blur"]) * s_blur
        + float(score_weights["cast"]) * s_cast
    )

    return {
        "score": score,
        "face_ratio": float(face_ratio),
        "cheek_sample_count": int(effective_cheek_count),
        "overexposed_pct": float(over_pct),
        "underexposed_pct": float(under_pct),
        "blur_laplacian_var": float(blur_var),
        "color_cast_imbalance_pre_wb": float(cast_imbalance),
        "reasons": reasons,
    }
