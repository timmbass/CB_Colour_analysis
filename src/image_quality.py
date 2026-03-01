"""Image quality scoring for portrait color analysis."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def evaluate_image_quality(
    image_rgb: np.ndarray,
    image_rgb_pre_wb: np.ndarray,
    landmarks_px: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    cheek_kept_count: Optional[int] = None,
) -> dict:
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
    if face_ratio < 0.08:
        reasons.append("tiny_face")
    if effective_cheek_count < 450:
        reasons.append("low_cheek_sample")
    if over_pct > 35.0:
        reasons.append("overexposed_cheeks")
    if under_pct > 35.0:
        reasons.append("underexposed_cheeks")
    if blur_var < 70.0:
        reasons.append("blurry_face")
    if cast_imbalance > 0.38:
        reasons.append("strong_color_cast_pre_wb")

    s_face = _clip01((face_ratio - 0.03) / 0.17)
    s_cheek = _clip01((effective_cheek_count - 300.0) / 1200.0)
    s_over = 1.0 - _clip01((over_pct - 8.0) / 40.0)
    s_under = 1.0 - _clip01((under_pct - 8.0) / 40.0)
    s_blur = _clip01((blur_var - 35.0) / 220.0)
    s_cast = 1.0 - _clip01((cast_imbalance - 0.08) / 0.35)
    score = float(0.22 * s_face + 0.20 * s_cheek + 0.14 * s_over + 0.14 * s_under + 0.20 * s_blur + 0.10 * s_cast)

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
