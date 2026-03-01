"""Additional facial-region diagnostics (hair and iris)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.color_features import ColorFeatures, compute_robust_lab_from_pixels


@dataclass
class RegionDiagnostics:
    hair_features: Optional[ColorFeatures]
    iris_features: Optional[ColorFeatures]
    hair_mask: np.ndarray
    iris_mask: np.ndarray


def _draw_rect(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> None:
    h, w = mask.shape[:2]
    ix0 = int(np.clip(round(min(x0, x1)), 0, w - 1))
    ix1 = int(np.clip(round(max(x0, x1)), 0, w - 1))
    iy0 = int(np.clip(round(min(y0, y1)), 0, h - 1))
    iy1 = int(np.clip(round(max(y0, y1)), 0, h - 1))
    if ix1 > ix0 and iy1 > iy0:
        mask[iy0:iy1, ix0:ix1] = 255


def _build_hair_candidate_mask(image_shape: tuple[int, int, int], landmarks_px: np.ndarray) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if landmarks_px.size == 0:
        return mask

    xs = landmarks_px[:, 0]
    ys = landmarks_px[:, 1]
    x_min = float(np.clip(np.min(xs), 0, w - 1))
    x_max = float(np.clip(np.max(xs), 0, w - 1))
    y_min = float(np.clip(np.min(ys), 0, h - 1))
    y_max = float(np.clip(np.max(ys), 0, h - 1))
    face_w = max(1.0, x_max - x_min)
    face_h = max(1.0, y_max - y_min)

    # Above-forehead strip.
    _draw_rect(
        mask,
        x_min + 0.08 * face_w,
        y_min - 0.38 * face_h,
        x_max - 0.08 * face_w,
        y_min + 0.10 * face_h,
    )
    # Temple bands to capture side hair.
    _draw_rect(
        mask,
        x_min - 0.20 * face_w,
        y_min + 0.02 * face_h,
        x_min + 0.20 * face_w,
        y_min + 0.56 * face_h,
    )
    _draw_rect(
        mask,
        x_max - 0.20 * face_w,
        y_min + 0.02 * face_h,
        x_max + 0.20 * face_w,
        y_min + 0.56 * face_h,
    )
    return mask


def _draw_circle(mask: np.ndarray, center: tuple[float, float], radius: float) -> None:
    if radius <= 1.0:
        return
    c = (int(round(center[0])), int(round(center[1])))
    cv2.circle(mask, c, int(round(radius)), 255, -1)


def _build_iris_mask(image_shape: tuple[int, int, int], landmarks_px: np.ndarray) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    n = landmarks_px.shape[0]

    if n >= 478:
        left_iris = landmarks_px[[468, 469, 470, 471, 472]]
        right_iris = landmarks_px[[473, 474, 475, 476, 477]]
        for iris in (left_iris, right_iris):
            center = np.mean(iris, axis=0)
            rad = float(np.median(np.linalg.norm(iris - center, axis=1)))
            # Slightly smaller than iris ring to reduce sclera/eyelid contamination.
            _draw_circle(mask, (center[0], center[1]), 0.55 * rad)
        return mask

    # Fallback from eye landmarks when dedicated iris points are absent.
    if n <= 386:
        return mask
    left_outer, left_inner = landmarks_px[33], landmarks_px[133]
    left_top, left_bottom = landmarks_px[159], landmarks_px[145]
    right_outer, right_inner = landmarks_px[263], landmarks_px[362]
    right_top, right_bottom = landmarks_px[386], landmarks_px[374]
    left_center = (left_outer + left_inner + left_top + left_bottom) / 4.0
    right_center = (right_outer + right_inner + right_top + right_bottom) / 4.0
    left_eye_w = float(np.linalg.norm(left_outer - left_inner))
    right_eye_w = float(np.linalg.norm(right_outer - right_inner))
    _draw_circle(mask, (left_center[0], left_center[1]), 0.11 * left_eye_w)
    _draw_circle(mask, (right_center[0], right_center[1]), 0.11 * right_eye_w)
    return mask


def _filter_non_skin_hair_pixels(
    image_rgb: np.ndarray,
    candidate_mask: np.ndarray,
    skin_features: ColorFeatures,
    min_lab_distance: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(candidate_mask > 0)
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.uint8), np.zeros_like(candidate_mask)

    pixels = image_rgb[ys, xs]
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    l_vals = lab[:, 0] * (100.0 / 255.0)
    a_vals = lab[:, 1] - 128.0
    b_vals = lab[:, 2] - 128.0
    d_lab = np.sqrt((l_vals - skin_features.l) ** 2 + (a_vals - skin_features.a) ** 2 + (b_vals - skin_features.b) ** 2)
    keep = d_lab >= min_lab_distance

    filtered_mask = np.zeros_like(candidate_mask)
    if np.count_nonzero(keep):
        filtered_mask[ys[keep], xs[keep]] = 255
    return pixels[keep], filtered_mask


def compute_region_diagnostics(
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray,
    skin_features: ColorFeatures,
) -> RegionDiagnostics:
    hair_candidate = _build_hair_candidate_mask(image_rgb.shape, landmarks_px)
    hair_pixels, hair_mask = _filter_non_skin_hair_pixels(image_rgb, hair_candidate, skin_features)
    hair_features = compute_robust_lab_from_pixels(
        hair_pixels,
        min_samples=180,
        trim_percentiles=(10.0, 90.0),
        min_kept_samples=60,
    )

    iris_mask = _build_iris_mask(image_rgb.shape, landmarks_px)
    iris_pixels = image_rgb[iris_mask > 0]
    iris_features = compute_robust_lab_from_pixels(
        iris_pixels,
        min_samples=20,
        trim_percentiles=(5.0, 95.0),
        min_kept_samples=10,
    )
    return RegionDiagnostics(
        hair_features=hair_features,
        iris_features=iris_features,
        hair_mask=hair_mask,
        iris_mask=iris_mask,
    )


def render_diagnostics_overlay(image_rgb: np.ndarray, hair_mask: np.ndarray, iris_mask: np.ndarray) -> np.ndarray:
    """Overlay diagnostic masks for hair and iris sampling."""
    overlay = image_rgb.copy()
    hair_color = np.array([30, 220, 120], dtype=np.uint8)  # green
    iris_color = np.array([245, 215, 66], dtype=np.uint8)  # yellow

    hair_bool = hair_mask > 0
    iris_bool = iris_mask > 0
    overlay[hair_bool] = (0.7 * overlay[hair_bool] + 0.3 * hair_color).astype(np.uint8)
    overlay[iris_bool] = (0.45 * overlay[iris_bool] + 0.55 * iris_color).astype(np.uint8)
    return overlay


def compute_definition_score(
    image_rgb: np.ndarray,
    iris_mask: np.ndarray,
    skin_mask: np.ndarray,
) -> float:
    """Definition axis from iris edge strength normalized by skin edge strength."""
    iris_bool = iris_mask > 0
    skin_bool = skin_mask > 0
    if int(np.count_nonzero(iris_bool)) < 10 or int(np.count_nonzero(skin_bool)) < 50:
        return 0.0

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    iris_mean = float(np.mean(mag[iris_bool]))
    skin_mean = float(np.mean(mag[skin_bool]))
    ratio = iris_mean / (skin_mean + 1e-6)

    # Center around ratio~1 as neutral; map to [-1,1].
    score = float(np.tanh((ratio - 1.0) / 0.40))
    return float(np.clip(score, -1.0, 1.0))
