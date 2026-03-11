"""Color sampling and robust Lab feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ColorFeatures:
    l: float
    a: float
    b: float
    chroma: float
    sample_count: int


def _extract_masked_pixels(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return image_rgb[mask > 0]


def compute_robust_lab_from_pixels(
    pixels: np.ndarray,
    min_samples: int = 500,
    trim_percentiles: tuple[float, float] = (10.0, 90.0),
    min_kept_samples: int | None = None,
) -> ColorFeatures | None:
    """Compute robust median Lab features from an RGB pixel array."""
    if pixels.size == 0 or pixels.shape[0] < min_samples:
        return None

    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    l_vals = lab[:, 0] * (100.0 / 255.0)
    a_vals = lab[:, 1] - 128.0
    b_vals = lab[:, 2] - 128.0

    low_p, high_p = trim_percentiles
    l_low, l_high = np.percentile(l_vals, [low_p, high_p])
    keep = (l_vals >= l_low) & (l_vals <= l_high)

    min_kept = min_kept_samples if min_kept_samples is not None else max(100, min_samples // 3)
    if np.count_nonzero(keep) < min_kept:
        return None

    l_med = float(np.median(l_vals[keep]))
    a_med = float(np.median(a_vals[keep]))
    b_med = float(np.median(b_vals[keep]))
    chroma = float(np.sqrt(a_med**2 + b_med**2))

    return ColorFeatures(l=l_med, a=a_med, b=b_med, chroma=chroma, sample_count=int(np.count_nonzero(keep)))


def compute_robust_lab_features(
    image_rgb: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    min_samples: int = 500,
) -> ColorFeatures | None:
    """Compute robust median Lab values from both cheek regions."""
    pixels_left = _extract_masked_pixels(image_rgb, left_mask)
    pixels_right = _extract_masked_pixels(image_rgb, right_mask)
    pixels = np.vstack([pixels_left, pixels_right]) if pixels_left.size and pixels_right.size else np.concatenate([pixels_left, pixels_right], axis=0)
    return compute_robust_lab_from_pixels(
        pixels,
        min_samples=min_samples,
        trim_percentiles=(10.0, 90.0),
    )


def compute_skin_chroma_variance(
    image_rgb: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    min_samples: int = 200,
) -> float | None:
    """Compute variance of skin chroma from trimmed cheek pixels."""
    pixels_left = _extract_masked_pixels(image_rgb, left_mask)
    pixels_right = _extract_masked_pixels(image_rgb, right_mask)
    pixels = np.vstack([pixels_left, pixels_right]) if pixels_left.size and pixels_right.size else np.concatenate([pixels_left, pixels_right], axis=0)
    if pixels.size == 0 or pixels.shape[0] < min_samples:
        return None

    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    l_vals = lab[:, 0] * (100.0 / 255.0)
    a_vals = lab[:, 1] - 128.0
    b_vals = lab[:, 2] - 128.0
    l_low, l_high = np.percentile(l_vals, [10, 90])
    keep = (l_vals >= l_low) & (l_vals <= l_high)
    if np.count_nonzero(keep) < max(60, min_samples // 3):
        return None

    chroma = np.sqrt(a_vals[keep] ** 2 + b_vals[keep] ** 2)
    return float(np.var(chroma))
