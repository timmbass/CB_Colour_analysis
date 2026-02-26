"""Color sampling and robust Lab feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


def compute_robust_lab_features(
    image_rgb: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    min_samples: int = 500,
) -> Optional[ColorFeatures]:
    """Compute robust median Lab values from both cheek regions."""
    pixels_left = _extract_masked_pixels(image_rgb, left_mask)
    pixels_right = _extract_masked_pixels(image_rgb, right_mask)
    pixels = np.vstack([pixels_left, pixels_right]) if pixels_left.size and pixels_right.size else np.concatenate([pixels_left, pixels_right], axis=0)

    if pixels.shape[0] < min_samples:
        return None

    # Convert sampled pixels to Lab.
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    # OpenCV Lab ranges: L in [0,255], a/b in [0,255] centered at 128.
    l_vals = lab[:, 0] * (100.0 / 255.0)
    a_vals = lab[:, 1] - 128.0
    b_vals = lab[:, 2] - 128.0

    # Exclude top/bottom 10% by L.
    l_low, l_high = np.percentile(l_vals, [10, 90])
    keep = (l_vals >= l_low) & (l_vals <= l_high)

    if np.count_nonzero(keep) < max(100, min_samples // 3):
        return None

    l_med = float(np.median(l_vals[keep]))
    a_med = float(np.median(a_vals[keep]))
    b_med = float(np.median(b_vals[keep]))
    chroma = float(np.sqrt(a_med**2 + b_med**2))

    return ColorFeatures(l=l_med, a=a_med, b=b_med, chroma=chroma, sample_count=int(np.count_nonzero(keep)))
