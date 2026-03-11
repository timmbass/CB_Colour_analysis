"""Image loading and conversion helpers."""

from __future__ import annotations

import cv2
import numpy as np


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray | None:
    """Decode uploaded file bytes into an RGB image."""
    if not file_bytes:
        return None

    np_buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        return None

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
