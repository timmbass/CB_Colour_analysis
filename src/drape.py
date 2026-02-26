"""Digital drape rendering utilities."""

from __future__ import annotations

from typing import List

import numpy as np


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    c = hex_color.strip().lstrip("#")
    if len(c) != 6:
        return (128, 128, 128)
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))


def render_drape_strip(image_rgb: np.ndarray, colors: List[str], strip_height: int = 70) -> np.ndarray:
    """Render a horizontal strip of season colors beneath the source image."""
    if not colors:
        return image_rgb

    h, w = image_rgb.shape[:2]
    strip = np.zeros((strip_height, w, 3), dtype=np.uint8)

    block_width = max(1, w // len(colors))
    for i, color_hex in enumerate(colors):
        start_x = i * block_width
        end_x = w if i == len(colors) - 1 else min(w, (i + 1) * block_width)
        strip[:, start_x:end_x] = _hex_to_rgb(color_hex)

    return np.vstack([image_rgb, strip])
