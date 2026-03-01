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
    strip = render_color_strip(colors, width=image_rgb.shape[1], height=strip_height, style="blocks")
    if strip is None:
        return image_rgb
    return np.vstack([image_rgb, strip])


def render_color_strip(colors: List[str], width: int, height: int, style: str = "blocks") -> np.ndarray | None:
    """Render a colour strip from hex values as equal-width blocks with thin borders."""
    if not colors or width <= 0 or height <= 0:
        return None
    if style != "blocks":
        raise ValueError("Only 'blocks' style is supported")

    strip = np.zeros((height, width, 3), dtype=np.uint8)
    block_width = max(1, width // len(colors))
    border = np.array([228, 228, 228], dtype=np.uint8)
    outer = np.array([180, 180, 180], dtype=np.uint8)

    for i, color_hex in enumerate(colors):
        start_x = i * block_width
        end_x = width if i == len(colors) - 1 else min(width, (i + 1) * block_width)
        strip[:, start_x:end_x] = _hex_to_rgb(color_hex)
        if i > 0 and start_x < width:
            strip[:, max(0, start_x - 1) : start_x] = border

    strip[0:1, :] = outer
    strip[-1:, :] = outer
    strip[:, 0:1] = outer
    strip[:, -1:] = outer
    return strip
