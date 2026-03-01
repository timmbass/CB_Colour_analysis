import numpy as np

from src.drape import render_color_strip


def test_render_color_strip_size_and_endpoints():
    colors = [
        "#112233",
        "#223344",
        "#334455",
        "#445566",
        "#556677",
        "#667788",
        "#778899",
        "#8899AA",
    ]
    width, height = 800, 64
    strip = render_color_strip(colors, width=width, height=height, style="blocks")
    assert strip is not None
    assert strip.shape == (height, width, 3)

    # Sample interior pixels to avoid border separators.
    assert np.array_equal(strip[height // 2, 10], np.array([17, 34, 51], dtype=np.uint8))
    assert np.array_equal(strip[height // 2, width - 10], np.array([136, 153, 170], dtype=np.uint8))


def test_render_color_strip_deterministic():
    colors = ["#0A0B0C", "#1A1B1C", "#2A2B2C", "#3A3B3C", "#4A4B4C", "#5A5B5C", "#6A6B6C", "#7A7B7C"]
    s1 = render_color_strip(colors, width=512, height=72, style="blocks")
    s2 = render_color_strip(colors, width=512, height=72, style="blocks")
    assert s1 is not None and s2 is not None
    assert np.array_equal(s1, s2)
