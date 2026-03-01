import re

from src.dynamic_colors import axes_to_hsl, hsl_to_hex, load_dynamic_colors_config, suggest_dynamic_colors


def test_simple_suggestions_are_deterministic_and_valid_hex():
    cfg = load_dynamic_colors_config("dynamic_colors_config.json")
    axes = {"temp": -0.4, "value": 0.25, "chroma": 0.35, "contrast": 0.15}

    out1 = suggest_dynamic_colors(face_context=None, axes=axes, mode="simple", config=cfg, diagnostics=True)
    out2 = suggest_dynamic_colors(face_context=None, axes=axes, mode="simple", config=cfg, diagnostics=True)

    assert out1 == out2
    assert out1["mode"] == "simple"
    assert len(out1["colors"]) == 3

    for item in out1["colors"]:
        assert re.match(r"^#[0-9A-F]{6}$", item["hex"]) is not None
        h = float(item["hsl"]["h"])
        s = float(item["hsl"]["s"])
        l = float(item["hsl"]["l"])
        assert 0.0 <= h <= 360.0
        assert 0.0 <= s <= 100.0
        assert 0.0 <= l <= 100.0


def test_axes_to_hsl_and_hex_clamped_ranges():
    cfg = load_dynamic_colors_config("dynamic_colors_config.json")
    h, s, l = axes_to_hsl(temp=2.0, value=-3.0, chroma=5.0, contrast=4.0, family="accent", config=cfg)
    assert 0.0 <= h <= 360.0
    assert cfg["mapping"]["sat_min"] <= s <= cfg["mapping"]["sat_max"]
    assert cfg["mapping"]["light_min"] <= l <= cfg["mapping"]["light_max"]

    hex_code = hsl_to_hex(h, s, l)
    assert re.match(r"^#[0-9A-F]{6}$", hex_code) is not None
