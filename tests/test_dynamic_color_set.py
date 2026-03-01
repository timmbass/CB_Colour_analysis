from src.dynamic_colors import load_dynamic_colors_config, suggest_dynamic_colors


def test_dynamic_color_set_has_minimum_size_and_unique_keys():
    cfg = load_dynamic_colors_config("dynamic_colors_config.json")
    axes = {"temp": -0.2, "value": 0.1, "chroma": 0.3, "contrast": 0.0}
    out = suggest_dynamic_colors(None, axes, mode="simple", config=cfg, diagnostics=False, return_set=True)

    colors = out["colors"]
    assert len(colors) >= 6
    keys = [c["key"] for c in colors]
    assert len(keys) == len(set(keys))


def test_dynamic_color_set_stable_order():
    cfg = load_dynamic_colors_config("dynamic_colors_config.json")
    axes = {"temp": 0.1, "value": -0.1, "chroma": 0.05, "contrast": 0.25}
    out1 = suggest_dynamic_colors(None, axes, mode="simple", config=cfg, diagnostics=False, return_set=True)
    out2 = suggest_dynamic_colors(None, axes, mode="simple", config=cfg, diagnostics=False, return_set=True)
    assert [c["key"] for c in out1["colors"]] == [c["key"] for c in out2["colors"]]
