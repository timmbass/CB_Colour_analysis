from copy import deepcopy

from src.dynamic_colors import axes_to_hsl, load_dynamic_colors_config, suggest_dynamic_colors


def test_everyday_green_family_returns_hsl_in_bounds():
    cfg = load_dynamic_colors_config("dynamic_colors_config.json")
    h, s, l = axes_to_hsl(
        temp=0.15,
        value=0.05,
        chroma=0.10,
        contrast=0.05,
        family="everyday_green",
        config=cfg,
    )
    assert 0.0 <= h <= 360.0
    assert cfg["mapping"]["sat_min"] <= s <= cfg["mapping"]["sat_max"]
    assert cfg["mapping"]["light_min"] <= l <= cfg["mapping"]["light_max"]


def test_unknown_family_falls_back_and_records_note():
    cfg = load_dynamic_colors_config("dynamic_colors_config.json")
    notes: list[str] = []
    h, s, l = axes_to_hsl(
        temp=-0.2,
        value=0.2,
        chroma=0.1,
        contrast=0.0,
        family="not_a_real_family",
        config=cfg,
        diagnostics_notes=notes,
    )
    assert 0.0 <= h <= 360.0
    assert cfg["mapping"]["sat_min"] <= s <= cfg["mapping"]["sat_max"]
    assert cfg["mapping"]["light_min"] <= l <= cfg["mapping"]["light_max"]
    assert notes
    assert "fallback" in notes[0].lower()


def test_suggest_dynamic_colors_missing_family_no_crash_with_note():
    cfg = load_dynamic_colors_config("dynamic_colors_config.json")
    cfg2 = deepcopy(cfg)
    cfg2["families"].pop("everyday_green", None)
    out = suggest_dynamic_colors(
        face_context=None,
        axes={"temp": 0.1, "value": 0.0, "chroma": 0.2, "contrast": 0.1},
        mode="simple",
        config=cfg2,
        diagnostics=True,
        return_set=True,
    )
    assert len(out["colors"]) >= 6
    notes = out.get("diagnostics", {}).get("family_notes", [])
    assert notes
    assert any("everyday_green" in n for n in notes)
