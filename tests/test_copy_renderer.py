from pathlib import Path

from src.copy_renderer import band_for, load_copy_rules, render_text_suggestions

RULES_PATH = Path("assets/copy_rules.v1.json")


def _base_payload() -> dict:
    return {
        "image_id": "img-123",
        "season_display": "Winter -> Cool Winter",
        "top1_key": "winter",
        "top2_key": "summer",
        "top1_variant_key": "cool_winter",
        "confidence": 0.9,
        "axes": {
            "temp": -0.4,
            "value": 0.2,
            "chroma": 0.35,
            "contrast": 0.7,
        },
        "dynamic_colors": [],
    }


def test_high_confidence_winter_has_no_close_call() -> None:
    payload = _base_payload()
    payload["confidence"] = 0.9

    rendered = render_text_suggestions(payload, RULES_PATH)
    ids = [block["id"] for block in rendered["blocks"]]

    assert rendered["headline"]
    assert "close_call" not in ids
    assert "axes_summary" in ids
    assert "core_guidance" in ids
    assert "best_colors" in ids


def test_low_confidence_summer_vs_winter_has_pair_tiebreaker() -> None:
    payload = _base_payload()
    payload["top1_key"] = "summer"
    payload["top1_variant_key"] = "cool_summer"
    payload["top2_key"] = "winter"
    payload["confidence"] = 0.52

    rendered = render_text_suggestions(payload, RULES_PATH)
    blocks = {block["id"]: block for block in rendered["blocks"]}

    assert rendered["headline"]
    assert "close_call" in blocks
    assert "tie_breaker_tests" in blocks
    joined = " | ".join(blocks["tie_breaker_tests"].get("bullets", []))
    assert "Stark white vs soft grey" in joined


def test_unknown_season_falls_back_without_crash() -> None:
    payload = _base_payload()
    payload["top1_key"] = "mystery_season"
    payload["top1_variant_key"] = ""
    payload["confidence"] = 0.8

    rendered = render_text_suggestions(payload, RULES_PATH)

    assert rendered["headline"]
    assert rendered["blocks"]
    assert rendered["diagnostics"].get("resolved_profile_key") in {"fallback_profile", "autumn", "spring", "summer", "winter"}


def test_axis_banding() -> None:
    rules = load_copy_rules(RULES_PATH)

    temp = band_for(0.7, rules["bands"]["temp"])
    chroma = band_for(-0.6, rules["bands"]["chroma"])
    contrast = band_for(0.8, rules["bands"]["contrast"])

    assert temp["id"] == "warm"
    assert chroma["id"] == "soft"
    assert contrast["id"] == "high"
