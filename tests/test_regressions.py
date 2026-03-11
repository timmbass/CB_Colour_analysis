import numpy as np

from src.color_features import ColorFeatures
from src.dynamic_colors import generate_candidates, load_dynamic_colors_config
from src.image_quality import evaluate_image_quality
from src.season_rules import classify_season


def test_classify_season_regression_winter_like_profile() -> None:
    decision = classify_season(
        skin_features=ColorFeatures(l=38.0, a=4.0, b=-3.0, chroma=24.0, sample_count=800),
        hair_features=ColorFeatures(l=15.0, a=2.0, b=-1.0, chroma=4.0, sample_count=200),
        delta_l_hair_skin=23.0,
        delta_l_iris_skin=18.0,
        skin_chroma_variance=1.2,
        definition_score=0.5,
    )
    assert decision.season == "Winter"


def test_evaluate_image_quality_regression_flags_obvious_low_quality() -> None:
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    landmarks = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    mask = np.ones((12, 12), dtype=np.uint8)
    result = evaluate_image_quality(
        image_rgb=image,
        image_rgb_pre_wb=image,
        landmarks_px=landmarks,
        left_mask=mask,
        right_mask=mask,
        cheek_kept_count=50,
    )
    assert "tiny_face" in result["reasons"]
    assert "low_cheek_sample" in result["reasons"]


def test_dynamic_color_candidates_regression_signature_blue_first_candidate_stable() -> None:
    cfg = load_dynamic_colors_config("dynamic_colors_config.json")
    candidates = generate_candidates(
        temp=-0.25,
        value=0.15,
        chroma=0.45,
        contrast=-0.1,
        family="signature_blue",
        config=cfg,
        n=5,
    )
    assert candidates[0].hex.startswith("#")
    assert candidates[0].meta["hue_offset_index"] == 0
