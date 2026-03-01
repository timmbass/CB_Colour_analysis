from src.dynamic_colors import generate_candidates, load_dynamic_colors_config


def test_candidate_generation_count_within_bounds_and_deduped():
    cfg = load_dynamic_colors_config("dynamic_colors_config.json")
    candidates = generate_candidates(
        temp=-0.25,
        value=0.15,
        chroma=0.45,
        contrast=-0.1,
        family="signature_blue",
        config=cfg,
        n=20,
    )
    assert 1 <= len(candidates) <= min(20, int(cfg["candidate_search"]["max_per_family"]))

    hexes = [c.hex for c in candidates]
    assert len(hexes) == len(set(hexes))


def test_candidate_generation_respects_n_cap():
    cfg = load_dynamic_colors_config("dynamic_colors_config.json")
    candidates = generate_candidates(
        temp=0.1,
        value=-0.1,
        chroma=0.0,
        contrast=0.0,
        family="accent",
        config=cfg,
        n=5,
    )
    assert len(candidates) <= 5
