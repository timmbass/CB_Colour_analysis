from src.stress_features import cool_stress_delta, summer_winter_nudge


def test_delta_sign_drives_expected_nudge_direction():
    # high-chroma cool drapes worse than low-chroma cool -> Winter down, Summer up => negative nudge
    metrics = {"cool_high_chroma_penalty": 0.8, "cool_low_chroma_penalty": 0.2}
    delta = cool_stress_delta(metrics)
    nudge = summer_winter_nudge(delta, scale=10.0)
    assert delta > 0
    assert nudge < 0


def test_nudge_clamps_correctly():
    assert summer_winter_nudge(delta=1000.0, scale=1.0) == -0.6
    assert summer_winter_nudge(delta=-1000.0, scale=1.0) == 0.6
