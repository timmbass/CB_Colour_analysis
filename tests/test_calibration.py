import numpy as np

from src.calibration import apply_calibration, margin_confidence, predict_topk


def test_apply_calibration_alpha_one():
    z_base = np.array([1.0, 2.0, 3.0, 4.0])
    z_drape = np.array([-1.0, -2.0, -3.0, -4.0])
    bias = [0.1, -0.2, 0.3, -0.4]
    out = apply_calibration(z_base, z_drape, alpha=1.0, bias=bias)
    np.testing.assert_allclose(out, z_base + np.array(bias), rtol=1e-9, atol=1e-9)


def test_apply_calibration_alpha_zero():
    z_base = np.array([1.0, 2.0, 3.0, 4.0])
    z_drape = np.array([-1.0, -2.0, -3.0, -4.0])
    bias = [0.1, -0.2, 0.3, -0.4]
    out = apply_calibration(z_base, z_drape, alpha=0.0, bias=bias)
    np.testing.assert_allclose(out, z_drape + np.array(bias), rtol=1e-9, atol=1e-9)


def test_margin_confidence_monotonic_with_margin():
    gamma = 3.0
    z_small = np.array([1.00, 0.99, 0.1, -0.2])
    z_big = np.array([1.00, 0.10, 0.1, -0.2])
    conf_small, _, _, m_small = margin_confidence(z_small, gamma=gamma)
    conf_big, _, _, m_big = margin_confidence(z_big, gamma=gamma)
    assert m_big > m_small
    assert conf_big > conf_small


def test_predict_top2_extraction_correct():
    z = np.array([0.1, 0.8, 0.5, -0.2])
    top2 = predict_topk(z, k=2)
    assert top2[0][0] == 1
    assert top2[1][0] == 2
