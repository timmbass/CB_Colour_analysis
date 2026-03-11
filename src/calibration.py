"""Tiny calibration helpers for season logits."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def apply_calibration(
    z_base: np.ndarray,
    z_drape: np.ndarray,
    alpha: float,
    bias: Iterable[float],
) -> np.ndarray:
    zb = np.asarray(z_base, dtype=np.float64)
    zd = np.asarray(z_drape, dtype=np.float64)
    b = np.asarray(list(bias), dtype=np.float64)
    if zb.shape != (4,) or zd.shape != (4,) or b.shape != (4,):
        raise ValueError("z_base, z_drape, and bias must all be shape (4,)")
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * zb + (1.0 - a) * zd + b


def predict_topk(z_cal: np.ndarray, k: int = 2) -> list[tuple[int, float]]:
    z = np.asarray(z_cal, dtype=np.float64)
    if z.shape != (4,):
        raise ValueError("z_cal must be shape (4,)")
    kk = int(max(1, min(k, 4)))
    idx = np.argsort(z)[::-1][:kk]
    return [(int(i), float(z[i])) for i in idx]


def margin_confidence(z_cal: np.ndarray, gamma: float) -> tuple[float, int, int, float]:
    top2 = predict_topk(z_cal, k=2)
    top1_idx, top1_logit = top2[0]
    top2_idx, top2_logit = top2[1]
    margin = float(top1_logit - top2_logit)
    g = max(1e-6, float(gamma))
    conf = float(1.0 / (1.0 + np.exp(-g * margin)))
    return conf, top1_idx, top2_idx, margin
