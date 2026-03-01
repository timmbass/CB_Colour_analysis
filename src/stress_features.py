"""Engineered separators from drape stress metrics."""

from __future__ import annotations

import numpy as np


def cool_stress_delta(drape_metrics: dict) -> float:
    hi = float(drape_metrics.get("cool_high_chroma_penalty", 0.0))
    lo = float(drape_metrics.get("cool_low_chroma_penalty", 0.0))
    return hi - lo


def summer_winter_nudge(delta: float, scale: float = 10.0) -> float:
    return float(np.clip(-float(delta) / float(max(scale, 1e-6)), -0.6, 0.6))
