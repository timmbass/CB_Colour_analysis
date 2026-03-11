"""Typed domain models used by the app orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from src.color_features import ColorFeatures
from src.face_regions import FaceRegionResult

AnalysisStatus = Literal["ok", "decode_failed", "no_face", "weak_sample"]


@dataclass(frozen=True)
class QualitySummary:
    score: float
    face_ratio: float
    cheek_sample_count: int
    overexposed_pct: float
    underexposed_pct: float
    blur_laplacian_var: float
    color_cast_imbalance_pre_wb: float
    reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> QualitySummary:
        return cls(
            score=float(payload.get("score", 0.0)),
            face_ratio=float(payload.get("face_ratio", 0.0)),
            cheek_sample_count=int(payload.get("cheek_sample_count", 0)),
            overexposed_pct=float(payload.get("overexposed_pct", 0.0)),
            underexposed_pct=float(payload.get("underexposed_pct", 0.0)),
            blur_laplacian_var=float(payload.get("blur_laplacian_var", 0.0)),
            color_cast_imbalance_pre_wb=float(payload.get("color_cast_imbalance_pre_wb", 0.0)),
            reasons=[str(item) for item in payload.get("reasons", [])],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "face_ratio": self.face_ratio,
            "cheek_sample_count": self.cheek_sample_count,
            "overexposed_pct": self.overexposed_pct,
            "underexposed_pct": self.underexposed_pct,
            "blur_laplacian_var": self.blur_laplacian_var,
            "color_cast_imbalance_pre_wb": self.color_cast_imbalance_pre_wb,
            "reasons": list(self.reasons),
        }


@dataclass
class ImageAnalysisResult:
    status: AnalysisStatus
    image_hash: str
    image_rgb: np.ndarray | None = None
    regions: FaceRegionResult | None = None
    features: ColorFeatures | None = None
    quality: QualitySummary | None = None
    wb_method: str = "none"
    wb_applied: bool = False
    pre_wb_skin_b: float | None = None
    post_wb_skin_b: float | None = None
    skin_chroma_var: float = 0.0

    def as_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "image_hash": self.image_hash,
            "image_rgb": self.image_rgb,
            "regions": self.regions,
            "features": self.features,
            "quality": None if self.quality is None else self.quality.as_dict(),
            "wb_method": self.wb_method,
            "wb_applied": self.wb_applied,
            "pre_wb_skin_b": self.pre_wb_skin_b,
            "post_wb_skin_b": self.post_wb_skin_b,
            "skin_chroma_var": self.skin_chroma_var,
        }
