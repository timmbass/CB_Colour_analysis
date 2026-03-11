"""Pure image-analysis pipeline helpers."""

from __future__ import annotations

import hashlib
import logging

import cv2
import numpy as np

from src.color_features import compute_robust_lab_features, compute_skin_chroma_variance
from src.face_regions import FaceMeshDetector, build_region_masks
from src.image_io import decode_uploaded_image
from src.image_quality import evaluate_image_quality
from src.models import ImageAnalysisResult, QualitySummary

logger = logging.getLogger(__name__)


def apply_white_balance(image_rgb: np.ndarray, wb_method_choice: str) -> tuple[np.ndarray, str, bool]:
    wb_method_used = wb_method_choice
    wb_applied = wb_method_choice == "grayworld"
    if wb_method_choice != "grayworld":
        return image_rgb, wb_method_used, wb_applied

    try:
        if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createGrayworldWB"):
            wb = cv2.xphoto.createGrayworldWB()
            corrected_bgr = wb.balanceWhite(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            if corrected_bgr is not None:
                return cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB), wb_method_used, wb_applied
    except cv2.error:
        logger.warning("OpenCV xphoto grayworld white balance failed", exc_info=True)
    except AttributeError:
        logger.warning("OpenCV grayworld white balance unavailable", exc_info=True)

    rgb_f = image_rgb.astype(np.float32)
    channel_means = np.mean(rgb_f.reshape(-1, 3), axis=0)
    gray_target = float(np.mean(channel_means))
    scale = gray_target / np.maximum(channel_means, 1e-6)
    corrected = np.clip(rgb_f * scale[None, None, :], 0, 255).astype(np.uint8)
    return corrected, "grayworld (fallback)", wb_applied


def analyze_image_bytes(file_bytes: bytes, wb_method_choice: str, detector: FaceMeshDetector) -> ImageAnalysisResult:
    image_hash = hashlib.sha256(file_bytes).hexdigest()
    image_rgb = decode_uploaded_image(file_bytes)
    if image_rgb is None:
        return ImageAnalysisResult(status="decode_failed", image_hash=image_hash)
    image_rgb_pre_wb = image_rgb.copy()
    image_rgb, wb_method_used, wb_applied = apply_white_balance(image_rgb, wb_method_choice)
    landmarks = detector.detect_single_face(image_rgb)
    if landmarks is None:
        return ImageAnalysisResult(
            status="no_face",
            image_hash=image_hash,
            image_rgb=image_rgb,
            wb_method=wb_method_used,
            wb_applied=wb_applied,
        )

    regions = build_region_masks(image_rgb.shape, landmarks)
    features = compute_robust_lab_features(image_rgb, regions.left_mask, regions.right_mask)
    pre_wb_features = compute_robust_lab_features(image_rgb_pre_wb, regions.left_mask, regions.right_mask)
    quality = QualitySummary.from_payload(
        evaluate_image_quality(
            image_rgb=image_rgb,
            image_rgb_pre_wb=image_rgb_pre_wb,
            landmarks_px=regions.landmarks_px.astype(np.float32),
            left_mask=regions.left_mask,
            right_mask=regions.right_mask,
            cheek_kept_count=features.sample_count if features is not None else None,
        )
    )
    skin_chroma_var = compute_skin_chroma_variance(image_rgb, regions.left_mask, regions.right_mask)
    if features is None:
        return ImageAnalysisResult(
            status="weak_sample",
            image_hash=image_hash,
            image_rgb=image_rgb,
            regions=regions,
            quality=quality,
            wb_method=wb_method_used,
            wb_applied=wb_applied,
        )

    return ImageAnalysisResult(
        status="ok",
        image_hash=image_hash,
        image_rgb=image_rgb,
        regions=regions,
        features=features,
        quality=quality,
        wb_method=wb_method_used,
        wb_applied=wb_applied,
        pre_wb_skin_b=None if pre_wb_features is None else float(pre_wb_features.b),
        post_wb_skin_b=float(features.b),
        skin_chroma_var=0.0 if skin_chroma_var is None else float(skin_chroma_var),
    )
