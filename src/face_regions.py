"""FaceMesh detection and cheek-region extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Stable cheek-region landmark sets for MediaPipe FaceMesh (468 landmarks).
LEFT_CHEEK_IDX = [50, 187, 205, 207, 213, 192, 147, 123, 116, 117, 118]
RIGHT_CHEEK_IDX = [280, 411, 425, 427, 436, 416, 376, 352, 345, 346, 347]


@dataclass
class FaceRegionResult:
    landmarks_px: np.ndarray
    left_cheek: np.ndarray
    right_cheek: np.ndarray
    left_mask: np.ndarray
    right_mask: np.ndarray


class FaceMeshDetector:
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        model_path: Optional[Path] = None,
    ) -> None:
        _ = static_image_mode
        model_path = model_path or (Path(__file__).resolve().parents[1] / "assets" / "models" / "face_landmarker.task")
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect_single_face(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Return (468,2) pixel landmarks for one face, or None if not found."""
        h, w = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self._detector.detect(mp_image)
        if not results.face_landmarks:
            return None

        landmarks = results.face_landmarks[0]
        points = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)
        return points

    def close(self) -> None:
        self._detector.close()


def _poly_from_indices(landmarks_px: np.ndarray, indices: List[int]) -> np.ndarray:
    poly = landmarks_px[indices]
    return np.round(poly).astype(np.int32)


def build_region_masks(
    image_shape: Tuple[int, int, int], landmarks_px: np.ndarray
) -> FaceRegionResult:
    """Build cheek polygons + masks from landmarks."""
    h, w = image_shape[:2]
    left_cheek = _poly_from_indices(landmarks_px, LEFT_CHEEK_IDX)
    right_cheek = _poly_from_indices(landmarks_px, RIGHT_CHEEK_IDX)

    left_mask = np.zeros((h, w), dtype=np.uint8)
    right_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(left_mask, [left_cheek], color=255)
    cv2.fillPoly(right_mask, [right_cheek], color=255)

    return FaceRegionResult(
        landmarks_px=np.round(landmarks_px).astype(np.int32),
        left_cheek=left_cheek,
        right_cheek=right_cheek,
        left_mask=left_mask,
        right_mask=right_mask,
    )


def render_debug_overlay(image_rgb: np.ndarray, regions: FaceRegionResult) -> np.ndarray:
    """Draw cheek polygons and semi-transparent masks for debugging."""
    overlay = image_rgb.copy()

    red = np.array([255, 60, 60], dtype=np.uint8)
    blue = np.array([60, 160, 255], dtype=np.uint8)

    left_mask_bool = regions.left_mask > 0
    right_mask_bool = regions.right_mask > 0

    overlay[left_mask_bool] = (0.65 * overlay[left_mask_bool] + 0.35 * red).astype(np.uint8)
    overlay[right_mask_bool] = (0.65 * overlay[right_mask_bool] + 0.35 * blue).astype(np.uint8)

    cv2.polylines(overlay, [regions.left_cheek], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(overlay, [regions.right_cheek], isClosed=True, color=(0, 128, 255), thickness=2)

    return overlay
