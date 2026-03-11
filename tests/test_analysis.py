import numpy as np

from src.analysis import analyze_image_bytes
from src.color_features import ColorFeatures
from src.face_regions import FaceRegionResult


class DummyDetector:
    def __init__(self, landmarks: np.ndarray | None) -> None:
        self._landmarks = landmarks

    def detect_single_face(self, image_rgb: np.ndarray) -> np.ndarray | None:
        return self._landmarks


def test_analyze_image_decode_failure(monkeypatch) -> None:
    monkeypatch.setattr("src.analysis.decode_uploaded_image", lambda file_bytes: None)
    result = analyze_image_bytes(b"bad", "none", DummyDetector(None))
    assert result.status == "decode_failed"


def test_analyze_image_no_face(monkeypatch) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr("src.analysis.decode_uploaded_image", lambda file_bytes: image)
    result = analyze_image_bytes(b"img", "none", DummyDetector(None))
    assert result.status == "no_face"
    assert result.image_rgb is not None


def test_analyze_image_weak_sample(monkeypatch) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    landmarks = np.array([[1.0, 1.0], [6.0, 1.0], [6.0, 6.0], [1.0, 6.0]] * 120, dtype=np.float32)
    regions = FaceRegionResult(
        landmarks_px=np.zeros((468, 2), dtype=np.int32),
        left_cheek=np.zeros((3, 2), dtype=np.int32),
        right_cheek=np.zeros((3, 2), dtype=np.int32),
        left_mask=np.ones((8, 8), dtype=np.uint8),
        right_mask=np.ones((8, 8), dtype=np.uint8),
    )
    monkeypatch.setattr("src.analysis.decode_uploaded_image", lambda file_bytes: image)
    monkeypatch.setattr("src.analysis.build_region_masks", lambda image_shape, landmarks_px: regions)
    monkeypatch.setattr("src.analysis.compute_robust_lab_features", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.analysis.compute_skin_chroma_variance", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.analysis.evaluate_image_quality", lambda **kwargs: {"score": 0.4, "reasons": ["low_cheek_sample"]})
    result = analyze_image_bytes(b"img", "none", DummyDetector(landmarks))
    assert result.status == "weak_sample"
    assert result.quality is not None


def test_analyze_image_success(monkeypatch) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    landmarks = np.array([[1.0, 1.0], [6.0, 1.0], [6.0, 6.0], [1.0, 6.0]] * 120, dtype=np.float32)
    regions = FaceRegionResult(
        landmarks_px=np.zeros((468, 2), dtype=np.int32),
        left_cheek=np.zeros((3, 2), dtype=np.int32),
        right_cheek=np.zeros((3, 2), dtype=np.int32),
        left_mask=np.ones((8, 8), dtype=np.uint8),
        right_mask=np.ones((8, 8), dtype=np.uint8),
    )
    features = ColorFeatures(l=50.0, a=10.0, b=12.0, chroma=15.6, sample_count=512)
    monkeypatch.setattr("src.analysis.decode_uploaded_image", lambda file_bytes: image)
    monkeypatch.setattr("src.analysis.build_region_masks", lambda image_shape, landmarks_px: regions)
    monkeypatch.setattr("src.analysis.compute_robust_lab_features", lambda *args, **kwargs: features)
    monkeypatch.setattr("src.analysis.compute_skin_chroma_variance", lambda *args, **kwargs: 1.5)
    monkeypatch.setattr("src.analysis.evaluate_image_quality", lambda **kwargs: {"score": 0.9, "reasons": []})
    result = analyze_image_bytes(b"img", "none", DummyDetector(landmarks))
    assert result.status == "ok"
    assert result.features == features
    assert result.skin_chroma_var == 1.5

