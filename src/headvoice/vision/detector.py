"""Face detection with YuNet (cv2.FaceDetectorYN)."""

from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "face_detection_yunet_2026may.onnx"


class FaceDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, score_threshold: float = 0.8):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} missing, run: python scripts/get_models.py")
        self._det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold)
        self._size: tuple[int, int] | None = None

    def detect(self, bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return the largest face as (x, y, w, h), or None."""
        h, w = bgr.shape[:2]
        if self._size != (w, h):
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = (int(round(v)) for v in best[:4])
        x, y = max(0, x), max(0, y)
        fw, fh = min(fw, w - x), min(fh, h - y)
        if fw <= 0 or fh <= 0:
            return None
        return x, y, fw, fh
