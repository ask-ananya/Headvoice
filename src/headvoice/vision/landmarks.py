"""68-point facial landmarks with cv2.face FacemarkLBF (iBUG 68 layout).

Index groups used elsewhere: subject's right eye 36-41, left eye 42-47, nose tip 30,
chin 8, mouth corners 48 and 54.
"""

from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "lbfmodel.yaml"

RIGHT_EYE = slice(36, 42)
LEFT_EYE = slice(42, 48)


class LandmarkModel:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} missing, run: python scripts/get_models.py")
        self._fm = cv2.face.createFacemarkLBF()
        self._fm.loadModel(str(model_path))

    def fit(self, gray: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        """Return a (68, 2) float32 array, or None if fitting failed."""
        ok, lms = self._fm.fit(gray, np.array([bbox], dtype=np.int32))
        if not ok or len(lms) == 0:
            return None
        pts = np.asarray(lms[0], dtype=np.float32).reshape(-1, 2)
        return pts if pts.shape[0] == 68 else None
