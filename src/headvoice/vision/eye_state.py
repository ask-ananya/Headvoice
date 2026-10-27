"""Eye open/closed from eye-region appearance: HOG features + a linear cv2.ml SVM.

LBF eyelid landmarks barely move when the eyes close (EAR separated open from closed
eyes at chance level on labeled recordings), so blink detection uses the eye image
instead. Each eye is cropped along its corner line, so head roll does not change the
crop, and the user's left eye is mirrored so one model serves both eyes.

Openness is mapped so a typical open eye reads about 1.0 and a closed eye about 0.0.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

CROP_W, CROP_H = 48, 24
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
DEFAULT_MODEL = MODEL_DIR / "eye_state_svm.yml"
DEFAULT_META = MODEL_DIR / "eye_state.json"

_HOG = cv2.HOGDescriptor((CROP_W, CROP_H), (16, 16), (8, 8), (8, 8), 9)


def eye_crop(gray: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Grayscale crop 1.5 eye-widths wide, aligned to the eye corner line."""
    c = (np.asarray(outer, np.float64) + np.asarray(inner, np.float64)) / 2
    d = np.asarray(inner, np.float64) - np.asarray(outer, np.float64)
    width = float(np.linalg.norm(d))
    if width < 2:
        return np.zeros((CROP_H, CROP_W), np.uint8)
    angle = math.degrees(math.atan2(d[1], d[0]))
    m = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), angle, CROP_W / (1.5 * width))
    m[0, 2] += CROP_W / 2 - c[0]
    m[1, 2] += CROP_H / 2 - c[1]
    return cv2.warpAffine(
        gray, m, (CROP_W, CROP_H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE
    )


def eye_crops(gray: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(user's left eye mirrored, user's right eye) from iBUG 68 landmarks."""
    lm = np.asarray(landmarks)
    left = eye_crop(gray, lm[45], lm[42])[:, ::-1]
    right = eye_crop(gray, lm[36], lm[39])
    return np.ascontiguousarray(left), right


def hog_features(crop: np.ndarray) -> np.ndarray:
    return _HOG.compute(cv2.equalizeHist(crop)).ravel()


def train(crops: list[np.ndarray], closed: list[bool], c: float = 0.1) -> tuple[object, dict]:
    """Train the SVM. Returns (svm, meta) where meta orients and scales the raw output."""
    x = np.array([hog_features(cr) for cr in crops], np.float32)
    y = np.asarray(closed, np.int32)
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(c)
    svm.train(x, cv2.ml.ROW_SAMPLE, y)
    raw = svm.predict(x, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1].ravel()
    # OpenCV's raw-output sign depends on label order: orient it so + means closed.
    sign = 1.0 if np.median(raw[y == 1]) > np.median(raw[y == 0]) else -1.0
    s = raw * sign
    meta = {
        "sign": sign,
        "open_ref": float(np.median(s[y == 0])),
        "closed_ref": float(np.median(s[y == 1])),
        "n_open": int((y == 0).sum()),
        "n_closed": int((y == 1).sum()),
    }
    return svm, meta


def save_compact(svm, path: str | Path) -> None:
    """Save the SVM without OpenCV's copy of the training support vectors.

    A linear SVM predicts from one compressed weight vector. OpenCV also stores the
    uncompressed support vectors (features of individual training eye crops) for
    inspection; they are not needed and are personal data, so they are dropped.
    """
    path = Path(path)
    svm.save(str(path))
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out, skipping = [], False
    for line in lines:
        key = line.strip().split(":", 1)[0]
        if key in ("uncompressed_sv_total", "uncompressed_support_vectors"):
            skipping = key == "uncompressed_support_vectors"
            continue
        if skipping and (line.startswith("      ") or line.strip().startswith("-")):
            continue
        skipping = False
        out.append(line)
    path.write_text("".join(out), encoding="utf-8")


class EyeStateModel:
    def __init__(
        self, model_path: str | Path = DEFAULT_MODEL, meta_path: str | Path = DEFAULT_META
    ):
        self.svm = cv2.ml.SVM_load(str(model_path))
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))

    @classmethod
    def load_default(cls) -> EyeStateModel | None:
        if DEFAULT_MODEL.exists() and DEFAULT_META.exists():
            return cls()
        return None

    def openness(self, gray: np.ndarray, landmarks: np.ndarray) -> tuple[float, float]:
        """(left, right) openness: about 1.0 open, about 0.0 closed (not clipped)."""
        crops = eye_crops(gray, landmarks)
        x = np.array([hog_features(c) for c in crops], np.float32)
        s = self.svm.predict(x, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1].ravel() * self.meta["sign"]
        o, c = self.meta["open_ref"], self.meta["closed_ref"]
        left, right = ((c - v) / (c - o) for v in s)
        return float(left), float(right)
