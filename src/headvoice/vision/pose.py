"""Head pose from 68 landmarks with cv2.solvePnP against a generic 3D face.

Camera assumption (documented in the report): focal length = image width, principal
point at the image center, zero distortion. The model uses camera-like axes (x right,
y down, z away from the camera) so a frontal face gives a rotation near identity.

Angle conventions for an unmirrored camera image:
  yaw   + = nose moves to image right (user turns to their left)
  pitch + = nose moves up (chin up)
  roll  + = top of head moves to image right (tilt toward the user's left shoulder)
"""

import math

import cv2
import numpy as np

# Generic face, arbitrary units (about 0.2 mm). Indices are iBUG 68.
MODEL_IDX = [30, 8, 36, 45, 48, 54]
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # 30 nose tip
        (0.0, 330.0, 65.0),  # 8 chin
        (-225.0, -170.0, 135.0),  # 36 outer corner of the eye on image left
        (225.0, -170.0, 135.0),  # 45 outer corner of the eye on image right
        (-150.0, 150.0, 125.0),  # 48 mouth corner, image left
        (150.0, 150.0, 125.0),  # 54 mouth corner, image right
    ],
    dtype=np.float64,
)


def camera_matrix(width: int, height: int) -> np.ndarray:
    f = float(width)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)


def rotation_to_angles(rvec: np.ndarray) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) in degrees using the conventions in the module docstring."""
    rmat, _ = cv2.Rodrigues(rvec)
    # R = Rz(rz) @ Ry(ry) @ Rx(rx); extract Tait-Bryan angles.
    sy = -rmat[2, 0]
    ry = math.asin(max(-1.0, min(1.0, sy)))
    rx = math.atan2(rmat[2, 1], rmat[2, 2])
    rz = math.atan2(rmat[1, 0], rmat[0, 0])
    return -math.degrees(ry), -math.degrees(rx), math.degrees(rz)


def _solve(
    model: np.ndarray, img_pts: np.ndarray, cam: np.ndarray, rvec0: np.ndarray, tvec0: np.ndarray
):
    return cv2.solvePnP(
        model,
        img_pts,
        cam,
        np.zeros(4),
        rvec0.copy(),
        tvec0.copy(),
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )


class HeadPoseEstimator:
    """solvePnP head pose. Absolute angles carry a per-person bias from the generic face
    model, so everything downstream uses pose relative to the user's neutral."""

    def __init__(self) -> None:
        self._rvec: np.ndarray | None = None
        self._tvec: np.ndarray | None = None

    def reset(self) -> None:
        self._rvec = self._tvec = None

    def estimate(
        self, landmarks: np.ndarray, width: int, height: int
    ) -> tuple[float, float, float] | None:
        img_pts = np.asarray(landmarks, dtype=np.float64)[MODEL_IDX]
        cam = camera_matrix(width, height)
        if self._rvec is None:
            # Start from a frontal face in front of the camera to avoid the mirrored solution.
            rvec0, tvec0 = np.zeros((3, 1)), np.array([[0.0], [0.0], [width * 3.0]])
        else:
            rvec0, tvec0 = self._rvec, self._tvec
        ok, rvec, tvec = _solve(MODEL_POINTS, img_pts, cam, rvec0, tvec0)
        if not ok or tvec[2, 0] <= 0:
            self.reset()
            return None
        self._rvec, self._tvec = rvec, tvec
        return rotation_to_angles(rvec)


# Nose-tip protrusion in front of the outer eye corners, as a fraction of the distance
# between them (adult anthropometry, roughly 35 to 40 mm over 90 to 100 mm).
NOSE_DEPTH_RATIO = 0.42


def yaw_from_landmarks(landmarks: np.ndarray) -> float:
    """Yaw in degrees from 2D geometry: the nose tip's offset along the eye line, divided by
    the outer eye corner distance. Unaffected by roll (measured along the eye line) and,
    unlike 6-point PnP, has a single solution. Same sign convention as rotation_to_angles."""
    pts = np.asarray(landmarks, dtype=np.float64)
    e1, e2, nose = pts[36], pts[45], pts[30]
    axis = e2 - e1
    dist = float(np.linalg.norm(axis))
    if dist < 1e-6:
        return 0.0
    offset = float(np.dot(nose - (e1 + e2) / 2, axis / dist)) / dist
    return math.degrees(math.asin(max(-1.0, min(1.0, offset / NOSE_DEPTH_RATIO))))
