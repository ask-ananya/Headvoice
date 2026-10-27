"""Eye openness and blink events.

Openness is eye aspect ratio (EAR, Soukupova and Cech 2016) divided by a rolling
per-eye baseline, so it reads about 1.0 when the eye is open for any face. The
long-blink threshold comes from the user's involuntary blink durations, not a constant.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from headvoice.vision.landmarks import LEFT_EYE, RIGHT_EYE


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """EAR for six eye points ordered as in iBUG 68 (corner, top, top, corner, bottom, bottom)."""
    eye = np.asarray(eye, dtype=np.float64)
    width = np.linalg.norm(eye[0] - eye[3])
    if width < 1e-6:
        return 0.0
    return float((np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / (2 * width))


def eye_ears(landmarks: np.ndarray) -> tuple[float, float]:
    """Return (left, right) EAR for the user's own left and right eyes."""
    pts = np.asarray(landmarks)
    return eye_aspect_ratio(pts[LEFT_EYE]), eye_aspect_ratio(pts[RIGHT_EYE])


class OpennessNormalizer:
    """Divides EAR by a high percentile of recent values (eyes are open most of the time)."""

    def __init__(self, window: int = 300, min_samples: int = 15, percentile: float = 80):
        self._vals: deque[float] = deque(maxlen=window)
        self.min_samples = min_samples
        self.percentile = percentile

    def __call__(self, ear: float) -> float | None:
        self._vals.append(ear)
        if len(self._vals) < self.min_samples:
            return None  # not enough history: blink cannot be measured yet (fail closed)
        base = float(np.percentile(self._vals, self.percentile))
        return ear / base if base > 1e-6 else None


def eyes_state(open_l: float | None, open_r: float | None, closed_ratio: float) -> str:
    """'both' if both eyes are closed, 'one' if they disagree, 'none' if open or unknown."""
    if open_l is None or open_r is None:
        return "none"
    cl, cr = open_l < closed_ratio, open_r < closed_ratio
    if cl and cr:
        return "both"
    return "one" if cl or cr else "none"


@dataclass
class BlinkEvent:
    start: float
    end: float
    agreed: bool  # False when only one eye closed (rejected)

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000


class BlinkDetector:
    """Turns per-frame openness into blink events for logging and calibration."""

    def __init__(self, closed_ratio: float = 0.7):
        self.closed_ratio = closed_ratio
        self._start: float | None = None
        self._agreed = True

    def update(self, open_l: float | None, open_r: float | None, t: float) -> BlinkEvent | None:
        s = eyes_state(open_l, open_r, self.closed_ratio)
        if s != "none":
            if self._start is None:
                self._start, self._agreed = t, s == "both"
            elif s == "one":
                self._agreed = False
            return None
        if self._start is None:
            return None
        ev = BlinkEvent(self._start, t, self._agreed)
        self._start = None
        return ev


def long_blink_threshold_ms(
    involuntary_ms: list[float],
    percentile: float = 95,
    margin_ms: float = 150,
    floor_ms: float = 400,
) -> float:
    """A long blink must last longer than nearly all of the user's involuntary blinks."""
    if not involuntary_ms:
        return max(floor_ms, 700.0)
    return max(floor_ms, float(np.percentile(involuntary_ms, percentile)) + margin_ms)
