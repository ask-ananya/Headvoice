"""Lucas-Kanade optical flow fallback when the face detector misses for a few frames."""

import cv2
import numpy as np


class FlowFallback:
    def __init__(self, max_frames: int = 10, min_good_frac: float = 0.7):
        self.max_frames = max_frames
        self.min_good_frac = min_good_frac
        self._prev_gray: np.ndarray | None = None
        self._pts: np.ndarray | None = None
        self.frames_since_model = 0

    def model_update(self, gray: np.ndarray, landmarks: np.ndarray) -> None:
        self._prev_gray = gray
        self._pts = landmarks.astype(np.float32).reshape(-1, 1, 2)
        self.frames_since_model = 0

    def track(self, gray: np.ndarray) -> np.ndarray | None:
        """Track the last landmarks into this frame, for a bounded number of frames."""
        self.frames_since_model += 1
        if (
            self._pts is None
            or self._prev_gray is None
            or self.frames_since_model > self.max_frames
        ):
            self._pts = None
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            self._pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if nxt is None or status is None or status.mean() < self.min_good_frac:
            self._pts = None
            return None
        self._prev_gray, self._pts = gray, nxt
        return nxt.reshape(-1, 2)
