"""Frame -> FrameFeatures. Detection, landmarks, pose, blink, quality, CLAHE and flow fallback."""

import time
from collections import deque

import cv2
import numpy as np

from headvoice.io.schemas import FrameFeatures, Neutral, Profile
from headvoice.vision.blink import OpennessNormalizer, eye_ears
from headvoice.vision.detector import FaceDetector
from headvoice.vision.eye_state import EyeStateModel
from headvoice.vision.flow import FlowFallback
from headvoice.vision.landmarks import LandmarkModel
from headvoice.vision.one_euro import OneEuroFilter
from headvoice.vision.pose import HeadPoseEstimator, yaw_from_landmarks
from headvoice.vision.quality import ClaheTrigger, StabilityMeter, assess

AUTO_NEUTRAL_FRAMES = 30  # about one second at 30 fps
AUTO_NEUTRAL_MAX_JITTER = 1.0  # degrees, filtered pose
FROZEN_FRAMES = 10  # identical frames in a row before the feed counts as frozen
DRIFT_TAU_S = 20.0  # neutral follows posture drift with this time constant
DRIFT_YAW_BAND_DEG = 20.0  # yaw is not a gesture axis unless lr_axis is yaw
MAX_BOX_TRACK_FRAMES = 15  # frames the landmark model may run on a carried box without YuNet


def _iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _bbox_of(points: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0 = np.maximum(points.min(axis=0), 0)
    x1, y1 = np.minimum(points.max(axis=0), (w - 1, h - 1))
    return int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))


def _expand(bbox, w: int, h: int, frac: float = 0.25) -> tuple[int, int, int, int]:
    x, y, bw, bh = bbox
    dx, dy = int(bw * frac), int(bh * frac)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(w, x + bw + dx), min(h, y + bh + dy)
    return x0, y0, x1, y1


class VisionPipeline:
    def __init__(
        self,
        profile: Profile,
        detector: FaceDetector | None = None,
        landmark_model: LandmarkModel | None = None,
        eye_model: EyeStateModel | None | str = "default",
        max_flow_frames: int = 10,
        filter_min_cutoff: float = 1.0,
        filter_beta: float = 0.05,
    ):
        self.profile = profile
        # 0.6, not 0.8: a tilted face scores lower, and landmarks fail closed anyway.
        self.detector = detector or FaceDetector(score_threshold=0.6)
        self.landmarks = landmark_model or LandmarkModel()
        self.pose = HeadPoseEstimator()
        self.flow = FlowFallback(max_frames=max_flow_frames)
        self.filters = [OneEuroFilter(filter_min_cutoff, filter_beta) for _ in range(3)]
        # Without the trained eye model, fall back to normalized EAR (much weaker).
        self.eye_model = EyeStateModel.load_default() if eye_model == "default" else eye_model
        self.norm_l, self.norm_r = OpennessNormalizer(), OpennessNormalizer()
        self.stability = StabilityMeter()
        self.clahe_trigger = ClaheTrigger()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._last_face_t: float | None = None
        self.frame_idx = 0
        self.stage_ms: dict[str, float] = {}
        self._frame_times: deque[float] = deque(maxlen=30)
        self._neutral_samples: list[tuple[float, float, float]] = []
        self._neutral_requested = False
        self._prev_sample: np.ndarray | None = None
        self._identical_frames = 0
        self._last_model_lms: np.ndarray | None = None
        self._box_rel = (-0.1, -0.35, 1.2, 1.4)  # replaced after the first detection
        self._box_misses = 0
        self._drift_t: float | None = None
        self._eye_hist: deque[tuple[float, float]] = deque(maxlen=3)

    @property
    def fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def _tracked_box(self, w: int, h: int) -> tuple[int, int, int, int] | None:
        """Detector-style box from the last landmarks, using the last landmark-to-box offset."""
        lx, ly, lw, lh = _bbox_of(self._last_model_lms, w, h)
        rx, ry, rw, rh = self._box_rel
        x, y = int(lx + rx * lw), int(ly + ry * lh)
        bw, bh = int(rw * lw), int(rh * lh)
        x, y = max(0, x), max(0, y)
        bw, bh = min(bw, w - x), min(bh, h - y)
        return (x, y, bw, bh) if bw > 10 and bh > 10 else None

    def request_neutral(self) -> None:
        """Take the next steady model-sourced pose as neutral. Safe to call from any thread."""
        self._neutral_requested = True

    def _update_neutral(
        self, filt: tuple[float, float, float], source: str, jitter: float, t: float
    ) -> None:
        """Operator request, or auto-neutral: median of the first steady second of frames.
        Afterwards the neutral slowly follows posture drift while the head is near rest.

        Auto-neutral assumes the user starts in their normal resting position, which is how
        the app is set up. Until a neutral exists, quality is 0 and nothing can be selected.
        """
        steady = source == "model" and jitter < AUTO_NEUTRAL_MAX_JITTER
        if self._neutral_requested and source == "model":
            self._neutral_requested = False
            self._set_neutral(filt)
            return
        if self.profile.neutral_set:
            if steady:
                self._follow_drift(filt, t)
            return
        if not steady:
            return
        self._neutral_samples.append(filt)
        if len(self._neutral_samples) >= AUTO_NEUTRAL_FRAMES:
            self._set_neutral(tuple(np.median(np.array(self._neutral_samples), axis=0)))

    def _follow_drift(self, filt: tuple[float, float, float], t: float) -> None:
        """Move the neutral toward the current pose with a slow time constant, but only
        inside a band well below the tilt thresholds, so a gesture is never absorbed."""
        last, self._drift_t = self._drift_t, t
        if last is None or not 0 < t - last < 0.5:
            return
        n, p = self.profile.neutral, self.profile
        d = (filt[0] - n.yaw, filt[1] - n.pitch, filt[2] - n.roll)
        lr_half = 0.5 * min(p.tilt_deg.left, p.tilt_deg.right)
        band = (
            DRIFT_YAW_BAND_DEG if p.lr_axis == "roll" else lr_half,
            0.5 * min(p.tilt_deg.up, p.tilt_deg.down),
            lr_half if p.lr_axis == "roll" else DRIFT_YAW_BAND_DEG,
        )
        if any(abs(x) > b for x, b in zip(d, band, strict=True)):
            return
        a = (t - last) / DRIFT_TAU_S
        self.profile.neutral = Neutral(
            yaw=n.yaw + a * d[0], pitch=n.pitch + a * d[1], roll=n.roll + a * d[2]
        )

    def _set_neutral(self, angles) -> None:
        y, p, r = (float(a) for a in angles)
        self.profile.neutral = Neutral(yaw=y, pitch=p, roll=r)
        self.profile.neutral_set = True
        self._neutral_samples.clear()

    def _reset_tracking(self) -> None:
        self.pose.reset()
        for f in self.filters:
            f.reset()
        self.stability.reset()

    def process(self, bgr: np.ndarray, t: float) -> FrameFeatures:
        timer = time.perf_counter
        stage: dict[str, float] = {}
        h, w = bgr.shape[:2]
        self._frame_times.append(t)
        idx = self.frame_idx
        self.frame_idx += 1

        t0 = timer()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        raw_gray = gray  # CLAHE below works on a copy
        # Some virtual cameras repeat the last image when the source stops. A live sensor
        # always has some noise, so identical frames mean a stale feed.
        sample = gray[::8, ::8].copy()
        same = self._prev_sample is not None and np.array_equal(sample, self._prev_sample)
        self._identical_frames = self._identical_frames + 1 if same else 0
        self._prev_sample = sample
        frozen = self._identical_frames >= FROZEN_FRAMES
        clahe_on = self.clahe_trigger.active and self._last_bbox is not None
        det_in = bgr
        if clahe_on:
            x0, y0, x1, y1 = _expand(self._last_bbox, w, h)
            gray = gray.copy()
            gray[y0:y1, x0:x1] = self._clahe.apply(gray[y0:y1, x0:x1])
            det_in = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        stage["preprocess"] = (timer() - t0) * 1000

        t0 = timer()
        bbox = self.detector.detect(det_in)
        stage["detect"] = (timer() - t0) * 1000

        t0 = timer()
        lms, source = None, "none"
        box_tracked = False
        if bbox is not None:
            lms = self.landmarks.fit(gray, bbox)
            if lms is not None:
                lx, ly, lw, lh = _bbox_of(lms, w, h)
                self._box_rel = (
                    (bbox[0] - lx) / lw,
                    (bbox[1] - ly) / lh,
                    bbox[2] / lw,
                    bbox[3] / lh,
                )
        elif self._last_model_lms is not None and self._box_misses < MAX_BOX_TRACK_FRAMES:
            # YuNet misses faces rolled about 20 degrees, which is exactly a tilt gesture.
            # Refit the landmark model on a box carried from the previous landmarks.
            bbox = self._tracked_box(w, h)
            lms = self.landmarks.fit(gray, bbox) if bbox else None
            if (
                lms is not None
                and _iou(_bbox_of(lms, w, h), _bbox_of(self._last_model_lms, w, h)) < 0.5
            ):
                lms = None  # the fit wandered off: do not trust it
            box_tracked = lms is not None
        if lms is not None:
            source = "model"
            self._box_misses = self._box_misses + 1 if box_tracked else 0
            self._last_model_lms = lms
            self.flow.model_update(gray, lms)
        else:
            self._last_model_lms = None  # lost: the next box must come from the detector
        if lms is None:
            lms = self.flow.track(gray)
            if lms is not None:
                source = "flow"
                bbox = _bbox_of(lms, w, h)
        stage["landmarks"] = (timer() - t0) * 1000

        face_found = lms is not None
        if face_found:
            self._last_bbox, self._last_face_t = bbox, t
        elif self._last_face_t is None or t - self._last_face_t > 0.5:
            self._reset_tracking()

        t0 = timer()
        pnp = self.pose.estimate(lms, w, h) if face_found else None
        # Pitch and roll from solvePnP. Yaw from 2D landmark geometry: with 6 points and a
        # generic face, PnP yaw flips between two near-equal solutions (logged: 155 jumps of
        # over 8 degrees in one frame with roll and pitch unchanged).
        raw = (yaw_from_landmarks(lms), pnp[1], pnp[2]) if pnp is not None else None
        yaw = pitch = roll = None
        jitter = 0.0
        if raw is not None:
            filt = tuple(f(v, t) for f, v in zip(self.filters, raw, strict=True))
            # Stability is judged on the filtered pose, which is what the gesture logic uses.
            jitter = self.stability.update(filt)
            if not frozen:  # never learn a neutral from a stale image
                self._update_neutral(filt, source, jitter, t)
            n = self.profile.neutral
            yaw, pitch, roll = filt[0] - n.yaw, filt[1] - n.pitch, filt[2] - n.roll
        stage["pose"] = (timer() - t0) * 1000

        t0 = timer()
        open_l = open_r = None
        if face_found:
            if self.eye_model is not None:
                # Eye-region appearance (HOG + SVM), on the frame before CLAHE, as trained.
                open_l, open_r = self.eye_model.openness(raw_gray, lms)
            else:
                ear_l, ear_r = eye_ears(lms)
                open_l, open_r = self.norm_l(ear_l), self.norm_r(ear_r)
            if open_l is not None and open_r is not None:
                # 3-frame median against single-frame flicker.
                self._eye_hist.append((open_l, open_r))
                open_l, open_r = (float(v) for v in np.median(self._eye_hist, axis=0))
        else:
            self._eye_hist.clear()
        brightness = contrast = None
        if bbox is not None:
            x, y, bw, bh = bbox
            roi = cv2.cvtColor(bgr[y : y + bh, x : x + bw], cv2.COLOR_BGR2GRAY)
            if roi.size:
                brightness, contrast = float(roi.mean()), float(roi.std())
                self.clahe_trigger.update(brightness, contrast)
        quality, flags = assess(
            face_found=face_found,
            bbox=bbox,
            frame_w=w,
            brightness=brightness,
            contrast=contrast,
            yaw_from_neutral=yaw,
            roll_from_neutral=roll,
            jitter_deg=jitter,
            source=source,
            neutral_set=self.profile.neutral_set,
        )
        if frozen:
            quality, flags = 0.0, [*flags, "frozen_feed"]
        if box_tracked:
            flags = [*flags, "tracked_box"]
        stage["blink_quality"] = (timer() - t0) * 1000
        self.stage_ms = stage

        return FrameFeatures(
            t=t,
            frame_idx=idx,
            face_found=face_found,
            bbox=bbox if face_found else None,
            landmarks=[(float(px), float(py)) for px, py in lms] if face_found else None,
            landmark_source=source,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            eye_open_l=open_l,
            eye_open_r=open_r,
            quality=quality,
            quality_flags=flags,
            clahe_applied=clahe_on,
        )
