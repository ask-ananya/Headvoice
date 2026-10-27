import math

import cv2
import numpy as np
import pytest

from headvoice.vision.blink import (
    BlinkDetector,
    OpennessNormalizer,
    eye_aspect_ratio,
    eyes_state,
    long_blink_threshold_ms,
)
from headvoice.vision.flow import FlowFallback
from headvoice.vision.one_euro import OneEuroFilter
from headvoice.vision.pose import MODEL_IDX, MODEL_POINTS, HeadPoseEstimator, camera_matrix
from headvoice.vision.quality import ClaheTrigger, StabilityMeter, assess

W, H = 640, 480


def project(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Synthetic 68-point array whose pose points come from a known rotation."""
    # Inverse of rotation_to_angles: R = Rz(roll) Ry(-yaw) Rx(-pitch).
    rx, ry, rz = math.radians(-pitch), math.radians(-yaw), math.radians(roll)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    rvec, _ = cv2.Rodrigues(Rz @ Ry @ Rx)
    tvec = np.array([[0.0], [0.0], [3000.0]])
    img, _ = cv2.projectPoints(MODEL_POINTS, rvec, tvec, camera_matrix(W, H), np.zeros(4))
    pts = np.zeros((68, 2), dtype=np.float32)
    pts[MODEL_IDX] = img.reshape(-1, 2)
    return pts


@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [(0, 0, 0), (15, 0, 0), (-15, 0, 0), (0, 12, 0), (0, -12, 0), (0, 0, 15), (10, -8, 6)],
)
def test_pose_recovers_known_rotation(yaw, pitch, roll):
    est = HeadPoseEstimator()
    got = est.estimate(project(yaw, pitch, roll), W, H)
    assert got is not None
    assert got == pytest.approx((yaw, pitch, roll), abs=0.5)


def test_pose_sign_conventions():
    # Positive yaw moves the nose tip (index 30) to image right of the eye corners;
    # positive pitch moves it up relative to them.
    def nose_offset(p):
        return p[30] - (p[36] + p[45]) / 2

    base = nose_offset(project(0, 0, 0))
    assert nose_offset(project(15, 0, 0))[0] > base[0]
    assert nose_offset(project(0, 15, 0))[1] < base[1]
    # Positive roll moves the eyes' midpoint (top of face) to image right relative to chin.
    p = project(0, 0, 15)
    eyes_mid = (p[36] + p[45]) / 2
    assert eyes_mid[0] > p[8][0]


def test_one_euro_reduces_jitter_and_tracks_steps():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    noisy = 10 + rng.normal(0, 1.0, 300)
    out = [f(x, i / 30) for i, x in enumerate(noisy)]
    assert np.std(out[50:]) < 0.5 * np.std(noisy[50:])
    f.reset()
    step = [0.0] * 30 + [20.0] * 60
    out = [f(x, i / 30) for i, x in enumerate(step)]
    assert out[-1] == pytest.approx(20.0, abs=0.5)


def _eye(openness: float) -> np.ndarray:
    return np.array(
        [(0, 0), (10, -openness), (20, -openness), (30, 0), (20, openness), (10, openness)]
    )


def test_ear_open_vs_closed():
    assert eye_aspect_ratio(_eye(10)) > eye_aspect_ratio(_eye(2))
    assert eye_aspect_ratio(_eye(0)) == 0


def test_normalizer_fails_closed_until_history():
    n = OpennessNormalizer(min_samples=15)
    assert all(n(0.3) is None for _ in range(14))
    assert n(0.3) == pytest.approx(1.0)
    assert n(0.1) < 0.7


def test_blink_events_from_synthetic_series():
    d = BlinkDetector(closed_ratio=0.7)
    fps = 30
    # 6 open, 5 closed (short), 10 open, 24 closed (long), 6 open
    series = [1.0] * 6 + [0.3] * 5 + [1.0] * 10 + [0.3] * 24 + [1.0] * 6
    events = [e for i, v in enumerate(series) if (e := d.update(v, v, i / fps))]
    assert len(events) == 2
    assert events[0].duration_ms == pytest.approx(5 / fps * 1000)
    assert events[1].duration_ms == pytest.approx(24 / fps * 1000)
    assert all(e.agreed for e in events)


def test_blink_one_eye_rejected():
    d = BlinkDetector(closed_ratio=0.7)
    evs = [d.update(1.0, 1.0, 0), d.update(0.2, 1.0, 0.1), d.update(1.0, 1.0, 0.3)]
    assert evs[2] is not None and not evs[2].agreed
    assert eyes_state(0.2, 1.0, 0.7) == "one"
    assert eyes_state(0.2, 0.2, 0.7) == "both"
    assert eyes_state(None, 0.2, 0.7) == "none"


def test_long_blink_threshold_from_distribution():
    inv = [120, 150, 180, 200, 220, 250, 300]
    thr = long_blink_threshold_ms(inv, percentile=95, margin_ms=150)
    assert thr > max(inv)
    assert long_blink_threshold_ms([50, 60], floor_ms=400) == 400


def test_quality_flags():
    q, flags = assess(
        face_found=False,
        bbox=None,
        frame_w=W,
        brightness=None,
        contrast=None,
        yaw_from_neutral=None,
        jitter_deg=0,
        source="none",
    )
    assert q == 0 and "no_face" in flags
    good = dict(
        face_found=True,
        bbox=(200, 100, 200, 200),
        frame_w=W,
        brightness=120.0,
        contrast=50.0,
        yaw_from_neutral=0.0,
        jitter_deg=0.5,
        source="model",
    )
    q, flags = assess(**good)
    assert q == pytest.approx(1.0) and flags == []
    q, flags = assess(**{**good, "yaw_from_neutral": 45.0})
    assert q == 0 and "turned_away" in flags
    q, flags = assess(**{**good, "brightness": 20.0})
    assert q < 0.5 and "low_light" in flags
    q, flags = assess(**{**good, "source": "flow"})
    assert q <= 0.5 and "flow" in flags


def test_stability_meter_separates_noise_from_motion():
    m = StabilityMeter()
    smooth = [m.update((i * 1.0, 0.0, 0.0)) for i in range(20)][-1]
    m.reset()
    rng = np.random.default_rng(1)
    noisy = [m.update((float(rng.normal(0, 3)), 0.0, 0.0)) for _ in range(20)][-1]
    assert smooth < 0.1 < noisy


def test_clahe_trigger_hysteresis():
    c = ClaheTrigger()
    assert not c.update(120, 50)
    assert c.update(50, 50)
    assert c.update(80, 50)  # between on and off thresholds: stays on
    assert not c.update(100, 50)


def test_flow_fallback_is_bounded():
    rng = np.random.default_rng(2)
    img = (rng.random((H, W)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    pts = np.array([[x, y] for x in range(200, 400, 25) for y in range(150, 350, 25)], np.float32)
    ff = FlowFallback(max_frames=3)
    ff.model_update(img, pts)
    shifted = np.roll(img, 2, axis=1)
    tracked = ff.track(shifted)
    assert tracked is not None
    assert np.median(tracked[:, 0] - pts[:, 0]) == pytest.approx(2, abs=0.5)
    assert ff.track(shifted) is not None
    assert ff.track(shifted) is not None
    assert ff.track(shifted) is None  # 4th frame without the model: give up


def test_quality_zero_until_neutral_set():
    q, flags = assess(
        face_found=True,
        bbox=(200, 100, 200, 200),
        frame_w=W,
        brightness=120.0,
        contrast=50.0,
        yaw_from_neutral=0.0,
        jitter_deg=0.5,
        source="model",
        neutral_set=False,
    )
    assert q == 0 and "no_neutral" in flags


class _FakeDetector:
    def detect(self, bgr):
        return (200, 100, 200, 200)


class _FakeLandmarks:
    """Returns pose points for a face turned 40 degrees: a biased but steady 'resting' pose."""

    def __init__(self):
        self.yaw = 40.0

    def fit(self, gray, bbox):
        return project(self.yaw, 0, 0)


def test_auto_neutral_removes_per_person_yaw_bias():
    from headvoice.io.schemas import Profile
    from headvoice.vision.pipeline import AUTO_NEUTRAL_FRAMES, VisionPipeline

    lm = _FakeLandmarks()
    pipe = VisionPipeline(Profile(), detector=_FakeDetector(), landmark_model=lm)
    rng = np.random.default_rng(3)

    def img():  # fresh sensor noise each frame, so the feed is not frozen
        return rng.integers(40, 220, (H, W, 3)).astype(np.uint8)

    feats = [pipe.process(img(), i / 30) for i in range(AUTO_NEUTRAL_FRAMES + 20)]
    assert "no_neutral" in feats[0].quality_flags and feats[0].quality == 0
    last = feats[-1]
    assert pipe.profile.neutral_set
    assert pipe.profile.neutral.yaw > 15  # the biased resting yaw became the neutral
    assert last.yaw == pytest.approx(0.0, abs=1.0)
    assert "turned_away" not in last.quality_flags and last.quality > 0.5
    # Operator neutral request re-centres on the current pose.
    lm.yaw = 30.0
    for i in range(30):
        pipe.process(img(), 10 + i / 30)
    pipe.request_neutral()
    ff = pipe.process(img(), 12.0)
    ff = pipe.process(img(), 12.04)
    assert ff.yaw == pytest.approx(0.0, abs=1.0)


def test_frozen_feed_fails_closed_and_never_sets_neutral():
    from headvoice.io.schemas import Profile
    from headvoice.vision.pipeline import FROZEN_FRAMES, VisionPipeline

    pipe = VisionPipeline(Profile(), detector=_FakeDetector(), landmark_model=_FakeLandmarks())
    still = np.random.default_rng(4).integers(40, 220, (H, W, 3)).astype(np.uint8)
    feats = [pipe.process(still, i / 30) for i in range(FROZEN_FRAMES + 40)]
    assert "frozen_feed" in feats[-1].quality_flags and feats[-1].quality == 0
    assert not pipe.profile.neutral_set


def test_yaw_from_landmarks_sign_monotonic_and_roll_invariant():
    from headvoice.vision.pose import yaw_from_landmarks

    ys = [yaw_from_landmarks(project(y, 0, 0)) for y in (-30, -15, 0, 15, 30)]
    assert ys[2] == pytest.approx(0, abs=0.5)
    assert all(a < b for a, b in zip(ys, ys[1:], strict=False))  # same sign as PnP yaw
    # A pure tilt (roll) does not read as a turn.
    assert yaw_from_landmarks(project(0, 0, 20)) == pytest.approx(0, abs=0.5)
    assert yaw_from_landmarks(project(15, 0, 20)) == pytest.approx(ys[3], abs=1.0)


def test_neutral_follows_slow_drift_but_not_a_held_tilt():
    from headvoice.io.schemas import Profile
    from headvoice.vision.pipeline import AUTO_NEUTRAL_FRAMES, VisionPipeline

    class Lm:
        roll = 0.0

        def fit(self, gray, bbox):
            return project(0, 0, self.roll)

    lm = Lm()
    pipe = VisionPipeline(Profile(), detector=_FakeDetector(), landmark_model=lm)
    rng = np.random.default_rng(5)

    def run(seconds, t0):
        out = None
        for i in range(int(seconds * 30)):
            out = pipe.process(rng.integers(40, 220, (H, W, 3)).astype(np.uint8), t0 + i / 30)
        return out

    run(AUTO_NEUTRAL_FRAMES / 30 + 0.5, 0)
    lm.roll = -4.0  # slow posture lean, inside the drift band
    ff = run(60, 10)
    assert abs(ff.roll) < 1.0  # absorbed after 3 time constants
    lm.roll = -4.0 + 20.0  # a held tilt, well outside the band
    ff = run(30, 80)
    assert ff.roll == pytest.approx(20.0, abs=1.5)  # not absorbed


def test_eye_state_model_trains_and_orients_on_synthetic_eyes():
    from headvoice.vision.eye_state import CROP_H, CROP_W, EyeStateModel, train

    rng = np.random.default_rng(6)

    def open_eye():
        c = rng.normal(150, 12, (CROP_H, CROP_W))
        cv2.circle(c, (CROP_W // 2, CROP_H // 2), 7, 40, -1)  # dark iris
        cv2.ellipse(c, (CROP_W // 2, CROP_H // 2), (20, 9), 0, 0, 360, 90, 1)
        return np.clip(c, 0, 255).astype(np.uint8)

    def closed_eye():
        c = rng.normal(150, 12, (CROP_H, CROP_W))
        cv2.line(c, (4, CROP_H // 2 + 2), (CROP_W - 4, CROP_H // 2 + 2), 70, 2)  # lid line
        return np.clip(c, 0, 255).astype(np.uint8)

    crops = [open_eye() for _ in range(60)] + [closed_eye() for _ in range(60)]
    svm, meta = train(crops, [False] * 60 + [True] * 60)
    assert meta["closed_ref"] > meta["open_ref"]
    model = EyeStateModel.__new__(EyeStateModel)
    model.svm, model.meta = svm, meta
    from headvoice.vision.eye_state import hog_features

    def openness(c):
        raw = svm.predict(
            np.array([hog_features(c)], np.float32), flags=cv2.ml.STAT_MODEL_RAW_OUTPUT
        )[1]
        v = float(raw.ravel()[0]) * meta["sign"]
        return (meta["closed_ref"] - v) / (meta["closed_ref"] - meta["open_ref"])

    assert openness(open_eye()) > 0.8 and openness(closed_eye()) < 0.2


def test_eye_crop_is_roll_invariant():
    from headvoice.vision.eye_state import eye_crop

    img = np.zeros((200, 200), np.uint8)
    cv2.ellipse(img, (100, 100), (30, 12), 0, 0, 360, 255, -1)
    flat = eye_crop(img, np.array([70.0, 100.0]), np.array([130.0, 100.0]))
    rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((100, 100), 25, 1.0), (200, 200))
    m = cv2.getRotationMatrix2D((100, 100), 25, 1.0)
    p1 = m @ np.array([70.0, 100.0, 1.0])
    p2 = m @ np.array([130.0, 100.0, 1.0])
    tilted = eye_crop(rot, p1, p2)
    assert np.abs(flat.astype(int) - tilted.astype(int)).mean() < 12
