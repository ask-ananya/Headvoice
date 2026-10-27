import numpy as np
import pytest

from headvoice.calibration.analyze import analyze, compare_profiles
from headvoice.calibration.protocol import STEPS, label_at, total_seconds
from headvoice.io.schemas import FrameFeatures, Profile

FPS = 30


def clip(ranges=None, jitter=0.4, inv_blink_s=0.15, closed_open=0.1, seed=0, lr_axis="roll"):
    """Synthetic calibration clip following the protocol."""
    ranges = ranges or {"left": 20, "right": 18, "up": 14, "down": 12}
    rng = np.random.default_rng(seed)
    frames = []
    n = int(total_seconds() * FPS)
    blink_times = [0.8, 2.2]  # within the "natural" step
    nat_start = sum(s.seconds for s in STEPS[: [s.label for s in STEPS].index("natural")])
    for i in range(n):
        t = i / FPS
        lab, idx, _ = label_at(t)
        step_start = sum(s.seconds for s in STEPS[:idx])
        ramp = min(1.0, (t - step_start) / 0.4)
        yaw, pitch, roll = (float(v) for v in rng.normal(0, jitter, 3))
        lr = roll if lr_axis == "roll" else yaw
        if lab == "left":
            lr += ranges["left"] * ramp
        elif lab == "right":
            lr -= ranges["right"] * ramp
        elif lab == "up":
            pitch += ranges["up"] * ramp
        elif lab == "down":
            pitch -= ranges["down"] * ramp
        if lr_axis == "roll":
            roll = lr
        else:
            yaw = lr
        eye = 1.0 + float(rng.normal(0, 0.05))
        if lab == "closed" and t - step_start > 0.25:
            eye = closed_open
        if lab == "natural" and any(0 <= t - nat_start - b < inv_blink_s for b in blink_times):
            eye = closed_open
        frames.append(
            FrameFeatures(
                t=t,
                frame_idx=i,
                face_found=True,
                landmark_source="model",
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                eye_open_l=eye,
                eye_open_r=eye,
                quality=0.9,
            )
        )
    return frames


def test_full_range_gives_four_way_and_thresholds_from_range():
    res = analyze(clip(), Profile())
    assert res.ok and res.profile.layout == "four_way"
    assert res.profile.tilt_deg.left == pytest.approx(0.55 * 20, abs=1.0)
    assert res.profile.tilt_deg.down == pytest.approx(0.55 * 12, abs=1.0)
    assert res.profile.neutral_set and res.profile.lr_axis == "roll"


def test_limited_up_down_gives_binary_scan_with_tilt_down_undo():
    res = analyze(clip(ranges={"left": 18, "right": 18, "up": 3, "down": 12}), Profile())
    assert res.profile.layout == "binary_scan"
    assert not res.report["directions"]["up"]["reliable"]
    assert res.profile.undo_gesture == "tilt_down"


def test_no_left_right_keeps_base_and_warns():
    res = analyze(clip(ranges={"left": 2, "right": 2, "up": 14, "down": 12}), Profile())
    assert not res.ok and "left_right_not_reliable" in res.warnings
    assert res.profile.layout == Profile().layout


def test_noisy_user_needs_more_range():
    # Jitter 2 deg -> a direction must exceed 12 deg; 8 deg tilts are not reliable.
    res = analyze(clip(ranges={"left": 8, "right": 8, "up": 8, "down": 8}, jitter=2.0), Profile())
    assert res.report["directions"]["left"]["needed_deg"] == pytest.approx(12, abs=1.5)
    assert not res.report["directions"]["left"]["reliable"]


def test_yaw_user_gets_yaw_axis():
    res = analyze(clip(lr_axis="yaw"), Profile())
    assert res.profile.lr_axis == "yaw"


def test_long_blink_between_involuntary_and_deliberate():
    res = analyze(clip(inv_blink_s=0.2), Profile())
    inv = res.report["involuntary_blinks_ms"]
    deliberate = res.report["deliberate_blinks_ms"]
    assert inv and deliberate
    assert max(inv) < res.profile.long_blink_ms < min(deliberate)


def test_eye_thresholds_follow_user_levels():
    res = analyze(clip(closed_open=0.3), Profile())
    assert 0.3 < res.profile.eye_closed_ratio < res.profile.eye_open_ratio < 1.0


def test_stability_report():
    a = analyze(clip(seed=1), Profile()).profile
    b = analyze(clip(seed=2), Profile()).profile
    cmp = compare_profiles(a, b)
    assert cmp["layout_same"] and max(cmp["tilt_deg_diff"].values()) < 1.5


def test_too_few_frames_is_rejected():
    res = analyze(clip()[:10], Profile())
    assert not res.ok and res.profile == Profile()
