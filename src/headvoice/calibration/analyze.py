"""Calibration analysis: labeled FrameFeatures from the guided clip -> a user Profile.

Shared by local calibration and the calibrate Lambda, so both give the same profile.
All rules are deterministic and documented here and in the report:

- Neutral: median pose over the settled "neutral" steps.
- Jitter: per-axis standard deviation over the neutral steps.
- Per-direction range: 90th percentile of displacement toward that direction during its
  step. A direction is reliable if its range is at least max(MIN_RANGE_DEG,
  JITTER_K * jitter) and at least REACHED_FRAC of the step's settled frames passed half of
  that range (reached consistently, not a single spike).
- Tilt threshold: THRESHOLD_FRAC of the range, clamped to [MIN_THR_DEG, MAX_THR_DEG].
- Left/right axis: roll or yaw, whichever moves more relative to its jitter.
- Layout: 4 reliable directions -> four_way; left and right only -> binary_scan;
  otherwise the base layout is kept and the result is flagged.
- Long blink: above the user's involuntary blinks (95th percentile + 150 ms, at least
  400 ms) and below most deliberate closures (80% of their median), clamped to
  [400, 1200] ms.
- Eye thresholds: between the user's open and closed openness levels (close at 45% of the
  gap above closed, reopen at 70%), only if the gap is large enough to trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from headvoice.calibration.protocol import label_at
from headvoice.io.schemas import FrameFeatures, Neutral, Profile, Thresholds
from headvoice.vision.blink import long_blink_threshold_ms

MIN_RANGE_DEG = 8.0
JITTER_K = 6.0
REACHED_FRAC = 0.4
THRESHOLD_FRAC = 0.55
MIN_THR_DEG, MAX_THR_DEG = 6.0, 20.0
MIN_EYE_GAP = 0.3


@dataclass
class CalibrationResult:
    profile: Profile
    report: dict = field(default_factory=dict)
    ok: bool = True
    warnings: list[str] = field(default_factory=list)


def _closures(t: np.ndarray, closed: np.ndarray) -> list[float]:
    """Durations (ms) of runs where both eyes are closed."""
    out, start = [], None
    for ti, c in zip(t, closed, strict=True):
        if c and start is None:
            start = ti
        elif not c and start is not None:
            out.append((ti - start) * 1000)
            start = None
    return out


def analyze(
    frames: list[FrameFeatures], base: Profile, labels: list[tuple[str, int, bool]] | None = None
) -> CalibrationResult:
    """frames: the clip's features with t measured from the clip start (or labels given)."""
    if labels is None:
        labels = [label_at(f.t) or ("after", -1, False) for f in frames]
    rows = [
        (f, lab)
        for f, lab in zip(frames, labels, strict=True)
        if lab[2] and f.face_found and f.landmark_source == "model" and f.roll is not None
    ]
    warnings: list[str] = []
    if len(rows) < 30:
        return CalibrationResult(base, {"frames_used": len(rows)}, False, ["too_few_usable_frames"])

    def arr(label: str, attr: str) -> np.ndarray:
        return np.array([getattr(f, attr) for f, lab in rows if lab[0] == label], dtype=float)

    # Neutral and jitter.
    neu = {a: arr("neutral", a) for a in ("yaw", "pitch", "roll")}
    if min(len(v) for v in neu.values()) < 10:
        return CalibrationResult(base, {}, False, ["no_neutral_frames"])
    offset = {a: float(np.median(v)) for a, v in neu.items()}
    jitter = {a: float(np.std(v)) for a, v in neu.items()}
    n = base.neutral
    neutral = Neutral(
        yaw=n.yaw + offset["yaw"], pitch=n.pitch + offset["pitch"], roll=n.roll + offset["roll"]
    )

    # Left/right axis: roll or yaw, whichever has the better range-to-jitter ratio.
    def lr_range(axis: str) -> float:
        left = arr("left", axis) - offset[axis]
        right = arr("right", axis) - offset[axis]
        if len(left) == 0 or len(right) == 0:
            return 0.0
        return float(np.percentile(np.abs(left), 90) + np.percentile(np.abs(right), 90)) / 2

    snr = {a: lr_range(a) / max(jitter[a], 0.3) for a in ("roll", "yaw")}
    lr_axis = "roll" if snr["roll"] >= snr["yaw"] else "yaw"
    # Sign: which way does "left" move the chosen axis?
    left_mean = (
        float(np.median(arr("left", lr_axis) - offset[lr_axis]))
        if len(arr("left", lr_axis))
        else 1.0
    )
    lr_sign = 1 if left_mean >= 0 else -1
    up_mean = (
        float(np.median(arr("up", "pitch") - offset["pitch"])) if len(arr("up", "pitch")) else 1.0
    )
    ud_sign = 1 if up_mean >= 0 else -1

    def disp(label: str) -> np.ndarray:
        if label in ("left", "right"):
            v = lr_sign * (arr(label, lr_axis) - offset[lr_axis])
            return v if label == "left" else -v
        v = ud_sign * (arr(label, "pitch") - offset["pitch"])
        return v if label == "up" else -v

    directions = {}
    thresholds = base.tilt_deg.model_dump()
    for d in ("left", "right", "up", "down"):
        v = disp(d)
        axis = lr_axis if d in ("left", "right") else "pitch"
        if len(v) == 0:
            directions[d] = {"range_deg": 0.0, "reliable": False, "frames": 0}
            continue
        rng = float(np.percentile(v, 90))
        need = max(MIN_RANGE_DEG, JITTER_K * jitter[axis])
        reached = float((v > 0.5 * rng).mean()) if rng > 0 else 0.0
        reliable = rng >= need and reached >= REACHED_FRAC
        directions[d] = {
            "range_deg": round(rng, 1),
            "needed_deg": round(need, 1),
            "reached_frac": round(reached, 2),
            "reliable": reliable,
            "frames": len(v),
        }
        if reliable:
            thresholds[d] = round(float(np.clip(THRESHOLD_FRAC * rng, MIN_THR_DEG, MAX_THR_DEG)), 1)

    rel = {d for d, v in directions.items() if v["reliable"]}
    if rel == {"left", "right", "up", "down"}:
        layout = "four_way"
    elif {"left", "right"} <= rel:
        layout = "binary_scan"
    else:
        layout = base.layout
        warnings.append("left_right_not_reliable")

    # Eyes. Openness levels for open frames (neutral and open steps) and closed steps.
    def eyes(label: str) -> np.ndarray:
        return np.array(
            [
                min(f.eye_open_l, f.eye_open_r)
                for f, lab in rows
                if lab[0] == label and f.eye_open_l is not None and f.eye_open_r is not None
            ]
        )

    open_vals = np.concatenate([eyes("neutral"), eyes("open")])
    closed_vals = eyes("closed")
    eye_closed_ratio, eye_open_ratio = base.eye_closed_ratio, base.eye_open_ratio
    eye_report: dict = {}
    if len(open_vals) and len(closed_vals):
        o, c = float(np.median(open_vals)), float(np.median(closed_vals))
        eye_report = {"open_level": round(o, 2), "closed_level": round(c, 2)}
        if o - c >= MIN_EYE_GAP:
            eye_closed_ratio = round(c + 0.45 * (o - c), 2)
            eye_open_ratio = round(c + 0.70 * (o - c), 2)
        else:
            warnings.append("eyes_not_separable")
    else:
        warnings.append("no_eye_frames")

    # Blink durations from the whole clip (all frames, not only settled ones).
    t_all = np.array([f.t for f in frames])
    closed_all = np.array(
        [
            f.eye_open_l is not None
            and f.eye_open_r is not None
            and max(f.eye_open_l, f.eye_open_r) < eye_closed_ratio
            for f in frames
        ]
    )
    lab_all = [lab[0] for lab in labels]
    nat = np.array([lab == "natural" for lab in lab_all])
    inv = [d for d in _closures(t_all[nat], closed_all[nat]) if d >= base.min_blink_ms]
    cls = np.array([lab in ("closed", "open") for lab in lab_all])
    deliberate = [d for d in _closures(t_all[cls], closed_all[cls]) if d >= 300]
    long_ms = long_blink_threshold_ms(inv) if inv else float(base.long_blink_ms)
    if deliberate:
        long_ms = min(long_ms, 0.8 * float(np.median(deliberate)))
    else:
        warnings.append("no_deliberate_blinks_seen")
    long_ms = int(np.clip(long_ms, 400, 1200))

    profile = base.model_copy(
        update={
            "profile_id": f"{base.profile_id.split(':')[0]}:calibrated",
            "neutral": neutral,
            "neutral_set": True,
            "lr_axis": lr_axis,
            "lr_sign": lr_sign,
            "ud_sign": ud_sign,
            "tilt_deg": Thresholds(**thresholds),
            "tilt_scale": 1.0,
            "calibrated_range_deg": Thresholds(
                **{d: max(v["range_deg"], 0.1) for d, v in directions.items()}
            ),
            "layout": layout,
            "eye_closed_ratio": eye_closed_ratio,
            "eye_open_ratio": eye_open_ratio,
            "long_blink_ms": long_ms,
            "involuntary_blink_ms_p95": round(float(np.percentile(inv, 95)), 0) if inv else None,
            "undo_gesture": "tilt_down"
            if layout != "four_way" and "down" in rel
            else "double_blink",
        }
    )
    report = {
        "frames_used": len(rows),
        "jitter_deg": {k: round(v, 2) for k, v in jitter.items()},
        "lr_axis_snr": {k: round(v, 1) for k, v in snr.items()},
        "directions": directions,
        "layout": layout,
        "eyes": eye_report,
        "involuntary_blinks_ms": [round(d) for d in inv],
        "deliberate_blinks_ms": [round(d) for d in deliberate],
        "long_blink_ms": long_ms,
    }
    return CalibrationResult(profile, report, "left_right_not_reliable" not in warnings, warnings)


def compare_profiles(a: Profile, b: Profile) -> dict:
    """Stability check: how much two calibrations of the same person differ."""
    return {
        "layout_same": a.layout == b.layout,
        "lr_axis_same": a.lr_axis == b.lr_axis,
        "tilt_deg_diff": {
            d: round(abs(getattr(a.tilt_deg, d) - getattr(b.tilt_deg, d)), 1)
            for d in ("left", "right", "up", "down")
        },
        "long_blink_ms_diff": abs(a.long_blink_ms - b.long_blink_ms),
        "eye_closed_ratio_diff": round(abs(a.eye_closed_ratio - b.eye_closed_ratio), 2),
    }
