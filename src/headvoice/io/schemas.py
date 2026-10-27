"""Pydantic models shared by the vision pipeline, interaction layer, logs and agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Direction = Literal["left", "right", "up", "down"]
LayoutName = Literal["four_way", "two_choice", "binary_scan"]
DIRECTIONS: tuple[Direction, ...] = ("left", "right", "up", "down")


class FrameFeatures(BaseModel):
    """Everything the pipeline measured for one frame. Pose is relative to the profile neutral."""

    t: float  # seconds since session start (monotonic)
    frame_idx: int
    face_found: bool
    bbox: tuple[int, int, int, int] | None = None
    landmarks: list[tuple[float, float]] | None = None
    landmark_source: Literal["model", "flow", "none"] = "none"
    yaw: float | None = None  # degrees, filtered; + = user turned to their left
    pitch: float | None = None  # + = chin up
    roll: float | None = None  # + = head tilted toward the user's left shoulder
    eye_open_l: float | None = None  # normalized openness, about 1.0 when open
    eye_open_r: float | None = None
    quality: float = 0.0  # 0..1
    quality_flags: list[str] = Field(default_factory=list)
    clahe_applied: bool = False


class Neutral(BaseModel):
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class Thresholds(BaseModel):
    """Displacement from neutral, in degrees, that counts as a tilt in each direction."""

    left: float = 12.0
    right: float = 12.0
    up: float = 10.0
    down: float = 10.0


class Profile(BaseModel):
    """Per-user settings. The default profile ships with the app; calibration replaces it."""

    profile_id: str = "default"
    neutral: Neutral = Field(default_factory=Neutral)
    # False until calibration, the operator, or the pipeline's auto-neutral has set it.
    neutral_set: bool = False
    # Which pose angle drives left/right: roll (ear toward shoulder) or yaw (turn).
    lr_axis: Literal["roll", "yaw"] = "roll"
    # Flip a sign if the camera image is mirrored. Applied before thresholds.
    lr_sign: Literal[1, -1] = 1
    ud_sign: Literal[1, -1] = 1
    tilt_deg: Thresholds = Field(default_factory=Thresholds)
    # Agent-tunable (bounded): every tilt threshold is multiplied by this. Lower = smaller
    # movements count, which is the main adaptation when movements shrink with fatigue.
    tilt_scale: float = 1.0
    # Reliable range per direction from calibration; None until calibrated.
    calibrated_range_deg: Thresholds | None = None
    tilt_return_frac: float = 0.5  # must come back below this fraction of threshold
    tilt_hold_ms: int = 300
    refractory_ms: int = 600
    highlight_timeout_ms: int = 4000
    # Openness is about 1.0 open, 0.0 closed (eye-state model). An eye closes below
    # eye_closed_ratio and only reopens above eye_open_ratio (hysteresis). 0.6 / 0.8 caught
    # 24 of 24 long blinks on held-out labeled recordings (docs/models.md).
    eye_closed_ratio: float = 0.6
    eye_open_ratio: float = 0.8
    min_blink_ms: int = 60  # closures shorter than this are landmark flicker, not blinks
    long_blink_ms: int = 700
    involuntary_blink_ms_p95: float | None = None  # from calibration, informational
    double_blink_gap_ms: int = 500
    undo_gesture: Literal["double_blink", "tilt_down"] = "double_blink"
    quality_min: float = 0.5
    recover_timeout_ms: int = 3000
    idle_grace_ms: int = 300  # tracking survives dropouts shorter than this
    speak_timeout_ms: int = 6000
    layout: LayoutName = "four_way"
    # "blink": confirm with a long blink (the product). "dwell": confirm by holding the head
    # at rest for dwell_ms after a highlight (evaluation baseline B only).
    confirm_mode: Literal["blink", "dwell"] = "blink"
    dwell_ms: int = 1500

    @classmethod
    def load(cls, path: str | Path) -> Profile:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class Phrase(BaseModel):
    id: str
    text: str
    group: str


class Phrasebook(BaseModel):
    language: str
    phrases: list[Phrase]

    @classmethod
    def load(cls, path: str | Path) -> Phrasebook:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def text(self, phrase_id: str) -> str:
        for p in self.phrases:
            if p.id == phrase_id:
                return p.text
        raise KeyError(phrase_id)


class Event(BaseModel):
    """A state machine or pipeline event, written to the session log."""

    model_config = ConfigDict(extra="allow")

    type: str
    t: float
