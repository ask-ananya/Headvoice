"""Signal quality score, flags, and the CLAHE trigger."""

from collections import deque

import numpy as np

TURNED_AWAY_DEG = 35.0  # yaw from neutral, head not tilted
TURNED_AWAY_TILTED_DEG = 50.0  # yaw from neutral during a tilt
TILTED_ROLL_DEG = 8.0


class StabilityMeter:
    """Pose jitter as the mean absolute second difference of raw angles (degrees).

    Smooth deliberate motion has a small second difference; landmark noise has a large one.
    """

    def __init__(self, window: int = 10):
        self._hist: deque[tuple[float, float, float]] = deque(maxlen=3)
        self._jit: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._hist.clear()
        self._jit.clear()

    def update(self, angles: tuple[float, float, float]) -> float:
        self._hist.append(angles)
        if len(self._hist) == 3:
            a, b, c = (np.array(x) for x in self._hist)
            self._jit.append(float(np.mean(np.abs(c - 2 * b + a))))
        return float(np.mean(self._jit)) if self._jit else 0.0


class ClaheTrigger:
    """Turns CLAHE on for dark or flat faces, with hysteresis so it does not flicker."""

    def __init__(
        self,
        on_brightness: float = 70.0,
        on_contrast: float = 30.0,
        off_brightness: float = 90.0,
        off_contrast: float = 40.0,
    ):
        self.on_b, self.on_c = on_brightness, on_contrast
        self.off_b, self.off_c = off_brightness, off_contrast
        self.active = False

    def update(self, brightness: float, contrast: float) -> bool:
        if not self.active and (brightness < self.on_b or contrast < self.on_c):
            self.active = True
        elif self.active and brightness > self.off_b and contrast > self.off_c:
            self.active = False
        return self.active


def assess(
    *,
    face_found: bool,
    bbox: tuple[int, int, int, int] | None,
    frame_w: int,
    brightness: float | None,
    contrast: float | None,
    yaw_from_neutral: float | None,
    roll_from_neutral: float | None = None,
    jitter_deg: float,
    source: str,
    neutral_set: bool = True,
) -> tuple[float, list[str]]:
    """Return (score in 0..1, flags). The score is the weakest component.

    "Turned away" is measured from the user's neutral, because absolute yaw from a generic
    face model carries a per-person bias of tens of degrees. A tilt (roll) gesture brings
    some real yaw with it, and the generic model exaggerates it (logged session: tilts of
    about 20 degrees read as about 40 degrees yaw), so a tilted head gets a wider limit.
    """
    if not face_found or bbox is None:
        return 0.0, ["no_face"]
    flags: list[str] = [] if neutral_set else ["no_neutral"]
    size = min(1.0, (bbox[2] / frame_w) / 0.2)
    if size < 0.75:
        flags.append("small_face")
    light = 1.0
    if brightness is not None and brightness < 60:
        flags.append("low_light")
        light = min(light, max(0.0, brightness / 60))
    if contrast is not None and contrast < 25:
        flags.append("low_contrast")
        light = min(light, max(0.0, contrast / 25))
    # Filtered-pose jitter on a live W100 webcam: median 0.35, p90 about 1 degree.
    stability = float(np.clip(1 - (jitter_deg - 1.0) / 2.0, 0, 1))
    if stability < 0.6:
        flags.append("unstable")
    if yaw_from_neutral is None:
        flags.append("no_pose")
        angle = 0.0
    else:
        tilted = roll_from_neutral is not None and abs(roll_from_neutral) >= TILTED_ROLL_DEG
        limit = TURNED_AWAY_TILTED_DEG if tilted else TURNED_AWAY_DEG
        angle = float(np.clip(1 - (abs(yaw_from_neutral) - (limit - 10)) / 15, 0, 1))
        if abs(yaw_from_neutral) > limit:
            flags.append("turned_away")
    score = min(size, light, stability, angle) if neutral_set else 0.0
    if source == "flow":
        flags.append("flow")
        score *= 0.5
    return score, flags
