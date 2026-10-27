"""Session monitor: rolling window summaries of what OpenCV measured, for the agents.

Only numbers and short labels go into a summary (checked by io/privacy.check_outbound).
A summary is due every `period_s`, and immediately on a notable change: a quality drop,
an amplitude decline or an undo spike.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from headvoice.calibration.analyze import THRESHOLD_FRAC
from headvoice.interaction.gestures import displacements
from headvoice.io.schemas import DIRECTIONS, FrameFeatures, Profile

# A tilt attempt: past this fraction of the threshold, held at least ATTEMPT_MIN_S. Shorter or
# smaller movements (glances, nods while blinking) are not attempts.
ATTEMPT_FRAC = 0.5
ATTEMPT_MIN_S = 0.2
WARMUP_S = 60.0  # no adaptation (except for poor signal) in the first minute of a session
MIN_ATTEMPTS = 6  # the amplitude trigger needs at least this many attempts in 60 s


@dataclass
class Attempt:
    t: float
    direction: str
    peak_deg: float
    counted: bool


class SessionMonitor:
    def __init__(self, window_s: float = 60.0, period_s: float = 60.0, every_selections: int = 5):
        self.window_s = window_s
        self.period_s = period_s
        self.every_selections = every_selections
        self.frames: deque[tuple[float, float, list[str], str]] = deque()  # t, quality, flags, src
        self.events: deque[dict] = deque()
        self.attempts: deque[Attempt] = deque()
        self._open: dict[str, tuple[float, float, float, float]] = {}  # start, peak, held s, last t
        self._first_t: float | None = None
        self._last_summary_t: float | None = None
        self._selections_since = 0
        self.notable: str | None = None

    # Inputs -----------------------------------------------------------------------------
    def add_frame(self, f: FrameFeatures, p: Profile) -> None:
        if self._first_t is None:
            self._first_t = f.t
        self.frames.append((f.t, f.quality, list(f.quality_flags), f.landmark_source))
        disp = displacements(f, p) if f.face_found else None
        for d in DIRECTIONS:
            thr = getattr(p.tilt_deg, d) * p.tilt_scale
            x = disp[d] if disp else 0.0
            if d in self._open:
                start, peak, held, last = self._open[d]
                if x >= ATTEMPT_FRAC * thr:
                    held += f.t - last
                if x < ATTEMPT_FRAC * thr * 0.5:
                    # Excursion ended: an attempt only if it was held long enough.
                    if held >= ATTEMPT_MIN_S:
                        self.attempts.append(Attempt(start, d, peak, peak >= thr))
                    del self._open[d]
                else:
                    self._open[d] = (start, max(peak, x), held, f.t)
            elif x >= ATTEMPT_FRAC * thr:
                self._open[d] = (f.t, x, 0.0, f.t)
        self._trim(f.t)

    def add_event(self, ev: dict) -> None:
        self.events.append(ev)
        if ev["type"] == "speak":
            self._selections_since += 1

    def _trim(self, now: float) -> None:
        lo = now - max(self.window_s, 300.0)
        while self.frames and self.frames[0][0] < lo:
            self.frames.popleft()
        while self.events and self.events[0]["t"] < lo:
            self.events.popleft()
        while self.attempts and self.attempts[0].t < lo:
            self.attempts.popleft()

    # Summary ----------------------------------------------------------------------------
    def summary(self, now: float, p: Profile, seconds: float | None = None) -> dict:
        w = seconds or self.window_s
        lo = now - w
        fr = [x for x in self.frames if x[0] >= lo]
        ev = [e for e in self.events if e["t"] >= lo]
        at = [a for a in self.attempts if a.t >= lo]
        count = lambda kind: sum(1 for e in ev if e["type"] == kind)  # noqa: E731
        q = np.array([x[1] for x in fr]) if fr else np.array([0.0])
        flags: dict[str, float] = {}
        for x in fr:
            for fl in x[2]:
                flags[fl] = flags.get(fl, 0) + 1
        amp = {}
        for d in DIRECTIONS:
            peaks = [a.peak_deg for a in at if a.direction == d]
            if peaks:
                amp[d] = round(float(np.median(peaks)) / _calibrated_range(p, d), 2)
        sel_times = _selection_times(ev)
        armed, confirmed = count("armed"), count("confirmed")
        return {
            "seconds": round(w),
            "selections": count("speak"),
            "undos": count("undo"),
            "timeouts": count("highlight_timeout"),
            "tilt_attempts": len(at),
            "tilt_near_misses": sum(1 for a in at if not a.counted),
            "tilt_amplitude_vs_calibrated": amp,
            "amplitude_trend_per_min": round(self._trend(now, p), 3),
            "mean_selection_time_s": round(float(np.mean(sel_times)), 1) if sel_times else None,
            "quality_mean": round(float(q.mean()), 2),
            "quality_flags": {k: round(v / max(len(fr), 1), 2) for k, v in flags.items()},
            "long_blink_success_rate": round(confirmed / armed, 2) if armed else None,
            "flow_fallback_fraction": round(
                sum(1 for x in fr if x[3] == "flow") / max(len(fr), 1), 3
            ),
        }

    def _trend(self, now: float, p: Profile) -> float:
        """Slope of attempt amplitude (fraction of calibrated range) per minute, last 5 min."""
        pts = [
            (a.t, a.peak_deg / _calibrated_range(p, a.direction))
            for a in self.attempts
            if a.t >= now - 300
        ]
        if len(pts) < 4:
            return 0.0
        t, y = np.array(pts).T
        if np.ptp(t) < 20:
            return 0.0
        return float(np.polyfit(t / 60.0, y, 1)[0])

    def due(self, now: float, p: Profile) -> str | None:
        """Reason a summary should be sent now, or None."""
        if self._last_summary_t is None:
            self._last_summary_t = now
            return None
        if now - self._last_summary_t < 10.0:
            return None  # at most one summary per 10 s
        s = self.summary(now, p, seconds=20)
        reason = None
        warming_up = self._first_t is None or now - self._first_t < WARMUP_S
        if s["quality_mean"] < 0.6 and len([x for x in self.frames if x[0] >= now - 20]) > 30:
            reason = "quality_drop"  # a signal problem may be handled at any time
        elif warming_up:
            return None  # too little evidence to change anything yet
        elif s["undos"] >= 3:
            reason = "undo_spike"
        elif (
            s["tilt_near_misses"] >= 3
            and self.summary(now, p, seconds=60)["tilt_attempts"] >= MIN_ATTEMPTS
        ):
            reason = "amplitude_decline"
        elif self._selections_since >= self.every_selections:
            reason = "selections"
        elif now - self._last_summary_t >= self.period_s:
            reason = "period"
        if reason:
            self._last_summary_t = now
            self._selections_since = 0
        return reason

    def direction_reliability(self, now: float, p: Profile) -> dict:
        """Per-direction counted / attempted this session (tool: get_direction_reliability)."""
        out = {}
        for d in DIRECTIONS:
            at = [a for a in self.attempts if a.direction == d]
            out[d] = {
                "attempts": len(at),
                "counted": sum(a.counted for a in at),
                "success_rate": round(sum(a.counted for a in at) / len(at), 2) if at else None,
            }
        return out


def _calibrated_range(p: Profile, d: str) -> float:
    if p.calibrated_range_deg is not None:
        return max(getattr(p.calibrated_range_deg, d), 1.0)
    return getattr(p.tilt_deg, d) / THRESHOLD_FRAC  # default profile: threshold implies range


def _selection_times(ev: list[dict]) -> list[float]:
    """Seconds from the first highlight after a reset to the speak that ends it."""
    out, start = [], None
    for e in ev:
        if e["type"] == "highlight" and start is None:
            start = e["t"]
        elif e["type"] == "speak" and start is not None:
            out.append(e["t"] - start)
            start = None
        elif e["type"] in ("undo", "highlight_timeout", "recovery_timeout"):
            start = None
    return out
