"""Tilt and blink extraction with debounce. Pure functions over small frozen records."""

from __future__ import annotations

from dataclasses import dataclass, replace

from headvoice.io.schemas import DIRECTIONS, Direction, FrameFeatures, Profile


def displacements(f: FrameFeatures, p: Profile) -> dict[Direction, float] | None:
    """Displacement from neutral toward each direction, in degrees (positive = toward it)."""
    lr_raw = f.roll if p.lr_axis == "roll" else f.yaw
    if lr_raw is None or f.pitch is None:
        return None
    lr = p.lr_sign * lr_raw  # + = toward the user's left
    ud = p.ud_sign * f.pitch  # + = up
    return {"left": lr, "right": -lr, "up": ud, "down": -ud}


@dataclass(frozen=True)
class TiltTrack:
    """idle -> out (past threshold) -> counted (held long enough) -> idle (returned)."""

    phase: str = "idle"
    out_since: float | None = None
    last_count: float = float("-inf")


def lock_tilts(tracks: tuple[TiltTrack, ...]) -> tuple[TiltTrack, ...]:
    """On a bad frame, any tilt in progress must return to neutral before it can count."""
    return tuple(t if t.phase == "idle" else replace(t, phase="counted") for t in tracks)


def update_tilts(
    tracks: tuple[TiltTrack, ...], disp: dict[Direction, float], p: Profile, now: float
) -> tuple[tuple[TiltTrack, ...], Direction | None]:
    """Advance each direction's tracker. Returns at most one counted direction."""
    out: list[TiltTrack] = []
    counted: list[tuple[float, Direction]] = []
    for d, tr in zip(DIRECTIONS, tracks, strict=True):
        thr = getattr(p.tilt_deg, d) * p.tilt_scale
        x = disp[d]
        if tr.phase == "idle":
            tr = replace(tr, phase="out", out_since=now) if x >= thr else tr
        elif tr.phase == "out":
            if x < thr:
                tr = replace(tr, phase="idle", out_since=None)
            elif (now - tr.out_since) * 1000 >= p.tilt_hold_ms and (
                now - tr.last_count
            ) * 1000 >= p.refractory_ms:
                tr = replace(tr, phase="counted", last_count=now)
                counted.append((x / thr, d))
        elif tr.phase == "counted" and x < p.tilt_return_frac * thr:
            tr = replace(tr, phase="idle", out_since=None)
        out.append(tr)
    if not counted:
        return tuple(out), None
    # If two directions count on the same frame (diagonal), keep the stronger one only.
    return tuple(out), max(counted)[1]


INITIAL_TILTS: tuple[TiltTrack, ...] = tuple(TiltTrack() for _ in DIRECTIONS)
