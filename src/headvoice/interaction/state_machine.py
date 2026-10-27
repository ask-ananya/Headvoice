"""Gesture state machine. Code decides what counts as a gesture; no agent output reaches here.

    step(state, features, profile, now) -> (new_state, events)

States: idle, tracking, highlighting, armed, speaking, recovering (plan section 5.1).
The only path to a "speak" event is armed -> speaking after both eyes stay closed for
profile.long_blink_ms on a single highlighted item, with every frame usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from headvoice.interaction.gestures import (
    INITIAL_TILTS,
    TiltTrack,
    displacements,
    lock_tilts,
    update_tilts,
)
from headvoice.interaction.layouts import Menu
from headvoice.io.schemas import FrameFeatures, LayoutName, Profile

IDLE, TRACKING, HIGHLIGHTING, ARMED, SPEAKING, RECOVERING = (
    "idle",
    "tracking",
    "highlighting",
    "armed",
    "speaking",
    "recovering",
)
NONE_OF_THESE = "none_of_these"
HEAD_DOWN_FRAC = 0.5  # eye closure is ignored past this fraction of the down threshold


@dataclass(frozen=True)
class MachineState:
    state: str = IDLE
    menu: Menu = field(default_factory=lambda: Menu.create("four_way", []))
    tilts: tuple[TiltTrack, ...] = INITIAL_TILTS
    closed_since: float | None = None  # both eyes closed since
    none_fired: bool = False  # "none of these" already fired for this closure
    last_short_blink_end: float | None = None
    one_eye_logged: bool = False
    highlight_since: float | None = None
    recover_since: float | None = None
    speaking_since: float | None = None
    speaking_item: str | None = None
    eye_closed: tuple[bool, bool] = (False, False)  # per-eye state, with hysteresis
    bad_since: float | None = None  # first unusable frame of the current dropout
    need_eyes_open: bool = False  # after a dropout or speech, eyes must open before a blink counts


def initial_state(layout: LayoutName, items: list[str]) -> MachineState:
    return MachineState(menu=Menu.create(layout, items))


def usable(f: FrameFeatures, p: Profile) -> tuple[bool, str]:
    """Fail closed: only good, model-sourced, facing-forward frames can drive a gesture."""
    if not f.face_found:
        return False, "no_face"
    if f.landmark_source != "model":
        return False, f"landmarks_{f.landmark_source}"
    if "turned_away" in f.quality_flags:
        return False, "turned_away"
    if f.quality < p.quality_min:
        return False, "low_quality"
    if f.yaw is None or f.pitch is None or f.roll is None:
        return False, "no_pose"
    return True, ""


def _eyes(s: MachineState, f: FrameFeatures, p: Profile) -> tuple[str, tuple[bool, bool]]:
    """Per-eye closed state with hysteresis: an eye closes below eye_closed_ratio and only
    reopens above eye_open_ratio, so LBF flicker near one threshold does not split a blink."""
    if f.eye_open_l is None or f.eye_open_r is None:
        return "none", (False, False)
    if f.pitch is not None and -p.ud_sign * f.pitch > HEAD_DOWN_FRAC * p.tilt_deg.down:
        # Looking down lowers the eyelids, so open eyes can read as closed: fail closed.
        return "none", (False, False)
    closed = tuple(
        v < (p.eye_open_ratio if was else p.eye_closed_ratio)
        for v, was in zip((f.eye_open_l, f.eye_open_r), s.eye_closed, strict=True)
    )
    if all(closed):
        return "both", closed
    return ("one" if any(closed) else "none"), closed


def step(
    s: MachineState, f: FrameFeatures, p: Profile, now: float
) -> tuple[MachineState, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []

    def emit(kind: str, **kw: Any) -> None:
        events.append({"type": kind, "t": now, **kw})

    def go(ns: MachineState, to: str, **kw: Any) -> MachineState:
        if ns.state != to:
            emit("transition", frm=ns.state, to=to, **kw)
        return replace(ns, state=to)

    # Speaking: gestures are ignored until speech ends (or a safety timeout).
    if s.state == SPEAKING:
        if s.speaking_since is not None and (now - s.speaking_since) * 1000 >= p.speak_timeout_ms:
            emit("speech_timeout", item=s.speaking_item)
            return _after_speech(s, go), events
        return s, events

    ok, why = usable(f, p)
    if not ok:
        # Every bad frame stops gesture progress: tilts must return to neutral, any closure
        # ends, and the eyes must be seen open again before a blink can count.
        bad_since = s.bad_since if s.bad_since is not None else now
        ns = replace(
            s,
            tilts=lock_tilts(s.tilts),
            closed_since=None,
            none_fired=False,
            eye_closed=(False, False),
            need_eyes_open=True,
            bad_since=bad_since,
        )
        if s.state == ARMED:
            emit("arm_cancelled", reason=why, item=s.menu.highlighted_item)
        if s.state in (HIGHLIGHTING, ARMED):
            emit("signal_lost", reason=why)
            ns = go(replace(ns, recover_since=now), RECOVERING, reason=why)
        elif s.state == TRACKING and (now - bad_since) * 1000 >= p.idle_grace_ms:
            ns = go(ns, IDLE, reason=why)
        elif s.state == RECOVERING and (now - s.recover_since) * 1000 >= p.recover_timeout_ms:
            emit("recovery_timeout")
            ns = go(replace(ns, menu=ns.menu.reset(), highlight_since=None), IDLE)
        return ns, events

    # Usable frame.
    s = replace(s, bad_since=None)
    if s.state == IDLE:
        return go(s, TRACKING), events
    if s.state == RECOVERING:
        # Signal is back within the recovery timeout: keep the highlight. Speaking still
        # needs a fresh, complete long blink on good frames (need_eyes_open is set).
        emit("signal_restored", item=s.menu.highlighted_item)
        ns = replace(s, recover_since=None)
        keep = s.menu.highlighted is not None or s.menu.narrowed
        return go(ns, HIGHLIGHTING if keep else TRACKING), events

    ns = s
    # 1. Eyes.
    eyes, eye_closed = _eyes(ns, f, p)
    ns = replace(ns, eye_closed=eye_closed)
    if ns.need_eyes_open:
        if eyes == "none":
            ns = replace(ns, need_eyes_open=False)
        eyes = "none" if eyes == "both" else eyes  # a closure that began earlier does not count
    undo = False
    if eyes == "both":
        closed_since = ns.closed_since if ns.closed_since is not None else now
        ns = replace(ns, closed_since=closed_since)
        dur_ms = (now - closed_since) * 1000
        item = ns.menu.highlighted_item
        if ns.state == HIGHLIGHTING and item is not None and p.confirm_mode == "blink":
            emit("armed", item=item)
            ns = go(ns, ARMED)
        if ns.state == ARMED and dur_ms >= p.long_blink_ms:
            return _confirm(ns, item, now, emit, go, blink_ms=dur_ms), events
        if ns.state == TRACKING and dur_ms >= p.long_blink_ms and not ns.none_fired:
            emit(NONE_OF_THESE, blink_ms=dur_ms)
            ns = replace(ns, none_fired=True)
    else:
        if eyes == "one" and not ns.one_eye_logged:
            emit("blink_rejected", reason="eyes_disagree")
            ns = replace(ns, one_eye_logged=True)
        elif eyes == "none":
            ns = replace(ns, one_eye_logged=False)
        if ns.closed_since is not None:
            dur_ms = (now - ns.closed_since) * 1000
            if ns.state == ARMED:
                emit("arm_cancelled", reason="blink_too_short", blink_ms=dur_ms)
                ns = go(replace(ns, highlight_since=now), HIGHLIGHTING)
            last = ns.last_short_blink_end
            if p.min_blink_ms <= dur_ms < p.long_blink_ms:  # shorter = landmark flicker
                gap_ms = None if last is None else (ns.closed_since - last) * 1000
                if (
                    p.undo_gesture == "double_blink"
                    and gap_ms is not None
                    and gap_ms <= p.double_blink_gap_ms
                ):
                    undo, last = True, None
                else:
                    last = now
            ns = replace(ns, closed_since=None, none_fired=False, last_short_blink_end=last)

    # 2. Tilts. Tracked while armed too, so a tilt held through a blink is not counted later.
    disp = displacements(f, p)
    tilts, counted = update_tilts(ns.tilts, disp, p, now)
    ns = replace(ns, tilts=tilts)
    if counted is not None:
        emit("tilt", direction=counted)
        if p.undo_gesture == "tilt_down" and counted == "down" and ns.menu.layout != "four_way":
            undo = True
        elif ns.state == ARMED:
            emit("tilt_ignored", direction=counted, reason="armed")
        else:
            menu = ns.menu.apply_tilt(counted)
            if menu is None:
                emit("tilt_ignored", direction=counted, reason="not_in_layout")
            else:
                emit("highlight", item=menu.highlighted_item, range=[menu.lo, menu.hi])
                ns = go(replace(ns, menu=menu, highlight_since=now), HIGHLIGHTING)

    # 3. Undo.
    if undo:
        if ns.state in (HIGHLIGHTING, ARMED):
            emit("undo", cancelled=ns.menu.highlighted_item)
            ns = replace(ns, menu=ns.menu.reset(), highlight_since=None, closed_since=None)
            ns = go(ns, TRACKING)
        else:
            emit("undo", cancelled=None)

    # 4. Dwell confirmation (evaluation baseline B only): the head is back at rest and has
    # stayed there, with one item highlighted, for dwell_ms.
    item = ns.menu.highlighted_item
    if (
        p.confirm_mode == "dwell"
        and ns.state == HIGHLIGHTING
        and item is not None
        and all(t.phase == "idle" for t in ns.tilts)
        and (now - ns.highlight_since) * 1000 >= p.dwell_ms
    ):
        return _confirm(ns, item, now, emit, go, dwell_ms=(now - ns.highlight_since) * 1000), events

    # 5. Highlight timeout.
    if (
        ns.state == HIGHLIGHTING
        and ns.highlight_since is not None
        and (now - ns.highlight_since) * 1000 >= p.highlight_timeout_ms
    ):
        emit("highlight_timeout")
        ns = go(replace(ns, menu=ns.menu.reset(), highlight_since=None), TRACKING)

    return ns, events


def _confirm(ns: MachineState, item: str, now: float, emit, go, **why: Any) -> MachineState:
    """The only place a speak event is emitted."""
    emit("confirmed", item=item, **why)
    emit("speak", item=item)
    ns = replace(ns, speaking_since=now, speaking_item=item, closed_since=None)
    return go(ns, SPEAKING)


def _after_speech(s: MachineState, go) -> MachineState:
    ns = replace(
        s,
        menu=s.menu.reset(),
        highlight_since=None,
        speaking_since=None,
        speaking_item=None,
        closed_since=None,
        tilts=lock_tilts(s.tilts),
        need_eyes_open=True,
    )
    return go(ns, TRACKING)


def speech_ended(s: MachineState, now: float) -> tuple[MachineState, list[dict[str, Any]]]:
    """Called when the UI reports the utterance finished."""
    if s.state != SPEAKING:
        return s, []
    events: list[dict[str, Any]] = [{"type": "speech_ended", "t": now, "item": s.speaking_item}]

    def go(ns: MachineState, to: str) -> MachineState:
        events.append({"type": "transition", "t": now, "frm": ns.state, "to": to})
        return replace(ns, state=to)

    return _after_speech(s, go), events


def set_menu(s: MachineState, layout: LayoutName, items: list[str]) -> tuple[MachineState, bool]:
    """Replace the menu only when no gesture is in progress. Returns (state, applied)."""
    if s.state not in (IDLE, TRACKING):
        return s, False
    return replace(s, menu=Menu.create(layout, items), highlight_since=None), True
