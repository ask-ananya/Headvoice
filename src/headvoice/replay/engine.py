"""Replay: logged FrameFeatures -> the real Controller -> events, traces and metrics.

Replay drives the same Controller class as the live app. When the log has speech_ended
events their times are reused, so replaying a session with its own profile and policy
reproduces the live events. When a different arm changes what gets selected, speech is
assumed to take SPEECH_S seconds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from headvoice.app.controller import Controller
from headvoice.interaction import state_machine as sm
from headvoice.io.logger import read_session
from headvoice.io.schemas import FrameFeatures, Phrasebook, Profile

SPEECH_S = 1.5
SKIP_EVENTS = {"session_start", "session_stop"}


@dataclass
class ReplayResult:
    events: list[dict]
    traces: list[dict] = field(default_factory=list)
    profile_after: dict = field(default_factory=dict)


def load(path: str | Path) -> tuple[dict, list[FrameFeatures], list[dict]]:
    return read_session(path)


def task_targets(events: list[dict]) -> list[str] | None:
    for e in events:
        if e["type"] == "task_start" and not e.get("practice"):
            return list(e["targets"])
    return None


def replay(
    frames: list[FrameFeatures],
    profile: Profile,
    phrasebook: Phrasebook,
    policy: str = "none",
    transport=None,
    targets: list[str] | None = None,
    speech_end_times: list[float] | None = None,
    session_id: str = "replay",
) -> ReplayResult:
    c = Controller(
        profile.model_copy(deep=True),
        phrasebook,
        policy=policy,
        transport=transport,
        background_agent=False,
    )
    c.events = deque()  # keep every event
    c.start(session_id)
    if targets:
        c.start_task(targets=targets)
    ends = deque(sorted(speech_end_times or []))
    speak_t = None
    for f in frames:
        c.handle_features(f, f.t)
        if c.state.state == sm.SPEAKING:
            if speak_t is None:
                speak_t = f.t
            while ends and ends[0] < speak_t:
                ends.popleft()  # logged speech end belonged to a speech this arm did not make
            end = ends[0] if ends and ends[0] - speak_t < 10 else speak_t + SPEECH_S
            if f.t >= end:
                if ends and ends[0] == end:
                    ends.popleft()
                c.speech_ended(f.t)
                speak_t = None
    traces = [r.model_dump() for r in c.adapter.traces.records] if c.adapter else []
    profile_after = c.profile.model_dump()
    c.stop()
    events = [e for e in c.events if e["type"] not in SKIP_EVENTS]
    return ReplayResult(events, traces, profile_after)


def replay_file(
    path: str | Path,
    phrasebook: Phrasebook,
    profile: Profile | None = None,
    policy: str = "none",
    transport=None,
    frames: list[FrameFeatures] | None = None,
    faithful_speech: bool = True,
) -> ReplayResult:
    header, logged_frames, logged_events = load(path)
    prof = profile or Profile.model_validate(header.get("profile", {}))
    ends = (
        [e["t"] for e in logged_events if e["type"] == "speech_ended"] if faithful_speech else None
    )
    return replay(
        frames if frames is not None else logged_frames,
        prof,
        phrasebook,
        policy=policy,
        transport=transport,
        targets=task_targets(logged_events),
        speech_end_times=ends,
        session_id=header.get("session_id", "replay"),
    )
