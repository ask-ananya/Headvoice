import pytest

from headvoice.app.controller import Controller
from headvoice.io.schemas import Phrasebook, Profile
from headvoice.replay import fatigue_sim
from headvoice.replay.engine import SKIP_EVENTS, replay, replay_file
from tests.helpers import FPS, frame

PB = Phrasebook.load("config/phrasebook_en.json")


def scripted_frames():
    """A short session: two selections, an undo, a none-of-these, a dropout."""
    fr, t = [], 0.0

    def run(seconds, **kw):
        nonlocal t
        for _ in range(round(seconds * FPS)):
            fr.append(frame(t, **kw))
            t += 1 / FPS

    run(0.5)
    run(0.5, roll=20.0)
    run(0.3)
    run(1.0, eyes=0.2)  # speak need_help
    run(2.0)
    run(0.5, pitch=15.0)
    run(0.3)
    run(0.15, eyes=0.2)
    run(0.2)
    run(0.15, eyes=0.2)  # double blink undo
    run(0.5)
    run(1.0, eyes=0.2)  # none of these
    run(0.5)
    run(0.4, face=False)
    run(0.5)
    run(0.5, roll=-20.0)
    run(0.3)
    run(1.0, eyes=0.2)  # speak
    run(2.0)
    return fr


def test_replay_reproduces_live_events(tmp_path):
    frames = scripted_frames()
    live = Controller(Profile(), PB, sessions_dir=tmp_path)
    live.events.__init__([], maxlen=None)
    live.start("s_test")
    speaking = None
    for f in frames:
        live.handle_features(f, f.t)
        if live.state.state == "speaking":
            speaking = speaking if speaking is not None else f.t
            if f.t - speaking >= 1.2:  # the UI reports speech end at its own time
                live.speech_ended(f.t)
                speaking = None
    live_events = [e for e in live.events if e["type"] not in SKIP_EVENTS]
    live.stop()
    log = next(tmp_path.glob("session_*.jsonl"))
    rep = replay_file(log, PB)
    assert [e["type"] for e in rep.events] == [e["type"] for e in live_events]
    assert [e["t"] for e in rep.events] == pytest.approx([e["t"] for e in live_events])
    assert sum(e["type"] == "speak" for e in rep.events) == 2


def test_replay_is_deterministic_with_rule_policy():
    frames = fatigue_sim.apply(scripted_frames(), final_scale=0.4)
    a = replay(frames, Profile(), PB, policy="rule")
    b = replay(frames, Profile(), PB, policy="rule")
    assert a.events == b.events
    assert [t["actions"] for t in a.traces] == [t["actions"] for t in b.traces]


def test_fatigue_scales_displacement_down_over_time():
    frames = [frame(i / FPS, roll=20.0) for i in range(300)]
    out = fatigue_sim.apply(frames, final_scale=0.5, jitter_deg=0.0, onset_frac=0.0)
    assert out[0].roll == pytest.approx(20.0)
    assert out[-1].roll == pytest.approx(10.0)
    assert out[150].roll == pytest.approx(15.0, abs=0.2)


def test_fatigue_keeps_missing_pose():
    frames = [frame(0, face=False), frame(0.1)]
    out = fatigue_sim.apply(frames)
    assert out[0].roll is None and out[1].roll is not None
