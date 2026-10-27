from headvoice.interaction import state_machine as sm
from headvoice.interaction.layouts import Menu
from headvoice.interaction.selection import MenuSource
from headvoice.io.schemas import Phrasebook, Profile
from tests.helpers import Sim


def test_binary_scan_narrows_then_speaks():
    items = [f"i{k}" for k in range(8)]
    sim = Sim(layout="binary_scan", items=items).settle()
    sim.tilt("right").tilt("left").tilt("right")  # [4,8) -> [4,6) -> [5,6)
    assert sim.s.menu.highlighted_item == "i5"
    sim.blink(1.0)
    assert sim.spoken() == ["i5"]


def test_binary_scan_odd_length():
    m = Menu.create("binary_scan", ["a", "b", "c"])
    m = m.apply_tilt("right")  # [2,3)
    assert m.highlighted_item == "c"
    m = Menu.create("binary_scan", ["a", "b", "c"]).apply_tilt("left").apply_tilt("right")
    assert m.highlighted_item == "b"


def test_two_choice_ignores_up_down():
    sim = Sim(layout="two_choice", items=["a", "b"]).settle().tilt("up")
    assert "tilt_ignored" in sim.types() and sim.s.state == sm.TRACKING
    sim.tilt("right").blink(1.0)
    assert sim.spoken() == ["b"]


def test_four_way_retarget_highlight():
    sim = Sim().settle().tilt("left").tilt("right")
    assert sim.s.menu.highlighted_item == "call_family"


def test_highlight_timeout_returns_to_tracking():
    sim = Sim(profile=Profile(highlight_timeout_ms=1000)).settle().tilt("up").run(1.5)
    assert "highlight_timeout" in sim.types() and sim.s.menu.highlighted is None


def test_short_dropout_keeps_highlight():
    sim = Sim().settle().tilt("left").run(0.5, face=False)
    assert sim.s.state == sm.RECOVERING
    sim.run(0.2)
    assert sim.s.state == sm.HIGHLIGHTING and sim.s.menu.highlighted_item == "need_help"


def test_one_bad_frame_does_not_drop_tracking():
    sim = Sim().settle().run(0.1, source="flow").run(0.3)
    assert "transition" not in [e["type"] for e in sim.events if e.get("to") == "idle"]
    assert sim.s.state == sm.TRACKING


def test_eye_hysteresis_keeps_a_flickering_long_blink_together():
    # Openness wobbles between 0.5 and 0.7 while the eyes are shut (closes below 0.6, only
    # reopens above 0.8): one blink, not many.
    sim = Sim().settle().tilt("left")
    for _ in range(6):
        sim.run(0.1, eyes=0.5).run(0.067, eyes=0.7)
    sim.run(0.3)
    assert sim.spoken() == ["need_help"]


def test_landmark_flicker_does_not_undo():
    sim = Sim().settle().tilt("left")
    for _ in range(4):  # single-frame dips: shorter than min_blink_ms
        sim.run(1 / 30, eyes=0.5).run(0.2)
    assert "undo" not in sim.types() and sim.s.state == sm.HIGHLIGHTING


def test_recovery_timeout_to_idle():
    sim = Sim(profile=Profile(recover_timeout_ms=1000)).settle().tilt("left").run(1.5, face=False)
    assert "recovery_timeout" in sim.types() and sim.s.state == sm.IDLE


def test_speech_ended_and_timeout():
    sim = Sim().settle().tilt("left").blink(1.0)
    assert sim.s.state == sm.SPEAKING
    s, ev = sm.speech_ended(sim.s, sim.t)
    assert s.state == sm.TRACKING and ev[0]["type"] == "speech_ended"
    sim2 = Sim(profile=Profile(speak_timeout_ms=1000)).settle().tilt("left").blink(1.0).run(1.5)
    assert "speech_timeout" in sim2.types() and sim2.s.state == sm.TRACKING


def test_gestures_ignored_while_speaking():
    sim = Sim().settle().tilt("left").blink(1.0)
    n = len(sim.spoken())
    sim.tilt("right").blink(1.0)
    assert len(sim.spoken()) == n


def test_mirrored_camera_sign_flip():
    sim = Sim(profile=Profile(lr_sign=-1)).settle().tilt("left")
    assert sim.s.menu.highlighted_item == "call_family"  # physical left reads as right


def test_menu_source_pages_and_suggestions():
    pb = Phrasebook.load("config/phrasebook_en.json")
    src = MenuSource(pb)
    first = src.current("four_way")
    assert len(first) == 4
    assert src.next_set("four_way") != first
    src.set_suggestions(["call_family", "not_a_phrase"])
    assert src.current("four_way")[0] == "call_family"
    assert "not_a_phrase" not in src.current("binary_scan")


def test_head_down_ignores_eye_closure():
    # Looking down lowers the lids: closure while pitched down must not arm or confirm.
    sim = Sim().settle().tilt("left")
    sim.run(1.5, pitch=-8.0, eyes=0.2)
    assert not sim.spoken() and "armed" not in sim.types()
