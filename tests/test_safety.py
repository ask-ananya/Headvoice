"""One or more tests per safety invariant in RULES.md (plan section 6)."""

import dataclasses
import inspect
import time

import numpy as np
import pytest

from headvoice.agent import tools as agent_tools
from headvoice.agent.tools import (
    ActionCap,
    AgentApplier,
    Bounds,
    StaleGuard,
    ToolCall,
    validate_call,
)
from headvoice.interaction import state_machine as sm
from headvoice.io.offline import candidates_with_fallback
from headvoice.io.privacy import PrivacyViolation, check_outbound
from headvoice.io.schemas import Profile
from tests.helpers import Sim

BOUNDS = Bounds.load()


# 1. Code owns the state ---------------------------------------------------------------


def test_1_agent_layer_cannot_reach_state_machine():
    src = inspect.getsource(agent_tools)
    assert "state_machine" not in src and "MachineState" not in src
    sim = Sim().settle().tilt("left")
    before = sim.s
    applier = AgentApplier(Profile(), BOUNDS, "s1")
    epoch = applier.guard.next_epoch()
    applier.apply(
        [
            ToolCall(tool="set_layout", args={"layout": "binary_scan"}),
            ToolCall(tool="set_state", args={"state": "speaking"}),
        ],
        "s1",
        epoch,
        0.0,
    )
    assert sim.s is before and sim.s.state == sm.HIGHLIGHTING
    assert dataclasses.is_dataclass(before) and before.__dataclass_params__.frozen


def test_1_menu_change_deferred_mid_gesture():
    sim = Sim().settle().tilt("left")
    s, applied = sm.set_menu(sim.s, "binary_scan", ["a", "b"])
    assert not applied and s is sim.s


# 2. Fail closed -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"face": False},
        {"quality": 0.2},
        {"source": "flow"},
        {"flags": ["turned_away"]},
    ],
)
def test_2_bad_signal_never_selects_or_speaks(bad):
    sim = Sim().settle()
    sim.run(0.5, roll=20.0, **bad).run(0.3, **bad)  # tilt during bad signal
    sim.run(1.5, eyes=0.2, **bad)  # long blink during bad signal
    assert "highlight" not in sim.types() and not sim.spoken()


@pytest.mark.parametrize("bad", [{"face": False}, {"source": "flow"}, {"quality": 0.1}])
def test_2_signal_loss_while_armed_cancels(bad):
    sim = Sim().settle().tilt("left")
    sim.run(0.3, eyes=0.2)  # armed
    assert sim.s.state == sm.ARMED
    sim.run(1.0, eyes=0.2, **bad)
    sim.run(1.0, eyes=0.2)  # signal back, eyes still closed
    assert not sim.spoken()
    assert "arm_cancelled" in sim.types()


def test_2_closure_spanning_dropout_never_confirms():
    # Eyes close, signal drops, signal returns with eyes still closed: no speech until the
    # eyes have been seen open and a complete new long blink happens on good frames.
    sim = Sim().settle().tilt("left")
    sim.run(0.4, eyes=0.2).run(0.2, eyes=0.2, face=False).run(2.0, eyes=0.2)
    assert not sim.spoken()
    sim.run(0.2).blink(1.0)
    assert sim.spoken() == ["need_help"]


def test_4_no_new_blink_counts_until_eyes_open_after_speech():
    sim = Sim().settle().tilt("left").run(1.0, eyes=0.2)
    s, _ = sm.speech_ended(sim.s, sim.t)
    sim.s = s
    sim.run(2.0, eyes=0.2)  # eyes still shut after speaking
    assert "none_of_these" not in sim.types()


# 3. Debounce ----------------------------------------------------------------------------


def test_3_one_long_tilt_counts_once():
    sim = Sim().settle().run(3.0, roll=25.0).run(0.5)
    assert sim.types().count("tilt") == 1


def test_3_jittery_tilt_around_threshold_counts_once():
    sim = Sim().settle()
    for _ in range(10):
        sim.run(0.4, roll=13.0).run(0.1, roll=10.0)  # dips below threshold, not below return
    assert sim.types().count("tilt") <= 1


def test_3_brief_twitch_does_not_count():
    sim = Sim().settle().run(0.15, roll=25.0).run(0.5)
    assert "tilt" not in sim.types()


# 4. Confirm before speaking -------------------------------------------------------------


def test_4_speech_only_after_long_blink_on_highlight():
    sim = Sim().settle().tilt("left")
    assert sim.s.state == sm.HIGHLIGHTING
    sim.blink(1.0)
    assert sim.spoken() == ["need_help"]  # four_way slots: up, left, right, down
    order = sim.types()
    assert order.index("armed") < order.index("confirmed") < order.index("speak")


def test_4_long_blink_without_highlight_never_speaks():
    sim = Sim().settle().blink(1.5)
    assert not sim.spoken()
    assert "none_of_these" in sim.types()


def test_4_short_blink_does_not_speak():
    sim = Sim().settle().tilt("left").blink(0.3)
    assert not sim.spoken() and sim.s.state == sm.HIGHLIGHTING


def test_4_one_eye_closed_does_not_speak():
    sim = Sim().settle().tilt("left").run(1.5, eye_l=0.2, eye_r=1.0)
    assert not sim.spoken() and "blink_rejected" in sim.types()


def test_4_binary_scan_needs_single_item():
    items = [f"i{k}" for k in range(8)]
    sim = Sim(layout="binary_scan", items=items).settle().tilt("left")
    assert sim.s.menu.highlighted is None and sim.s.menu.hi - sim.s.menu.lo == 4
    sim.blink(1.5)
    assert not sim.spoken()


def test_4_only_path_to_speak_event_is_confirm():
    # One emit site for speech, in _confirm; step reaches it only from Armed (long blink)
    # or, in the dwell evaluation baseline, from a held highlight.
    src = inspect.getsource(sm)
    assert src.count('emit("speak"') == 1
    assert 'emit("speak"' in inspect.getsource(sm._confirm)


def test_4_dwell_mode_still_fails_closed_and_needs_a_highlight():
    p = Profile(confirm_mode="dwell", dwell_ms=1000)
    sim = Sim(profile=p).settle().run(3.0)
    assert not sim.spoken()  # nothing highlighted: holding still does nothing
    sim.tilt("left").run(1.5, face=False)
    assert not sim.spoken()  # signal lost during the dwell
    sim2 = Sim(profile=p).settle().tilt("left").run(1.2)
    assert sim2.spoken() == ["need_help"]
    sim3 = Sim(profile=p).settle().tilt("left").blink(1.0)
    assert "armed" not in sim3.types()  # no blink arming in dwell mode


# 5. Undo --------------------------------------------------------------------------------


def test_5_double_blink_undo_cancels_highlight():
    sim = Sim().settle().tilt("left")
    sim.run(0.15, eyes=0.2).run(0.2).run(0.15, eyes=0.2).run(0.3)
    assert "undo" in sim.types() and sim.s.state == sm.TRACKING
    sim.blink(1.5)
    assert not sim.spoken()


def test_5_tilt_down_undo_in_binary_layout():
    p = Profile(undo_gesture="tilt_down", layout="two_choice")
    sim = Sim(profile=p, layout="two_choice", items=["a", "b"]).settle().tilt("left")
    sim.run(0.3, eyes=0.2)  # armed
    sim.run(0.2, eyes=1.0)
    sim.tilt("down")
    assert "undo" in sim.types() and sim.s.state == sm.TRACKING
    assert not sim.spoken()


# 6. Stale-result guard ------------------------------------------------------------------


def test_6_stale_epoch_discarded():
    g = StaleGuard("s1")
    e1 = g.next_epoch()
    e2 = g.next_epoch()
    assert g.accept("s1", e1) == (False, "stale_epoch")
    assert g.accept("other", e2) == (False, "wrong_session")
    assert g.accept("s1", e2) == (True, None)


def test_6_applier_ignores_stale_response():
    a = AgentApplier(Profile(), BOUNDS, "s1")
    old = a.guard.next_epoch()
    a.guard.next_epoch()
    recs = a.apply([ToolCall(tool="set_layout", args={"layout": "binary_scan"})], "s1", old, 0)
    assert recs[0]["rejected_reason"] == "stale_epoch" and a.profile.layout == "four_way"


# 7. Bounded agent -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "call,reason",
    [
        (
            ToolCall(tool="adjust_timing", args={"param": "long_blink_ms", "value": 50}),
            "value_out_of_bounds",
        ),
        (
            ToolCall(tool="adjust_timing", args={"param": "quality_min", "value": 0.1}),
            "param_not_tunable",
        ),
        (ToolCall(tool="set_layout", args={"layout": "free_text"}), "layout_out_of_bounds"),
        (ToolCall(tool="rm_rf"), "unknown_tool"),
    ],
)
def test_7_out_of_bounds_rejected(call, reason):
    assert validate_call(call, BOUNDS) == (False, reason)


def test_7_in_bounds_applied_and_rollback():
    a = AgentApplier(Profile(), BOUNDS, "s1")
    e = a.guard.next_epoch()
    recs = a.apply(
        [ToolCall(tool="adjust_timing", args={"param": "long_blink_ms", "value": 900})], "s1", e, 0
    )
    assert recs[0]["applied"] and a.profile.long_blink_ms == 900
    assert a.rollback() and a.profile.long_blink_ms == 700


def test_7_max_tool_calls_per_invocation():
    a = AgentApplier(Profile(), BOUNDS, "s1")
    e = a.guard.next_epoch()
    calls = [ToolCall(tool="no_change")] * (BOUNDS.max_tool_calls_per_invocation + 1)
    recs = a.apply(calls, "s1", e, 0)
    assert recs[-1]["rejected_reason"] == "too_many_calls"


# 8. Tool tiers ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool,reason",
    [
        ("speak_message", "agent_may_not_call"),
        ("send_caregiver_alert", "agent_may_not_call"),
        ("access_raw_video", "forbidden_tool"),
    ],
)
def test_8_tiers_enforced(tool, reason):
    a = AgentApplier(Profile(), BOUNDS, "s1")
    e = a.guard.next_epoch()
    recs = a.apply([ToolCall(tool=tool, args={"message_id": "need_water"})], "s1", e, 0)
    assert recs[0] == {
        "tool": tool,
        "args": {"message_id": "need_water"},
        "applied": False,
        "rejected_reason": reason,
    }


def test_8_recalibration_only_prompts_user():
    a = AgentApplier(Profile(), BOUNDS, "s1")
    e = a.guard.next_epoch()
    a.apply([ToolCall(tool="request_recalibration", reason="undos up")], "s1", e, 0)
    assert a.ui_requests and a.profile == Profile()


# 9. Action cap ---------------------------------------------------------------------------


def test_9_action_cap():
    cap = ActionCap(max_actions=3, window_s=120)
    assert [cap.allow(t) for t in (0, 1, 2, 3)] == [True, True, True, False]
    assert cap.allow(130)


def test_9_cap_applies_to_adaptations_only():
    a = AgentApplier(Profile(), BOUNDS, "s1")
    results = []
    for i in range(BOUNDS.max_actions_per_window + 1):
        e = a.guard.next_epoch()
        layout = "binary_scan" if i % 2 == 0 else "four_way"
        results += a.apply([ToolCall(tool="set_layout", args={"layout": layout})], "s1", e, i)
    assert results[-1]["rejected_reason"] == "action_cap"
    e = a.guard.next_epoch()
    assert a.apply([ToolCall(tool="no_change")], "s1", e, 10)[0]["applied"]


# 10. Offline fallback --------------------------------------------------------------------

FALLBACK = ["need_water", "need_help"]
VALID = {"need_water", "need_help", "call_family"}


def test_10_timeout_falls_back_without_blocking():
    t0 = time.monotonic()
    ids, src = candidates_with_fallback(
        lambda: time.sleep(2) or ["call_family"], FALLBACK, VALID, timeout_s=0.2
    )
    assert ids == FALLBACK and src == "offline:timeout"
    assert time.monotonic() - t0 < 0.6


def test_10_error_and_invalid_fall_back():
    def boom():
        raise ConnectionError("no network")

    assert candidates_with_fallback(boom, FALLBACK, VALID, 1)[1] == "offline:error"
    assert candidates_with_fallback(lambda: ["free text!"], FALLBACK, VALID, 1) == (
        FALLBACK,
        "offline:invalid",
    )
    assert candidates_with_fallback(lambda: ["call_family"], FALLBACK, VALID, 1) == (
        ["call_family"],
        "cloud",
    )


# 11. Privacy -----------------------------------------------------------------------------


def test_11_summary_payload_allowed():
    check_outbound(
        {
            "session_id": "s1",
            "epoch": 3,
            "window": {
                "undos": 2,
                "quality_mean": 0.8,
                "tilt_amplitude_vs_calibrated": {"left": 0.6},
            },
        }
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"landmarks": [[1.0, 2.0]]},
        {"window": {"frame": "x"}},
        {"blob": b"\xff\xd8"},
        {"arr": np.zeros((4, 4))},
        {"series": list(range(500))},
    ],
)
def test_11_raw_data_blocked(payload):
    with pytest.raises(PrivacyViolation):
        check_outbound(payload)
