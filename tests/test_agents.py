import pytest

from headvoice.agent.suggest import run_suggest, validate
from headvoice.agent.supervisor import MAX_TOOL_CALLS, run_supervisor
from headvoice.io.schemas import FrameFeatures, Profile
from headvoice.replay.arms import arm_setup
from headvoice.replay.metrics import abstention, agent_metrics, task_metrics


def fake_converse(script):
    """Returns tool uses from `script` in order; records the messages it saw."""
    seen = []

    def converse(**kw):
        seen.append(kw)
        name, args = script[min(len(seen) - 1, len(script) - 1)]
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": f"t{len(seen)}",
                                "name": name,
                                "input": dict(args),
                            }
                        }
                    ],
                }
            },
            "usage": {"inputTokens": 100, "outputTokens": 20},
            "stopReason": "tool_use",
        }

    converse.seen = seen
    return converse


REQ = {
    "session_id": "s1",
    "epoch": 4,
    "current_layout": "four_way",
    "current_params": {},
    "window": {"quality_mean": 0.9, "tilt_near_misses": 5},
    "evidence": {
        "windows": {"300": {"tilt_attempts": 12}},
        "direction_reliability": {"up": {"success_rate": 0.2}},
    },
}


def test_supervisor_queries_evidence_then_acts():
    conv = fake_converse(
        [
            ("get_direction_reliability", {}),
            ("set_layout", {"layout": "binary_scan", "reason": "up success 0.2, 5 near misses"}),
        ]
    )
    out = run_supervisor(REQ, conv, "m")
    assert out["actions"] == [
        {
            "tool": "set_layout",
            "args": {"layout": "binary_scan"},
            "reason": "up success 0.2, 5 near misses",
        }
    ]
    assert out["tool_calls"][0] == {
        "tool": "get_direction_reliability",
        "args": {},
        "result": {"up": {"success_rate": 0.2}},
    }
    # The evidence pack is not shown up front: only answered when asked.
    assert "evidence" not in conv.seen[0]["messages"][0]["content"][0]["text"]
    assert out["epoch"] == 4 and out["usage"]["inputTokens"] == 200


def test_supervisor_call_limit_gives_no_change():
    conv = fake_converse([("get_window_summary", {"seconds": 300})] * 10)
    out = run_supervisor(REQ, conv, "m")
    assert len(conv.seen) <= MAX_TOOL_CALLS
    assert out["actions"][0]["tool"] == "no_change"


def test_supervisor_passes_forbidden_tool_through_for_device_rejection():
    conv = fake_converse([("speak_message", {"message_id": "need_water", "reason": "x"})])
    out = run_supervisor(REQ, conv, "m")
    assert out["actions"][0]["tool"] == "speak_message"  # the device rejects it (test_safety 8)


PHRASES = [{"id": "need_water", "text": "I need water", "group": "needs"}]


def test_suggest_validation():
    ok = validate(
        {
            "abstain": False,
            "candidates": [{"message_id": "need_water", "confidence": 0.6}],
            "reason": "r",
        },
        {"need_water"},
    )
    assert ok.candidates[0].message_id == "need_water"
    with pytest.raises(ValueError):
        validate(
            {
                "abstain": False,
                "candidates": [{"message_id": "free text", "confidence": 0.6}],
                "reason": "r",
            },
            {"need_water"},
        )
    with pytest.raises(ValueError):
        validate(
            {
                "abstain": True,
                "candidates": [{"message_id": "need_water", "confidence": 0.6}],
                "reason": "r",
            },
            {"need_water"},
        )


def test_run_suggest_rejects_invalid_output():
    conv = fake_converse(
        [
            (
                "propose_candidates",
                {
                    "abstain": False,
                    "candidates": [{"message_id": "call_the_police", "confidence": 0.9}],
                    "reason": "r",
                },
            )
        ]
    )
    out = run_suggest({"context": {}, "phrasebook": PHRASES}, conv, "m", "evidence_first")
    assert not out["valid"] and "unknown message ids" in out["error"]


def test_task_metrics_and_not_measured():
    ev = [
        {"type": "task_start", "t": 0, "targets": ["a", "b"]},
        {"type": "target", "t": 0, "item": "a"},
        {"type": "undo", "t": 2},
        {"type": "task_hit", "t": 5, "item": "a"},
        {"type": "target", "t": 5, "item": "b"},
        {"type": "task_miss", "t": 7, "item": "c", "target": "b"},
    ]
    m = task_metrics(ev)
    assert m["task_success_rate"] == 0.5 and m["false_triggers"] == 1
    assert m["selection_time_s_median"] == 5 and m["retries_per_correct"] == 1
    assert task_metrics([])["task_success_rate"] is None


def test_abstention_metric():
    frames = [
        FrameFeatures(t=i / 10, frame_idx=i, face_found=True, quality=0.1 if i < 40 else 0.9)
        for i in range(100)
    ]
    a = abstention([{"type": "speak", "t": 8.0}], frames, 0.5)
    assert a["low_quality_windows"] == 2 and a["abstention_correct"] == 1.0
    b = abstention([{"type": "speak", "t": 1.0}], frames, 0.5)
    assert b["abstention_correct"] == 0.5


def test_agent_metrics_cost():
    m = agent_metrics(
        [
            {
                "latency_ms": 1000,
                "usage": {"inputTokens": 1_000_000, "outputTokens": 0},
                "actions": [{"tool": "no_change"}],
                "applied": [],
            }
        ],
        1.0,
        5.0,
    )
    assert m["cost_usd"] == 1.0 and m["no_change_rate"] == 1.0


def test_arms():
    cal = Profile(layout="binary_scan", profile_id="u:calibrated")
    p, pol = arm_setup("A", Profile(), cal)
    assert p.layout == "four_way" and pol == "none"
    p, pol = arm_setup("B", Profile(), cal)
    assert p.confirm_mode == "dwell"
    p, pol = arm_setup("E", Profile(), cal)
    assert p.layout == "binary_scan" and pol == "agent"
