import json

from fastapi.testclient import TestClient

from headvoice.app.server import Controller, create_app
from headvoice.io.schemas import Phrasebook, Profile
from tests.helpers import FPS, frame

PB = Phrasebook.load("config/phrasebook_en.json")


def feed(c: Controller, t0: float, seconds: float, **kw) -> float:
    t = t0
    for _ in range(round(seconds * FPS)):
        c.handle_features(frame(t, **kw), t)
        t += 1 / FPS
    return t


def test_speak_message_only_after_confirmed_blink():
    c = Controller(Profile(), PB)
    c.start()
    t = feed(c, 0, 0.5)
    t = feed(c, t, 0.5, roll=20.0)
    t = feed(c, t, 0.3)
    assert c.drain_outbox() == []  # highlighted, nothing spoken
    t = feed(c, t, 0.3, eyes=0.2)  # blink too short
    t = feed(c, t, 0.2)
    assert c.drain_outbox() == []
    t = feed(c, t, 1.0, eyes=0.2)
    out = c.drain_outbox()
    assert len(out) == 1 and out[0]["type"] == "speak"
    assert out[0]["text"] == PB.text(out[0]["item"])
    c.speech_ended(t)
    assert c.snapshot()["state"] == "tracking"


def test_not_running_does_nothing():
    c = Controller(Profile(), PB)
    t = feed(c, 0, 0.5)
    t = feed(c, t, 0.5, roll=20.0)
    feed(c, t, 1.5, eyes=0.2)
    assert c.drain_outbox() == [] and c.snapshot()["state"] == "idle"


def test_none_of_these_changes_menu():
    c = Controller(Profile(), PB)
    c.start()
    before = c.snapshot()["menu"]["items"]
    t = feed(c, 0, 0.5)
    feed(c, t, 1.2, eyes=0.2)
    assert c.snapshot()["menu"]["items"] != before
    assert c.drain_outbox() == []


def test_set_neutral():
    c = Controller(Profile(), PB)
    c.handle_features(frame(0, roll=5.0, pitch=-3.0), 0)
    assert c.set_neutral()
    assert (c.profile.neutral.roll, c.profile.neutral.pitch) == (5.0, -3.0)


def test_websocket_snapshot_and_controls():
    c = Controller(Profile(), PB)
    app = create_app(controller=c, camera=None)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        with client.websocket_connect("/ws") as ws:
            snap = json.loads(ws.receive_text())
            assert snap["type"] == "snapshot" and not snap["running"]
            ws.send_text(json.dumps({"type": "start"}))
            for _ in range(20):
                snap = json.loads(ws.receive_text())
                if snap.get("running"):
                    break
            assert snap["running"]
            ws.send_text(json.dumps({"type": "task_start", "n": 3}))
            for _ in range(20):
                snap = json.loads(ws.receive_text())
                if snap.get("task"):
                    break
            assert snap["task"]["n"] == 3 and snap["task"]["target"]


def test_task_hit_and_miss_events():
    c = Controller(Profile(), PB)
    c.start()
    c.start_task(targets=["need_help", "need_water"])
    t = feed(c, 0, 0.5)
    t = feed(c, t, 0.5, roll=20.0)  # left = slot 1 = need_help on the first page
    t = feed(c, t, 0.3)
    t = feed(c, t, 1.0, eyes=0.2)
    c.speech_ended(t)
    kinds = [e["type"] for e in c.events]
    assert "task_hit" in kinds and c.snapshot()["task"]["target"] == "need_water"


def test_rule_policy_adapts_and_traces_outcome():
    c = Controller(Profile(), PB, policy="rule")
    c.start()
    t = feed(c, 0, 61.0)  # no adaptation in the first minute
    # Small tilts that never reach the threshold: near misses, low amplitude.
    for _ in range(8):
        t = feed(c, t, 0.6, roll=7.0)
        t = feed(c, t, 0.6)
    t = feed(c, t, 70.0)  # long enough for the outcome window
    recs = c.adapter.traces.records
    acted = [r for r in recs if r.actions and r.actions[0]["tool"] == "adjust_sensitivity"]
    assert acted and acted[0].applied[0]["applied"]
    assert c.profile.tilt_scale < 1.0
    assert acted[0].outcome_after is not None


def test_agent_waits_for_outcome_before_deciding_again():
    calls = []

    def transport(req):
        calls.append(req)
        return {
            "session_id": req["session_id"],
            "epoch": req["epoch"],
            "tool_calls": [],
            "actions": [
                {"tool": "adjust_sensitivity", "args": {"scale": 0.8}, "reason": "near misses"}
            ],
            "reason": "near misses",
            "latency_ms": 1,
        }

    c = Controller(Profile(), PB, policy="agent", transport=transport, background_agent=False)
    c.start()
    t = feed(c, 0, 61.0)  # no adaptation in the first minute
    for _ in range(30):  # 42 s of near misses: the monitor keeps asking
        t = feed(c, t, 0.6, roll=7.0)
        t = feed(c, t, 0.8)
    assert len(calls) == 1  # one change, then waiting for its outcome window
    t = feed(c, t, 30.0)
    rec = c.adapter.traces.records[0]
    assert rec.outcome_after is not None
    if len(calls) > 1:  # the next decision sees the measured outcome of the first
        assert calls[1]["history"][0]["outcome_after"] is not None


def test_pages_and_assets_served():
    app = create_app(controller=Controller(Profile(), PB), camera=None)
    with TestClient(app) as client:
        for path in ("/", "/start", "/app", "/ui.css", "/common.js", "/session.js", "/app.js"):
            assert client.get(path).status_code == 200, path
        assert "HeadVoice" in client.get("/").text


def test_condition_endpoint():
    c = Controller(Profile(), PB)
    app = create_app(controller=c, camera=None)
    with TestClient(app) as client:
        assert client.post("/api/condition", json={"condition": "small_moves"}).json()["ok"]
        assert not client.post("/api/condition", json={"condition": "bogus"}).json()["ok"]
    assert c.condition == "small_moves"


def test_cameras_endpoint(monkeypatch):
    import headvoice.vision.camera as cam

    monkeypatch.setattr(
        cam,
        "probe_cameras",
        lambda in_use=None: [{"index": 0, "status": "frozen"}, {"index": 1, "status": "in_use"}],
    )
    c = Controller(Profile(), PB)
    c.camera_index = 1
    with TestClient(create_app(controller=c, camera=None)) as client:
        r = client.get("/api/cameras").json()
    assert r["current"] == 1 and r["cameras"][0]["status"] == "frozen"


def test_practice_targets_are_on_screen_and_not_evaluated():
    from headvoice.replay.metrics import task_metrics

    c = Controller(Profile(), PB)
    c.start_practice()
    assert c.running
    snap = c.snapshot()
    assert snap["task"]["practice"] and snap["task"]["target"] in snap["menu"]["items"]
    assert task_metrics(list(c.events))["task_success_rate"] is None  # practice only


def test_custom_messages_add_show_first_remove(tmp_path):
    path = tmp_path / "custom_messages.json"
    c = Controller(Profile(), PB, custom_path=path)
    app = create_app(controller=c, camera=None)
    with TestClient(app) as client:
        assert client.get("/messages").status_code == 200
        r = client.post("/api/messages", json={"text": "  Turn on   the cricket "}).json()
        assert r["ok"] and r["message"]["text"] == "Turn on the cricket"
        assert client.post("/api/messages", json={"text": "turn on the cricket"}).status_code == 400
        assert client.post("/api/messages", json={"text": ""}).status_code == 400
        assert client.post("/api/messages", json={"text": "x" * 81}).status_code == 400
        mid = r["message"]["id"]
        snap = c.snapshot()
        assert snap["menu"]["items"][0] == mid  # custom messages come first
        assert snap["menu"]["texts"][0] == "Turn on the cricket"
        # Persisted and reloaded.
        assert Controller(Profile(), PB, custom_path=path).custom.items[0].id == mid
        assert client.delete(f"/api/messages/{mid}").json()["ok"]
        assert client.get("/api/messages").json()["custom"] == []


def test_custom_message_change_waits_for_gesture_and_tasks_use_builtin(tmp_path):
    c = Controller(Profile(), PB, custom_path=tmp_path / "m.json")
    c.start()
    t = feed(c, 0, 0.5)
    t = feed(c, t, 0.5, roll=20.0)
    t = feed(c, t, 0.3)  # a message is highlighted
    before = c.snapshot()["menu"]["items"]
    c.add_message("Call Priya")
    assert c.snapshot()["menu"]["items"] == before  # not swapped mid-gesture
    t = feed(c, t, 5.0)  # highlight times out, menu refreshes
    assert c.snapshot()["menu"]["texts"][0] == "Call Priya"
    c.start_task(n=26, seed=1)
    assert not any(i.startswith("custom_") for i in c.task)


def test_no_adaptation_in_first_minute_and_glances_are_not_attempts():
    calls = []
    c = Controller(
        Profile(),
        PB,
        policy="agent",
        background_agent=False,
        transport=lambda r: (
            calls.append(r)
            or {
                "session_id": r["session_id"],
                "epoch": r["epoch"],
                "actions": [{"tool": "no_change", "args": {}, "reason": "x"}],
            }
        ),
    )
    c.start()
    t = feed(c, 0, 0.5)
    for _ in range(12):  # 30 s of clear near misses: still inside the warm-up
        t = feed(c, t, 0.6, roll=7.0)
        t = feed(c, t, 0.6)
    assert calls == []
    c2 = Controller(Profile(), PB, policy="rule")
    c2.start()
    t = feed(c2, 0, 61.0)
    for _ in range(20):  # quick glances (0.1 s) are not tilt attempts
        t = feed(c2, t, 0.1, pitch=-6.0)
        t = feed(c2, t, 0.5)
    assert c2.adapter.monitor.summary(t, c2.profile)["tilt_attempts"] == 0


def test_helper_layout_choice_locks_out_the_agent():
    def transport(req):
        assert req["layout_locked"] is True
        return {
            "session_id": req["session_id"],
            "epoch": req["epoch"],
            "tool_calls": [],
            "actions": [{"tool": "set_layout", "args": {"layout": "binary_scan"}, "reason": "x"}],
        }

    c = Controller(Profile(), PB, policy="agent", transport=transport, background_agent=False)
    c.start()
    assert c.set_layout("four_way") and c.snapshot()["layout_locked"]
    t = feed(c, 0, 61.0)
    for _ in range(10):
        t = feed(c, t, 0.6, roll=7.0)
        t = feed(c, t, 0.6)
    recs = c.adapter.traces.records
    assert recs and all(
        a["rejected_reason"] == "layout_set_by_helper" for r in recs for a in r.applied
    )
    assert c.profile.layout == "four_way"
