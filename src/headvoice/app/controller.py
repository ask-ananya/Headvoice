"""Controller: owns the profile, menu, state machine, adaptation, tasks and calibration.

Thread-safe; no camera code. The live app and the replay engine both drive this class, so
replayed sessions produce the same events as live ones.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from headvoice import LOG_FORMAT
from headvoice.agent.adapter import Adapter
from headvoice.agent.trace import TraceStore
from headvoice.calibration.analyze import analyze
from headvoice.calibration.record import CalibrationRecorder
from headvoice.interaction import state_machine as sm
from headvoice.interaction.selection import MenuSource
from headvoice.io.custom_messages import CustomMessages, merged
from headvoice.io.logger import SessionLogger
from headvoice.io.schemas import FrameFeatures, Neutral, Phrasebook, Profile

CONDITIONS = ("normal", "adverse_light", "glare", "off_angle", "small_moves")

CALIBRATED_FIELDS = (
    "profile_id",
    "neutral",
    "neutral_set",
    "lr_axis",
    "lr_sign",
    "ud_sign",
    "tilt_deg",
    "tilt_scale",
    "calibrated_range_deg",
    "layout",
    "eye_closed_ratio",
    "eye_open_ratio",
    "long_blink_ms",
    "involuntary_blink_ms_p95",
    "undo_gesture",
)


class Controller:
    def __init__(
        self,
        profile: Profile,
        phrasebook: Phrasebook,
        sessions_dir: Path | None = None,
        policy: str = "none",  # none, rule, agent
        transport=None,  # agent transport (CloudClient.supervise or a local function)
        cloud=None,  # CloudClient for calibration upload and audit
        background_agent: bool = True,
        clock=time.monotonic,
        custom_path: Path | None = None,  # the user's own messages (local file)
    ):
        self.lock = threading.RLock()
        self.profile = profile
        self.builtin = phrasebook
        self.custom = CustomMessages(custom_path)
        self.phrasebook = merged(phrasebook, self.custom.items)
        self.menu_source = MenuSource(self.phrasebook)
        self._retired_text: dict[str, str] = {}  # removed messages still on screen
        self.menu_dirty = False
        self.sessions_dir = sessions_dir
        self.state = sm.initial_state(profile.layout, self.menu_source.current(profile.layout))
        self.running = False
        self.logger: SessionLogger | None = None
        self.last_features: FrameFeatures | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=30)
        self.outbox: deque[dict[str, Any]] = deque()  # messages that must reach the UI
        self.policy = policy
        self.transport = transport
        self.cloud = cloud
        self.background_agent = background_agent
        self.online = cloud is not None or transport is not None
        self.camera_ok = True
        self.fps = 0.0
        self.neutral_hook = None  # set by the camera loop to the pipeline's request_neutral
        self.preview_jpeg: bytes | None = None  # local operator preview only, never sent out
        self.adapter: Adapter | None = None
        self.session_id: str | None = None
        self.task: list[str] = []
        self.task_i = 0
        self.calib: CalibrationRecorder | None = None
        self.calib_upload = False
        self.calib_result: dict | None = None
        self.clock = clock
        # Recording condition for evaluation: normal, adverse_light, glare, off_angle, small_moves
        self.condition = os.environ.get("HEADVOICE_CONDITION", "normal")
        self.task_practice = False
        self.layout_locked = False  # a helper chose the layout
        self.camera_index: int | None = None

    # Session control (operator) ---------------------------------------------------------
    def start(self, session_id: str | None = None) -> None:
        with self.lock:
            if self.running:
                return
            self.session_id = session_id or datetime.now().strftime("s%Y%m%d_%H%M%S")
            self.menu_source.reset()
            self.state = sm.initial_state(
                self.profile.layout, self.menu_source.current(self.profile.layout)
            )
            if self.sessions_dir is not None:
                self.logger = SessionLogger(
                    self.sessions_dir / f"session_{self.session_id[1:]}.jsonl",
                    {
                        "source": "app",
                        "session_id": self.session_id,
                        "policy": self.policy,
                        "format": LOG_FORMAT,
                        "condition": self.condition,
                        "layout_locked": self.layout_locked,
                        "profile": self.profile.model_dump(),
                    },
                )
            if self.policy in ("rule", "agent"):
                traces = TraceStore(
                    self.sessions_dir / f"traces_{self.session_id[1:]}.jsonl"
                    if self.sessions_dir
                    else None,
                    remote=self.cloud.audit_put if self.cloud else None,
                )
                self.adapter = Adapter(
                    self.profile,
                    self.session_id,
                    self.policy,
                    transport=self.transport,
                    traces=traces,
                    background=self.background_agent,
                )
                if self.layout_locked:
                    self.adapter.applier.locked.add("layout")
            self.running = True
            self._event({"type": "session_start", "t": self._t(), "policy": self.policy})

    def stop(self) -> None:
        with self.lock:
            if not self.running:
                return
            self._event({"type": "session_stop", "t": self._t()})
            self.running = False
            self.adapter = None
            self.task, self.task_i = [], 0
            if self.logger:
                self.logger.close()
                self.logger = None

    def set_neutral(self) -> bool:
        """Operator: take the current head pose as neutral."""
        with self.lock:
            f = self.last_features
            if f is None or f.yaw is None:
                return False
            if self.neutral_hook is not None:
                self.neutral_hook()  # the camera thread applies it on its next frame
                self._event({"type": "neutral_requested", "t": self._t()})
                return True
            n = self.profile.neutral
            self.profile.neutral = Neutral(
                yaw=n.yaw + f.yaw, pitch=n.pitch + f.pitch, roll=n.roll + f.roll
            )
            self.profile.neutral_set = True
            self._event(
                {"type": "neutral_set", "t": self._t(), **self.profile.neutral.model_dump()}
            )
            return True

    # Scripted task (evaluation ground truth) --------------------------------------------
    def start_task(
        self,
        n: int = 10,
        seed: int | None = None,
        targets: list[str] | None = None,
        practice: bool = False,
    ) -> None:
        with self.lock:
            ids = [p.id for p in self.builtin.phrases]  # built-in only: comparable recordings
            rnd = random.Random(seed)
            self.task = targets or rnd.sample(ids, min(n, len(ids)))
            self.task_i = 0
            self.task_practice = practice
            ev = {"type": "task_start", "t": self._t(), "targets": self.task}
            if practice:
                ev["practice"] = True  # onboarding practice: not an evaluation task
            self._event(ev)
            self._target_event()

    def start_practice(self) -> None:
        """Onboarding: two easy targets from the menu on screen (left and right in
        four_way, so no paging is needed)."""
        with self.lock:
            if not self.running:
                self.start()
            items = list(self.state.menu.items)
            pick = [1, 2] if self.state.menu.layout == "four_way" and len(items) >= 3 else [0, 1]
            self.start_task(targets=[items[i] for i in pick if i < len(items)], practice=True)

    def set_layout(self, layout: str) -> bool:
        """Helper panel: choose the layout. The agent will not change it for this session."""
        if layout not in ("four_way", "two_choice", "binary_scan"):
            return False
        with self.lock:
            self.profile.layout = layout
            if self.adapter is not None:
                self.adapter.applier.locked.add("layout")
            self.layout_locked = True
            self._event({"type": "layout_set_by_helper", "t": self._t(), "layout": layout})
            self._sync_layout(self._t())
            return True

    def set_condition(self, condition: str) -> bool:
        """Research drawer: label the recording condition (written to the next session log)."""
        if condition not in CONDITIONS:
            return False
        with self.lock:
            self.condition = condition
            self._event({"type": "condition", "t": self._t(), "condition": condition})
            return True

    def skip_target(self) -> None:
        with self.lock:
            if self.task_i < len(self.task):
                self._event({"type": "task_skip", "t": self._t(), "item": self.task[self.task_i]})
                self.task_i += 1
                self._target_event()

    def _target_event(self) -> None:
        if self.task_i < len(self.task):
            self._event({"type": "target", "t": self._t(), "item": self.task[self.task_i]})
        elif self.task:
            self._event({"type": "task_done", "t": self._t()})

    # Calibration ----------------------------------------------------------------------
    def start_calibration(self, upload: bool, tmp_dir: Path | None = None) -> None:
        """Operator, after the consent screen. upload=False keeps everything on the device."""
        with self.lock:
            clip = None
            if upload and self.cloud is not None and tmp_dir is not None:
                clip = tmp_dir / f"calib_{int(time.time())}.avi"
            self.calib = CalibrationRecorder(self._t(), clip)
            self.calib_upload = clip is not None
            self.calib_result = None
            self._event({"type": "calibration_start", "t": self._t(), "upload": self.calib_upload})

    def cancel_calibration(self) -> None:
        with self.lock:
            if self.calib:
                path = self.calib.finish()
                _delete_clip(path)
                self.calib = None
                self._event({"type": "calibration_cancelled", "t": self._t()})

    def _finish_calibration(self) -> None:
        rec, self.calib = self.calib, None
        frames = rec.frames
        clip = rec.finish()
        base = self.profile.model_copy(deep=True)
        local = analyze(frames, base)
        result, source = local, "local"
        if clip is not None:
            try:
                remote = self.cloud.calibrate(clip, rec.timestamps, base.model_dump())
                if remote.get("ok"):
                    result = type(local)(
                        Profile.model_validate(remote["profile"]),
                        remote["report"],
                        True,
                        remote.get("warnings", []),
                    )
                    source = "cloud"
            except Exception as e:  # offline: keep the local result
                self._event(
                    {"type": "calibration_cloud_failed", "t": self._t(), "error": type(e).__name__}
                )
            finally:
                _delete_clip(clip)
        if result.ok:
            for k in CALIBRATED_FIELDS:
                setattr(self.profile, k, getattr(result.profile, k))
            if self.sessions_dir is not None:
                out = self.sessions_dir.parent / "profiles"
                out.mkdir(parents=True, exist_ok=True)
                (out / f"profile_{int(time.time())}.json").write_text(
                    self.profile.model_dump_json(indent=2)
                )
            self.state, _ = sm.set_menu(
                self.state, self.profile.layout, self.menu_source.current(self.profile.layout)
            )
        self.calib_result = {
            "ok": result.ok,
            "source": source,
            "warnings": result.warnings,
            "report": result.report,
        }
        self._event(
            {
                "type": "calibration_done",
                "t": self._t(),
                **self.calib_result,
                "layout": self.profile.layout,
            }
        )
        self.outbox.append(
            {
                "type": "notice",
                "text": f"Calibration {'done' if result.ok else 'not applied'}: "
                f"layout {self.profile.layout}"
                + (f" ({', '.join(result.warnings)})" if result.warnings else ""),
            }
        )

    # Frame and UI inputs -------------------------------------------------------------
    def handle_features(self, f: FrameFeatures, now: float, bgr=None) -> None:
        with self.lock:
            self.last_features = f
            if self.calib is not None:
                self.calib.add(bgr, f)
                if self.calib.done_at(f.t):
                    if self.calib_upload:
                        # Cloud calibration can take seconds: run it off the camera thread.
                        threading.Thread(target=self._finish_locked, daemon=True).start()
                    else:
                        self._finish_calibration()
                return
            if not self.running:
                return
            if self.logger:
                self.logger.frame(f)
            self.state, events = sm.step(self.state, f, self.profile, now)
            for ev in events:
                self._on_event(ev)
            if self.adapter is not None:
                self.adapter.on_frame(f)
                for rec in self.adapter.tick(now):
                    self._event(
                        {
                            "type": "agent_decision",
                            "t": now,
                            "trace_id": rec.trace_id,
                            "policy": rec.policy,
                            "trigger": rec.trigger,
                            "actions": [a["tool"] for a in rec.actions],
                            "applied": [a["applied"] for a in rec.applied],
                            "rejected": rec.rejected_reason,
                            "reason": rec.reason,
                        }
                    )
                while self.adapter.notices:
                    n = self.adapter.notices.pop(0)
                    text = (
                        "Please hold still"
                        if n["tool"] == "request_hold_still"
                        else "Recalibration suggested"
                    )
                    self.outbox.append({"type": "notice", "text": text})
            self._sync_layout(now)

    def _finish_locked(self) -> None:
        with self.lock:
            if self.calib is not None:
                self._finish_calibration()

    # The user's own messages ------------------------------------------------------------
    def text(self, phrase_id: str) -> str:
        try:
            return self.phrasebook.text(phrase_id)
        except KeyError:
            return self._retired_text.get(phrase_id, "")

    def messages(self) -> dict[str, Any]:
        with self.lock:
            return {
                "custom": [p.model_dump() for p in self.custom.items],
                "builtin": [p.model_dump() for p in self.builtin.phrases],
            }

    def add_message(self, text: str) -> dict:
        with self.lock:
            phrase = self.custom.add(text)  # raises ValueError with a friendly message
            self._rebuild_phrases()
            self._event({"type": "message_added", "t": self._t(), "item": phrase.id})
            return phrase.model_dump()

    def remove_message(self, phrase_id: str) -> bool:
        with self.lock:
            text = self.text(phrase_id)
            if not self.custom.remove(phrase_id):
                return False
            self._retired_text[phrase_id] = text
            self._rebuild_phrases()
            self._event({"type": "message_removed", "t": self._t(), "item": phrase_id})
            return True

    def _rebuild_phrases(self) -> None:
        self.phrasebook = merged(self.builtin, self.custom.items)
        self.menu_source = MenuSource(self.phrasebook)
        self.menu_dirty = True
        self._sync_layout(self._t())

    def _sync_layout(self, now: float) -> None:
        """Apply a layout change (calibration or agent) once no gesture is in progress."""
        if self.state.menu.layout != self.profile.layout or self.menu_dirty:
            items = self.menu_source.current(self.profile.layout)
            self.state, applied = sm.set_menu(self.state, self.profile.layout, items)
            if applied:
                kind = "menu_refreshed" if self.menu_dirty else "layout_changed"
                self.menu_dirty = False
                self._event({"type": kind, "t": now, "layout": self.profile.layout})

    def speech_ended(self, now: float) -> None:
        with self.lock:
            self.state, events = sm.speech_ended(self.state, now)
            for ev in events:
                self._on_event(ev)

    def _on_event(self, ev: dict[str, Any]) -> None:
        self._event(ev)
        if self.adapter is not None:
            self.adapter.on_event(ev)
        if ev["type"] == "speak":
            item = ev["item"]
            self.outbox.append({"type": "speak", "item": item, "text": self.text(item)})
            if self.task_i < len(self.task):
                target = self.task[self.task_i]
                if item == target:
                    self._event({"type": "task_hit", "t": ev["t"], "item": item})
                    self.task_i += 1
                    self._target_event()
                else:
                    self._event({"type": "task_miss", "t": ev["t"], "item": item, "target": target})
        elif ev["type"] == sm.NONE_OF_THESE:
            items = self.menu_source.next_set(self.profile.layout)
            self.state, applied = sm.set_menu(self.state, self.profile.layout, items)
            self._event({"type": "menu_changed", "t": ev["t"], "items": items, "applied": applied})

    def notify(self, text: str) -> None:
        with self.lock:
            self.outbox.append({"type": "notice", "text": text})

    def _event(self, ev: dict[str, Any]) -> None:
        self.events.append(ev)
        if self.logger:
            self.logger.event(ev)

    def _t(self) -> float:
        return self.last_features.t if self.last_features else 0.0

    # Snapshot for the UI ------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            s, f, p = self.state, self.last_features, self.profile
            armed = 0.0
            if s.state == sm.ARMED and s.closed_since is not None and f is not None:
                armed = min(1.0, (f.t - s.closed_since) * 1000 / p.long_blink_ms)
            menu = s.menu.view()
            menu["texts"] = [self.text(i) for i in menu["items"]]
            feats = f.model_dump(exclude={"landmarks", "bbox"}) if f is not None else None
            decisions = []
            if self.adapter is not None:
                for r in self.adapter.traces.records[-5:]:
                    decisions.append(
                        {
                            "t": round(r.t, 1),
                            "trigger": r.trigger,
                            "policy": r.policy,
                            "actions": [
                                {"tool": a["tool"], "args": a.get("args", {})} for a in r.actions
                            ],
                            "applied": [a["applied"] for a in r.applied],
                            "rejected": r.rejected_reason,
                            "reason": r.reason,
                            "tool_calls": [c["tool"] for c in r.tool_calls],
                            "evidence": {
                                k: r.input_summary.get(k)
                                for k in (
                                    "quality_mean",
                                    "undos",
                                    "timeouts",
                                    "tilt_near_misses",
                                    "tilt_amplitude_vs_calibrated",
                                    "amplitude_trend_per_min",
                                )
                            },
                            "outcome_after": r.outcome_after
                            and {
                                k: r.outcome_after.get(k)
                                for k in (
                                    "selections",
                                    "undos",
                                    "timeouts",
                                    "tilt_near_misses",
                                    "mean_selection_time_s",
                                )
                            },
                        }
                    )
            calib = None
            if self.calib is not None and f is not None:
                calib = self.calib.prompt_at(f.t)
            target = self.task[self.task_i] if self.task_i < len(self.task) else None
            return {
                "type": "snapshot",
                "running": self.running,
                "session_id": self.session_id,
                "state": s.state,
                "menu": menu,
                "armed_progress": armed,
                "quality": f.quality if f else 0.0,
                "flags": f.quality_flags if f else [],
                "features": feats,
                "online": self.online,
                "policy": self.policy,
                "camera_ok": self.camera_ok,
                "fps": round(self.fps, 1),
                "layout": p.layout,
                "params": {
                    "tilt_hold_ms": p.tilt_hold_ms,
                    "highlight_timeout_ms": p.highlight_timeout_ms,
                    "long_blink_ms": p.long_blink_ms,
                    "tilt_scale": p.tilt_scale,
                    "undo_gesture": p.undo_gesture,
                    "profile": p.profile_id,
                },
                "events": list(self.events)[-15:],
                "agent_decisions": decisions,
                "calibration": calib,
                "camera_index": self.camera_index,
                "condition": self.condition,
                "layout_locked": self.layout_locked,
                "calibration_result": self.calib_result,
                "task": {
                    "target": target,
                    "text": self.text(target) if target else None,
                    "i": self.task_i,
                    "n": len(self.task),
                    "practice": self.task_practice,
                }
                if self.task
                else None,
            }

    def drain_outbox(self) -> list[dict[str, Any]]:
        with self.lock:
            out = list(self.outbox)
            self.outbox.clear()
            return out


def _delete_clip(path: Path | None) -> None:
    if path is None:
        return
    for p in (path, path.with_suffix(".json")):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def controller_from_env(root: Path) -> Controller:
    from headvoice.io.cloud import CloudClient

    profile_path = os.environ.get(
        "HEADVOICE_PROFILE", str(root / "config" / "default_profile.json")
    )
    cloud = CloudClient.from_env()
    # Offline installs still adapt on the device with the rule policy.
    policy = os.environ.get("HEADVOICE_POLICY", "agent" if cloud else "rule")
    transport = cloud.supervise if (cloud and policy == "agent") else None
    if policy == "agent" and transport is None:
        policy = "rule"  # no cloud: the deterministic baseline still adapts offline
    return Controller(
        Profile.load(profile_path),
        Phrasebook.load(root / "config" / "phrasebook_en.json"),
        sessions_dir=root / "data" / "sessions",
        custom_path=root / "data" / "custom_messages.json",
        policy=policy,
        transport=transport,
        cloud=cloud,
    )
