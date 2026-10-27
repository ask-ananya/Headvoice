"""Adaptation loop on the device: perceive (monitor) -> decide (rule or agent) ->
validate and act (tools.AgentApplier) -> verify (outcome window) -> trace.

The same class runs live (agent calls in a background thread, never blocking frames) and
in replay (synchronous, deterministic for the rule policy).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from headvoice.agent import rule_policy
from headvoice.agent.monitor import SessionMonitor
from headvoice.agent.tools import READ_ONLY, TIERS, AgentApplier, Bounds, ToolCall
from headvoice.agent.trace import TraceRecord, TraceStore
from headvoice.io.privacy import check_outbound
from headvoice.io.schemas import FrameFeatures, Profile

# transport(request) -> supervisor response dict (raises on network failure)
Transport = Callable[[dict], dict]

PARAMS = ("tilt_hold_ms", "highlight_timeout_ms", "long_blink_ms", "refractory_ms", "tilt_scale")


class Adapter:
    def __init__(
        self,
        profile: Profile,
        session_id: str,
        policy: str,  # "rule" or "agent"
        transport: Transport | None = None,
        bounds: Bounds | None = None,
        traces: TraceStore | None = None,
        background: bool = False,
        outcome_window_s: float = 60.0,
        monitor: SessionMonitor | None = None,
    ):
        assert policy in ("rule", "agent")
        self.policy = policy
        self.transport = transport
        self.bounds = bounds or Bounds.load()
        self.applier = AgentApplier(profile, self.bounds, session_id)
        self.monitor = monitor or SessionMonitor()
        self.traces = traces or TraceStore()
        self.background = background
        self.outcome_window_s = outcome_window_s
        self.session_id = session_id
        self._pending: list[TraceRecord] = []  # waiting for outcome_after
        self._inbox: queue.Queue = queue.Queue()  # (record, response or exception)
        self._in_flight = False
        self.notices: list[dict] = []  # UI messages (hold still, recalibration prompt)
        self.stale_discards = 0

    @property
    def profile(self) -> Profile:
        return self.applier.profile

    def on_frame(self, f: FrameFeatures) -> None:
        self.monitor.add_frame(f, self.profile)

    def on_event(self, ev: dict) -> None:
        self.monitor.add_event(ev)

    # The loop -------------------------------------------------------------------------
    def tick(self, now: float) -> list[TraceRecord]:
        """Call once per frame. Returns traces finished this tick (applied or rejected)."""
        done = self._drain(now)
        self._outcomes(now)
        if self._in_flight:
            return done
        trigger = self.monitor.due(now, self.profile)
        if trigger is None:
            return done
        if self._pending and trigger != "quality_drop":
            # Verify before acting again: wait until the last change's outcome window has
            # been measured. A signal problem (quality drop) is a different problem.
            return done
        request = self._request(now, trigger)
        rec = TraceRecord(
            session_id=self.session_id,
            epoch=request["epoch"],
            t=now,
            trigger=trigger,
            policy=self.policy,
            input_summary=request["window"],
        )
        if self.policy == "rule":
            calls = rule_policy.decide(request["window"], self.profile)
            resp = {
                "epoch": request["epoch"],
                "session_id": self.session_id,
                "actions": [c.model_dump() for c in calls],
                "tool_calls": [],
                "reason": calls[0].reason,
            }
            done.append(self._apply(rec, resp, now))
        elif self.transport is not None:
            self._in_flight = True
            if self.background:
                threading.Thread(target=self._call, args=(rec, request), daemon=True).start()
            else:
                self._call(rec, request)
                done += self._drain(now)
        return done

    def _request(self, now: float, trigger: str) -> dict:
        p = self.profile
        epoch = self.applier.guard.next_epoch()
        history = [
            {
                "epoch": r.epoch,
                "actions": [a["tool"] for a in r.actions],
                "applied": [a.get("applied") for a in r.applied],
                "outcome_after": r.outcome_after,
            }
            for r in self.traces.records[-3:]
        ]
        req = {
            "session_id": self.session_id,
            "epoch": epoch,
            "profile_id": p.profile_id,
            "trigger": trigger,
            "current_layout": p.layout,
            "layout_locked": "layout" in self.applier.locked,
            "current_params": {k: getattr(p, k) for k in PARAMS},
            "window": self.monitor.summary(now, p),
            "history": history,
            "evidence": {
                "windows": {str(s): self.monitor.summary(now, p, seconds=s) for s in (20, 60, 300)},
                "direction_reliability": self.monitor.direction_reliability(now, p),
            },
        }
        check_outbound(req)  # numbers and short labels only
        return req

    def _call(self, rec: TraceRecord, request: dict) -> None:
        try:
            self._inbox.put((rec, self.transport(request)))
        except Exception as e:  # network or model failure: keep current settings
            self._inbox.put((rec, e))

    def _drain(self, now: float) -> list[TraceRecord]:
        done = []
        while True:
            try:
                rec, resp = self._inbox.get_nowait()
            except queue.Empty:
                return done
            self._in_flight = False
            if isinstance(resp, Exception):
                rec.rejected_reason = f"request_failed: {type(resp).__name__}"
                self.traces.add(rec)
                done.append(rec)
            else:
                done.append(self._apply(rec, resp, now))

    def _apply(self, rec: TraceRecord, resp: dict, now: float) -> TraceRecord:
        calls = [ToolCall(**a) for a in resp.get("actions", [])]
        rec.actions = [c.model_dump() for c in calls]
        rec.tool_calls = resp.get("tool_calls", [])
        rec.reason = resp.get("reason", "")
        rec.latency_ms = resp.get("latency_ms")
        rec.usage = resp.get("usage", {})
        before = {k: getattr(self.profile, k) for k in ("layout", *PARAMS)}
        records = self.applier.apply(
            calls, resp.get("session_id", ""), int(resp.get("epoch", -1)), now
        )
        rec.applied = records
        rec.validated = all(r["rejected_reason"] is None for r in records)
        rec.rejected_reason = next(
            (r["rejected_reason"] for r in records if r["rejected_reason"]), None
        )
        if rec.rejected_reason == "stale_epoch":
            self.stale_discards += 1
        for r in records:
            if r["applied"] and r["tool"] in ("request_hold_still", "request_recalibration"):
                self.notices.append({"tool": r["tool"], "reason": rec.reason, "t": now})
        changed = any(r["applied"] and TIERS.get(r["tool"]) != READ_ONLY for r in records)
        rec.input_summary = {**rec.input_summary, "params_before": before}
        self.traces.add(rec)
        if changed:
            self._pending.append(rec)
        return rec

    def _outcomes(self, now: float) -> None:
        for rec in list(self._pending):
            if now - rec.t >= self.outcome_window_s:
                rec.outcome_after = {
                    **self.monitor.summary(now, self.profile, seconds=self.outcome_window_s),
                    "window_s": self.outcome_window_s,
                }
                self.traces.update(rec)
                self._pending.remove(rec)
