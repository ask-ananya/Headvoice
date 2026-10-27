"""Agent tool tiers, bounds, stale-result guard and action cap, enforced in code.

The agent can only change a Profile's tunable parameters and layout, within
config/agent_bounds.json. Nothing here can read or write the gesture state machine.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from headvoice.io.schemas import Profile

READ_ONLY = "read_only"
REVERSIBLE = "reversible"
NEEDS_CONFIRMATION = "needs_confirmation"
FORBIDDEN = "forbidden"

TIERS: dict[str, str] = {
    "get_window_summary": READ_ONLY,
    "get_direction_reliability": READ_ONLY,
    "propose_candidates": READ_ONLY,
    "no_change": READ_ONLY,
    "set_layout": REVERSIBLE,
    "adjust_timing": REVERSIBLE,
    "adjust_sensitivity": REVERSIBLE,
    "request_hold_still": REVERSIBLE,
    "update_calibration_profile": REVERSIBLE,
    "request_recalibration": NEEDS_CONFIRMATION,
    "speak_message": NEEDS_CONFIRMATION,
    "send_caregiver_alert": NEEDS_CONFIRMATION,
    "access_raw_video": FORBIDDEN,
}
# Needs-confirmation tools the agent may never call at all; only the user's gesture unlocks them.
AGENT_NEVER: frozenset[str] = frozenset({"speak_message", "send_caregiver_alert"})
# Tools that change the interface and count against the action cap.
ADAPTATIONS: frozenset[str] = frozenset(
    {
        "set_layout",
        "adjust_timing",
        "adjust_sensitivity",
        "update_calibration_profile",
        "request_recalibration",
    }
)

ROLLBACK_FIELDS = (
    "layout",
    "tilt_scale",
    "tilt_hold_ms",
    "highlight_timeout_ms",
    "long_blink_ms",
    "refractory_ms",
)

DEFAULT_BOUNDS = Path(__file__).resolve().parents[3] / "config" / "agent_bounds.json"


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class Bounds(BaseModel):
    layouts: list[str]
    params: dict[str, dict[str, float]]
    max_actions_per_window: int
    action_window_s: float
    max_tool_calls_per_invocation: int
    request_timeout_s: float

    @classmethod
    def load(cls, path: str | Path = DEFAULT_BOUNDS) -> Bounds:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_call(call: ToolCall, bounds: Bounds) -> tuple[bool, str | None]:
    """Return (ok, rejection reason)."""
    tier = TIERS.get(call.tool)
    if tier is None:
        return False, "unknown_tool"
    if tier == FORBIDDEN:
        return False, "forbidden_tool"
    if call.tool in AGENT_NEVER:
        return False, "agent_may_not_call"
    if call.tool == "set_layout":
        if call.args.get("layout") not in bounds.layouts:
            return False, "layout_out_of_bounds"
    elif call.tool == "adjust_sensitivity":
        scale = call.args.get("scale")
        rng = bounds.params["tilt_scale"]
        if not isinstance(scale, int | float) or isinstance(scale, bool):
            return False, "value_not_numeric"
        if not rng["min"] <= scale <= rng["max"]:
            return False, "value_out_of_bounds"
    elif call.tool == "adjust_timing":
        param, value = call.args.get("param"), call.args.get("value")
        rng = bounds.params.get(str(param))
        if rng is None or param == "tilt_scale":
            return False, "param_not_tunable"
        if not isinstance(value, int | float) or isinstance(value, bool):
            return False, "value_not_numeric"
        if not rng["min"] <= value <= rng["max"]:
            return False, "value_out_of_bounds"
    elif call.tool == "update_calibration_profile":
        return False, "not_implemented"
    return True, None


class StaleGuard:
    """Every request carries session_id and a rising epoch; older answers are discarded."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.epoch = 0

    def next_epoch(self) -> int:
        self.epoch += 1
        return self.epoch

    def accept(self, session_id: str, epoch: int) -> tuple[bool, str | None]:
        if session_id != self.session_id:
            return False, "wrong_session"
        if epoch < self.epoch:
            return False, "stale_epoch"
        if epoch > self.epoch:
            return False, "future_epoch"
        return True, None


class ActionCap:
    def __init__(self, max_actions: int, window_s: float):
        self.max_actions = max_actions
        self.window_s = window_s
        self._times: deque[float] = deque()

    def allow(self, now: float) -> bool:
        while self._times and now - self._times[0] > self.window_s:
            self._times.popleft()
        if len(self._times) >= self.max_actions:
            return False
        self._times.append(now)
        return True


class AgentApplier:
    """Applies validated agent tool calls to a Profile. Keeps the previous one for rollback."""

    def __init__(self, profile: Profile, bounds: Bounds, session_id: str):
        self.profile = profile
        self.previous: Profile | None = None
        self.bounds = bounds
        self.guard = StaleGuard(session_id)
        self.cap = ActionCap(bounds.max_actions_per_window, bounds.action_window_s)
        self.ui_requests: list[dict[str, Any]] = []  # hold-still and recalibration prompts
        self.locked: set[str] = set()  # settings a person chose; the agent may not change them

    def apply(
        self, calls: list[ToolCall], session_id: str, epoch: int, now: float
    ) -> list[dict[str, Any]]:
        """Validate and apply. Returns one record per call for the trace."""
        ok, why = self.guard.accept(session_id, epoch)
        if not ok:
            return [{"tool": c.tool, "applied": False, "rejected_reason": why} for c in calls]
        records = []
        for i, call in enumerate(calls):
            rec: dict[str, Any] = {"tool": call.tool, "args": call.args, "applied": False}
            if i >= self.bounds.max_tool_calls_per_invocation:
                rec["rejected_reason"] = "too_many_calls"
                records.append(rec)
                continue
            valid, why = validate_call(call, self.bounds)
            if valid and call.tool == "set_layout" and "layout" in self.locked:
                valid, why = False, "layout_set_by_helper"
            if valid and call.tool in ADAPTATIONS and not self.cap.allow(now):
                valid, why = False, "action_cap"
            if not valid:
                rec["rejected_reason"] = why
                records.append(rec)
                continue
            if call.tool == "set_layout":
                self._set(layout=call.args["layout"])
            elif call.tool == "adjust_timing":
                self._set(**{call.args["param"]: int(call.args["value"])})
            elif call.tool == "adjust_sensitivity":
                self._set(tilt_scale=float(call.args["scale"]))
            elif call.tool in ("request_hold_still", "request_recalibration"):
                self.ui_requests.append({"tool": call.tool, "reason": call.reason, "t": now})
            rec["applied"] = True
            rec["rejected_reason"] = None
            records.append(rec)
        return records

    def _set(self, **changes: Any) -> None:
        # In place: the vision pipeline and controller share this profile object.
        self.previous = self.profile.model_copy(deep=True)
        for k, v in changes.items():
            setattr(self.profile, k, v)

    def rollback(self) -> bool:
        """One-step rollback of the last reversible change (neutral is left alone)."""
        if self.previous is None:
            return False
        for k in ROLLBACK_FIELDS:
            setattr(self.profile, k, getattr(self.previous, k))
        self.previous = None
        return True
