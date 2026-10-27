"""Decision trace records: what was measured, what the agent asked and did, what happened.

`outcome_after` is filled in by the device one window after an action was applied. This is
the evidence that OpenCV output changed a later decision and that the change was checked.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field


class TraceRecord(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str
    epoch: int
    ts: float = Field(default_factory=time.time)
    t: float  # session time
    trigger: str
    policy: str  # "agent" or "rule"
    input_summary: dict
    tool_calls: list[dict] = Field(default_factory=list)
    actions: list[dict] = Field(default_factory=list)
    reason: str = ""
    applied: list[dict] = Field(default_factory=list)  # validation result per action
    validated: bool = False
    rejected_reason: str | None = None
    latency_ms: int | None = None
    usage: dict = Field(default_factory=dict)
    outcome_after: dict | None = None


class TraceStore:
    """Local JSONL (always) plus an optional remote sink (the /audit endpoint)."""

    def __init__(self, path: str | Path | None = None, remote=None):
        self.path = Path(path) if path else None
        self.remote = remote
        self.records: list[TraceRecord] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, rec: TraceRecord) -> None:
        self.records.append(rec)
        self._write(rec)

    def update(self, rec: TraceRecord) -> None:
        self._write(rec)

    def _write(self, rec: TraceRecord) -> None:
        data = rec.model_dump()
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        if self.remote:
            try:
                self.remote(data)
            except Exception:  # the audit sink must never break the device
                pass
