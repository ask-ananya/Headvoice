"""JSONL session logs: one header line, then frame and event records. Never raw video."""

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from headvoice.io.schemas import FrameFeatures


class SessionLogger:
    def __init__(self, path: str | Path, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self._write({"kind": "header", "created": time.time(), **(meta or {})})

    def _write(self, rec: dict[str, Any]) -> None:
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def frame(self, ff: FrameFeatures) -> None:
        self._write({"kind": "frame", **ff.model_dump()})

    def event(self, ev: BaseModel | dict[str, Any]) -> None:
        data = ev.model_dump() if isinstance(ev, BaseModel) else ev
        self._write({"kind": "event", **data})

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


def read_session(path: str | Path) -> tuple[dict, list[FrameFeatures], list[dict]]:
    header: dict = {}
    frames: list[FrameFeatures] = []
    events: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            kind = rec.pop("kind")
            if kind == "header":
                header = rec
            elif kind == "frame":
                frames.append(FrameFeatures.model_validate(rec))
            else:
                events.append(rec)
    return header, frames, events
