"""Recording the guided calibration clip in the app, and turning a clip back into features.

Local calibration uses the live features directly. With the user's consent the raw clip
is also written (with per-frame timestamps) so it can be uploaded for cloud calibration;
the file is deleted after upload, and the cloud deletes it after processing.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from headvoice.calibration.protocol import STEPS, label_at, total_seconds
from headvoice.io.schemas import FrameFeatures, Profile


class CalibrationRecorder:
    def __init__(self, start_t: float, clip_path: Path | None = None):
        self.start_t = start_t
        self.clip_path = clip_path  # None = no consent to upload: nothing is written
        self.frames: list[FrameFeatures] = []
        self.timestamps: list[float] = []
        self._writer: cv2.VideoWriter | None = None

    def add(self, bgr, features: FrameFeatures) -> None:
        rel = features.t - self.start_t
        if rel < 0 or self.done_at(features.t):
            return
        self.frames.append(features.model_copy(update={"t": rel}))
        if self.clip_path is not None:
            if self._writer is None:
                h, w = bgr.shape[:2]
                self.clip_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self._writer = cv2.VideoWriter(str(self.clip_path), fourcc, 30.0, (w, h))
            self._writer.write(bgr)
            self.timestamps.append(rel)

    def done_at(self, t: float) -> bool:
        return t - self.start_t >= total_seconds()

    def prompt_at(self, t: float) -> dict:
        rel = t - self.start_t
        lab = label_at(rel)
        if lab is None:
            return {"step": len(STEPS), "of": len(STEPS), "label": "done", "prompt": "Done"}
        i = lab[1]
        return {
            "step": i + 1,
            "of": len(STEPS),
            "label": STEPS[i].label,
            "prompt": STEPS[i].prompt,
            "progress": min(1.0, rel / total_seconds()),
        }

    def finish(self) -> Path | None:
        """Close the clip and write its timestamps beside it. Returns the clip path."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            ts = self.clip_path.with_suffix(".json")
            ts.write_text(json.dumps({"t": self.timestamps}), encoding="utf-8")
            return self.clip_path
        return None


def features_from_clip(
    clip: str | Path, timestamps: list[float], base: Profile
) -> list[FrameFeatures]:
    """Run the vision pipeline over a recorded clip. Used by the calibrate Lambda."""
    from headvoice.vision.pipeline import VisionPipeline

    profile = base.model_copy(update={"neutral_set": False})  # relearn from the clip
    pipe = VisionPipeline(profile)
    cap = cv2.VideoCapture(str(clip))
    out: list[FrameFeatures] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(timestamps):
            break
        out.append(pipe.process(frame, timestamps[i]))
        i += 1
    cap.release()
    return out
