"""Webcam window with landmarks, pose, eye openness, quality and FPS overlaid.

Keys: q quit, n take current pose as neutral, r start/stop JSONL recording.

    python scripts/live_debug.py            # first live camera
    python scripts/live_debug.py --camera 1 # a specific one
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from headvoice.io.logger import SessionLogger
from headvoice.io.schemas import Profile
from headvoice.vision.blink import BlinkDetector
from headvoice.vision.camera import open_live_camera
from headvoice.vision.pipeline import VisionPipeline

ROOT = Path(__file__).resolve().parent.parent


def fmt(v: float | None, spec: str = "+6.1f") -> str:
    return "  --  " if v is None else format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None, help="default: first live camera")
    ap.add_argument("--profile", default=str(ROOT / "config" / "default_profile.json"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    profile = Profile.load(args.profile)
    pipe = VisionPipeline(profile)
    blinks = BlinkDetector(profile.eye_closed_ratio)
    cap, index = open_live_camera(args.camera, args.width, args.height)
    print(f"using camera {index}")
    t_start = time.monotonic()
    logger: SessionLogger | None = None
    last_blink = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break
        t = time.monotonic() - t_start
        ff = pipe.process(frame, t)
        ev = blinks.update(ff.eye_open_l, ff.eye_open_r, t)
        if ev:
            kind = "LONG" if ev.duration_ms >= profile.long_blink_ms else "short"
            last_blink = (
                f"{kind} {ev.duration_ms:.0f} ms{'' if ev.agreed else ' (one eye, rejected)'}"
            )
            print(f"blink t={t:.2f} {last_blink}")
            if logger:
                logger.event(
                    {"type": "blink", "t": t, "duration_ms": ev.duration_ms, "agreed": ev.agreed}
                )
        if logger:
            logger.frame(ff)

        vis = frame.copy()
        if ff.bbox:
            x, y, w, h = ff.bbox
            color = (0, 200, 0) if ff.landmark_source == "model" else (0, 165, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        for px, py in ff.landmarks or []:
            cv2.circle(vis, (int(px), int(py)), 1, (255, 255, 0), -1)
        lines = [
            f"yaw {fmt(ff.yaw)}  pitch {fmt(ff.pitch)}  roll {fmt(ff.roll)}",
            f"eye L {fmt(ff.eye_open_l, '5.2f')}  R {fmt(ff.eye_open_r, '5.2f')}"
            f"  last blink: {last_blink}",
            f"quality {ff.quality:.2f} {','.join(ff.quality_flags)}  src {ff.landmark_source}"
            f"  clahe {'on' if ff.clahe_applied else 'off'}",
            f"fps {pipe.fps:4.1f}  "
            + "  ".join(f"{k} {v:.1f}" for k, v in pipe.stage_ms.items())
            + (f"  REC {logger.path.name}" if logger else ""),
        ]
        for i, line in enumerate(lines):
            y = 22 + 22 * i
            cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("HeadVoice live debug", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("n"):
            pipe.request_neutral()
            print("neutral will be taken from the next steady frame")
        if key == ord("r"):
            if logger:
                logger.close()
                print(f"saved {logger.path}")
                logger = None
            else:
                name = datetime.now().strftime("debug_%Y%m%d_%H%M%S.jsonl")
                logger = SessionLogger(ROOT / "data" / "sessions" / name, {"source": "live_debug"})

    if logger:
        logger.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
